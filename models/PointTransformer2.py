"""Point Transformer V2 — Grouped Vector Attention and Partition-based Pooling.
Wu et al. NeurIPS 2022. arXiv:2210.05666
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List
from timm.models.layers import trunc_normal_, DropPath
from .build import MODELS
from utils.logger import print_log
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.cls_losses import ClassificationLossMixin
from utils.feature_hooks import FeatureExtractionMixin
from utils import misc
import einops

def masked_knn(coord, mask, k):
    """kNN over valid points only -> (B, N, k). GridPool's repeated-cluster padding must
    never be picked as a neighbour (square_distance + topk, no [B, N, N, 3] tensor)."""
    d = misc.square_distance(coord, coord)                       # (B, N, N)
    d = d.masked_fill(~mask.unsqueeze(1), float("inf"))          # drop padded columns
    k = min(k, int(mask.sum(dim=1).min().item()))
    return torch.topk(d, k, dim=-1, largest=False, sorted=True).indices

class PointBatchNorm(nn.Module):
    """BatchNorm over [B*N, C] / [B, N, C] / [B, N, K, C]."""

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 4:  # [B, N, K, C] format
            B, N, K, C = input.shape
            input = input.reshape(B*N*K, C)
            output = self.norm(input)
            return output.reshape(B, N, K, C)
        elif input.dim() == 3:  # [B, N, C] format
            B, N, C = input.shape
            input = input.reshape(B*N, C)
            output = self.norm(input)
            return output.reshape(B, N, C)
        elif input.dim() == 2:  # [B*N, C] format
            return self.norm(input)
        else:
            raise NotImplementedError(f"PointBatchNorm not implemented for {input.dim()}D tensors")

class GroupedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, groups: int, device=None, dtype=None):
        super(GroupedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        assert in_features % groups == 0
        assert out_features % groups == 0
        assert out_features == groups
        self.weight = nn.Parameter(torch.empty((1, in_features), device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (input * self.weight).reshape(
            list(input.shape[:-1]) + [self.groups, input.shape[-1] // self.groups]
        ).sum(-1)

class GroupedVectorAttention(nn.Module):
    def __init__(self, embed_channels, groups, attn_drop_rate=0.0, qkv_bias=True,
                 pe_multiplier=False, pe_bias=True, weight_encoding="linear"):
        super(GroupedVectorAttention, self).__init__()
        self.embed_channels = embed_channels
        self.groups = groups
        assert embed_channels % groups == 0
        self.attn_drop_rate = attn_drop_rate
        self.qkv_bias = qkv_bias
        self.pe_multiplier = pe_multiplier
        self.pe_bias = pe_bias

        self.linear_q = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )
        self.linear_k = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )
        self.linear_v = nn.Linear(embed_channels, embed_channels, bias=qkv_bias)

        if self.pe_multiplier:
            self.linear_p_multiplier = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        if self.pe_bias:
            self.linear_p_bias = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        
        # v2m2 ("recommend", blueprint-primary) weight encoding starts with a
        # plain Linear(C, groups); weight_encoding="grouped" restores the
        # v2m1 GroupedLinear variant for ablation.
        if weight_encoding == "grouped":
            first_layer = GroupedLinear(embed_channels, groups, groups)
        else:
            first_layer = nn.Linear(embed_channels, groups)
        self.weight_encoding = nn.Sequential(
            first_layer,
            PointBatchNorm(groups),
            nn.ReLU(inplace=True),
            nn.Linear(groups, groups),
        )
        self.softmax = nn.Softmax(dim=2)   # normalize over the neighbour axis (dense layout is B, N, K, groups)
        self.attn_drop = nn.Dropout(attn_drop_rate)

    def forward(self, feat, coord, reference_index):
        B, N, C = feat.shape
        K = reference_index.shape[-1]

        query = self.linear_q(feat)
        key = self.linear_k(feat)
        value = self.linear_v(feat)
        
        key = misc.index_points(key, reference_index)              # [B, N, K, C]
        value = misc.index_points(value, reference_index)          # [B, N, K, C]
        coord_expanded = misc.index_points(coord, reference_index)  # [B, N, K, 3]
        pos = coord_expanded - coord.unsqueeze(2)                  # relative positions
        
        relation_qk = key - query.unsqueeze(2)
        
        if self.pe_multiplier:
            pem = self.linear_p_multiplier(pos)
            relation_qk = relation_qk * pem
        if self.pe_bias:
            peb = self.linear_p_bias(pos)
            relation_qk = relation_qk + peb
            value = value + peb

        weight = self.weight_encoding(relation_qk)
        weight = self.attn_drop(self.softmax(weight))

        value = einops.rearrange(value, "b n k (g i) -> b n k g i", g=self.groups)
        feat = torch.einsum("b n k g i, b n k g -> b n g i", value, weight)
        feat = einops.rearrange(feat, "b n g i -> b n (g i)")
        
        return feat

class Block(nn.Module):
    def __init__(self, embed_channels, groups, qkv_bias=True, pe_multiplier=False,
                 pe_bias=True, attn_drop_rate=0.0, drop_path_rate=0.0,
                 weight_encoding="linear"):
        super(Block, self).__init__()
        self.attn = GroupedVectorAttention(
            embed_channels=embed_channels,
            groups=groups,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            weight_encoding=weight_encoding,
        )
        self.fc1 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc3 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.norm1 = PointBatchNorm(embed_channels)
        self.norm2 = PointBatchNorm(embed_channels)
        self.norm3 = PointBatchNorm(embed_channels)
        self.act = nn.ReLU(inplace=True)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, coord, feat, reference_index):
        identity = feat
        feat = self.act(self.norm1(self.fc1(feat)))
        feat = self.attn(feat, coord, reference_index)
        feat = self.act(self.norm2(feat))
        feat = self.norm3(self.fc3(feat))
        feat = identity + self.drop_path(feat)
        feat = self.act(feat)
        return feat

class BlockSequence(nn.Module):
    def __init__(self, depth, embed_channels, groups, neighbours=16, qkv_bias=True,
                 pe_multiplier=False, pe_bias=True, attn_drop_rate=0.0, drop_path_rate=0.0,
                 weight_encoding="linear"):
        super(BlockSequence, self).__init__()
        
        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        elif isinstance(drop_path_rate, float):
            drop_path_rates = [drop_path_rate for _ in range(depth)]
        else:
            drop_path_rates = [0.0 for _ in range(depth)]

        self.neighbours = neighbours
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                embed_channels=embed_channels,
                groups=groups,
                qkv_bias=qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                weight_encoding=weight_encoding,
            )
            self.blocks.append(block)

    def forward(self, coord, feat, mask=None):
        # one kNN per sequence, shared by every block; clamped to the valid point count
        if mask is None:
            mask = coord.new_ones(coord.shape[:2], dtype=torch.bool)
        reference_index = masked_knn(coord, mask, self.neighbours)
        for block in self.blocks:
            feat = block(coord, feat, reference_index)
        return feat

class GridPool(nn.Module):
    """Metric voxel-grid pooling (dense adaptation): MAX-reduce features, MEAN-reduce coords per
    voxel. Clusters right-padded to the batch max; a (B, M) mask marks real ones; the (B, N)
    cluster index is returned for exact map-based unpooling."""

    def __init__(self, in_channels, out_channels, grid_size=0.12, bias=False):
        super(GridPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, coord, feat):
        B, N, _ = coord.shape
        feat = self.act(self.norm(self.fc(feat)))                 # (B, N, C_out)
        C = feat.shape[-1]

        # per-sample min-corner shift, then integer voxel coordinates
        start = coord.min(dim=1, keepdim=True)[0]
        vox = torch.div(coord - start, self.grid_size, rounding_mode="floor").long()  # (B, N, 3)

        pooled_coords, pooled_feats, clusters, counts = [], [], [], []
        for b in range(B):
            code = (vox[b, :, 0] << 42) | (vox[b, :, 1] << 21) | vox[b, :, 2]
            unique, cluster = torch.unique(code, sorted=True, return_inverse=True)
            m = unique.shape[0]

            # max-reduce features, mean-reduce coords per voxel
            p_feat = feat.new_full((m, C), float("-inf"))
            p_feat = p_feat.scatter_reduce(
                0, cluster.unsqueeze(1).expand(-1, C), feat[b],
                reduce="amax", include_self=True)
            p_coord = coord.new_zeros((m, 3)).index_add_(0, cluster, coord[b])
            cnt = torch.bincount(cluster, minlength=m).clamp(min=1).unsqueeze(1)
            p_coord = p_coord / cnt

            pooled_coords.append(p_coord)
            pooled_feats.append(p_feat)
            clusters.append(cluster)
            counts.append(m)

        # pad to the batch-max cluster count by repeating the last cluster
        M = max(counts)
        new_coord = coord.new_zeros((B, M, 3))
        new_feat = feat.new_zeros((B, M, C))
        for b in range(B):
            m = counts[b]
            new_coord[b, :m] = pooled_coords[b]
            new_feat[b, :m] = pooled_feats[b]
            if m < M:
                new_coord[b, m:] = pooled_coords[b][-1]
                new_feat[b, m:] = pooled_feats[b][-1]

        valid = torch.zeros((B, M), dtype=torch.bool, device=coord.device)
        for b in range(B):
            valid[b, :counts[b]] = True

        cluster_map = torch.stack(clusters, dim=0)               # (B, N)
        return new_coord, new_feat, cluster_map, valid

class UnpoolWithSkip(nn.Module):
    """Unpool + skip. backend="map" (default): broadcast each coarse cluster feature back to its
    member points via GridPool's cluster index. backend="interp": 3-NN inverse-distance."""

    def __init__(self, in_channels, skip_channels, out_channels, bias=True,
                 skip=True, backend="map"):
        super(UnpoolWithSkip, self).__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.skip = skip
        assert backend in ("map", "interp")
        self.backend = backend

        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )
        self.proj_skip = nn.Sequential(
            nn.Linear(skip_channels, out_channels, bias=bias),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, coord, feat, skip_coord, skip_feat, cluster=None):
        feat = self.proj(feat)

        if self.backend == "map" and cluster is not None:
            interpolated_feat = misc.index_points(feat, cluster)   # exact inverse of GridPool
        else:
            # 3-NN inverse-distance interpolation (k clamped if the coarse stage has < 3 points)
            _, idx = misc.knn(coord, skip_coord, k=min(3, coord.shape[1]))
            weights = 1.0 / (torch.norm(
                misc.index_points(coord, idx) - skip_coord.unsqueeze(2), dim=-1) + 1e-8)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            interpolated_feat = torch.sum(
                misc.index_points(feat, idx) * weights.unsqueeze(-1), dim=2)

        if self.skip:
            interpolated_feat = interpolated_feat + self.proj_skip(skip_feat)

        return skip_coord, interpolated_feat

@MODELS.register_module()
class PointTransformerV2_Classification(nn.Module, ClassificationLossMixin, FeatureExtractionMixin):
    """PointTransformer V2 classifier (encoder-only by default; the reference has no PT-V2 classifier)."""

    def __init__(self, config, **kwargs):
        super(PointTransformerV2_Classification, self).__init__()
        self.config = config
        self.name = 'PointTransformerV2_Classification'
        self.num_classes = config.cls_dim
        self.num_points = getattr(config, 'num_points', 1024)

        in_channels = getattr(config, 'in_channels', 3)
        patch_embed_depth = getattr(config, 'patch_embed_depth', 1)
        patch_embed_channels = getattr(config, 'patch_embed_channels', 48)
        patch_embed_groups = getattr(config, 'patch_embed_groups', 6)
        patch_embed_neighbours = getattr(config, 'patch_embed_neighbours', 8)
        
        enc_depths = getattr(config, 'enc_depths', [2, 2, 6, 2])
        enc_channels = getattr(config, 'enc_channels', [96, 192, 384, 512])
        enc_groups = getattr(config, 'enc_groups', [12, 24, 48, 64])
        enc_neighbours = getattr(config, 'enc_neighbours', [16, 16, 16, 16])
        
        dec_depths = getattr(config, 'dec_depths', [1, 1, 1, 1])
        dec_channels = getattr(config, 'dec_channels', [48, 96, 192, 384])
        dec_groups = getattr(config, 'dec_groups', [6, 12, 24, 48])
        dec_neighbours = getattr(config, 'dec_neighbours', [16, 16, 16, 16])
        
        # metric voxel sizes per encoder stage (PT-V2 grid pooling)
        grid_sizes = getattr(config, 'grid_sizes', [0.06, 0.12, 0.24, 0.48])
        # use_decoder: true also runs the V2 decoder and classifies from dec_channels[0]
        self.use_decoder = getattr(config, 'use_decoder', False)
        unpool_backend = getattr(config, 'unpool_backend', 'map')
        weight_encoding = getattr(config, 'weight_encoding', 'linear')  # 'linear' (v2m2) | 'grouped' (v2m1)
        attn_qkv_bias = getattr(config, 'attn_qkv_bias', True)
        pe_multiplier = getattr(config, 'pe_multiplier', False)
        pe_bias = getattr(config, 'pe_bias', True)
        attn_drop_rate = getattr(config, 'attn_drop_rate', 0.0)
        drop_path_rate = getattr(config, 'drop_path_rate', 0.3)

        print_log(f'[PointTransformer2] Building model for {self.num_classes} classes', logger='PTv2')
        self.num_stages = len(enc_depths)

        self.patch_embed = nn.Sequential(
            nn.Linear(in_channels, patch_embed_channels, bias=False),
            PointBatchNorm(patch_embed_channels),
            nn.ReLU(inplace=True)
        )
        
        self.patch_blocks = BlockSequence(
            depth=patch_embed_depth,
            embed_channels=patch_embed_channels,
            groups=patch_embed_groups,
            neighbours=patch_embed_neighbours,
            qkv_bias=attn_qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            weight_encoding=weight_encoding,
        )

        enc_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))]
        dec_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(dec_depths))]
        
        enc_channels = [patch_embed_channels] + list(enc_channels)
        # dec_channels is the OUTPUT width per decoder stage, ascending like enc_channels
        # (reference default (48, 96, 192, 384)); a descending list builds an inverted decoder
        assert list(dec_channels) == sorted(dec_channels), (
            f'dec_channels must be ascending (reference order, e.g. [48, 96, 192, 384]); '
            f'got {list(dec_channels)}')
        assert list(dec_groups) == sorted(dec_groups), (
            f'dec_groups must be ascending to match dec_channels; got {list(dec_groups)}')
        dec_channels = list(dec_channels) + [enc_channels[-1]]
        
        self.enc_stages = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        
        for i in range(self.num_stages):
            enc_down = GridPool(
                in_channels=enc_channels[i],
                out_channels=enc_channels[i + 1],
                grid_size=grid_sizes[i],
            )
            enc_blocks = BlockSequence(
                depth=enc_depths[i],
                embed_channels=enc_channels[i + 1],
                groups=enc_groups[i],
                neighbours=enc_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=enc_dp_rates[sum(enc_depths[:i]):sum(enc_depths[:i+1])],
                weight_encoding=weight_encoding,
            )
            self.enc_stages.append(nn.ModuleList([enc_down, enc_blocks]))

            if not self.use_decoder:
                continue

            dec_up = UnpoolWithSkip(
                in_channels=dec_channels[i + 1],
                skip_channels=enc_channels[i],
                out_channels=dec_channels[i],
                backend=unpool_backend,
            )
            dec_blocks = BlockSequence(
                depth=dec_depths[i],
                embed_channels=dec_channels[i],
                groups=dec_groups[i],
                neighbours=dec_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dec_dp_rates[sum(dec_depths[:i]):sum(dec_depths[:i+1])],
                weight_encoding=weight_encoding,
            )
            self.dec_stages.append(nn.ModuleList([dec_up, dec_blocks]))
        
        # classification head (PT-V1 recipe): C -> 256 -> 128 -> num_classes
        head_in = dec_channels[0] if self.use_decoder else enc_channels[-1]
        self.feature_dim = head_in
        self.classifier = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, self.num_classes)
        )
        
        # ce | focal
        self._init_cls_loss(config, self.num_classes)
        self._init_feature_extraction()

    def load_model_from_ckpt(self, ckpt_path):
        if ckpt_path is None:
            print_log('Training PointTransformer2 from scratch!!!', logger='PTv2')
            self.apply(self._init_weights)
            return

        print_log(f'Loading PointTransformer2 checkpoint from {ckpt_path}', logger='PTv2')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if 'base_model' in ckpt:
            state_dict = ckpt['base_model']
        elif 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
        processed_state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

        incompatible = self.load_state_dict(processed_state_dict, strict=False)
        if incompatible.missing_keys:
            print_log('missing_keys', logger='PTv2')
            print_log(get_missing_parameters_message(incompatible.missing_keys), logger='PTv2')
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='PTv2')
            print_log(get_unexpected_parameters_message(incompatible.unexpected_keys), logger='PTv2')
        print_log(f'[PointTransformer2] Successful Loading the ckpt from {ckpt_path}', logger='PTv2')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        """pts: [B, N, 3] or [B, 3, N] -> logits [B, cls_dim]."""
        if pts.dim() == 3:
            if pts.shape[1] == 3:                                  # (B, 3, N)
                coord = pts.permute(0, 2, 1)
            else:                                                 # (B, N, 3)
                coord = pts

        B, N, _ = coord.shape

        feat = coord
        mask = coord.new_ones((B, N), dtype=torch.bool)            # GridPool introduces padding from here on

        feat = self.patch_embed(feat)
        feat = self.patch_blocks(coord, feat, mask)

        # encoder; each skip keeps its pooling cluster index and validity mask
        skips = []
        for i in range(self.num_stages):
            down, blocks = self.enc_stages[i]
            new_coord, new_feat, cluster, new_mask = down(coord, feat)
            skips.append((coord, feat, cluster, mask))
            coord, feat, mask = new_coord, new_feat, new_mask
            feat = blocks(coord, feat, mask)

        # decoder (optional)
        if self.use_decoder:
            for i in reversed(range(self.num_stages)):
                skip_coord, skip_feat, cluster, skip_mask = skips[i]
                up, blocks = self.dec_stages[i]
                coord, feat = up(coord, feat, skip_coord, skip_feat, cluster)
                mask = skip_mask
                feat = blocks(coord, feat, mask)

        # per-sample mean over the valid points (padding excluded)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(feat.dtype)
        feat = (feat * mask.unsqueeze(-1)).sum(dim=1) / denom

        self._store_features(feat)
        return self.classifier(feat)