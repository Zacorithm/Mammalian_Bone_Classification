"""PointGPT — Auto-regressively Generative Pre-training from Point Clouds.
Chen et al. NeurIPS 2023. arXiv:2305.11487
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
from utils.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from utils.cls_losses import ClassificationLossMixin
from utils.feature_hooks import FeatureExtractionMixin
import math

class Encoder_large(nn.Module):  # Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 1024, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(2048, 2048, 1),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Conv1d(2048, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        """point_groups [B, G, N, 3] -> feature_global [B, G, C]."""
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat(
            [feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)

class Encoder_small(nn.Module):  # Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        """point_groups [B, G, N, 3] -> feature_global [B, G, C]."""
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat(
            [feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)
        
class Group(nn.Module):
    """FPS + kNN + greedy-chain ordering."""
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
    def greedy_chain_sorting(self, xyz, center):
        """Serialize patches by a greedy nearest-neighbour chain over centres
        (PointGPT's "simplified Morton sorting": from the first FPS centre, repeatedly visit the nearest unvisited)."""
        batch_size, num_points, _ = xyz.shape
        distances_batch = torch.cdist(center, center)
        distances_batch[:, torch.eye(self.num_group).bool()] = float("inf")
        idx_base = torch.arange(0, batch_size, device=xyz.device) * self.num_group
        sorted_indices_list = []
        sorted_indices_list.append(idx_base)
        distances_batch = distances_batch.view(batch_size, self.num_group, self.num_group).transpose(
            1, 2).contiguous().view(batch_size * self.num_group, self.num_group)
        distances_batch[idx_base] = float("inf")
        distances_batch = distances_batch.view(
            batch_size, self.num_group, self.num_group).transpose(1, 2).contiguous()
        for i in range(self.num_group - 1):
            distances_batch = distances_batch.view(
                batch_size * self.num_group, self.num_group)
            distances_to_last_batch = distances_batch[sorted_indices_list[-1]]
            closest_point_idx = torch.argmin(distances_to_last_batch, dim=-1)
            closest_point_idx = closest_point_idx + idx_base
            sorted_indices_list.append(closest_point_idx)
            distances_batch = distances_batch.view(batch_size, self.num_group, self.num_group).transpose(
                1, 2).contiguous().view(batch_size * self.num_group, self.num_group)
            distances_batch[closest_point_idx] = float("inf")
            distances_batch = distances_batch.view(
                batch_size, self.num_group, self.num_group).transpose(1, 2).contiguous()
        sorted_indices = torch.stack(sorted_indices_list, dim=-1)
        sorted_indices = sorted_indices.view(-1)
        return sorted_indices

    def forward(self, xyz):
        """
        input: B N 3
        ---------------------------
        output: B G M 3
        center : B G 3
        """
        batch_size, num_points, _ = xyz.shape
        center = misc.fps(xyz, self.num_group)  # B G 3
        _, idx = misc.knn(xyz, center, self.group_size)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        # auto-regressive sequence order
        sorted_indices = self.greedy_chain_sorting(xyz, center)
        neighborhood = neighborhood.view(batch_size * self.num_group, self.group_size, 3)[sorted_indices, :, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        center = center.view(batch_size * self.num_group, 3)[sorted_indices, :]
        center = center.view(batch_size, self.num_group, 3).contiguous()

        return neighborhood, center

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PositionEmbeddingCoordsSine(nn.Module):
    """Sine positional encoding generalized to arbitrary dimensions and continuous coordinates."""

    def __init__(self, n_dim: int = 1, d_model: int = 256, temperature=10000, scale=None):
        super().__init__()

        self.n_dim = n_dim
        self.num_pos_feats = d_model // n_dim // 2 * 2
        self.temperature = temperature
        self.padding = d_model - self.num_pos_feats * self.n_dim

        if scale is None:
            scale = 1.0
        self.scale = scale * 2 * math.pi

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """xyz (*, d_in) -> pos_emb (*, d_out)."""
        assert xyz.shape[-1] == self.n_dim

        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t,
                                     2, rounding_mode='trunc') / self.num_pos_feats)

        xyz = xyz * self.scale
        pos_divided = xyz.unsqueeze(-1) / dim_t
        pos_sin = pos_divided[..., 0::2].sin()
        pos_cos = pos_divided[..., 1::2].cos()
        pos_emb = torch.stack([pos_sin, pos_cos], dim=-
                              1).reshape(*xyz.shape[:-1], -1)

        pos_emb = F.pad(pos_emb, (0, self.padding))
        return pos_emb

class GPTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(GPTBlock, self).__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x, attn_mask):

        x = self.ln_1(x)
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x

class GPT_extractor(nn.Module):
    def __init__(
        self, embed_dim, num_heads, num_layers, num_classes, trans_dim, group_size, pretrained=False
    ):
        super(GPT_extractor, self).__init__()
        self.embed_dim = embed_dim
        self.trans_dim = trans_dim
        self.group_size = group_size
        self.sos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GPTBlock(embed_dim, num_heads))
        self.ln_f = nn.LayerNorm(embed_dim)
        self.increase_dim = nn.Sequential(nn.Conv1d(self.trans_dim, 3*(self.group_size), 1))
        if pretrained == False:
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
            self.cls_norm = nn.LayerNorm(self.trans_dim)

    def forward(self, h, pos, attn_mask, classify=False):
        """h, pos: [batch, seq_len, C]. If classify, returns classification logits."""
        batch, length, C = h.shape
        h = h.transpose(0, 1)
        pos = pos.transpose(0, 1)
        sos = torch.ones(1, batch, self.embed_dim, device=h.device) * self.sos
        if not classify:
            h = torch.cat([sos, h[:-1, :, :]], axis=0)
        else:
            h = torch.cat([sos, h], axis=0)
        for layer in self.layers:
            h = layer(h + pos, attn_mask)
        h = self.ln_f(h)
        encoded_points = h.transpose(0, 1)
        if not classify:
            return encoded_points
        h = h.transpose(0, 1)
        h = self.cls_norm(h)
        concat_f = torch.cat([h[:, 1], h[:, 2:].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret, encoded_points, concat_f

class GPT_generator(nn.Module):
    def __init__(
        self, embed_dim, num_heads, num_layers, trans_dim, group_size):
        super(GPT_generator, self).__init__()
        self.embed_dim = embed_dim
        self.trans_dim = trans_dim
        self.group_size = group_size
        self.sos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GPTBlock(embed_dim, num_heads))
        self.ln_f = nn.LayerNorm(embed_dim)
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3*(self.group_size), 1)
        )

    def forward(self, h, pos, attn_mask):
        """h, pos: [batch, seq_len, C]. If classify, returns classification logits."""
        batch, length, C = h.shape
        h = h.transpose(0, 1)
        pos = pos.transpose(0, 1)
        for layer in self.layers:
            h = layer(h + pos, attn_mask)
        h = self.ln_f(h)
        rebuild_points = self.increase_dim(h.transpose(1, 2)).transpose(
            1, 2).transpose(0, 1).reshape(batch * length, -1, 3)
        return rebuild_points

class GPT_Transformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.decoder_depth = config.transformer_config.decoder_depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads
        self.group_size = config.group_size
        print_log(f'[args] {config.transformer_config}', logger='Transformer')

        self.encoder_dims = config.transformer_config.encoder_dims

        assert self.encoder_dims in [384, 768, 1024]
        if self.encoder_dims == 384:
            self.encoder = Encoder_small(encoder_channel=self.encoder_dims)
        else:
            self.encoder = Encoder_large(encoder_channel=self.encoder_dims)

        self.pos_embed = PositionEmbeddingCoordsSine(3, self.encoder_dims, 1.0)

        self.blocks = GPT_extractor(
            embed_dim=self.encoder_dims,
            num_heads=self.num_heads,
            num_layers=self.depth,
            num_classes=config.cls_dim,
            trans_dim=self.trans_dim,
            group_size=self.group_size,
            pretrained=True,
        )

        self.generator_blocks = GPT_generator(
            embed_dim=self.encoder_dims,
            num_heads=self.num_heads,
            num_layers=self.decoder_depth,
            trans_dim=self.trans_dim,
            group_size=self.group_size
        )

        # do not perform additional mask on the first (self.keep_attend) tokens
        self.keep_attend = 10
        self.num_groups = config.num_group
        self.num_mask = int(
            (self.num_groups - self.keep_attend) * self.mask_ratio)

        self.sos_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, neighborhood, center, noaug=False, classify=False):

        group_input_tokens = self.encoder(neighborhood)  # B G C

        batch_size, seq_len, C = group_input_tokens.size()

        relative_position = center[:, 1:, :] - center[:, :-1, :]
        relative_norm = torch.norm(relative_position, dim=-1, keepdim=True)
        relative_direction = relative_position / relative_norm
        position = torch.cat(
            [center[:, 0, :].unsqueeze(1), relative_direction], dim=1)
        pos_relative = self.pos_embed(position)

        sos_pos = self.sos_pos.expand(group_input_tokens.size(0), -1, -1)
        pos_absolute = self.pos_embed(center[:, :-1, :])
        pos_absolute = torch.cat([sos_pos, pos_absolute], dim=1)

        attn_mask = torch.full(
            (seq_len, seq_len), -float("Inf"), device=group_input_tokens.device, dtype=group_input_tokens.dtype
        ).to(torch.bool)

        with torch.no_grad():
            attn_mask = torch.triu(attn_mask, diagonal=1)


            overall_mask = np.hstack([
                np.zeros(self.num_groups-self.keep_attend-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(overall_mask)
            overall_mask = np.hstack([
                np.zeros(self.keep_attend),
                overall_mask,
            ])
            # device-aware
            mask_device = group_input_tokens.device
            overall_mask = torch.from_numpy(
                overall_mask).to(torch.bool).to(mask_device)

            eye_mask = torch.eye(self.num_groups).to(torch.bool).to(mask_device)

            attn_mask = attn_mask | overall_mask.unsqueeze(0) & ~eye_mask

        if classify == False:
            encoded_features = self.blocks(
                group_input_tokens, pos_absolute, attn_mask, classify=classify)
            generated_points = self.generator_blocks(
                encoded_features, pos_relative, attn_mask)
            return generated_points
        else:
            print('----error---- This code is detached ----error----')
            logits, generated_points = self.blocks(
                group_input_tokens, pos_absolute, classify=classify)
            return logits, generated_points

@MODELS.register_module()
class PointGPT_Pretrain(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[PointGPT_Pretrain] ', logger='PointGPT_Pretrain')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.GPT_Transformer = GPT_Transformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate

        print_log(
            f'[PointGPT_Pretrain] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger='PointGPT_Pretrain')
        self.group_divider = Group(
            num_group=self.num_group, group_size=self.group_size)

        self.loss = config.loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = [ChamferDistanceL1()]
        elif loss_type =='cdl2':
            self.loss_func = [ChamferDistanceL2()]
        elif loss_type == 'cdl12':
            self.loss_func = [ChamferDistanceL1(), ChamferDistanceL2()]
        else:
            raise NotImplementedError
        self.loss_func_c = nn.MSELoss()

    def forward(self, pts, vis=False, **kwargs):
        loss = 0
        neighborhood, center = self.group_divider(pts)

        B = neighborhood.shape[0]

        generated_points = self.GPT_Transformer(
            neighborhood, center)

        gt_points = neighborhood.reshape(
            B*(self.num_group), self.group_size, 3)
        for l in self.loss_func:
            loss += l(generated_points, gt_points)

        if vis:  # visualization
            gt_points = gt_points.reshape(
                B, self.num_group, self.group_size, 3)
            gt_points = (gt_points + center.unsqueeze(-2)
                         ).reshape(-1, 3).unsqueeze(0)
            generated_points = generated_points.reshape(
                B, self.num_group, self.group_size, 3) + center.unsqueeze(-2)
            generated_points = generated_points.reshape(-1, 3).unsqueeze(0)
            return generated_points, gt_points, center
        return loss
        
@MODELS.register_module()
class PointGPT_Classification(nn.Module, ClassificationLossMixin, FeatureExtractionMixin):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.decoder_depth = config.decoder_depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.freeze_encoder = getattr(config, 'freeze_encoder', False)

        # total = cls_loss_weight * <cls loss> + gen_loss_weight * (CDL1 + CDL2, next-patch gen)
        # gen_loss_weight: 0 disables the aux loss (generator_blocks not built). Upstream FT = 1 / 3.
        # legacy use_generation_loss: true maps to gen_loss_weight=3.
        self.cls_loss_weight = float(getattr(config, 'cls_loss_weight', 1.0))
        _legacy_gen = getattr(config, 'use_generation_loss', False)
        self.gen_loss_weight = float(
            getattr(config, 'gen_loss_weight', 3.0 if _legacy_gen else 0.0))
        assert self.gen_loss_weight >= 0, \
            f'gen_loss_weight must be >= 0, got {self.gen_loss_weight}'
        self.use_generation_loss = self.gen_loss_weight > 0

        self.group_divider = Group(
            num_group=self.num_group, group_size=self.group_size)

        assert self.encoder_dims in [384, 768, 1024]
        if self.encoder_dims == 384:
            self.encoder = Encoder_small(encoder_channel=self.encoder_dims)
        else:
            self.encoder = Encoder_large(encoder_channel=self.encoder_dims)

        self.pos_embed = PositionEmbeddingCoordsSine(3, self.encoder_dims, 1.0)

        self.blocks = GPT_extractor(
            embed_dim=self.encoder_dims,
            num_heads=self.num_heads,
            num_layers=self.depth,
            num_classes=config.cls_dim,
            trans_dim=self.trans_dim,
            group_size=self.group_size
        )

        if self.use_generation_loss:
            self.generator_blocks = GPT_generator(
                embed_dim=self.encoder_dims,
                num_heads=self.num_heads,
                num_layers=self.decoder_depth,
                trans_dim=self.trans_dim,
                group_size=self.group_size
            )
            self.build_loss_func()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.sos_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))

        self.norm = nn.LayerNorm(self.trans_dim)

        # ce | focal
        self._init_cls_loss(config, self.cls_dim)
        self._init_feature_extraction()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def build_loss_func(self, loss_type='cdl12'):
        # Reconstruction losses for the auxiliary generation branch.
        # NOTE: the hardcoded 'cdl12' default matches upstream PointGPT, which
        # ignores the config's `loss` key at fine-tune time.
        if loss_type == "cdl1":
            self.loss_func = [ChamferDistanceL1()]
        elif loss_type =='cdl2':
            self.loss_func = [ChamferDistanceL2()]
        elif loss_type == 'cdl12':
            self.loss_func = [ChamferDistanceL1(), ChamferDistanceL2()]
        else:
            raise NotImplementedError

    def freeze_encoder_weights(self):
        """Freeze encoder and GPT_extractor blocks if specified"""
        if self.freeze_encoder:
            print_log('Freezing encoder weights...', logger='PointGPT_Classification')
            
            # Freeze the point encoder
            for param in self.encoder.parameters():
                param.requires_grad = False 
                
            # Freeze the GPT_extractor blocks (the main encoder)
            for param in self.blocks.parameters():
                param.requires_grad = False
                
            # Freeze positional embeddings
            for param in self.pos_embed.parameters():
                param.requires_grad = False
                
            # Keep generator_blocks, cls_token, cls_pos, sos_pos, norm trainable
            print_log('PointGPT Encoder (GPT_extractor) frozen. Generator blocks and cls tokens remain trainable.', 
                     logger='PointGPT_Classification')

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k,
                         v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('GPT_Transformer'):
                    base_ckpt[k[len('GPT_Transformer.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                if 'cls_head_finetune' in k:
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='PointGPT_Classification')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='PointGPT_Classification'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='PointGPT_Classification')
                print_log(
                    get_unexpected_parameters_message(
                        incompatible.unexpected_keys),
                    logger='PointGPT_Classification'
                )

            print_log(
                f'[PointGPT_Classification] Successful Loading the ckpt from {bert_ckpt_path}', logger='PointGPT_Classification')
            # Freeze encoder after loading pretrained weights
            self.freeze_encoder_weights()
        else:
            print_log('Training from scratch!!!', logger='PointGPT_Classification')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N

        B, L, _ = group_input_tokens.shape

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)
        sos_pos = self.sos_pos.expand(group_input_tokens.size(0), -1, -1)
        pos = torch.cat([sos_pos, pos], dim=1)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)

        attn_mask = torch.full(
            (L+2, L+2), -float("Inf"), device=group_input_tokens.device, dtype=group_input_tokens.dtype
        ).to(torch.bool)

        attn_mask = torch.triu(attn_mask, diagonal=1)

        ret, encoded_features, concat_f = self.blocks(x, pos, attn_mask, classify=True)

        self._store_features(concat_f)

        if not self.use_generation_loss:
            return ret, None

        relative_position = center[:, 1:, :] - center[:, :-1, :]
        relative_norm = torch.norm(relative_position, dim=-1, keepdim=True)
        relative_direction = relative_position / relative_norm
        position = torch.cat(
            [center[:, 0, :].unsqueeze(1), relative_direction], dim=1)
        pos_relative = self.pos_embed(position)

        encoded_features = torch.cat(
            [encoded_features[:, 0, :].unsqueeze(1), encoded_features[:, 2:-1, :]], dim=1)

        attn_mask = torch.full(
            (L, L), -float("Inf"), device=group_input_tokens.device, dtype=group_input_tokens.dtype
        ).to(torch.bool)

        attn_mask = torch.triu(attn_mask, diagonal=1)

        generated_points = self.generator_blocks(
            encoded_features, pos_relative, attn_mask)

        neighborhood = neighborhood + center.unsqueeze(2)

        gt_points = neighborhood.reshape(
            B*(self.num_group), self.group_size, 3)

        loss = 0
        for l in self.loss_func:
            loss += l(generated_points, gt_points)
        return ret, loss

@MODELS.register_module()
class PointGPT_ClassificationOnly(nn.Module, ClassificationLossMixin, FeatureExtractionMixin):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.freeze_encoder = getattr(config, 'freeze_encoder', False)

        self.group_divider = Group(
            num_group=self.num_group, group_size=self.group_size)

        assert self.encoder_dims in [384, 768, 1024]
        if self.encoder_dims == 384:
            self.encoder = Encoder_small(encoder_channel=self.encoder_dims)
        else:
            self.encoder = Encoder_large(encoder_channel=self.encoder_dims)

        self.pos_embed = PositionEmbeddingCoordsSine(3, self.encoder_dims, 1.0)

        # Classification-only GPT extractor (no decoder needed)
        self.blocks = GPT_extractor(
            embed_dim=self.encoder_dims,
            num_heads=self.num_heads,
            num_layers=self.depth,
            num_classes=config.cls_dim,
            trans_dim=self.trans_dim,
            group_size=self.group_size
        )

        # Classification tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.sos_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))

        self.norm = nn.LayerNorm(self.trans_dim)

        # ce | focal
        self._init_cls_loss(config, self.cls_dim)
        self._init_feature_extraction()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def freeze_encoder_weights(self):
        """Freeze encoder and GPT_extractor blocks if specified"""
        if self.freeze_encoder:
            print_log('Freezing encoder weights...', logger='PointGPT_Classification')
            
            # Freeze the point encoder
            for param in self.encoder.parameters():
                param.requires_grad = False 
                
            # Freeze the GPT_extractor blocks (the main encoder)
            for param in self.blocks.parameters():
                param.requires_grad = False
                
            # Freeze positional embeddings
            for param in self.pos_embed.parameters():
                param.requires_grad = False
                
            print_log('PointGPT Encoder frozen. Only classification head remains trainable.', 
                     logger='PointGPT_Classification')

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k,
                         v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('GPT_Transformer'):
                    base_ckpt[k[len('GPT_Transformer.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                if 'cls_head_finetune' in k:
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='PointGPT_Classification')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='PointGPT_Classification'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='PointGPT_Classification')
                print_log(
                    get_unexpected_parameters_message(
                        incompatible.unexpected_keys),
                    logger='PointGPT_Classification'
                )

            print_log(
                f'[PointGPT_Classification] Successful Loading the ckpt from {bert_ckpt_path}', logger='PointGPT_Classification')
            # Freeze encoder after loading pretrained weights
            self.freeze_encoder_weights()
        else:
            print_log('Training from scratch!!!', logger='PointGPT_Classification')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        # Process input point cloud
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G C

        B, L, _ = group_input_tokens.shape

        # Add classification token
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        # Positional embeddings
        pos = self.pos_embed(center)
        sos_pos = self.sos_pos.expand(group_input_tokens.size(0), -1, -1)
        pos = torch.cat([sos_pos, pos], dim=1)

        # Concatenate classification token with group tokens
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)

        # Create attention mask for autoregressive processing
        attn_mask = torch.full(
            (L+2, L+2), -float("Inf"), device=group_input_tokens.device, dtype=group_input_tokens.dtype
        ).to(torch.bool)
        attn_mask = torch.triu(attn_mask, diagonal=1)

        # Forward through transformer blocks for classification
        ret, _ , concat_f= self.blocks(x, pos, attn_mask, classify=True)

        self._store_features(concat_f)

        # Return only classification logits
        return ret, None