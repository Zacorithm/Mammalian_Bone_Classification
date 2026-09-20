"""Point Transformer V3 — Simpler, Faster, Stronger.
Wu et al. CVPR 2024. doi:10.1109/CVPR52733.2024.00463
"""

import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from functools import partial

try:
    import flash_attn
except ImportError:
    flash_attn = None

from .build import MODELS
from utils.cls_losses import ClassificationLossMixin
from utils.feature_hooks import FeatureExtractionMixin
from utils.serialization import encode as sfc_encode

# Serialization helpers
def serialize(grid_coord, batch_indices, orders, depth):
    """Stack one serialization code per curve order -> (num_orders, N) int64."""
    return torch.stack(
        [sfc_encode(grid_coord, batch=batch_indices, depth=depth, order=o) for o in orders]
    )

def order_and_inverse(code):
    """argsort each code row plus its inverse permutation (reference `Point.serialization`)."""
    order = torch.argsort(code, dim=1)
    inverse = torch.zeros_like(order).scatter_(
        dim=1,
        index=order,
        src=torch.arange(0, code.shape[1], device=order.device).repeat(code.shape[0], 1),
    )
    return order, inverse

def shuffle_rows(code, order, inverse):
    """Randomly permute the curve orders (reference `shuffle_orders`, train and eval)."""
    perm = torch.randperm(code.shape[0], device=code.device)
    return code[perm], order[perm], inverse[perm]

def build_patches(batch_counts, patch_size, device):
    """Per-sample patch bookkeeping over the batch-contiguous sorted sequence.
    Pads each sample to a multiple of patch_size by repeating the region one patch
    earlier (samples shorter than a patch wrap cyclically, flash path only).
    Returns pad (N_pad,) building the padded seq, unpad (N,) undoing it."""
    pad_chunks, unpad_chunks = [], []
    base, pbase = 0, 0
    for cnt in batch_counts.tolist():
        if cnt == 0:
            continue
        n_pad = int(math.ceil(cnt / patch_size)) * patch_size
        idx = torch.arange(cnt, device=device)
        extra = n_pad - cnt
        if extra > 0:
            if cnt >= patch_size:
                start = cnt - patch_size
                fill = torch.arange(start, start + extra, device=device)
            else:
                fill = torch.arange(extra, device=device) % cnt
            pad_chunks.append(base + torch.cat([idx, fill]))
        else:
            pad_chunks.append(base + idx)
        unpad_chunks.append(pbase + idx)
        base += cnt
        pbase += n_pad
    return torch.cat(pad_chunks), torch.cat(unpad_chunks)

# Pure-torch submanifold sparse convolution
def build_submanifold_neighbors(grid_coord, batch_indices, kernel_size):
    """Neighbour index map -> (N, kernel_size**3) int64. out[i, k] = index of the point in the
    voxel at kernel offset k from point i, or -1 if empty (pure-torch spconv indice pairs;
    on a voxel collision the first occupant in sorted key order wins)."""
    r = kernel_size // 2
    lo = grid_coord.min(0).values - r
    ext = (grid_coord.max(0).values + r - lo + 1).tolist()
    base = grid_coord - lo

    def key_of(coord):
        return ((batch_indices * ext[0] + coord[:, 0]) * ext[1] + coord[:, 1]) * ext[2] \
            + coord[:, 2]

    key = key_of(base)
    skey, sidx = torch.sort(key)
    n = key.shape[0]

    rng = torch.arange(-r, r + 1, device=grid_coord.device)
    offsets = torch.stack(torch.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)

    neighbors = torch.empty((n, offsets.shape[0]), dtype=torch.long, device=grid_coord.device)
    for k in range(offsets.shape[0]):
        q = key_of(base + offsets[k])
        pos = torch.searchsorted(skey, q).clamp(max=n - 1)
        hit = skey[pos] == q
        neighbors[:, k] = torch.where(hit, sidx[pos], torch.full_like(pos, -1))
    return neighbors

class SubmanifoldConv3d(nn.Module):
    """Submanifold sparse 3-D convolution, pure torch (reference: ``spconv.SubMConv3d``)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_volume = kernel_size ** 3
        self.weight = nn.Parameter(torch.empty(self.kernel_volume, in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        # spconv's default: kaiming_uniform_(a=sqrt(5)) over fan_in = volume * in_channels
        bound = 1.0 / math.sqrt(self.kernel_volume * in_channels)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, neighbors):
        """x: (N, in_channels); neighbors: (N, kernel_volume) from build_submanifold_neighbors."""
        src = neighbors.clamp(min=0)
        valid = neighbors >= 0
        out = None
        for k in range(self.kernel_volume):
            # project first, then gather: (x @ W)[src] == (x[src]) @ W, and this order
            # keeps autograd from stashing one (N, C) gathered tensor per kernel offset
            y = x @ self.weight[k]
            y = y[src[:, k]] * valid[:, k].unsqueeze(-1)
            out = y if out is None else out + y
        if self.bias is not None:
            out = out + self.bias
        return out

class Embedding(nn.Module):
    """Sparse-conv stem (reference ``Embedding``: SubMConv3d k=5 -> BN -> GELU)."""

    def __init__(self, in_channels, embed_channels,
                 norm_layer=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01),
                 act_layer=nn.GELU):
        super().__init__()
        self.conv = SubmanifoldConv3d(in_channels, embed_channels, kernel_size=5, bias=False)
        self.norm = norm_layer(embed_channels)
        self.act = act_layer()

    def forward(self, x, neighbors):
        return self.act(self.norm(self.conv(x, neighbors)))

class xCPE(nn.Module):
    """Conditional positional encoding (reference ``Block.cpe``)."""

    def __init__(self, channels, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = SubmanifoldConv3d(channels, channels, kernel_size=3, bias=True)
        self.linear = nn.Linear(channels, channels)
        self.norm = norm_layer(channels)

    def forward(self, x, neighbors):
        return self.norm(self.linear(self.conv(x, neighbors)))

# Attention
class RPE(nn.Module):
    """Grid-relative positional bias table (reference ``RPE``, math path only)."""

    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, rel_pos):
        """rel_pos: (P, K, K, 3) int64 pairwise grid-coord differences within a patch."""
        idx = rel_pos.clamp(-self.pos_bnd, self.pos_bnd) + self.pos_bnd
        idx = idx + torch.arange(3, device=idx.device) * self.rpe_num
        bias = self.rpe_table.index_select(0, idx.reshape(-1))
        bias = bias.view(*idx.shape, self.num_heads).sum(3)   # (P, K, K, H)
        return bias.permute(0, 3, 1, 2)                       # (P, H, K, K)

class SerializedAttention(nn.Module):
    """Multi-head attention within fixed-size serialized patches."""

    def __init__(self, channels, num_heads, patch_size, qkv_bias=True, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0, enable_flash=True, enable_rpe=False,
                 upcast_attention=False, upcast_softmax=False):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.patch_size_max = patch_size
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        # RPE needs per-point grid coords and a fixed window, so it is incompatible
        # with flash's varlen kernel -- upstream asserts enable_rpe is False there.
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash and flash_attn is not None and not enable_rpe
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax

        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

        if not self.enable_flash:
            self.attn_drop = nn.Dropout(attn_drop)
            self.softmax = nn.Softmax(dim=-1)
            self.rpe = RPE(patch_size, num_heads) if enable_rpe else None
        else:
            self.attn_drop = attn_drop

    def forward(self, x, order, inverse, pad, unpad, patch_size, grid_coord=None):
        H, C, K = self.num_heads, self.channels, patch_size

        qkv = self.qkv(x)[order][pad]                # (N_pad, 3C) sorted + padded
        num_patches = qkv.shape[0] // K

        if not self.enable_flash:
            q, k, v = (
                qkv.reshape(num_patches, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(0)
            )                                        # each (P, H, K, C/H)
            if self.upcast_attention:
                q, k = q.float(), k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)
            if self.enable_rpe:
                gc = grid_coord[order][pad].reshape(num_patches, K, 3)
                attn = attn + self.rpe(gc.unsqueeze(2) - gc.unsqueeze(1))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn).to(qkv.dtype)
            attn = self.attn_drop(attn)
            out = (attn @ v).transpose(1, 2).reshape(num_patches * K, C)
        else:
            flash_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            cu_seqlens = torch.arange(
                0, (num_patches + 1) * K, K, dtype=torch.int32, device=x.device
            )
            out = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.to(flash_dtype).reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=K,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            )
            # flash returns (N_pad, H, C/H) point-major -- no transpose here
            out = out.reshape(-1, C).to(qkv.dtype)

        out = out[unpad]        # drop padded slots -> sorted order
        out = out[inverse]      # undo serialization -> original point order
        return self.proj_drop(self.proj(out))

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))

class TransformerBlock(nn.Module):
    """PTv3 block: xCPE residual -> pre-norm attention -> pre-norm MLP."""

    def __init__(self, channels, num_heads, patch_size=1024, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, attn_drop=0.0, proj_drop=0.0, drop_path=0.0,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU, enable_flash=True,
                 enable_rpe=False, upcast_attention=False, upcast_softmax=False,
                 order_index=0):
        super().__init__()
        self.channels = channels
        self.order_index = order_index

        self.cpe = xCPE(channels, norm_layer=norm_layer)
        self.norm1 = norm_layer(channels)
        self.attn = SerializedAttention(
            channels=channels, num_heads=num_heads, patch_size=patch_size,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=proj_drop, enable_flash=enable_flash, enable_rpe=enable_rpe,
            upcast_attention=upcast_attention, upcast_softmax=upcast_softmax)
        self.norm2 = norm_layer(channels)
        self.mlp = MLP(in_channels=channels, hidden_channels=int(channels * mlp_ratio),
                       out_channels=channels, act_layer=act_layer, drop=proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, order, inverse, pad, unpad, patch_size, neighbors, grid_coord=None):
        i = self.order_index % order.shape[0]
        x = x + self.cpe(x, neighbors)
        x = x + self.drop_path(
            self.attn(self.norm1(x), order[i], inverse[i], pad, unpad, patch_size, grid_coord)
        )
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# Pooling
class SerializedPooling(nn.Module):
    """Octree-parent pooling: clusters from code >> 3*pooling_depth (a coarse voxel is exactly
    the union of its finer voxels), features max-reduced, coords mean-reduced, grid_coord/codes
    derived by shifting. scatter_reduce replaces torch_scatter.segment_csr."""

    def __init__(self, in_channels, out_channels, stride=2,
                 norm_layer=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01),
                 act_layer=nn.GELU, reduce="max", shuffle_orders=True):
        super().__init__()
        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()   # 2, 4, 8
        assert reduce in ("sum", "mean", "min", "max")
        self.stride = stride
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = norm_layer(out_channels) if norm_layer is not None else nn.Identity()
        self.act = act_layer() if act_layer is not None else nn.Identity()

    _REDUCE = {"max": "amax", "min": "amin", "mean": "mean", "sum": "sum"}

    def forward(self, x, coords, grid_coord, batch_indices, code, depth):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > depth:
            pooling_depth = 0

        code = code >> (pooling_depth * 3)
        _, cluster, counts = torch.unique(
            code[0], sorted=True, return_inverse=True, return_counts=True
        )
        m = counts.shape[0]
        # first point of each cluster, for attributes that are constant within a voxel
        _, indices = torch.sort(cluster, stable=True)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]

        code = code[:, head_indices]
        order, inverse = order_and_inverse(code)
        if self.shuffle_orders:
            code, order, inverse = shuffle_rows(code, order, inverse)

        feat = self.proj(x)
        C = feat.shape[1]
        init = float("-inf") if self.reduce == "max" else (
            float("inf") if self.reduce == "min" else 0.0)
        pooled = feat.new_full((m, C), init).scatter_reduce(
            0, cluster.unsqueeze(1).expand(-1, C), feat,
            reduce=self._REDUCE[self.reduce], include_self=True)

        coord_sum = coords.new_zeros((m, 3)).index_add_(0, cluster, coords)
        pooled_coords = coord_sum / counts.unsqueeze(1).to(coords.dtype)

        pooled = self.act(self.norm(pooled))
        return (pooled, pooled_coords, grid_coord[head_indices] >> pooling_depth,
                batch_indices[head_indices], code, order, inverse, depth - pooling_depth)

# Registered model
@MODELS.register_module()
class PointTransformerV3_Classification(nn.Module, ClassificationLossMixin, FeatureExtractionMixin):
    """PointTransformerV3 encoder (reference ``cls_mode=True``) + Pointcept's DefaultClassifier."""

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.name = getattr(config, 'NAME', 'PointTransformerV3_Classification')
        self.num_classes = config.cls_dim

        # with raw xyz input the model builds 7-d features (xyz, ||xyz||, xyz^2)
        self.in_channels = getattr(config, 'in_channels', 7)
        self.enc_depths = list(getattr(config, 'enc_depths', [2, 2, 2, 6, 2]))
        self.enc_channels = list(getattr(config, 'enc_channels', [32, 64, 128, 256, 512]))
        self.enc_num_head = list(getattr(config, 'enc_num_head', [2, 4, 8, 16, 32]))
        self.enc_patch_size = list(getattr(config, 'enc_patch_size', [1024] * 5))
        self.stride = list(getattr(config, 'stride', [2, 2, 2, 2]))
        self.orders = tuple(getattr(config, 'orders',
                                    ('z', 'z-trans', 'hilbert', 'hilbert-trans')))
        self.grid_size = getattr(config, 'grid_size', 0.02)
        self.shuffle_orders = getattr(config, 'shuffle_orders', True)

        self.mlp_ratio = getattr(config, 'mlp_ratio', 4.0)
        self.qkv_bias = getattr(config, 'qkv_bias', True)
        self.attn_drop = getattr(config, 'attn_drop', 0.0)
        self.proj_drop = getattr(config, 'proj_drop', 0.0)
        self.drop_path = getattr(config, 'drop_path', 0.3)
        self.enable_flash = getattr(config, 'enable_flash', True)
        self.enable_rpe = getattr(config, 'enable_rpe', False)
        self.upcast_attention = getattr(config, 'upcast_attention', False)
        self.upcast_softmax = getattr(config, 'upcast_softmax', False)

        self.num_stages = len(self.enc_depths)
        assert self.num_stages == len(self.stride) + 1
        assert self.num_stages == len(self.enc_channels)
        assert self.num_stages == len(self.enc_num_head)
        assert self.num_stages == len(self.enc_patch_size)

        # the math path is the only one available without flash_attn installed
        self.use_flash = self.enable_flash and flash_attn is not None and not self.enable_rpe

        print(f'[PTv3] {self.num_stages} stages, orders={self.orders}, '
              f'flash={"on" if self.use_flash else "off (math path)"}')

        self.embedding = Embedding(self.in_channels, self.enc_channels[0])

        self.stages = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()
        enc_drop_path = [x.item() for x in torch.linspace(0, self.drop_path, sum(self.enc_depths))]

        for s in range(self.num_stages):
            stage_drop_path = enc_drop_path[sum(self.enc_depths[:s]): sum(self.enc_depths[:s + 1])]
            self.pooling_layers.append(
                SerializedPooling(self.enc_channels[s - 1], self.enc_channels[s],
                                  stride=self.stride[s - 1],
                                  shuffle_orders=self.shuffle_orders)
                if s > 0 else nn.Identity()
            )
            self.stages.append(nn.ModuleList([
                TransformerBlock(
                    channels=self.enc_channels[s],
                    num_heads=self.enc_num_head[s],
                    patch_size=self.enc_patch_size[s],
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    attn_drop=self.attn_drop,
                    proj_drop=self.proj_drop,
                    drop_path=stage_drop_path[b],
                    enable_flash=self.enable_flash,
                    enable_rpe=self.enable_rpe,
                    upcast_attention=self.upcast_attention,
                    upcast_softmax=self.upcast_softmax,
                    order_index=b,
                ) for b in range(self.enc_depths[s])
            ]))

        # Pointcept DefaultClassifier
        self.classifier = nn.Sequential(
            nn.Linear(self.enc_channels[-1], 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, self.num_classes),
        )

        # ce | focal
        self._init_cls_loss(config, self.num_classes)
        self._init_feature_extraction()

    def load_model_from_ckpt(self, ckpt_path):
        if ckpt_path is None:
            print('Training PTv3 from scratch!')
            self.apply(self._init_weights)
            return

        print(f'Loading PTv3 checkpoint from {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if 'base_model' in ckpt:
            state_dict = ckpt['base_model']
        elif 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
        state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

        incompatible = self.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            print('missing_keys in PTv3:', incompatible.missing_keys)
        if incompatible.unexpected_keys:
            print('unexpected_keys in PTv3:', incompatible.unexpected_keys)
        print(f'[PTv3] Successfully loaded checkpoint from {ckpt_path}')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, pts):
        """pts: (B, N, 3) -- or (B, 3, N). Returns (B, num_classes) logits."""
        if pts.dim() == 3 and pts.shape[1] == 3 and pts.shape[2] != 3:
            pts = pts.permute(0, 2, 1)
        batch_size, num_points, channels = pts.shape

        flat = pts.reshape(-1, channels)
        coords = flat[:, :3].contiguous()
        batch_indices = torch.arange(batch_size, device=pts.device).repeat_interleave(num_points)

        if channels > 3:
            features = flat
        else:
            features = torch.cat(
                [coords, torch.norm(coords, dim=1, keepdim=True), coords ** 2], dim=1)
        assert features.shape[-1] == self.in_channels, (
            f"in_channels={self.in_channels} but built {features.shape[-1]}-d features")

        # quantize once; every later stage derives grid_coord by right-shifting,
        # exactly as the reference does from Point.grid_coord
        grid_coord = torch.div(coords, self.grid_size, rounding_mode="floor").long()
        grid_coord = grid_coord - grid_coord.min(dim=0, keepdim=True)[0]
        depth = max(int(grid_coord.max().item()).bit_length(), 1)
        assert depth * 3 + int(batch_size).bit_length() <= 63, \
            "serialization code exceeds the int64 budget; lower grid_size resolution"
        assert depth <= 16

        code = serialize(grid_coord, batch_indices, self.orders, depth)
        order, inverse = order_and_inverse(code)
        if self.shuffle_orders:
            code, order, inverse = shuffle_rows(code, order, inverse)

        # the stem uses its own kernel (5^3), i.e. the reference's indice_key="stem"
        x = self.embedding(features, build_submanifold_neighbors(grid_coord, batch_indices, 5))

        for s, stage_blocks in enumerate(self.stages):
            if s > 0:
                (x, coords, grid_coord, batch_indices,
                 code, order, inverse, depth) = self.pooling_layers[s](
                    x, coords, grid_coord, batch_indices, code, depth)

            counts = torch.bincount(batch_indices, minlength=batch_size)
            patch_size = self.enc_patch_size[s]
            if not self.use_flash:
                # upstream shrinks the window to the smallest sample on the math path,
                # so no sample is ever shorter than one patch
                patch_size = min(int(counts[counts > 0].min().item()), patch_size)
            pad, unpad = build_patches(counts, patch_size, x.device)

            # one neighbour map per stage, shared by its blocks (spconv indice_key)
            neighbors = build_submanifold_neighbors(grid_coord, batch_indices, 3)

            for block in stage_blocks:
                x = block(x, order, inverse, pad, unpad, patch_size, neighbors, grid_coord)

        # global mean pooling per sample (Pointcept DefaultClassifier)
        pooled = x.new_zeros((batch_size, x.shape[1])).index_add_(0, batch_indices, x)
        counts = torch.bincount(batch_indices, minlength=batch_size).clamp(min=1)
        global_features = pooled / counts.unsqueeze(1).to(x.dtype)

        self._store_features(global_features)
        return self.classifier(global_features)
