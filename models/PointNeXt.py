"""PointNeXt — Revisiting PointNet++ with Improved Training and Scaling Strategies.
Qian et al. NeurIPS 2022. arXiv:2206.04670
"""

import copy
from typing import List

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.cls_losses import ClassificationLossMixin
from utils.feature_hooks import FeatureExtractionMixin
from utils.logger import print_log

# Normalization / activation / conv-block factories
class LayerNorm1d(nn.Module):
    """LayerNorm over the channel axis of (B, C, N) / (B, C) tensors."""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        if x.dim() == 3:                       # (B, C, N)
            return x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return x * self.weight.view(1, -1) + self.bias.view(1, -1)   # (B, C)

class FastBatchNorm1d(nn.Module):
    """BatchNorm1d accepting either (B, C) or (B, N, C)."""

    def __init__(self, num_features, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, **kwargs)

    def forward(self, x):
        if x.dim() == 2:
            return self.bn(x)
        if x.dim() == 3:
            return self.bn(x.transpose(1, 2)).transpose(2, 1)
        raise ValueError(f"Unsupported dimensions: {x.dim()}")

_NORM_LAYER = {
    'bn': nn.BatchNorm1d,
    'bn1d': nn.BatchNorm1d,
    'bn2d': nn.BatchNorm2d,
    'in1d': nn.InstanceNorm1d,
    'in2d': nn.InstanceNorm2d,
    'gn': nn.GroupNorm,
    'syncbn': nn.SyncBatchNorm,
    'ln': nn.LayerNorm,
    'ln1d': LayerNorm1d,
    'fastbn': FastBatchNorm1d,
    'fastbn1d': FastBatchNorm1d,
}

_ACT_LAYER = {
    'relu': nn.ReLU,
    'relu6': nn.ReLU6,
    'leaky_relu': nn.LeakyReLU,
    'leakyrelu': nn.LeakyReLU,
    'elu': nn.ELU,
    'prelu': nn.PReLU,
    'selu': nn.SELU,
    'celu': nn.CELU,
    'gelu': nn.GELU,
    'silu': nn.SiLU,
    'swish': nn.SiLU,
    'mish': nn.Mish,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'hard_swish': nn.Hardswish,
    'hard_sigmoid': nn.Hardsigmoid,
}

# activations whose constructor takes no `inplace` argument
_NO_INPLACE_ACT = ('gelu', 'sigmoid', 'tanh', 'prelu', 'hard_swish', 'hard_sigmoid')

def create_norm(norm_args, channels, dimension=None):
    """Build a normalization layer. Raises on an unknown name rather than silently
    returning ``None`` (which would drop the norm *and* re-enable the conv bias)."""
    if norm_args is None:
        return None
    if isinstance(norm_args, dict):
        norm_args = copy.deepcopy(dict(norm_args))
        norm = norm_args.pop('norm', None)
    else:
        norm, norm_args = norm_args, {}
    if norm is None:
        return None
    if not isinstance(norm, str):
        return norm(channels, **norm_args)

    norm = norm.lower()
    if dimension is not None:
        dimension = str(dimension).lower()
        if dimension not in norm:
            norm += dimension
    assert norm in _NORM_LAYER, \
        f"normalization '{norm}' is not supported; pick one of {sorted(_NORM_LAYER)}"
    return _NORM_LAYER[norm](channels, **norm_args)

def create_act(act_args):
    """Build an activation layer. Raises on an unknown name."""
    if act_args is None:
        return None
    if isinstance(act_args, str):
        act_args = {'act': act_args}
    act_args = copy.deepcopy(dict(act_args))
    act = act_args.pop('act', None)
    if act is None:
        return None
    act = act.lower()
    assert act in _ACT_LAYER, \
        f"activation '{act}' is not supported; pick one of {sorted(_ACT_LAYER)}"
    if act in _NO_INPLACE_ACT:
        act_args.pop('inplace', None)
        return _ACT_LAYER[act](**act_args)
    inplace = act_args.pop('inplace', True)
    return _ACT_LAYER[act](inplace=inplace, **act_args)

def _create_convblock(conv_cls, dimension, *args, norm_args=None, act_args=None,
                      order='conv-norm-act', **kwargs):
    out_channels = args[1]
    bias = kwargs.pop('bias', True)
    norm_layer = create_norm(norm_args, out_channels, dimension=dimension)
    act_layer = create_act(act_args)
    bias = False if norm_layer is not None else bias

    if order == 'conv-norm-act':
        layers = [conv_cls(*args, bias=bias, **kwargs), norm_layer, act_layer]
    elif order == 'conv-act-norm':
        layers = [conv_cls(*args, bias=bias, **kwargs), act_layer, norm_layer]
    else:
        raise NotImplementedError(f"conv order '{order}' is not supported")
    return nn.Sequential(*[l for l in layers if l is not None])

def create_convblock1d(*args, **kwargs):
    return _create_convblock(nn.Conv1d, '1d', *args, **kwargs)

def create_convblock2d(*args, **kwargs):
    return _create_convblock(nn.Conv2d, '2d', *args, **kwargs)

def create_linearblock(in_channels, out_channels, norm_args=None, act_args=None):
    norm_layer = create_norm(norm_args, out_channels, dimension='1d')
    act_layer = create_act(act_args)
    layers = [nn.Linear(in_channels, out_channels, bias=norm_layer is None),
              norm_layer, act_layer]
    return nn.Sequential(*[l for l in layers if l is not None])

# Neighbourhood features
CHANNEL_MAP = {
    'dp_fj': lambda x: 3 + x,
    'dp_df': lambda x: x + 3,
    'dp_fj_df': lambda x: x * 2 + 3,
}

def get_aggregation_features(p, dp, f, fj, feature_type='dp_fj'):
    """Per-neighbour input of a local MLP. dp (B, 3, M, K), fj (B, C, M, K), f (B, C, M) for the df types."""
    if feature_type == 'dp_fj':
        return torch.cat([dp, fj], 1)
    if feature_type == 'dp_df':
        return torch.cat([dp, fj - f.unsqueeze(-1)], 1)
    if feature_type == 'dp_fj_df':
        return torch.cat([dp, fj, fj - f.unsqueeze(-1)], 1)
    raise NotImplementedError(f"feature type '{feature_type}' is not implemented")

def get_reduction_fn(reduction):
    reduction = 'mean' if reduction.lower() == 'avg' else reduction.lower()
    assert reduction in ('sum', 'max', 'mean')
    if reduction == 'max':
        return lambda x: torch.max(x, dim=-1)[0]
    if reduction == 'mean':
        return lambda x: torch.mean(x, dim=-1)
    return lambda x: torch.sum(x, dim=-1)

# Pure-torch sampling and grouping
def _random_sample(xyz, npoint):
    """(B, N, 3) -> (B, npoint) random indices, one independent draw per batch item."""
    B, N, _ = xyz.shape
    return torch.argsort(torch.rand(B, N, device=xyz.device), dim=-1)[:, :npoint]

def ball_query(radius, nsample, support_xyz, query_xyz):
    """Pure-torch ball query -> (B, M, K) indices: the nsample nearest supports inside radius,
    short balls padded with the query's nearest neighbour. One (B, M, N) distance matrix."""
    k = min(nsample, support_xyz.shape[1])
    dist = misc.square_distance(query_xyz, support_xyz)          # (B, M, N)
    dist.masked_fill_(dist > radius ** 2, float('inf'))
    dist, idx = torch.topk(dist, k, dim=-1, largest=False, sorted=True)
    return torch.where(torch.isinf(dist), idx[..., :1].expand_as(idx), idx)

def knn_point(nsample, support_xyz, query_xyz):
    """Pure-torch kNN -> (B, M, K) neighbour indices."""
    k = min(nsample, support_xyz.shape[1])
    dist = misc.square_distance(query_xyz, support_xyz)          # (B, M, N)
    return torch.topk(dist, k, dim=-1, largest=False, sorted=True)[1]

def _gather_grouped(x_bnc, idx):
    """(B, N, C) + (B, M, K) -> (B, C, M, K)"""
    return misc.index_points(x_bnc, idx).permute(0, 3, 1, 2).contiguous()

class QueryAndGroup(nn.Module):
    """Ball-query grouping (openpoints ``QueryAndGroup``)."""

    def __init__(self, radius, nsample, relative_xyz=True, normalize_dp=False, **kwargs):
        super().__init__()
        self.radius, self.nsample = radius, nsample
        self.relative_xyz, self.normalize_dp = relative_xyz, normalize_dp

    def forward(self, query_xyz, support_xyz, features=None):
        idx = ball_query(self.radius, self.nsample, support_xyz, query_xyz)
        grouped_xyz = _gather_grouped(support_xyz, idx)                 # (B, 3, M, K)
        if self.relative_xyz:
            grouped_xyz = grouped_xyz - query_xyz.transpose(1, 2).unsqueeze(-1)
            if self.normalize_dp:
                grouped_xyz = grouped_xyz / self.radius
        grouped_features = None
        if features is not None:
            grouped_features = _gather_grouped(features.transpose(1, 2).contiguous(), idx)
        return grouped_xyz, grouped_features

class KNNGroup(nn.Module):
    """kNN grouping (openpoints ``KNNGroup``)."""

    def __init__(self, nsample, relative_xyz=True, normalize_dp=False, **kwargs):
        super().__init__()
        self.nsample = nsample
        self.relative_xyz, self.normalize_dp = relative_xyz, normalize_dp

    def forward(self, query_xyz, support_xyz, features=None):
        idx = knn_point(self.nsample, support_xyz, query_xyz)
        grouped_xyz = _gather_grouped(support_xyz, idx)                 # (B, 3, M, K)
        if self.relative_xyz:
            grouped_xyz = grouped_xyz - query_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_dp:
            scale = torch.amax(torch.sqrt(torch.sum(grouped_xyz ** 2, dim=1)),
                               dim=(1, 2)).view(-1, 1, 1, 1)
            grouped_xyz = grouped_xyz / scale.clamp(min=1e-8)
        grouped_features = None
        if features is not None:
            grouped_features = _gather_grouped(features.transpose(1, 2).contiguous(), idx)
        return grouped_xyz, grouped_features

class GroupAll(nn.Module):
    """Treat the whole cloud as one neighbourhood (openpoints ``GroupAll``).

    Note that -- as upstream -- the coordinates are *not* made relative here.
    """

    def forward(self, query_xyz, support_xyz, features=None):
        grouped_xyz = support_xyz.transpose(1, 2).unsqueeze(2)          # (B, 3, 1, N)
        grouped_features = features.unsqueeze(2) if features is not None else None
        return grouped_xyz, grouped_features

def create_grouper(group_args):
    group_args = copy.deepcopy(dict(group_args))
    method = group_args.pop('NAME', 'ballquery').lower()
    radius = group_args.pop('radius', 0.1)
    nsample = group_args.pop('nsample', 32)
    if nsample is None:
        return GroupAll()
    if method == 'ballquery':
        return QueryAndGroup(radius, nsample, **group_args)
    if method == 'knn':
        return KNNGroup(nsample, **group_args)
    raise NotImplementedError(f"grouper '{method}' is not supported")

# Blocks
class LocalAggregation(nn.Module):
    """Aggregate features from a point set onto *itself* (no sub-sampling)."""

    def __init__(self, channels: List[int],
                 norm_args={'norm': 'bn'}, act_args={'act': 'relu'},
                 group_args={'NAME': 'ballquery', 'radius': 0.1, 'nsample': 32},
                 conv_args=None, feature_type='dp_fj', reduction='max', last_act=True):
        super().__init__()
        channels = list(channels)
        channels[0] = CHANNEL_MAP[feature_type](channels[0])
        convs = []
        for i in range(len(channels) - 1):
            last = i == len(channels) - 2
            convs.append(create_convblock2d(
                channels[i], channels[i + 1], 1,
                norm_args=norm_args,
                act_args=None if (last and not last_act) else act_args,
                **(conv_args or {})))
        self.convs = nn.Sequential(*convs)
        self.grouper = create_grouper(group_args)
        self.pool = get_reduction_fn(reduction)
        self.feature_type = feature_type

    def forward(self, p, f):
        dp, fj = self.grouper(p, p, f)
        fj = get_aggregation_features(p, dp, f, fj, self.feature_type)
        return self.pool(self.convs(fj))

class SetAbstraction(nn.Module):
    """Set abstraction + PointNeXt residual. Modes: is_head (per-point MLP stem, no sampling),
    all_aggr (stride 1: group all points, the global stage), else FPS to N//stride + grouping."""

    def __init__(self, in_channels, out_channels, layers=1, stride=1,
                 group_args={'NAME': 'ballquery', 'radius': 0.1, 'nsample': 32},
                 norm_args={'norm': 'bn'}, act_args={'act': 'relu'}, conv_args=None,
                 sampler='fps', feature_type='dp_fj', reduction='max',
                 use_res=False, is_head=False):
        super().__init__()
        self.stride = stride
        self.is_head = is_head
        self.all_aggr = not is_head and stride == 1
        self.use_res = use_res and not self.all_aggr and not self.is_head
        self.feature_type = feature_type
        self.sampler = sampler.lower()

        mid_channel = out_channels // 2 if stride > 1 else out_channels
        channels = [in_channels] + [mid_channel] * (layers - 1) + [out_channels]
        channels[0] = in_channels if is_head else CHANNEL_MAP[feature_type](channels[0])

        if self.use_res:
            self.skipconv = (create_convblock1d(in_channels, channels[-1], 1,
                                                norm_args=None, act_args=None)
                             if in_channels != channels[-1] else nn.Identity())
            self.act = create_act(act_args)

        create_conv = create_convblock1d if is_head else create_convblock2d
        convs = []
        for i in range(len(channels) - 1):
            last = i == len(channels) - 2
            convs.append(create_conv(
                channels[i], channels[i + 1], 1,
                norm_args=norm_args if not is_head else None,
                act_args=None if (last and (self.use_res or is_head)) else act_args,
                **(conv_args or {})))
        self.convs = nn.Sequential(*convs)

        if not is_head:
            if self.all_aggr:
                group_args = dict(group_args)
                group_args['radius'] = None
                group_args['nsample'] = None
            self.grouper = create_grouper(group_args)
            # upstream hard-codes max here and silently drops `reduction`; honouring it
            # is behaviour-identical for the default (max) and correct otherwise
            self.pool = get_reduction_fn(reduction)

    def forward(self, p, f):
        if self.is_head:
            return p, self.convs(f)

        if self.all_aggr:
            new_p, idx = p, None
        else:
            npoint = max(p.shape[1] // self.stride, 1)
            idx = (misc.farthest_point_sample(p, npoint) if self.sampler == 'fps'
                   else _random_sample(p, npoint))
            new_p = misc.index_points(p, idx)

        fi = None
        if idx is not None and (self.use_res or 'df' in self.feature_type):
            fi = torch.gather(f, -1, idx.unsqueeze(1).expand(-1, f.shape[1], -1))
        elif self.all_aggr and 'df' in self.feature_type:
            raise NotImplementedError(
                "feature types containing 'df' are undefined for the all-aggregation stage")

        identity = self.skipconv(fi) if self.use_res else None
        dp, fj = self.grouper(new_p, p, f)
        fj = get_aggregation_features(new_p, dp, fi, fj, self.feature_type)
        f = self.pool(self.convs(fj))
        if self.use_res:
            f = self.act(f + identity)
        return new_p, f

class InvResMLP(nn.Module):
    """Inverted-residual MLP (the depth-scaling block): LocalAggregation, then an inverted
    bottleneck C -> C*expansion -> C of 1-D convs, then a residual add."""

    def __init__(self, in_channels, norm_args={'norm': 'bn'}, act_args={'act': 'relu'},
                 aggr_args={'feature_type': 'dp_fj', 'reduction': 'max'},
                 group_args={'NAME': 'ballquery'}, conv_args=None,
                 expansion=4, use_res=True, num_posconvs=2, less_act=False):
        super().__init__()
        self.use_res = use_res
        mid_channels = int(in_channels * expansion)
        self.convs = LocalAggregation(
            [in_channels, in_channels],
            norm_args=norm_args, act_args=act_args if num_posconvs > 0 else None,
            group_args=group_args, conv_args=conv_args, **aggr_args)

        if num_posconvs < 1:
            channels = []
        elif num_posconvs == 1:
            channels = [in_channels, in_channels]
        else:
            channels = [in_channels, mid_channels, in_channels]

        pwconv = []
        for i in range(len(channels) - 1):
            last = i == len(channels) - 2
            pwconv.append(create_convblock1d(
                channels[i], channels[i + 1], 1,
                norm_args=norm_args,
                act_args=None if (last or less_act) else act_args,
                **(conv_args or {})))
        self.pwconv = nn.Sequential(*pwconv)
        self.act = create_act(act_args)

    def forward(self, p, f):
        identity = f
        f = self.convs(p, f)
        f = self.pwconv(f)
        if self.use_res and f.shape == identity.shape:
            f = f + identity
        return p, self.act(f)

class ResBlock(nn.Module):
    """Non-inverted residual variant (openpoints ``ResBlock``), selectable via ``block``."""

    def __init__(self, in_channels, norm_args={'norm': 'bn'}, act_args={'act': 'relu'},
                 aggr_args={'feature_type': 'dp_fj', 'reduction': 'max'},
                 group_args={'NAME': 'ballquery'}, conv_args=None,
                 expansion=1, use_res=True, **kwargs):
        super().__init__()
        self.use_res = use_res
        mid_channels = int(in_channels * expansion)
        # NB: upstream passes ``act_args=None`` here, i.e. the local-aggregation convs
        # of a ResBlock carry no activation at all -- reproduced verbatim. ResBlock is
        # not used by any released PointNeXt recipe; ``block: InvResMLP`` is the default.
        self.convs = LocalAggregation(
            [in_channels, in_channels, mid_channels, in_channels],
            norm_args=norm_args, act_args=None, group_args=group_args,
            conv_args=conv_args, **aggr_args)
        self.act = create_act(act_args)

    def forward(self, p, f):
        identity = f
        f = self.convs(p, f)
        if self.use_res and f.shape == identity.shape:
            f = f + identity
        return p, self.act(f)

_BLOCKS = {'InvResMLP': InvResMLP, 'ResBlock': ResBlock}

# Encoder
class PointNeXtEncoder(nn.Module):
    """PointNeXt encoder. A stage = one SetAbstraction + (blocks[i] - 1) residual blocks;
    depth scales with blocks, width with width (channels double at every strided stage)."""

    def __init__(self,
                 in_channels: int = 3,
                 width: int = 32,
                 blocks: List[int] = [1, 2, 3, 2, 2, 1],
                 strides: List[int] = [1, 2, 2, 2, 2, 1],
                 block: str = 'InvResMLP',
                 nsample=32,
                 radius=0.15,
                 radius_scaling: float = 1.5,
                 nsample_scaling: float = 1,
                 aggr_args: dict = {'feature_type': 'dp_fj', 'reduction': 'max'},
                 group_args: dict = {'NAME': 'ballquery', 'normalize_dp': True},
                 conv_args: dict = None,
                 norm_args: dict = {'norm': 'bn'},
                 act_args: dict = {'act': 'relu'},
                 sa_layers: int = 1,
                 sa_use_res: bool = False,
                 expansion: int = 4,
                 sampler: str = 'fps',
                 use_res: bool = True):
        super().__init__()
        assert len(blocks) == len(strides), \
            f"blocks ({len(blocks)}) and strides ({len(strides)}) must have the same length"
        block_cls = _BLOCKS[block] if isinstance(block, str) else block

        self.blocks = list(blocks)
        self.strides = list(strides)
        self.in_channels = in_channels
        self.aggr_args = dict(aggr_args)
        self.norm_args = dict(norm_args)
        self.act_args = dict(act_args)
        self.conv_args = dict(conv_args) if conv_args else None
        self.sampler = sampler
        self.expansion = expansion
        self.sa_layers = sa_layers
        self.sa_use_res = sa_use_res
        self.use_res = use_res

        self.radii = self._to_full_list(radius, radius_scaling)
        self.nsample = self._to_full_list(nsample, nsample_scaling)

        # width doubles after every down-sampling stage
        channels, w = [], width
        for stride in self.strides:
            if stride != 1:
                w *= 2
            channels.append(w)

        cur_channels = in_channels
        stages = []
        for i in range(len(self.blocks)):
            stage, cur_channels = self._make_enc(
                block_cls, cur_channels, channels[i], self.blocks[i], self.strides[i],
                group_args=dict(group_args, radius=self.radii[i], nsample=self.nsample[i]),
                is_head=(i == 0 and self.strides[i] == 1))
            stages.append(stage)
        self.encoder = nn.ModuleList(stages)
        self.out_channels = channels[-1]
        self.channel_list = channels

    def _to_full_list(self, param, param_scaling=1):
        """Expand a scalar (or per-stage list) into a per-block list of lists.

        The scaling is applied once per *down-sampling* stage, so a stage's later
        blocks already query at the next stage's scale -- upstream behaviour.
        """
        param_list = []
        if isinstance(param, (list, tuple)):
            for i, value in enumerate(param):
                value = list(value) if isinstance(value, (list, tuple)) else [value]
                if len(value) != self.blocks[i]:
                    value = value + [value[-1]] * (self.blocks[i] - len(value))
                param_list.append(value)
        else:
            for i, stride in enumerate(self.strides):
                if stride == 1:
                    param_list.append([param] * self.blocks[i])
                else:
                    param_list.append([param] + [param * param_scaling] * (self.blocks[i] - 1))
                    param *= param_scaling
        return param_list

    def _make_enc(self, block_cls, in_channels, channels, blocks, stride, group_args,
                  is_head=False):
        radii, nsample = group_args['radius'], group_args['nsample']
        layers = [SetAbstraction(
            in_channels, channels,
            layers=self.sa_layers if not is_head else 1,
            stride=stride,
            group_args=dict(group_args, radius=radii[0], nsample=nsample[0]),
            sampler=self.sampler,
            norm_args=self.norm_args, act_args=self.act_args, conv_args=self.conv_args,
            is_head=is_head, use_res=self.sa_use_res, **self.aggr_args)]
        for i in range(1, blocks):
            layers.append(block_cls(
                channels,
                norm_args=self.norm_args, act_args=self.act_args,
                aggr_args=self.aggr_args,
                group_args=dict(group_args, radius=radii[i], nsample=nsample[i]),
                conv_args=self.conv_args, expansion=self.expansion, use_res=self.use_res))
        return nn.ModuleList(layers), channels

    def forward(self, p, f=None):
        """p: (B, N, 3) coordinates, f: (B, C, N) features -> (B, out_channels)."""
        if f is None:
            f = p.transpose(1, 2).contiguous()
        for stage in self.encoder:
            for layer in stage:
                p, f = layer(p, f)
        # the last stage aggregates all points (N == 1); max() also covers configs
        # whose final stride is not 1, instead of silently returning a point map
        return f.max(dim=-1)[0]

class Classifier(nn.Module):
    """openpoints ClsHead (the encoder already returns a global vector, so no pooling modes)."""

    def __init__(self, num_classes, in_channels, mlps: List[int] = [512, 256],
                 norm_args={'norm': 'bn1d'}, act_args={'act': 'relu'}, dropout=0.5):
        super().__init__()
        mlps = [in_channels] + list(mlps) + [num_classes] if mlps else [in_channels, num_classes]
        heads = []
        for i in range(len(mlps) - 2):
            heads.append(create_linearblock(mlps[i], mlps[i + 1],
                                            norm_args=norm_args, act_args=act_args))
            if dropout:
                heads.append(nn.Dropout(dropout))
        heads.append(nn.Linear(mlps[-2], mlps[-1]))
        self.head = nn.Sequential(*heads)
        self.apply(self._init_linear)

    @staticmethod
    def _init_linear(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.head(x)

# Registered model
@MODELS.register_module()
class PointNeXt_Classification(nn.Module, ClassificationLossMixin, FeatureExtractionMixin):
    """PointNeXt classifier. Defaults reproduce the **PointNeXt-B** recipe."""

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.name = getattr(config, 'NAME', 'PointNeXt')
        self.num_classes = config.cls_dim

        # encoder
        self.in_channels = getattr(config, 'in_channels', 3)
        self.width = getattr(config, 'width', 32)
        self.blocks = list(getattr(config, 'blocks', [1, 2, 3, 2, 2, 1]))
        self.strides = list(getattr(config, 'strides', [1, 2, 2, 2, 2, 1]))
        self.feature_type = getattr(config, 'feature_type', 'dp_fj')

        group_args = {
            'NAME': getattr(config, 'grouper', 'ballquery'),
            'normalize_dp': getattr(config, 'normalize_dp', True),
        }
        aggr_args = {
            'feature_type': self.feature_type,
            'reduction': getattr(config, 'reduction', 'max'),
        }

        self.encoder = PointNeXtEncoder(
            in_channels=self.in_channels,
            width=self.width,
            blocks=self.blocks,
            strides=self.strides,
            block=getattr(config, 'block', 'InvResMLP'),
            nsample=getattr(config, 'nsample', 32),
            radius=getattr(config, 'radius', 0.15),
            radius_scaling=getattr(config, 'radius_scaling', 1.5),
            nsample_scaling=getattr(config, 'nsample_scaling', 1),
            aggr_args=aggr_args,
            group_args=group_args,
            conv_args={'order': getattr(config, 'conv_order', 'conv-norm-act')},
            norm_args={'norm': getattr(config, 'norm', 'bn')},
            act_args={'act': getattr(config, 'act', 'relu')},
            sa_layers=getattr(config, 'sa_layers', 1),
            sa_use_res=getattr(config, 'sa_use_res', False),
            expansion=getattr(config, 'expansion', 4),
            sampler=getattr(config, 'sampler', 'fps'),
            use_res=getattr(config, 'use_res', True),
        )

        # --- head ---------------------------------------------------------------------
        self.classifier = Classifier(
            num_classes=self.num_classes,
            in_channels=self.encoder.out_channels,
            mlps=list(getattr(config, 'head_mlps', [512, 256])),
            norm_args={'norm': getattr(config, 'head_norm', 'bn1d')},
            act_args={'act': getattr(config, 'act', 'relu')},
            dropout=getattr(config, 'dropout', 0.5),
        )

        print_log(
            f'[PointNeXt] width={self.width}, blocks={self.blocks}, strides={self.strides}, '
            f'channels={self.encoder.channel_list}, radii={self.encoder.radii}, '
            f'params={sum(p.numel() for p in self.parameters()):,}',
            logger='PointNeXt')

        # ce | focal
        self._init_cls_loss(config, self.num_classes)
        self._init_feature_extraction()

    def load_model_from_ckpt(self, ckpt_path):
        if ckpt_path is None:
            print_log('Training PointNeXt from scratch!!!', logger='PointNeXt')
            return

        print_log(f'Loading PointNeXt checkpoint from {ckpt_path}', logger='PointNeXt')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if 'base_model' in ckpt:
            state_dict = ckpt['base_model']
        elif 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt

        state_dict = {(k[7:] if k.startswith('module.') else k): v
                      for k, v in state_dict.items()}
        incompatible = self.load_state_dict(state_dict, strict=False)

        if incompatible.missing_keys:
            print_log('missing_keys', logger='PointNeXt')
            print_log(get_missing_parameters_message(incompatible.missing_keys),
                      logger='PointNeXt')
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='PointNeXt')
            print_log(get_unexpected_parameters_message(incompatible.unexpected_keys),
                      logger='PointNeXt')
        print_log(f'[PointNeXt] Successful Loading the ckpt from {ckpt_path}',
                  logger='PointNeXt')

    def forward(self, pts):
        """pts: [B, N, 3] / [B, 3, N] / [B, N, C] (C = in_channels) -> logits [B, cls_dim]."""
        if pts.dim() != 3:
            raise ValueError(f"expected a 3-D point tensor, got shape {tuple(pts.shape)}")
        if pts.shape[1] == 3 and pts.shape[2] != 3:      # (B, 3, N) -> (B, N, 3)
            pts = pts.permute(0, 2, 1)

        p = pts[:, :, :3].contiguous()
        if pts.shape[-1] != self.in_channels:
            raise ValueError(
                f"model was built with in_channels={self.in_channels} but received "
                f"{pts.shape[-1]} input channels")
        f = pts.transpose(1, 2).contiguous()             # (B, C, N)

        features = self.encoder(p, f)                    # (B, out_channels)
        self._store_features(features)                   # UMAP / t-SNE hook
        return self.classifier(features)
