"""PointConv — Deep Convolutional Networks on 3D Point Clouds.
Wu et al. CVPR 2019. doi:10.1109/CVPR.2019.00985
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .build import MODELS
from utils import misc
from utils.cls_losses import ClassificationLossMixin
from utils.feature_hooks import FeatureExtractionMixin
from utils.logger import print_log
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message


def compute_density(xyz, bandwidth):
    """Gaussian density at every point — mean over ALL points, no ball query.
    xyz [B, N, C] -> [B, N]."""
    sqrdists = misc.square_distance(xyz, xyz)
    gaussion_density = torch.exp(-sqrdists / (2.0 * bandwidth * bandwidth)) / (2.5 * bandwidth)
    return gaussion_density.mean(dim=-1)


def sample_and_group(npoint, nsample, xyz, points, density_scale=None):
    """FPS to npoint centres, then kNN with nsample neighbours.
    Returns new_xyz, new_points, grouped_xyz_norm, idx (+ grouped density if given)."""
    B, N, C = xyz.shape
    S = min(npoint, N)
    k = min(nsample, N)

    fps_idx = misc.farthest_point_sample(xyz, S)
    new_xyz = misc.index_points(xyz, fps_idx)
    # neighbour order irrelevant: the nsample axis is contracted by a matmul with same-idx weights
    _, idx = misc.knn(xyz, new_xyz, k)

    grouped_xyz = misc.index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = misc.index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    if density_scale is None:
        return new_xyz, new_points, grouped_xyz_norm, idx
    grouped_density = misc.index_points(density_scale, idx)
    return new_xyz, new_points, grouped_xyz_norm, idx, grouped_density


def sample_and_group_all(xyz, points, density_scale=None):
    """Last stage: one centre at xyz.mean(1) (not the origin, not a point), grouping every point."""
    B, N, C = xyz.shape
    new_xyz = xyz.mean(dim=1, keepdim=True)
    grouped_xyz = xyz.view(B, 1, N, C) - new_xyz.view(B, 1, 1, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz

    if density_scale is None:
        return new_xyz, new_points, grouped_xyz
    grouped_density = density_scale.view(B, 1, N, 1)
    return new_xyz, new_points, grouped_xyz, grouped_density


class DensityNet(nn.Module):
    """Normalized inverse density -> per-neighbour scale. Final sigmoid is dead code; every layer gets ReLU."""

    def __init__(self, hidden_unit=(16, 8), activation='relu'):
        super(DensityNet, self).__init__()
        self.activation = activation
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        self.mlp_convs.append(nn.Conv2d(1, hidden_unit[0], 1))
        self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
        for i in range(1, len(hidden_unit)):
            self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
        self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], 1, 1))
        self.mlp_bns.append(nn.BatchNorm2d(1))

    def forward(self, density_scale):
        last = len(self.mlp_convs) - 1
        for i, conv in enumerate(self.mlp_convs):
            density_scale = self.mlp_bns[i](conv(density_scale))
            if i == last and self.activation == 'sigmoid':
                density_scale = torch.sigmoid(density_scale)
            else:
                density_scale = F.relu(density_scale)
        return density_scale


class WeightNet(nn.Module):
    """Local coordinates -> convolution weights. Shipped as 3 -> 8 -> 8 -> 16, each layer conv+BN+ReLU."""

    def __init__(self, in_channel, out_channel, hidden_unit=(8, 8)):
        super(WeightNet, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))

    def forward(self, localized_xyz):
        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            weights = F.relu(self.mlp_bns[i](conv(weights)))
        return weights


class PointConvDensitySetAbstraction(nn.Module):
    """One PointConv set-abstraction layer."""

    def __init__(self, npoint, nsample, in_channel, mlp, bandwidth, group_all,
                 weightnet_hidden=(8, 8), weightnet_out=16, density_hidden=(16, 8),
                 density_activation='relu'):
        super(PointConvDensitySetAbstraction, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet = WeightNet(3, weightnet_out, weightnet_hidden)
        self.linear = nn.Linear(weightnet_out * mlp[-1], mlp[-1])
        self.bn_linear = nn.BatchNorm1d(mlp[-1])
        self.densitynet = DensityNet(density_hidden, density_activation)
        self.group_all = group_all
        self.bandwidth = bandwidth
        self.out_channels = mlp[-1]

    def forward(self, xyz, points):
        """xyz [B, 3, N], points [B, D, N] or None -> new_xyz [B, 3, S], new_points [B, mlp[-1], S]."""
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        xyz_density = compute_density(xyz, self.bandwidth)
        inverse_density = 1.0 / xyz_density

        if self.group_all:
            new_xyz, new_points, grouped_xyz_norm, grouped_density = sample_and_group_all(
                xyz, points, inverse_density.view(B, N, 1))
        else:
            new_xyz, new_points, grouped_xyz_norm, _, grouped_density = sample_and_group(
                self.npoint, self.nsample, xyz, points, inverse_density.view(B, N, 1))
        S = new_xyz.shape[1]

        new_points = new_points.permute(0, 3, 2, 1)          # [B, C+D, nsample, S]
        for i, conv in enumerate(self.mlp_convs):
            new_points = F.relu(self.mlp_bns[i](conv(new_points)))

        inverse_max_density = grouped_density.max(dim=2, keepdim=True)[0]
        density_scale = grouped_density / inverse_max_density
        density_scale = self.densitynet(density_scale.permute(0, 3, 2, 1))
        new_points = new_points * density_scale

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = torch.matmul(new_points.permute(0, 3, 1, 2),
                                  weights.permute(0, 3, 2, 1)).view(B, S, -1)
        new_points = self.linear(new_points)
        new_points = self.bn_linear(new_points.permute(0, 2, 1))
        new_points = F.relu(new_points)
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points


@MODELS.register_module()
class PointConv_Classification(nn.Module, ClassificationLossMixin, FeatureExtractionMixin):
    """PointConv (PointConvDensityClsSsg) whole-cloud classifier, scratch-only.
    sa1/sa2/sa3/fc1/bn1/fc2/bn2/fc3 keep the reference names so its checkpoints load directly.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        print_log('[PointConv] Initializing PointConv model', logger='PointConv')

        self.config = config
        self.name = config.NAME
        self.cls_dim = config.cls_dim

        self.feature_dim = getattr(config, 'feature_dim', 3)
        stages = getattr(config, 'stages', None) or [
            {'npoint': 512, 'nsample': 32, 'mlp': [64, 64, 128], 'bandwidth': 0.1,
             'group_all': False},
            {'npoint': 128, 'nsample': 64, 'mlp': [128, 128, 256], 'bandwidth': 0.2,
             'group_all': False},
            {'npoint': 1, 'nsample': None, 'mlp': [256, 512, 1024], 'bandwidth': 0.4,
             'group_all': True},
        ]
        weightnet_hidden = list(getattr(config, 'weightnet_hidden', [8, 8]))
        weightnet_out = getattr(config, 'weightnet_out', 16)
        density_hidden = list(getattr(config, 'density_hidden', [16, 8]))
        density_activation = getattr(config, 'density_activation', 'relu')
        assert density_activation in ('relu', 'sigmoid'), \
            f"density_activation must be 'relu' or 'sigmoid', got {density_activation!r}"
        fc_dims = list(getattr(config, 'fc_dims', [512, 256]))
        self.dropout = getattr(config, 'dropout', 0.7)

        # in_channel = prev feature width + 3 local coords (grouping always concats xyz)
        sa_layers = []
        prev = self.feature_dim
        for s in stages:
            layer = PointConvDensitySetAbstraction(
                npoint=s['npoint'], nsample=s['nsample'], in_channel=prev + 3,
                mlp=list(s['mlp']), bandwidth=s['bandwidth'],
                group_all=s.get('group_all', False),
                weightnet_hidden=weightnet_hidden, weightnet_out=weightnet_out,
                density_hidden=density_hidden, density_activation=density_activation)
            sa_layers.append(layer)
            prev = layer.out_channels
        self.sa_layers = nn.ModuleList(sa_layers)
        for i, layer in enumerate(sa_layers, start=1):
            setattr(self, f'sa{i}', layer)                   # sa1/sa2/sa3 names for reference state_dicts
        self.feature_out = prev

        c = prev
        self.fc1 = nn.Linear(c, fc_dims[0])
        self.bn1 = nn.BatchNorm1d(fc_dims[0])
        self.drop1 = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(fc_dims[0], fc_dims[1])
        self.bn2 = nn.BatchNorm1d(fc_dims[1])
        self.drop2 = nn.Dropout(self.dropout)
        self.fc3 = nn.Linear(fc_dims[1], self.cls_dim)

        self._init_cls_loss(config, self.cls_dim)            # ce | focal
        self._init_feature_extraction()
        self.apply(self._init_weights)

        print_log(f'[PointConv] {len(sa_layers)} set-abstraction layers '
                  f'(npoint={[s["npoint"] for s in stages]}, '
                  f'nsample={[s["nsample"] for s in stages]}, '
                  f'bandwidth={[s["bandwidth"] for s in stages]}), '
                  f'feature {self.feature_out}, {self.cls_dim} classes',
                  logger='PointConv')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def load_model_from_ckpt(self, ckpt_path):
        if ckpt_path is None:
            print_log('Training from scratch!!!', logger='PointConv')
            return

        ckpt = torch.load(ckpt_path, map_location='cpu')
        if 'base_model' in ckpt:
            base_ckpt = {k.replace('module.', ''): v for k, v in ckpt['base_model'].items()}
        elif 'model' in ckpt:
            base_ckpt = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
        elif 'model_state_dict' in ckpt:                     # reference checkpoints
            base_ckpt = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
        else:
            base_ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}

        incompatible = self.load_state_dict(base_ckpt, strict=False)
        if incompatible.missing_keys:
            print_log('missing_keys', logger='PointConv')
            print_log(get_missing_parameters_message(incompatible.missing_keys), logger='PointConv')
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='PointConv')
            print_log(get_unexpected_parameters_message(incompatible.unexpected_keys), logger='PointConv')
        print_log(f'[PointConv] Successful Loading the ckpt from {ckpt_path}', logger='PointConv')

    def forward(self, pts):
        """pts: [B, N, 3] or [B, 3, N] -> raw logits [B, cls_dim] (loss applies its own log-softmax)."""
        if pts.size(-1) != 3 and pts.size(1) == 3:
            pts = pts.transpose(1, 2)
        xyz = pts.transpose(1, 2).contiguous()               # [B, 3, N]
        # no normals in BoneDataset: feed coords as the feature (feature_dim 0 -> None)
        feat = xyz if self.feature_dim else None

        for layer in self.sa_layers:
            xyz, feat = layer(xyz, feat)

        global_feat = feat.view(feat.shape[0], -1)           # [B, mlp[-1]]

        self._store_features(global_feat)
        x = self.drop1(F.relu(self.bn1(self.fc1(global_feat))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)
