"""PCT — Point Cloud Transformer.
Guo et al. Computational Visual Media 7(2), 2021. doi:10.1007/s41095-021-0229-5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_
from .build import MODELS
from utils.cls_losses import ClassificationLossMixin
from utils.feature_hooks import FeatureExtractionMixin
from utils.logger import print_log
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils import misc


def sample_and_group(npoint, nsample, xyz, points, use_xyz=False):
    B, N, C = xyz.shape
    S = npoint

    fps_idx = misc.farthest_point_sample(xyz, npoint)             # [B, S]
    new_xyz = misc.index_points(xyz, fps_idx)
    _, idx = misc.knn(xyz, new_xyz, nsample)                      # [B, S, K]

    grouped_xyz = misc.index_points(xyz, idx)                     # [B, S, K, 3]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, 3)

    if points is not None:
        new_points_sampled = misc.index_points(points, fps_idx)   # [B, S, D]
        grouped_points = misc.index_points(points, idx)           # [B, S, K, D]
        grouped_points_norm = grouped_points - new_points_sampled.view(B, S, 1, -1)

        new_points = torch.cat([
            grouped_points_norm,
            new_points_sampled.view(B, S, 1, -1).repeat(1, 1, nsample, 1)
        ], dim=-1)                                                # [B, S, K, 2*D]
        if use_xyz:
            new_points = torch.cat([grouped_xyz_norm, new_points], dim=-1)
    else:
        new_points = grouped_xyz_norm if use_xyz else None

    return new_xyz, new_points


class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        b, n, s, d = x.size()                                     # [B, N, S, D]
        x = x.permute(0, 1, 3, 2).reshape(-1, d, s)              # [B*N, D, S]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)          # [B*N, out_channels]
        return x.reshape(b, n, -1).permute(0, 2, 1)             # [B, out_channels, N]


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, xyz):
        if xyz is not None:
            x = x + xyz
        x_q = self.q_conv(x).permute(0, 2, 1)                    # [B, N, C//4]
        x_k = self.k_conv(x)                                     # [B, C//4, N]
        x_v = self.v_conv(x)                                     # [B, C, N]
        energy = torch.bmm(x_q, x_k)                             # [B, N, N]
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        x_r = torch.bmm(x_v, attention)                          # [B, C, N]
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        return x + x_r


class Point_Transformer_Last(nn.Module):
    def __init__(self, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv_pos = nn.Conv1d(3, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x, xyz):
        xyz = xyz.permute(0, 2, 1)                               # [B, 3, N]
        xyz = self.conv_pos(xyz)                                 # [B, channels, N]
        x = self.relu(self.bn1(self.conv1(x)))                   # [B, channels, N]

        x1 = self.sa1(x, xyz)
        x2 = self.sa2(x1, xyz)
        x3 = self.sa3(x2, xyz)
        x4 = self.sa4(x3, xyz)
        return torch.cat((x1, x2, x3, x4), dim=1)


@MODELS.register_module()
class PCT_Classification(nn.Module, ClassificationLossMixin, FeatureExtractionMixin):
    def __init__(self, config, **kwargs):
        super().__init__()
        print_log('[PointCloudTransformer] Initializing Point Transformer model', logger='PointCloudTransformer')

        self.config = config
        self.name = config.NAME
        self.cls_dim = config.cls_dim

        self.simple_pct = getattr(config, 'simple_pct', False)
        self.npoint1 = getattr(config, 'npoint1', 512)
        self.npoint2 = getattr(config, 'npoint2', 256)
        self.nsample = getattr(config, 'nsample', 32)
        self.sa_layers = getattr(config, 'sa_layers', 4)
        self.dim = getattr(config, 'dim', 256)

        if self.simple_pct:
            # SPCT: attention-only, no sampling/grouping, no position embedding
            assert self.sa_layers == 4, "sa_layers must be 4 (SA_Layer count is not configurable)"
            self.simple_dim = 128
            self.conv1 = nn.Conv1d(3, self.simple_dim, kernel_size=1, bias=False)
            self.conv2 = nn.Conv1d(self.simple_dim, self.simple_dim, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm1d(self.simple_dim)
            self.bn2 = nn.BatchNorm1d(self.simple_dim)
            self.relu = nn.ReLU()

            self.sa1 = SA_Layer(self.simple_dim)
            self.sa2 = SA_Layer(self.simple_dim)
            self.sa3 = SA_Layer(self.simple_dim)
            self.sa4 = SA_Layer(self.simple_dim)

            fusion_channels = self.simple_dim * self.sa_layers
        else:
            # full PCT: hierarchical sampling + local aggregation
            assert self.dim == 256, "dim must be 256 for Full PCT (Local_op channel widths are fixed to 128/256)"
            assert self.sa_layers == 4, "sa_layers must be 4 (Point_Transformer_Last hardcodes 4 SA layers)"

            self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
            self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(64)
            self.relu = nn.ReLU()

            self.gather_local_0 = Local_op(in_channels=self.dim // 2, out_channels=128)
            self.gather_local_1 = Local_op(in_channels=self.dim, out_channels=256)
            self.pt_last = Point_Transformer_Last(channels=self.dim)

            fusion_channels = self.dim * self.sa_layers + self.dim

        self.conv_fuse = nn.Sequential(
            nn.Conv1d(fusion_channels, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.cls_head = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, self.cls_dim)
        )

        self._init_cls_loss(config, self.cls_dim)                # ce | focal
        self._init_feature_extraction()
        self.apply(self._init_weights)

        print_log(f'[PointCloudTransformer] Model initialized with {self.cls_dim} classes', logger='PointCloudTransformer')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def load_model_from_ckpt(self, ckpt_path):
        if ckpt_path is None:
            print_log('Training from scratch!!!', logger='PointCloudTransformer')
            return

        ckpt = torch.load(ckpt_path, map_location='cpu')
        if 'base_model' in ckpt:
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        elif 'model' in ckpt:
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['model'].items()}
        else:
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}

        incompatible = self.load_state_dict(base_ckpt, strict=False)
        if incompatible.missing_keys:
            print_log('missing_keys', logger='PointCloudTransformer')
            print_log(get_missing_parameters_message(incompatible.missing_keys), logger='PointCloudTransformer')
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='PointCloudTransformer')
            print_log(get_unexpected_parameters_message(incompatible.unexpected_keys), logger='PointCloudTransformer')

        print_log(f'[PointCloudTransformer] Successful Loading the ckpt from {ckpt_path}', logger='PointCloudTransformer')

    def forward(self, pts):
        """pts: [B, N, 3] or [B, 3, N] -> logits [B, cls_dim]."""
        if pts.dim() == 3 and pts.size(1) == 3:
            pts = pts.transpose(2, 1)                            # [B, 3, N] -> [B, N, 3]

        xyz = pts                                                # [B, N, 3]
        batch_size, _, _ = xyz.shape

        if self.simple_pct:
            x = xyz.transpose(2, 1)                              # [B, 3, N]
            x = self.relu(self.bn1(self.conv1(x)))               # [B, 128, N]
            x = self.relu(self.bn2(self.conv2(x)))               # [B, 128, N]

            x1 = self.sa1(x, None)
            x2 = self.sa2(x1, None)
            x3 = self.sa3(x2, None)
            x4 = self.sa4(x3, None)
            x = torch.cat((x1, x2, x3, x4), dim=1)               # [B, 512, N]
        else:
            x = xyz.transpose(2, 1)                              # [B, 3, N]
            x = self.relu(self.bn1(self.conv1(x)))               # [B, 64, N]
            x = self.relu(self.bn2(self.conv2(x)))               # [B, 64, N]
            x = x.permute(0, 2, 1)                               # [B, N, 64]

            new_xyz, new_feature = sample_and_group(
                npoint=self.npoint1, nsample=self.nsample, xyz=xyz, points=x
            )                                                   # new_feature: [B, 512, 32, 128]
            feature_0 = self.gather_local_0(new_feature)         # [B, 128, 512]

            feature = feature_0.permute(0, 2, 1)                 # [B, 512, 128]
            new_xyz, new_feature = sample_and_group(
                npoint=self.npoint2, nsample=self.nsample, xyz=new_xyz, points=feature
            )                                                   # new_feature: [B, 256, 32, 256]
            feature_1 = self.gather_local_1(new_feature)         # [B, 256, 256]

            x = self.pt_last(feature_1, new_xyz)                 # [B, 1024, 256]
            x = torch.cat([x, feature_1], dim=1)                 # [B, 1280, 256]

        x = self.conv_fuse(x)                                    # [B, 1024, 256]
        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)          # [B, 1024]

        self._store_features(x)
        return self.cls_head(x)
