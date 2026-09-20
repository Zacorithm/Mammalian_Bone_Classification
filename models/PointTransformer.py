"""Point Transformer.
Zhao et al. ICCV 2021. doi:10.1109/ICCV48922.2021.01595
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
import numpy as np
from timm.models.layers import trunc_normal_
from .build import MODELS
from utils.logger import print_log
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.cls_losses import ClassificationLossMixin
from utils.feature_hooks import FeatureExtractionMixin
from utils import misc


class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample

        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)

        self.linear_p = nn.Sequential(                    # BatchNorm1d, not LayerNorm1d
            nn.Linear(3, 3),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_planes),
        )
        self.linear_w = nn.Sequential(
            nn.BatchNorm1d(mid_planes),
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes, out_planes // share_planes),
            nn.BatchNorm1d(out_planes // share_planes),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // share_planes, out_planes // share_planes),
        )
        self.softmax = nn.Softmax(dim=2)

    def forward(self, xyz, features):
        """xyz [B, N, 3], features [B, N, C] -> [B, N, out_planes]."""
        B, N, _ = xyz.shape
        actual_nsample = min(self.nsample, N)

        x_q = self.linear_q(features)                     # (B, N, mid_planes)
        x_k = self.linear_k(features)
        x_v = self.linear_v(features)                     # (B, N, out_planes)

        _, idx = misc.knn(xyz, xyz, actual_nsample)       # (B, N, actual_nsample)
        grouped_xyz = misc.index_points(xyz, idx)         # (B, N, actual_nsample, 3)
        x_k_grouped = misc.index_points(x_k, idx)
        x_v_grouped = misc.index_points(x_v, idx)

        xyz_expanded = xyz.unsqueeze(2).expand(-1, -1, actual_nsample, -1)
        p_r = grouped_xyz - xyz_expanded                  # (B, N, actual_nsample, 3)

        p_r_flat = p_r.view(-1, 3)
        p_encoded = self.linear_p[0](p_r_flat)
        p_encoded = self.linear_p[1](p_encoded.unsqueeze(-1)).squeeze(-1)
        p_encoded = self.linear_p[2](p_encoded)
        p_encoded = self.linear_p[3](p_encoded)
        p_r_encoded = p_encoded.view(B, N, actual_nsample, self.out_planes)

        # PT-V1 vector attention: r_qk = k_j - q_i + delta(p_r). With
        # mid_planes == out_planes the blueprint's einops.reduce is an identity
        # reshape, so the plain add is exact.
        x_q_expanded = x_q.unsqueeze(2).expand(-1, -1, actual_nsample, -1)
        r_qk = x_k_grouped - x_q_expanded + p_r_encoded

        r_qk_flat = r_qk.view(-1, self.mid_planes)
        w = self.linear_w[0](r_qk_flat.unsqueeze(-1)).squeeze(-1)
        w = self.linear_w[1](w)
        w = self.linear_w[2](w)
        w = self.linear_w[3](w.unsqueeze(-1)).squeeze(-1)
        w = self.linear_w[4](w)
        w = self.linear_w[5](w)
        w = w.view(B, N, actual_nsample, -1)              # (B, N, actual_nsample, out_planes//share_planes)
        w = self.softmax(w)

        x_v_combined = x_v_grouped + p_r_encoded
        x_v_reshaped = x_v_combined.view(B, N, actual_nsample, self.share_planes, -1)
        w_expanded = w.unsqueeze(3).expand(-1, -1, -1, self.share_planes, -1)
        x = (x_v_reshaped * w_expanded).sum(dim=2)
        return x.view(B, N, -1)                           # (B, N, out_planes)


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(Bottleneck, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, xyz, features):
        """xyz [B, N, 3], features [B, N, C] -> xyz unchanged, features [B, N, C]."""
        identity = features

        x = self.linear1(features)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)   # BatchNorm expects (B, C, N)
        x = self.relu(x)

        x = self.transformer(xyz, x)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x = self.relu(x)

        x = self.linear3(x)
        x = self.bn3(x.transpose(1, 2)).transpose(1, 2)

        x += identity
        x = self.relu(x)
        return xyz, x


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride = stride
        self.nsample = nsample
        self.out_planes = out_planes

        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, xyz, features):
        """xyz [B, N, 3], features [B, N, C] -> new_xyz [B, M, 3], new_features [B, M, out_planes]."""
        B, N, C = features.shape

        if self.stride != 1:
            M = N // self.stride
            fps_indices = misc.farthest_point_sample(xyz, M)
            new_xyz = misc.index_points(xyz, fps_indices)         # (B, M, 3)

            _, idx = misc.knn(xyz, new_xyz, self.nsample)         # (B, M, nsample)
            grouped_features = misc.index_points(features, idx)   # (B, M, nsample, C)
            grouped_xyz = misc.index_points(xyz, idx)

            new_xyz_expanded = new_xyz.unsqueeze(2).expand(-1, -1, self.nsample, -1)
            relative_pos = grouped_xyz - new_xyz_expanded
            # blueprint layout: relative xyz first, then features (3 + C)
            grouped_features = torch.cat([relative_pos, grouped_features], dim=-1)

            x = self.linear(grouped_features)
            x = x.transpose(2, 3)                                 # (B, M, out_planes, nsample)
            x = self.bn(x.transpose(1, 2).contiguous().view(-1, self.out_planes, self.nsample)).view(B, self.out_planes, M, self.nsample).transpose(1, 2)
            x = self.relu(x)
            x = F.max_pool2d(x, (1, self.nsample)).squeeze(-1)    # (B, M, out_planes)
        else:
            new_xyz = xyz
            x = self.linear(features)
            x = self.bn(x.transpose(1, 2)).transpose(1, 2)
            x = self.relu(x)

        return new_xyz, x


@MODELS.register_module()
class PointTransformerV1_Classification(nn.Module, ClassificationLossMixin, FeatureExtractionMixin):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.name = 'PointTransformerV1_Classification'
        self.num_classes = config.cls_dim
        self.num_points = getattr(config, 'num_points', 1024)

        print_log(f'[PointTransformerV1_Classification] Building model for {self.num_classes} classes',
                  logger='PointTransformerV1_Classification')

        self.in_channels = getattr(config, 'in_channels', 3)
        self.in_planes = self.in_channels
        planes = [32, 64, 128, 256, 512]
        # blueprint's cls file builds blocks-1 attention blocks per stage (likely an
        # upstream bug); the full count is kept here on purpose.
        blocks = [1, 1, 1, 1, 1]
        share_planes = 8
        strides = [1, 4, 4, 4, 4]
        nsamples = [8, 16, 16, 16, 16]                    # blueprint: stage 1 uses 8 neighbours

        # stage 0's stride-1 TransitionDown is the stem (Linear(in_channels -> 32) + BN + ReLU)
        self.stages = nn.ModuleList()
        for i in range(len(planes)):
            stage_modules = nn.ModuleList()
            if i == 0:
                transition = TransitionDown(self.in_channels, planes[0], stride=strides[i], nsample=nsamples[i])
            else:
                transition = TransitionDown(self.in_planes, planes[i], stride=strides[i], nsample=nsamples[i])
            stage_modules.append(transition)

            self.in_planes = planes[i]
            for _ in range(blocks[i]):
                bottleneck = Bottleneck(self.in_planes, self.in_planes, share_planes, nsamples[i])
                stage_modules.append(bottleneck)

            self.stages.append(stage_modules)

        self.classifier = nn.Sequential(
            nn.Linear(planes[-1], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, self.num_classes),
        )

        self._init_cls_loss(config, self.num_classes)    # ce | focal
        self._init_feature_extraction()

    def load_model_from_ckpt(self, ckpt_path):
        if ckpt_path is None:
            print_log('Training PointTransformer from scratch!!!', logger='PointTransformer')
            self.apply(self._init_weights)
            return

        print_log(f'Loading PointTransformer checkpoint from {ckpt_path}', logger='PointTransformer')
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
            print_log('missing_keys', logger='PointTransformer')
            print_log(get_missing_parameters_message(incompatible.missing_keys), logger='PointTransformer')
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='PointTransformer')
            print_log(get_unexpected_parameters_message(incompatible.unexpected_keys), logger='PointTransformer')
        print_log(f'[PointTransformer] Successful Loading the ckpt from {ckpt_path}', logger='PointTransformer')

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
        """pts: dict{coord[,feat]} or tensor [B, N, 3] / [B, 3, N] -> logits [B, cls_dim]."""
        if isinstance(pts, dict):
            if 'coord' in pts:
                xyz = pts['coord']
                if 'feat' in pts and pts['feat'] is not None:
                    feat = pts['feat']
                    features = torch.cat([xyz, feat], dim=-1) if self.in_channels > 3 else xyz
                else:
                    features = xyz
            else:
                features = pts
                xyz = features[..., :3]
        else:
            if pts.dim() == 2:
                pts = pts.unsqueeze(0)                            # (N, 3) -> (1, N, 3)

            if pts.dim() == 3:
                if pts.shape[-1] == 3 or pts.shape[-1] > 3:       # (B, N, 3) or (B, N, C)
                    xyz = pts[..., :3]
                    features = pts if self.in_channels > 3 else xyz
                elif pts.shape[1] == 3:                           # (B, 3, N)
                    xyz = pts.permute(0, 2, 1)
                    features = xyz
                elif pts.shape[1] > 3:                            # (B, C, N)
                    features = pts.permute(0, 2, 1)
                    xyz = features[..., :3]
                else:
                    features = pts
                    xyz = features[..., :3]

        B, N, C = features.shape
        if C != self.in_channels:
            if C > self.in_channels:
                features = features[..., :self.in_channels]
            else:
                padding = torch.zeros(B, N, self.in_channels - C, device=features.device)
                features = torch.cat([features, padding], dim=-1)

        for stage in self.stages:
            for module in stage:
                if isinstance(module, (TransitionDown, Bottleneck)):
                    xyz, features = module(xyz, features)

        x = features.mean(dim=1)                                  # blueprint cls variant: per-sample mean
        self._store_features(x)
        return self.classifier(x)


def _build_v1_stages(model, blocks):
    """(Re)build the PT-V1 encoder stages for a given block-count schedule."""
    planes = [32, 64, 128, 256, 512]
    share_planes = 8
    strides = [1, 4, 4, 4, 4]
    nsamples = [8, 16, 16, 16, 16]

    model.stages = nn.ModuleList()
    model.in_planes = model.in_channels

    for i in range(len(planes)):
        stage_modules = nn.ModuleList()
        if i == 0:
            transition = TransitionDown(model.in_channels, planes[0], stride=strides[i], nsample=nsamples[i])
        else:
            transition = TransitionDown(model.in_planes, planes[i], stride=strides[i], nsample=nsamples[i])
        stage_modules.append(transition)

        model.in_planes = planes[i]
        for _ in range(blocks[i]):
            bottleneck = Bottleneck(model.in_planes, model.in_planes, share_planes, nsamples[i])
            stage_modules.append(bottleneck)

        model.stages.append(stage_modules)


@MODELS.register_module()
class PointTransformerV1_Classification_38(PointTransformerV1_Classification):
    def __init__(self, config, **kwargs):
        super(PointTransformerV1_Classification_38, self).__init__(copy.deepcopy(config), **kwargs)
        _build_v1_stages(self, blocks=[1, 2, 2, 2, 2])


@MODELS.register_module()
class PointTransformerV1_Classification_50(PointTransformerV1_Classification):
    def __init__(self, config, **kwargs):
        super(PointTransformerV1_Classification_50, self).__init__(copy.deepcopy(config), **kwargs)
        _build_v1_stages(self, blocks=[1, 2, 3, 5, 2])
