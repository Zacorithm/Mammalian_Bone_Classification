"""PointMLP — Rethinking Network Design and Local Geometry in Point Cloud: A Simple Residual MLP Framework.
Ma et al. ICLR 2022. arXiv:2202.07123
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from typing import Optional, Tuple, List
import numpy as np
from timm.models.layers import trunc_normal_
from .build import MODELS
from utils.cls_losses import ClassificationLossMixin
from utils.feature_hooks import FeatureExtractionMixin
from utils.logger import print_log
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils import misc


def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)


class LocalGrouper(nn.Module):
    """xyz [b,p,3], fea [b,p,d] -> new_xyz [b,g,3], new_fea [b,g,k,d]."""

    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="center", radius=None):
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel = 3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))
        self.radius = radius

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()

        fps_indices = misc.farthest_point_sample(xyz, self.groups)   # [B, npoint]
        new_xyz = misc.index_points(xyz, fps_indices)
        new_points = misc.index_points(points, fps_indices)

        if self.radius is not None:
            _, idx = misc.query_ball_point(xyz, new_xyz, self.radius, self.kneighbors)
        else:
            _, idx = misc.knn(xyz, new_xyz, self.kneighbors)

        grouped_xyz = misc.index_points(xyz, idx)
        grouped_points = misc.index_points(points, idx)

        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)

        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize == "anchor":
                mean = torch.cat([new_points, new_xyz], dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)
            std = torch.std((grouped_points-mean).reshape(B, -1), dim=-1, keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points-mean)/(std + 1e-5)
            grouped_points = self.affine_alpha*grouped_points + self.affine_beta

        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_xyz, new_points


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):
    """[b,g,k,d] -> [b,d,g]."""

    def __init__(self, channels, out_channels, blocks=1, groups=1, res_expansion=1, bias=True,
                 activation='relu', use_xyz=True):
        super(PreExtraction, self).__init__()
        in_channels = 3+2*channels if use_xyz else 2*channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,
                                bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class PosExtraction(nn.Module):
    """[b,d,g] -> [b,d,g]."""

    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        return self.operation(x)


@MODELS.register_module()
class PointMLP_Classification(nn.Module, ClassificationLossMixin, FeatureExtractionMixin):
    """PointMLP classifier."""

    def __init__(self, config, **kwargs):
        super(PointMLP_Classification, self).__init__()
        self.config = config
        self.name = 'PointMLP'
        self.num_classes = config.cls_dim
        self.num_points = getattr(config, 'num_points', 1024)

        print_log(f'[PointMLP] Building model for {self.num_classes} classes', logger='PointMLP')

        dim_expansion = [2, 2, 2, 2]
        pre_blocks = [2, 2, 2, 2]
        pos_blocks = [2, 2, 2, 2]
        k_neighbors = [24, 24, 24, 24]
        reducers = [2, 2, 2, 2]

        self.stages = len(pre_blocks)
        self.embedding = ConvBNReLU1D(3, 64, bias=False, activation="relu")
        self.radius = None                              # kNN, not ball query

        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()

        last_channel = 64
        last_radius = self.radius
        anchor_points = self.num_points

        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce

            local_grouper = LocalGrouper(last_channel, anchor_points, kneighbor, False, "anchor", last_radius)
            self.local_grouper_list.append(local_grouper)

            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=1,
                                           res_expansion=1.0, bias=False, activation="relu", use_xyz=False)
            self.pre_blocks_list.append(pre_block_module)

            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=1,
                                           res_expansion=1.0, bias=False, activation="relu")
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel
            if self.radius is not None:
                last_radius = 2 * last_radius

        self.act = get_activation("relu")
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, 512),
            nn.BatchNorm1d(512),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)
        )

        self._init_cls_loss(config, getattr(self, 'cls_dim', None) or self.num_classes)   # ce | focal
        self._init_feature_extraction()
        self.features_for_visualization = None

    def load_model_from_ckpt(self, ckpt_path):
        if ckpt_path is None:
            print_log('Training PointMLP from scratch!!!', logger='PointMLP')
            self.apply(self._init_weights)
            return

        print_log(f'Loading PointMLP checkpoint from {ckpt_path}', logger='PointMLP')
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
            print_log('missing_keys', logger='PointMLP')
            print_log(get_missing_parameters_message(incompatible.missing_keys), logger='PointMLP')
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='PointMLP')
            print_log(get_unexpected_parameters_message(incompatible.unexpected_keys), logger='PointMLP')
        print_log(f'[PointMLP] Successful Loading the ckpt from {ckpt_path}', logger='PointMLP')

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
            if pts.shape[1] == 3:                       # (B, 3, N)
                x = pts
                xyz = pts.permute(0, 2, 1)
            else:                                       # (B, N, 3)
                x = pts.permute(0, 2, 1)
                xyz = pts

        device = x.device
        batch_size, _, _ = x.size()
        x = self.embedding(x).to(device)                # [B, D, N]

        for i in range(self.stages):
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))   # [b,g,3], [b,g,k,d]
            x = self.pre_blocks_list[i](x).to(device)   # [b,d,g]
            x = self.pos_blocks_list[i](x).to(device)   # [b,d,g]

        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1).to(device)
        self._store_features(x)
        return self.classifier(x).to(device)


@MODELS.register_module()
class PointMLPElite_Classification(nn.Module, ClassificationLossMixin, FeatureExtractionMixin):
    """PointMLP-Elite: efficient PointMLP variant."""

    def __init__(self, config, **kwargs):
        super(PointMLPElite_Classification, self).__init__()
        self.config = config
        self.name = 'PointMLPElite'
        self.num_classes = config.cls_dim
        self.num_points = getattr(config, 'num_points', 1024)

        print_log(f'[PointMLPElite] Building model for {self.num_classes} classes', logger='PointMLPElite')

        dim_expansion = [2, 2, 2, 1]
        pre_blocks = [1, 1, 2, 1]
        pos_blocks = [1, 1, 2, 1]
        k_neighbors = [24, 24, 24, 24]
        reducers = [2, 2, 2, 2]

        self.stages = len(pre_blocks)
        self.embedding = ConvBNReLU1D(3, 32, bias=False, activation="relu")

        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()

        last_channel = 32
        anchor_points = self.num_points

        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce

            local_grouper = LocalGrouper(last_channel, anchor_points, kneighbor, False, "anchor")
            self.local_grouper_list.append(local_grouper)

            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=1,
                                           res_expansion=0.25, bias=False, activation="relu", use_xyz=False)
            self.pre_blocks_list.append(pre_block_module)

            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=1,
                                           res_expansion=0.25, bias=False, activation="relu")
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel

        self.act = get_activation("relu")
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, 512),
            nn.BatchNorm1d(512),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)
        )

        self._init_cls_loss(config, getattr(self, 'cls_dim', None) or self.num_classes)   # ce | focal
        self._init_feature_extraction()

    def load_model_from_ckpt(self, ckpt_path):
        if ckpt_path is None:
            print_log('Training PointMLPElite from scratch!!!', logger='PointMLPElite')
            self.apply(self._init_weights)
            return

        print_log(f'Loading PointMLPElite checkpoint from {ckpt_path}', logger='PointMLPElite')
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
            print_log('missing_keys', logger='PointMLPElite')
            print_log(get_missing_parameters_message(incompatible.missing_keys), logger='PointMLPElite')
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='PointMLPElite')
            print_log(get_unexpected_parameters_message(incompatible.unexpected_keys), logger='PointMLPElite')
        print_log(f'[PointMLPElite] Successful Loading the ckpt from {ckpt_path}', logger='PointMLPElite')

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
            if pts.shape[1] == 3:                       # (B, 3, N)
                x = pts
                xyz = pts.permute(0, 2, 1)
            else:                                       # (B, N, 3)
                x = pts.permute(0, 2, 1)
                xyz = pts

        device = x.device
        batch_size, _, _ = x.size()
        x = self.embedding(x).to(device)                # [B, D, N]

        for i in range(self.stages):
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))   # [b,g,3], [b,g,k,d]
            x = self.pre_blocks_list[i](x).to(device)   # [b,d,g]
            x = self.pos_blocks_list[i](x).to(device)   # [b,d,g]

        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1).to(device)
        self._store_features(x)
        return self.classifier(x).to(device)
