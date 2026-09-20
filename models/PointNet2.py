"""PointNet++ — Deep Hierarchical Feature Learning on Point Sets in a Metric Space.
Qi et al. NeurIPS 2017. arXiv:1706.02413
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


class GroupingSamplingLayer(object):
    @staticmethod
    def __call__(points, num_samples=None, radius=None, max_points=None, features=None, sample_all=False):
        device = points.device
        B, N, _ = points.shape

        if sample_all:
            sampled_points = torch.zeros(B, 1, 3, device=device)
            grouped_points = points.view(B, 1, N, 3)
            if features is not None:
                grouped_features = features.view(B, 1, N, -1)
                sampled_features = torch.cat([grouped_points, grouped_features], dim=-1)
            else:
                sampled_features = grouped_points
        else:
            sampled_points = misc.fps(points, num_samples)

            if radius is None:
                _, grouped_points_indices = misc.knn(points, sampled_points, max_points)
            else:
                _, grouped_points_indices = misc.query_ball_point(points, sampled_points, radius, max_points)

            grouped_points = misc.index_points(points, grouped_points_indices)
            grouped_points = grouped_points - sampled_points.view(B, num_samples, 1, 3)

            if features is not None:
                grouped_features = misc.index_points(features, grouped_points_indices)
                sampled_features = torch.cat([grouped_points, grouped_features], dim=-1)
            else:
                sampled_features = grouped_points

        return sampled_points, sampled_features


class MLP(nn.Module):
    def __init__(self, input_size, layers_list):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.layers_list = layers_list
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        input_layer_size = input_size
        for layer in layers_list:
            self.conv_layers.append(nn.Conv2d(input_layer_size, layer, 1))
            self.bn_layers.append(nn.BatchNorm2d(layer))
            input_layer_size = layer

    def forward(self, input):
        layer_output = input
        for i in range(len(self.conv_layers)):
            conv = self.conv_layers[i]
            bn = self.bn_layers[i]
            layer_output = F.relu(bn(conv(layer_output)))
        return layer_output


class SetAbstraction(nn.Module):
    def __init__(self, input_size, mlp_layers, num_samples=None, radius=None, max_points=None, sample_all=False):
        super(SetAbstraction, self).__init__()
        self.MLP = MLP(input_size, mlp_layers)
        self.num_samples = num_samples
        self.radius = radius
        self.max_points = max_points
        self.sample_all = sample_all

    def forward(self, points, features):
        """points [B, N, 3], features [B, N, C] or None -> new_points, flattened new_features [B, feat]."""
        new_points, new_features = GroupingSamplingLayer()(
            points, self.num_samples, self.radius, self.max_points, features, self.sample_all
        )

        new_features = new_features.permute(0, 3, 2, 1)          # (B, feat, max_points, num_samples)
        new_features = self.MLP(new_features)
        new_features = torch.max(new_features, 2)[0]             # (B, feat, num_samples)
        new_features = new_features.transpose(1, 2).contiguous()
        new_features = new_features.view(new_features.size(0), -1)
        return new_points, new_features


class SetAbstractionMSG(nn.Module):
    def __init__(self, input_size, mlp_layers_list, num_samples=None, radius_list=None, max_points_list=None):
        super(SetAbstractionMSG, self).__init__()
        self.MLP_list = nn.ModuleList()
        for mlp_layers_item in mlp_layers_list:
            self.MLP_list.append(MLP(input_size, mlp_layers_item))

        self.num_samples = num_samples
        self.radius_list = radius_list
        self.max_points_list = max_points_list

    def forward(self, points, features):
        """points [B, N, 3], features [B, N, C] or None -> new_points, concatenated new_features [B, feat]."""
        device = points.device
        B, N, C = points.shape

        sampled_points = misc.fps(points, self.num_samples)
        sampled_features_list = []

        for i in range(len(self.radius_list)):
            radius = self.radius_list[i]
            max_points = self.max_points_list[i]
            MLP = self.MLP_list[i]

            _, grouped_points_indices = misc.query_ball_point(points, sampled_points, radius, max_points)
            grouped_points = misc.index_points(points, grouped_points_indices)
            grouped_points = grouped_points - sampled_points.view(B, self.num_samples, 1, 3)

            if features is not None:
                grouped_features = misc.index_points(features, grouped_points_indices)
                scale_features = torch.cat([grouped_points, grouped_features], dim=-1)
            else:
                scale_features = grouped_points

            scale_features = scale_features.permute(0, 3, 2, 1)  # (B, feat, max_points, num_samples)
            scale_features = MLP(scale_features)
            scale_features = torch.max(scale_features, 2)[0]     # (B, feat, num_samples)
            scale_features = scale_features.transpose(1, 2)      # (B, num_samples, feat)

            sampled_features_list.append(scale_features)

        sampled_features = torch.cat(sampled_features_list, dim=2)
        sampled_features = sampled_features.reshape(B, -1)
        return sampled_points, sampled_features


class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.5):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(p=dropout)
        self.drop2 = nn.Dropout(p=dropout)

    def forward(self, input):
        xb = self.drop1(F.relu(self.bn1(self.fc1(input))))
        xb = self.drop2(F.relu(self.bn2(self.fc2(xb))))
        return self.fc3(xb)


def _init_pn2_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def _load_pn2_ckpt(model, ckpt_path, tag, verbose=False):
    if ckpt_path is None:
        print_log(f'Training {tag} from scratch!!!', logger=tag)
        model.apply(model._init_weights)
        return

    print_log(f'Loading {tag} checkpoint from {ckpt_path}', logger=tag)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'base_model' in ckpt:
        state_dict = ckpt['base_model']
    elif 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    processed = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}
    incompatible = model.load_state_dict(processed, strict=False)
    if verbose and incompatible.missing_keys:
        print_log('missing_keys', logger=tag)
        print_log(get_missing_parameters_message(incompatible.missing_keys), logger=tag)
    if verbose and incompatible.unexpected_keys:
        print_log('unexpected_keys', logger=tag)
        print_log(get_unexpected_parameters_message(incompatible.unexpected_keys), logger=tag)
    print_log(f'[{tag}] Successful Loading the ckpt from {ckpt_path}', logger=tag)


@MODELS.register_module()
class PointNet2SSG_Classification(nn.Module, ClassificationLossMixin, FeatureExtractionMixin):
    """PointNet++ Single-Scale Grouping (SSG), ball query."""

    def __init__(self, config, **kwargs):
        super(PointNet2SSG_Classification, self).__init__()
        self.config = config
        self.name = 'PointNet2SSG'
        self.num_classes = config.cls_dim

        print_log(f'[PointNet2SSG] Building model for {self.num_classes} classes', logger='PointNet2SSG')

        self.sa1 = SetAbstraction(3, [64, 64, 128], 512, 0.2, 32, sample_all=False)
        self.sa2 = SetAbstraction(3+128, [128, 128, 256], 128, 0.4, 64, sample_all=False)
        self.sa3 = SetAbstraction(3+256, [256, 512, 1024], sample_all=True)
        self.classifier = Classifier(1024, self.num_classes)

        self._init_cls_loss(config, getattr(self, 'cls_dim', None) or self.num_classes)   # ce | focal
        self._init_feature_extraction()

    def load_model_from_ckpt(self, ckpt_path):
        _load_pn2_ckpt(self, ckpt_path, 'PointNet2SSG', verbose=True)

    def _init_weights(self, m):
        _init_pn2_weights(m)

    def forward(self, pts):
        """pts: [B, N, 3] or [B, 3, N] -> logits [B, cls_dim]."""
        if pts.dim() == 3 and pts.shape[1] == 3:
            pts = pts.permute(0, 2, 1)                           # (B, 3, N) -> (B, N, 3)

        B, N, C = pts.shape
        points = pts[:, :, :3]
        features = None
        points, features = self.sa1(points, features)

        features = features.view(B, points.shape[1], -1).contiguous()
        points_with_features = torch.cat([points, features], dim=-1)
        points, features = self.sa2(points, points_with_features[:, :, 3:])

        features = features.view(B, points.shape[1], -1).contiguous()
        points_with_features = torch.cat([points, features], dim=-1)
        points, features = self.sa3(points, points_with_features[:, :, 3:])

        self._store_features(features)
        return self.classifier(features)


@MODELS.register_module()
class PointNet2SSGKNN_Classification(nn.Module, ClassificationLossMixin, FeatureExtractionMixin):
    """PointNet++ Single-Scale Grouping (SSG), kNN."""

    def __init__(self, config, **kwargs):
        super(PointNet2SSGKNN_Classification, self).__init__()
        self.config = config
        self.name = 'PointNet2SSGKNN'
        self.num_classes = config.cls_dim

        print_log(f'[PointNet2SSGKNN] Building model for {self.num_classes} classes', logger='PointNet2SSGKNN')

        self.sa1 = SetAbstraction(3, [64, 64, 128], 512, None, 32, sample_all=False)
        self.sa2 = SetAbstraction(3+128, [128, 128, 256], 128, None, 64, sample_all=False)
        self.sa3 = SetAbstraction(3+256, [256, 512, 1024], sample_all=True)
        self.classifier = Classifier(1024, self.num_classes)

        self._init_cls_loss(config, getattr(self, 'cls_dim', None) or self.num_classes)   # ce | focal
        self._init_feature_extraction()

    def load_model_from_ckpt(self, ckpt_path):
        _load_pn2_ckpt(self, ckpt_path, 'PointNet2SSGKNN')

    def _init_weights(self, m):
        _init_pn2_weights(m)

    def forward(self, pts):
        """pts: [B, N, 3] or [B, 3, N] -> logits [B, cls_dim]."""
        if pts.dim() == 3 and pts.shape[1] == 3:
            pts = pts.permute(0, 2, 1)

        B, N, C = pts.shape
        points = pts[:, :, :3]
        features = None
        points, features = self.sa1(points, features)

        features = features.view(B, points.shape[1], -1)
        points_with_features = torch.cat([points, features], dim=-1)
        points, features = self.sa2(points, points_with_features[:, :, 3:])

        features = features.view(B, points.shape[1], -1)
        points_with_features = torch.cat([points, features], dim=-1)
        points, features = self.sa3(points, points_with_features[:, :, 3:])

        self._store_features(features)
        return self.classifier(features)


@MODELS.register_module()
class PointNet2MSG_Classification(nn.Module, ClassificationLossMixin, FeatureExtractionMixin):
    """PointNet++ Multi-Scale Grouping (MSG)."""

    def __init__(self, config, **kwargs):
        super(PointNet2MSG_Classification, self).__init__()
        self.config = config
        self.name = 'PointNet2MSG'
        self.num_classes = config.cls_dim

        print_log(f'[PointNet2MSG] Building model for {self.num_classes} classes', logger='PointNet2MSG')

        self.sa1 = SetAbstractionMSG(
            3,
            [[32, 32, 64], [64, 64, 128], [64, 96, 128]],
            512,
            [0.1, 0.2, 0.4],
            [16, 32, 128]
        )
        self.sa2 = SetAbstractionMSG(
            320 + 3,                                             # 64 + 128 + 128 = 320
            [[64, 64, 128], [128, 128, 256], [128, 128, 256]],
            128,
            [0.2, 0.4, 0.8],
            [32, 64, 128]
        )
        self.sa3 = SetAbstraction(640 + 3, [256, 512, 1024], sample_all=True)   # 128 + 256 + 256 = 640
        self.classifier = Classifier(1024, self.num_classes)

        self._init_cls_loss(config, getattr(self, 'cls_dim', None) or self.num_classes)   # ce | focal
        self._init_feature_extraction()

    def load_model_from_ckpt(self, ckpt_path):
        _load_pn2_ckpt(self, ckpt_path, 'PointNet2MSG')

    def _init_weights(self, m):
        _init_pn2_weights(m)

    def forward(self, pts):
        """pts: [B, N, 3] or [B, 3, N] -> logits [B, cls_dim]."""
        if pts.dim() == 3 and pts.shape[1] == 3:
            pts = pts.permute(0, 2, 1)

        B, N, C = pts.shape
        points = pts[:, :, :3]
        features = pts if C > 3 else None

        # SA layers return flattened (B, S*C); reshape to (B, S, C) between layers
        points, features = self.sa1(points, features)
        features = features.view(B, points.shape[1], -1)
        points, features = self.sa2(points, features)
        features = features.view(B, points.shape[1], -1)
        points, features = self.sa3(points, features)

        self._store_features(features.view(B, -1))
        return self.classifier(features.view(B, -1))


@MODELS.register_module()
class PointNet2MRG_Classification(nn.Module, ClassificationLossMixin, FeatureExtractionMixin):
    """PointNet++ Multi-Resolution Grouping (MRG)."""

    def __init__(self, config, **kwargs):
        super(PointNet2MRG_Classification, self).__init__()
        self.config = config
        self.name = 'PointNet2MRG'
        self.num_classes = config.cls_dim

        print_log(f'[PointNet2MRG] Building model for {self.num_classes} classes', logger='PointNet2MRG')

        # branch 1: standard hierarchical abstraction
        self.sa1 = SetAbstraction(3, [64, 64, 128], 512, 0.2, 32, sample_all=False)
        self.sa2 = SetAbstraction(128+3, [128, 128, 256], 64, 0.4, 64, sample_all=False)
        # branch 2: alternative resolution path
        self.sa3 = SetAbstraction(3, [64, 128, 256], 512, 0.4, 64, sample_all=False)
        # branch 3: global context
        self.sa4 = SetAbstraction(3, [64, 128, 256, 512], sample_all=True)
        # fusion
        self.sa5 = SetAbstraction(256+3, [256, 512, 1024], sample_all=True)
        self.classifier = Classifier(1536, self.num_classes)    # 512 + 1024

        self._init_cls_loss(config, getattr(self, 'cls_dim', None) or self.num_classes)   # ce | focal
        self._init_feature_extraction()

    def load_model_from_ckpt(self, ckpt_path):
        _load_pn2_ckpt(self, ckpt_path, 'PointNet2MRG')

    def _init_weights(self, m):
        _init_pn2_weights(m)

    def forward(self, pts):
        """pts: [B, N, 3] or [B, 3, N] -> logits [B, cls_dim]."""
        if pts.dim() == 3 and pts.shape[1] == 3:
            pts = pts.permute(0, 2, 1)

        B, N, C = pts.shape
        points = pts[:, :, :3]
        features = pts if C > 3 else None

        # SA layers return flattened (B, S*C); reshape to (B, S, C) between layers
        B1_points, B1_features = self.sa1(points, features)
        B1_features = B1_features.view(B, B1_points.shape[1], -1)
        B1_points, B1_features = self.sa2(B1_points, B1_features)
        B1_features = B1_features.view(B, B1_points.shape[1], -1)

        B2_points, B2_features = self.sa3(points, features)
        B2_features = B2_features.view(B, B2_points.shape[1], -1)

        B3_points, B3_features = self.sa4(points, features)

        # merge the two point sets along the point axis (both carry 256-d features)
        B4_input_points = torch.cat([B1_points, B2_points], dim=1)
        B4_input_features = torch.cat([B1_features, B2_features], dim=1)
        B4_points, B4_features = self.sa5(B4_input_points, B4_input_features)

        classifier_input_features = torch.cat([B3_features, B4_features], dim=1)
        self._store_features(classifier_input_features.view(B, -1))
        return self.classifier(classifier_input_features.view(B, -1))
