"""PointNet — Deep Learning on Point Sets for 3D Classification and Segmentation.
Qi et al. CVPR 2017. doi:10.1109/CVPR.2017.16
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


class Tnet(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, input):
        device = input.device
        batch_size = input.size(0)
        xb = F.relu(self.bn1(self.conv1(input)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        pool = nn.MaxPool1d(xb.size(-1))(xb)
        flat = nn.Flatten(1)(pool)
        xb = F.relu(self.bn4(self.fc1(flat)))
        xb = F.relu(self.bn5(self.fc2(xb)))

        init = torch.eye(self.k, requires_grad=True).repeat(batch_size, 1, 1)
        if xb.is_cuda:
            init = init.cuda(device=device)
        return self.fc3(xb).view(-1, self.k, self.k) + init


class PointNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

    def forward(self, input):
        matrix3x3 = self.input_transform(input)
        xb = torch.bmm(torch.transpose(input, 1, 2), matrix3x3).transpose(1, 2)

        xb = F.relu(self.bn1(self.conv1(xb)))
        xb = F.relu(self.bn2(self.conv2(xb)))

        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb, 1, 2), matrix64x64).transpose(1, 2)

        xb = F.relu(self.bn3(self.conv3(xb)))
        xb = F.relu(self.bn4(self.conv4(xb)))
        xb = self.bn5(self.conv5(xb))

        xb = nn.MaxPool1d(xb.size(-1))(xb)
        global_feature = nn.Flatten(1)(xb)
        return global_feature, matrix3x3, matrix64x64


@MODELS.register_module()
class PointNet_Classification(nn.Module, ClassificationLossMixin, FeatureExtractionMixin):
    def __init__(self, config, **kwargs):
        super().__init__()
        print_log('[PointNet] Initializing PointNet model', logger='PointNet')

        self.config = config
        self.name = config.NAME
        self.cls_dim = config.cls_dim

        # state_dict buffer, grown on demand in forward
        max_batch_size = getattr(config, 'total_bs', 32)
        self.register_buffer('matrix64x64', torch.eye(64).unsqueeze(0).repeat(max_batch_size, 1, 1))
        self.features_for_visualization = None
        self.alpha = getattr(config, 'alpha', 0.001)      # T-net regularization weight

        self.backbone = PointNetBackbone()
        self.cls_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self._init_cls_loss(config, getattr(self, 'cls_dim', None) or self.num_classes)   # ce | focal
        self._init_feature_extraction()
        self.apply(self._init_weights)

        print_log(f'[PointNet] Model initialized with {self.cls_dim} classes', logger='PointNet')

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
            print_log('Training from scratch!!!', logger='PointNet')
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
            print_log('missing_keys', logger='PointNet')
            print_log(get_missing_parameters_message(incompatible.missing_keys), logger='PointNet')
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='PointNet')
            print_log(get_unexpected_parameters_message(incompatible.unexpected_keys), logger='PointNet')

        print_log(f'[PointNet] Successful Loading the ckpt from {ckpt_path}', logger='PointNet')

    def forward(self, pts):
        """pts: [B, N, 3] -> logits [B, cls_dim]."""
        if pts.dim() == 3 and pts.size(-1) == 3:
            pts = pts.transpose(2, 1)                      # [B, 3, N]

        global_feature, matrix3x3, matrix64x64 = self.backbone(pts)

        batch_size = matrix64x64.size(0)
        if batch_size > self.matrix64x64.size(0):
            new_size = max(batch_size, self.matrix64x64.size(0) * 2)
            new_buffer = torch.eye(64, device=self.matrix64x64.device).unsqueeze(0).repeat(new_size, 1, 1)
            new_buffer[:self.matrix64x64.size(0)] = self.matrix64x64
            self.register_buffer('matrix64x64', new_buffer)
        self.matrix64x64[:batch_size].copy_(matrix64x64.detach())   # copy_ keeps it a registered buffer

        self._store_features(global_feature)
        return self.cls_head(global_feature)
