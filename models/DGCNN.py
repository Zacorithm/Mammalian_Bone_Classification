"""DGCNN — Dynamic Graph CNN for Learning on Point Clouds.
Wang et al. ACM TOG 38(5), 2019. doi:10.1145/3326362
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


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]          # (B, N, k)


def get_graph_feature(x, k, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    device = x.device
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        idx = knn(x, k=k)
    idx = idx + torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()                     # (B, N, C)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    return torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()


class DGCNNBackbone(nn.Module):
    def __init__(self, k=20, emb_dims=1024):
        super(DGCNNBackbone, self).__init__()
        self.k = k
        self.emb_dims = emb_dims
        self.out_channels = emb_dims * 2                   # concat(max, avg)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        batch_size = x.size(0)

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        return torch.cat((x1, x2), dim=1)                  # [B, emb_dims * 2]


@MODELS.register_module()
class DGCNN_Classification(nn.Module, ClassificationLossMixin, FeatureExtractionMixin):
    def __init__(self, config, **kwargs):
        super().__init__()
        print_log('[DGCNN] Initializing DGCNN model', logger='DGCNN')

        self.config = config
        self.name = config.NAME
        self.cls_dim = config.cls_dim
        self.k = getattr(config, 'k', 20)
        self.emb_dims = getattr(config, 'emb_dims', 1024)
        self.dropout = getattr(config, 'dropout', 0.5)

        self.backbone = DGCNNBackbone(k=self.k, emb_dims=self.emb_dims)
        self.feature_dim = self.backbone.out_channels

        self.cls_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(self.dropout),
            nn.Linear(256, self.cls_dim)
        )

        self._init_cls_loss(config, self.cls_dim)          # ce | focal
        self._init_feature_extraction()
        self.apply(self._init_weights)

        print_log(f'[DGCNN] Model initialized with {self.cls_dim} classes and k={self.k}', logger='DGCNN')

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
            print_log('Training from scratch!!!', logger='DGCNN')
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
            print_log('missing_keys', logger='DGCNN')
            print_log(get_missing_parameters_message(incompatible.missing_keys), logger='DGCNN')
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='DGCNN')
            print_log(get_unexpected_parameters_message(incompatible.unexpected_keys), logger='DGCNN')

        print_log(f'[DGCNN] Successful Loading the ckpt from {ckpt_path}', logger='DGCNN')

    def forward(self, pts):
        """pts: [B, N, 3] or [B, 3, N] -> logits [B, cls_dim]."""
        if pts.dim() == 3 and pts.size(-1) == 3:
            pts = pts.transpose(2, 1)                      # [B, N, 3] -> [B, 3, N]

        global_feature = self.backbone(pts)
        self._store_features(global_feature)
        return self.cls_head(global_feature)
