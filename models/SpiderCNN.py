"""SpiderCNN — Deep Learning on Point Sets with Parameterized Convolutional Filters.
Xu et al. ECCV 2018. doi:10.1007/978-3-030-01237-3_6
"""

import math

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


# 20 monomials of a degree-3 Taylor expansion, in the reference's g1..g6 order;
# index 7 ('1') is the constant term, standing in for its separate bias tensor.
TAYLOR_TERMS = ('x', 'y', 'z', 'xyz',
                'xy', 'yz', 'xz', '1',
                'xx', 'yy', 'zz',
                'xxy', 'xyy', 'xxz',
                'xzz', 'yyz', 'yzz',
                'xxx', 'yyy', 'zzz')
_CONST_TERM_INDEX = TAYLOR_TERMS.index('1')


def taylor_basis(delta):
    """delta [B, N, k, 3] -> monomials [B, N, k, 20]."""
    x = delta[..., 0]
    y = delta[..., 1]
    z = delta[..., 2]
    xx, yy, zz = x * x, y * y, z * z
    return torch.stack([
        x, y, z, x * y * z,
        x * y, y * z, x * z, torch.ones_like(x),
        xx, yy, zz,
        xx * y, x * yy, xx * z,
        x * zz, yy * z, y * zz,
        xx * x, yy * y, zz * z,
    ], dim=-1)


class SpiderConv(nn.Module):
    """feat [B, N, C] -> [B, N, num_conv], over a precomputed neighbour graph."""

    def __init__(self, in_channels, num_conv, k, taylor_channel=5, G=16,
                 gn_eps=1e-6, activation=True):
        super().__init__()
        self.in_channels = in_channels
        self.num_conv = num_conv
        self.k = k
        self.taylor_channel = taylor_channel

        self.taylor = nn.Parameter(torch.empty(len(TAYLOR_TERMS), taylor_channel))
        # reference inits each coeff separately with xavier (fan_in=1), so the
        # bound is computed explicitly — torch's xavier_uniform_ on the stacked
        # matrix would read fan_in=taylor_channel and give a ~2x narrower range.
        bound = math.sqrt(6.0 / (1.0 + taylor_channel))
        with torch.no_grad():
            self.taylor.uniform_(-bound, bound)
            self.taylor[_CONST_TERM_INDEX].zero_()

        self.conv = nn.Conv2d(in_channels * taylor_channel, num_conv,
                              kernel_size=(1, k), bias=True)   # (1, k) sums over the neighbourhood
        self.gn = nn.GroupNorm(min(G, num_conv), num_conv, eps=gn_eps)
        self.act = nn.ReLU(inplace=True) if activation else None

    def forward(self, feat, idx, delta):
        b, n, _ = feat.shape
        grouped = misc.index_points(feat, idx)                   # [B,N,k,C]
        g = taylor_basis(delta) @ self.taylor                    # [B,N,k,taylor]

        out = grouped.unsqueeze(-1) * g.unsqueeze(3)             # [B,N,k,C,taylor]
        out = out.reshape(b, n, idx.shape[-1], self.in_channels * self.taylor_channel)

        out = self.conv(out.permute(0, 3, 1, 2))                 # [B,num_conv,N,1]
        out = self.gn(out)
        if self.act is not None:
            out = self.act(out)
        return out.squeeze(-1).transpose(1, 2)                   # [B,N,num_conv]


@MODELS.register_module()
class SpiderCNN_Classification(nn.Module, ClassificationLossMixin, FeatureExtractionMixin):
    """SpiderCNN whole-cloud classifier (xyz input), scratch-only."""

    def __init__(self, config, **kwargs):
        super().__init__()
        print_log('[SpiderCNN] Initializing SpiderCNN model', logger='SpiderCNN')

        self.config = config
        self.name = config.NAME
        self.cls_dim = config.cls_dim

        self.k = getattr(config, 'k', 20)
        self.taylor_channel = getattr(config, 'taylor_channel', 5)
        G = getattr(config, 'G', 16)
        gn_eps = getattr(config, 'gn_eps', 1e-6)
        channels = list(getattr(config, 'channels', [32, 64, 128, 256]))
        in_channels = getattr(config, 'in_channels', 3)
        self.topk = getattr(config, 'topk', 2)
        fc_dims = list(getattr(config, 'fc_dims', [1024, 512]))
        self.dropout = getattr(config, 'dropout', 0.7)           # reference keep_prob 0.3

        convs = []
        c = in_channels
        for out_c in channels:
            convs.append(SpiderConv(c, out_c, self.k, self.taylor_channel, G, gn_eps))
            c = out_c
        self.spider_convs = nn.ModuleList(convs)

        self.feature_dim = sum(channels) * self.topk             # concat all layers, then top-k pool

        head = []
        c = self.feature_dim
        for d in fc_dims:
            head += [nn.Linear(c, d), nn.BatchNorm1d(d), nn.ReLU(inplace=True),
                     nn.Dropout(self.dropout)]
            c = d
        head.append(nn.Linear(c, self.cls_dim))
        self.cls_head = nn.Sequential(*head)

        self._init_cls_loss(config, self.cls_dim)                # ce | focal
        self._init_feature_extraction()
        self.apply(self._init_weights)

        print_log(f'[SpiderCNN] {len(self.spider_convs)} SpiderConv layers '
                  f'(channels={channels}, k={self.k}, taylor={self.taylor_channel}, '
                  f'G={G}), top-{self.topk} pool -> {self.feature_dim}, '
                  f'{self.cls_dim} classes', logger='SpiderCNN')

    def _init_weights(self, m):
        # Taylor matrices keep their own xavier init (apply() doesn't reach bare Parameters)
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def load_model_from_ckpt(self, ckpt_path):
        if ckpt_path is None:
            print_log('Training from scratch!!!', logger='SpiderCNN')
            return

        ckpt = torch.load(ckpt_path, map_location='cpu')
        if 'base_model' in ckpt:
            base_ckpt = {k.replace('module.', ''): v for k, v in ckpt['base_model'].items()}
        elif 'model' in ckpt:
            base_ckpt = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
        else:
            base_ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}

        incompatible = self.load_state_dict(base_ckpt, strict=False)
        if incompatible.missing_keys:
            print_log('missing_keys', logger='SpiderCNN')
            print_log(get_missing_parameters_message(incompatible.missing_keys), logger='SpiderCNN')
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='SpiderCNN')
            print_log(get_unexpected_parameters_message(incompatible.unexpected_keys), logger='SpiderCNN')
        print_log(f'[SpiderCNN] Successful Loading the ckpt from {ckpt_path}', logger='SpiderCNN')

    def forward(self, pts):
        """pts: [B, N, 3] or [B, 3, N] -> logits [B, cls_dim]."""
        if pts.size(-1) != 3 and pts.size(1) == 3:
            pts = pts.transpose(1, 2)
        xyz = pts.contiguous()
        n = xyz.shape[1]

        k = min(self.k, n)
        _, idx = misc.knn(xyz, xyz, k)                           # [B,N,k], one static graph for all layers
        if k < self.k:
            # keep exactly self.k slots: each SpiderConv kernel is (1, self.k), fixed at build
            idx = torch.cat([idx, idx[..., -1:].expand(-1, -1, self.k - k)], dim=-1)
        delta = misc.index_points(xyz, idx) - xyz.unsqueeze(2)   # [B,N,self.k,3]

        feats = []
        feat = xyz
        for layer in self.spider_convs:
            feat = layer(feat, idx, delta)
            feats.append(feat)
        feat = torch.cat(feats, dim=-1)                          # [B,N,sum(channels)]

        topk = min(self.topk, n)
        pooled = feat.transpose(1, 2).topk(topk, dim=-1).values   # [B,C,topk]
        if topk < self.topk:
            pooled = torch.cat(
                [pooled, pooled[..., -1:].expand(-1, -1, self.topk - topk)], dim=-1)
        global_feat = pooled.reshape(feat.shape[0], -1)           # [B, C*topk]

        self._store_features(global_feat)
        return self.cls_head(global_feat)
