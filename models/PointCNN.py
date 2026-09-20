"""PointCNN — Convolution On X-Transformed Points.
Li et al. NeurIPS 2018. arXiv:1801.07791
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

def _activation(name):
    """PointCNN uses ELU everywhere, never ReLU (§4)."""
    if name == 'elu':
        return nn.ELU(inplace=True)
    if name == 'relu':
        return nn.ReLU(inplace=True)
    raise ValueError(f"activation must be 'elu' or 'relu', got {name!r}")

class Dense(nn.Module):
    """pointfly.dense: Linear -> activation -> BatchNorm on the last axis (any rank).
    Norm sees post-activation values (reference order); BatchNorm1d over the flattened [..., C] view.
    """

    def __init__(self, in_channels, out_channels, bn=True, act='elu',
                 bn_momentum=0.01, bn_eps=1e-3):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=not bn)
        self.bn = nn.BatchNorm1d(out_channels, eps=bn_eps, momentum=bn_momentum) if bn else None
        self.act = _activation(act) if act else None
        self.out_channels = out_channels

    def forward(self, x):
        x = self.linear(x)
        if self.act is not None:
            x = self.act(x)
        if self.bn is not None:
            shape = x.shape
            x = self.bn(x.reshape(-1, shape[-1])).reshape(shape)
        return x

class XConv(nn.Module):
    """One X-Conv layer. Public tensors are channels-last [B, P, K, C]; convs transpose to NCHW internally."""

    def __init__(self, K, D, C, C_pts_fts, dm, in_fts_channels, with_global=False,
                 with_X_transformation=True, act='elu', bn_momentum=0.01, bn_eps=1e-3):
        super().__init__()
        self.K, self.D, self.C = K, D, C
        self.dm = dm
        self.with_global = with_global
        self.with_X = with_X_transformation

        bn_kw = dict(bn_momentum=bn_momentum, bn_eps=bn_eps, act=act)

        # step 1 — lift the local coordinates to features (§3.1)
        self.dense0 = Dense(3, C_pts_fts, **bn_kw)
        self.dense1 = Dense(C_pts_fts, C_pts_fts, **bn_kw)
        # channels entering the X product / separable conv
        self.in_channels = C_pts_fts + (in_fts_channels or 0)

        # step 2 — the K x K X-transformation, predicted from local coords only
        if self.with_X:
            self.x_conv0 = nn.Conv2d(3, K * K, (1, K), bias=False)
            self.x_bn0 = nn.BatchNorm2d(K * K, eps=bn_eps, momentum=bn_momentum)
            # depthwise: groups=K, depth_multiplier=K -> K*K out channels (§10)
            self.x_conv1 = nn.Conv2d(K, K * K, (1, K), groups=K, bias=False)
            self.x_bn1 = nn.BatchNorm2d(K * K, eps=bn_eps, momentum=bn_momentum)
            self.x_conv2 = nn.Conv2d(K, K * K, (1, K), groups=K, bias=False)
            # X_2 has BatchNorm but NO activation (§3, step 2)
            self.x_bn2 = nn.BatchNorm2d(K * K, eps=bn_eps, momentum=bn_momentum)
            self.x_act = _activation(act)

        # step 3 — depthwise-separable convolution over the K slots
        cin = self.in_channels
        self.sep_depth = nn.Conv2d(cin, cin * dm, (1, K), groups=cin, bias=False)
        self.sep_point = nn.Conv2d(cin * dm, C, 1, bias=False)
        self.sep_bn = nn.BatchNorm2d(C, eps=bn_eps, momentum=bn_momentum)
        self.sep_act = _activation(act)

        # step 4 — optional absolute-coordinate branch (last layer only)
        if with_global:
            self.global0 = Dense(3, C // 4, **bn_kw)
            self.global1 = Dense(C // 4, C // 4, **bn_kw)
            self.out_channels = C + C // 4
        else:
            self.out_channels = C

    def _to_kk(self, x):
        """[B, K*K, P, 1] (NCHW conv output) -> [B, P, K, K]."""
        b, _, p, _ = x.shape
        return x.squeeze(-1).transpose(1, 2).reshape(b, p, self.K, self.K)

    def _neighbor_indices(self, pts, qrs):
        """Dilated kNN: query K*D neighbours by distance, then stride by D."""
        n = pts.shape[1]
        k = min(self.K * self.D, n)                      # §9.9 guard
        _, idx = misc.knn(pts, qrs, k)                   # [B, P, k]
        idx = idx[..., ::self.D][..., :self.K]
        if idx.shape[-1] < self.K:                       # fewer points than K*D
            pad = self.K - idx.shape[-1]
            idx = torch.cat([idx, idx[..., -1:].expand(-1, -1, pad)], dim=-1)
        return idx

    def forward(self, pts, fts, qrs):
        """pts [B, N, 3], fts [B, N, C_prev] or None, qrs [B, P, 3] -> [B, P, out_channels]."""
        idx = self._neighbor_indices(pts, qrs)
        nn_pts = misc.index_points(pts, idx)                     # [B, P, K, 3]
        nn_pts_local = nn_pts - qrs.unsqueeze(2)

        # ---- step 1: local coordinates -> features ----
        nn_fts_input = self.dense1(self.dense0(nn_pts_local))    # [B,P,K,C_pts_fts]
        if fts is not None:
            nn_fts_input = torch.cat([nn_fts_input, misc.index_points(fts, idx)], dim=-1)

        # ---- step 2: the X transformation ----
        if self.with_X:
            x = nn_pts_local.permute(0, 3, 1, 2)                 # [B, 3, P, K]
            x = self.x_bn0(self.x_act(self.x_conv0(x)))
            x = self._to_kk(x).permute(0, 3, 1, 2)               # [B, K, P, K]
            x = self.x_bn1(self.x_act(self.x_conv1(x)))
            x = self._to_kk(x).permute(0, 3, 1, 2)
            # X_2 is built with activation=None upstream, so order is moot here
            x = self.x_bn2(self.x_conv2(x))                      # no activation
            X = self._to_kk(x)                                   # [B, P, K, K]
            fts_X = torch.matmul(X, nn_fts_input)                # mixes the K axis
        else:
            fts_X = nn_fts_input

        # ---- step 3: depthwise-separable convolution over K ----
        y = fts_X.permute(0, 3, 1, 2)                            # [B, Cin, P, K]
        y = self.sep_point(self.sep_depth(y))                    # [B, C, P, 1]
        y = self.sep_bn(self.sep_act(y))
        fts_conv = y.squeeze(-1).transpose(1, 2)                 # [B, P, C]

        # ---- step 4: absolute-coordinate branch, concatenated global-first ----
        if self.with_global:
            fts_global = self.global1(self.global0(qrs))         # [B, P, C//4]
            return torch.cat([fts_global, fts_conv], dim=-1)
        return fts_conv

@MODELS.register_module()
class PointCNN_Classification(nn.Module, ClassificationLossMixin, FeatureExtractionMixin):
    """PointCNN whole-cloud classifier, scratch-only."""

    def __init__(self, config, **kwargs):
        super().__init__()
        print_log('[PointCNN] Initializing PointCNN model', logger='PointCNN')

        self.config = config
        self.name = config.NAME
        self.cls_dim = config.cls_dim

        xconv_params = getattr(config, 'xconv_params', None) or [
            {'K': 8, 'D': 1, 'P': -1, 'C': 48},
            {'K': 12, 'D': 2, 'P': 384, 'C': 96},
            {'K': 16, 'D': 2, 'P': 128, 'C': 192},
            {'K': 16, 'D': 3, 'P': 128, 'C': 384},
        ]
        self.xconv_params = [dict(p) for p in xconv_params]
        fc_params = getattr(config, 'fc_params', None) or [
            {'C': 384, 'dropout_rate': 0.0},
            {'C': 192, 'dropout_rate': 0.8},
        ]
        self.with_X_transformation = getattr(config, 'with_X_transformation', True)
        self.with_global = getattr(config, 'with_global', True)
        self.sampling = getattr(config, 'sampling', 'fps')
        assert self.sampling in ('fps', 'random'), \
            f"sampling must be 'fps' or 'random', got {self.sampling!r}"
        sorting_method = getattr(config, 'sorting_method', None)
        assert sorting_method in (None, 'none', 'null'), \
            'sorting_method is not implemented (unused by every classification setting, §11.7)'

        # per-point supervision (§11.3)
        self.per_point_supervision = getattr(config, 'per_point_supervision', True)
        self.per_point_only = getattr(config, 'per_point_only', True)
        self.aux_weight = getattr(config, 'aux_weight', 1.0)
        self._pp_logits = None
        self._warned_pp = False

        act = getattr(config, 'activation', 'elu')
        bn_momentum = getattr(config, 'bn_momentum', 0.01)   # == TF momentum 0.99 (§9.3)
        bn_eps = getattr(config, 'bn_eps', 1e-3)             # TF BN default

        # ---- trunk ----
        layers = []
        prev_C = None                        # None == "no input features" (layer 0)
        for i, p in enumerate(self.xconv_params):
            K, D, C = p['K'], p['D'], p['C']
            # C_pts_fts and dm are DERIVED, exactly as pointcnn.py:107-113 (§6)
            if i == 0:
                C_pts_fts = C // 2 if prev_C is None else C // 4
                dm = 4
            else:
                C_pts_fts = self.xconv_params[i - 1]['C'] // 4
                dm = int(math.ceil(C / self.xconv_params[i - 1]['C']))
            layers.append(XConv(
                K=K, D=D, C=C, C_pts_fts=C_pts_fts, dm=dm,
                in_fts_channels=prev_C,
                with_global=self.with_global and i == len(self.xconv_params) - 1,
                with_X_transformation=self.with_X_transformation,
                act=act, bn_momentum=bn_momentum, bn_eps=bn_eps))
            prev_C = layers[-1].out_channels
        self.xconvs = nn.ModuleList(layers)

        # ---- head (§5) ----
        c = self.xconvs[-1].out_channels
        head = []
        for p in fc_params:
            head.append(Dense(c, p['C'], act=act, bn_momentum=bn_momentum, bn_eps=bn_eps))
            head.append(nn.Dropout(p.get('dropout_rate', 0.0)))
            c = p['C']
        self.fc = nn.Sequential(*head)
        # the only layer with a bias and without BN/activation
        self.logits = nn.Linear(c, self.cls_dim, bias=True)
        self.feature_dim = c

        # ce | focal
        self._init_cls_loss(config, self.cls_dim)
        self._init_feature_extraction()
        # see set_class_weights — alignment padding for DataParallel
        self.register_buffer('_cw_align_pad', None)

        self.apply(self._init_weights)

        print_log(f'[PointCNN] {len(self.xconvs)} X-Conv layers '
                  f'(K={[p["K"] for p in self.xconv_params]}, '
                  f'P={[p["P"] for p in self.xconv_params]}, sampling={self.sampling}, '
                  f'X={self.with_X_transformation}), {self.cls_dim} classes',
                  logger='PointCNN')

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
            print_log('Training from scratch!!!', logger='PointCNN')
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
            print_log('missing_keys', logger='PointCNN')
            print_log(get_missing_parameters_message(incompatible.missing_keys), logger='PointCNN')
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='PointCNN')
            print_log(get_unexpected_parameters_message(incompatible.unexpected_keys), logger='PointCNN')
        print_log(f'[PointCNN] Successful Loading the ckpt from {ckpt_path}', logger='PointCNN')

    # class_weights (mixin buffer) is the model's first buffer; if its length
    # isn't a multiple of 16 floats, nn.DataParallel's coalesced buffer views
    # land misaligned and cuDNN grouped convs crash. Pad it back to alignment.
    _BUFFER_ALIGN_FLOATS = 16

    def set_class_weights(self, class_weights):
        super().set_class_weights(class_weights)
        if self.class_weights is None:
            self._cw_align_pad = None
            return
        pad = (-self.class_weights.numel()) % self._BUFFER_ALIGN_FLOATS
        self._cw_align_pad = torch.zeros(
            pad, dtype=self.class_weights.dtype, device=self.class_weights.device) if pad else None

    def _sample(self, pts, P, prev_P, layer_idx):
        """Representative points for one layer (§2.2, §11.2)."""
        n = pts.shape[1]
        if P == -1 or P >= n or (layer_idx > 0 and P == prev_P):
            return pts                                   # no sampling
        if self.sampling == 'fps':
            return misc.index_points(pts, misc.farthest_point_sample(pts, P))
        # 'random': the original relies on the dataloader having shuffled the
        # points, so shuffle explicitly at train time and take the prefix at eval
        if self.training:
            idx = torch.rand(pts.shape[0], n, device=pts.device).argsort(dim=1)[:, :P]
            return misc.index_points(pts, idx)
        return pts[:, :P, :].contiguous()

    def forward(self, pts):
        """pts: [B, N, 3] or [B, 3, N] -> logits [B, cls_dim] (pooled over the last layer's points)."""
        if pts.size(-1) != 3 and pts.size(1) == 3:
            pts = pts.transpose(1, 2)
        pts = pts.contiguous()

        fts = None
        prev_P = None
        for i, (layer, param) in enumerate(zip(self.xconvs, self.xconv_params)):
            qrs = self._sample(pts, param['P'], prev_P, i)
            fts = layer(pts, fts, qrs)
            pts = qrs
            prev_P = qrs.shape[1]

        fts = self.fc(fts)                               # [B, P_last, C_fc]

        # the trunk has no global pooling; this mean is the original's eval path
        pooled = fts.mean(dim=1)                         # [B, C_fc]
        self._store_features(pooled)

        # final layer is affine, so logits(mean(fts)) == mean(logits(fts))
        self._pp_logits = self.logits(fts) if (self.training and self.per_point_supervision) else None
        return self.logits(pooled)

    def get_loss_acc(self, ret, gt, split='train'):
        """Repo loss + PointCNN per-point supervision. Accuracy always on the pooled logits."""
        loss, acc = super().get_loss_acc(ret, gt, split)

        pp = self._pp_logits
        if pp is None:
            if self.per_point_supervision and self.training and not self._warned_pp:
                self._warned_pp = True
                print_log(
                    '[PointCNN] per_point_supervision is enabled but the auxiliary '
                    'logits are missing — this happens under nn.DataParallel, which '
                    'discards attributes written by replica forwards. Falling back to '
                    'the pooled loss. Run on a single GPU for the original objective, '
                    'or set per_point_supervision: false to make this explicit.',
                    logger='PointCNN')
            return loss, acc

        b, p, c = pp.shape
        pp_loss, _ = super().get_loss_acc(pp.reshape(b * p, c),
                                          gt.repeat_interleave(p), split)
        loss = pp_loss if self.per_point_only else loss + self.aux_weight * pp_loss
        return loss, acc
