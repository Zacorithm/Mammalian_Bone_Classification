"""DeepGCN — DeepGCNs: Can GCNs Go as Deep as CNNs?
Li et al. ICCV 2019. doi:10.1109/ICCV.2019.00936
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .build import MODELS
from utils.cls_losses import ClassificationLossMixin
from utils.feature_hooks import FeatureExtractionMixin
from utils.logger import print_log
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU(inplace)
    if act == 'leakyrelu':
        return nn.LeakyReLU(neg_slope, inplace)
    if act == 'prelu':
        return nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    raise NotImplementedError(f'activation layer [{act}] is not found')


def norm_layer(norm, nc):
    norm = norm.lower()
    if norm == 'batch':
        return nn.BatchNorm2d(nc, affine=True)
    if norm == 'instance':
        return nn.InstanceNorm2d(nc, affine=False)
    raise NotImplementedError(f'normalization layer [{norm}] is not found')


class BasicConv(nn.Sequential):
    """1x1 Conv2d -> activation -> normalization -> dropout (activation before norm, as in the reference)."""

    def __init__(self, channels, act='relu', norm=None, bias=True, drop=0.):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Conv2d(channels[i - 1], channels[i], 1, bias=bias))
            if act is not None and act.lower() != 'none':
                m.append(act_layer(act))
            if norm is not None and norm.lower() != 'none':
                m.append(norm_layer(norm, channels[-1]))
            if drop > 0:
                m.append(nn.Dropout2d(drop))
        super(BasicConv, self).__init__(*m)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()


def batched_index_select(x, idx):
    """x [B, C, N, 1], idx [B, N, k] -> [B, C, N, k]."""
    batch_size, num_dims, num_vertices = x.shape[:3]
    k = idx.shape[-1]
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)

    x = x.transpose(2, 1)
    feature = x.contiguous().view(batch_size * num_vertices, -1)[idx, :]
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()
    return feature


def pairwise_distance(x):
    """x [B, N, C] -> squared distances [B, N, N]."""
    x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
    x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
    return x_square + x_inner + x_square.transpose(2, 1)


def dense_knn_matrix(x, k=16):
    """x [B, C, N, 1] -> edge_index [2, B, N, k] as (neighbour_idx, centre_idx).
    Nearest neighbour of a point is itself (self-loop), like the reference matrix impl.
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        k = min(k, n_points)                             # k*dilation may exceed N
        _, nn_idx = torch.topk(-pairwise_distance(x.detach()), k=k)
        center_idx = torch.arange(0, n_points, device=x.device).expand(batch_size, k, -1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


class DenseDilated(nn.Module):
    """Pick k neighbours from the k*dilation candidates: [2, B, N, k*dilation] -> [2, B, N, k]."""

    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.stochastic:
            # reference evaluates torch.rand(1) before checking self.training
            if torch.rand(1) < self.epsilon and self.training:
                num = min(self.k * self.dilation, edge_index.shape[-1])
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, :, randnum]
            else:
                edge_index = edge_index[:, :, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, :, ::self.dilation]
        if edge_index.shape[-1] < self.k:
            pad = self.k - edge_index.shape[-1]          # fewer points than k*dilation: repeat last
            edge_index = torch.cat(
                [edge_index, edge_index[:, :, :, -1:].expand(-1, -1, -1, pad)], dim=-1)
        return edge_index


class DenseDilatedKnnGraph(nn.Module):
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)
        self.knn = dense_knn_matrix

    def forward(self, x):
        edge_index = self.knn(x, self.k * self.dilation)
        return self._dilated(edge_index)


class MRConv2d(nn.Module):
    """Max-Relative graph convolution."""

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index):
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        return self.nn(torch.cat([x, x_j], dim=1))


class EdgeConv2d(nn.Module):
    """max over neighbours of MLP([x_i, x_j - x_i])."""

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index):
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class GraphConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'mr':
            self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError(f'conv:{conv} is not supported')

    def forward(self, x, edge_index):
        return self.gconv(x, edge_index)


class DynConv2d(GraphConv2d):
    """Rebuilds the graph from the features each call."""

    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge',
                 act='relu', norm=None, bias=True, stochastic=False, epsilon=0.0):
        super(DynConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x, edge_index=None):
        if edge_index is None:
            edge_index = self.dilated_knn_graph(x)
        return super(DynConv2d, self).forward(x, edge_index)


class PlainDynBlock2d(nn.Module):
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0):
        super(PlainDynBlock2d, self).__init__()
        self.body = DynConv2d(in_channels, in_channels, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon)

    def forward(self, x, edge_index=None):
        return self.body(x, edge_index)


class ResDynBlock2d(nn.Module):
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, res_scale=1):
        super(ResDynBlock2d, self).__init__()
        self.body = DynConv2d(in_channels, in_channels, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon)
        self.res_scale = res_scale

    def forward(self, x, edge_index=None):
        return self.body(x, edge_index) + x * self.res_scale


class DenseDynBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel_size=9, dilation=1, conv='edge',
                 act='relu', norm=None, bias=True, stochastic=False, epsilon=0.0):
        super(DenseDynBlock2d, self).__init__()
        self.body = DynConv2d(in_channels, out_channels, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon)

    def forward(self, x, edge_index=None):
        dense = self.body(x, edge_index)
        return torch.cat((x, dense), 1)


@MODELS.register_module()
class DeepGCN_Classification(nn.Module, ClassificationLossMixin, FeatureExtractionMixin):
    """DeepGCN whole-cloud classifier (ResGCN / PlainGCN / DenseGCN), scratch-only.
    head/backbone/fusion_block/prediction keep the reference names so its checkpoints load directly.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        print_log('[DeepGCN] Initializing DeepGCN model', logger='DeepGCN')

        self.config = config
        self.name = config.NAME
        self.cls_dim = config.cls_dim

        channels = getattr(config, 'n_filters', 64)
        k = getattr(config, 'k', 9)
        act = getattr(config, 'act', 'relu')
        norm = getattr(config, 'norm', 'batch')
        bias = getattr(config, 'bias', True)
        epsilon = getattr(config, 'epsilon', 0.2)
        stochastic = getattr(config, 'use_stochastic', True)
        conv = getattr(config, 'conv', 'edge')
        emb_dims = getattr(config, 'emb_dims', 1024)
        in_channels = getattr(config, 'in_channels', 3)
        dropout = getattr(config, 'dropout', 0.5)
        use_dilation = getattr(config, 'use_dilation', True)
        block = getattr(config, 'block', 'res').lower()
        self.n_blocks = getattr(config, 'n_blocks', 28)
        # 'instance' norm is unusable for this head (broken upstream too): weight is
        # None on InstanceNorm2d(affine=False), and it can't norm [B, C, 1, 1] pooled tensors
        assert norm is None or norm.lower() in ('batch', 'none'), (
            f"norm must be 'batch' or 'none', got {norm!r}: 'instance' is "
            f"unusable for this classification head (broken upstream too)")
        self.in_channels = in_channels
        c_growth = channels

        self.knn = DenseDilatedKnnGraph(k, 1, stochastic, epsilon)
        self.head = GraphConv2d(in_channels, channels, conv, act, norm, bias=False)

        if block == 'dense':
            self.backbone = nn.Sequential(*[
                DenseDynBlock2d(channels + c_growth * i, c_growth, k, 1 + i, conv, act,
                                norm, bias, stochastic, epsilon)
                for i in range(self.n_blocks - 1)])
            fusion_dims = int((channels + channels + c_growth * (self.n_blocks - 1))
                              * self.n_blocks // 2)
        elif block == 'res':
            self.backbone = nn.Sequential(*[
                ResDynBlock2d(channels, k, i + 1 if use_dilation else 1, conv, act, norm,
                              bias, stochastic, epsilon)
                for i in range(self.n_blocks - 1)])
            fusion_dims = int(channels + c_growth * (self.n_blocks - 1))
        elif block == 'plain':
            stochastic = False                          # plain GCN: no dilation/stochastic/residual
            self.backbone = nn.Sequential(*[
                PlainDynBlock2d(channels, k, 1, conv, act, norm, bias, stochastic, epsilon)
                for _ in range(self.n_blocks - 1)])
            fusion_dims = int(channels + c_growth * (self.n_blocks - 1))
        else:
            raise NotImplementedError(f"block must be 'res', 'dense' or 'plain', got {block!r}")

        self.fusion_block = BasicConv([fusion_dims, emb_dims], 'leakyrelu', norm, bias=False)
        self.prediction = nn.Sequential(*[
            BasicConv([emb_dims * 2, 512], 'leakyrelu', norm, drop=dropout),
            BasicConv([512, 256], 'leakyrelu', norm, drop=dropout),
            BasicConv([256, self.cls_dim], None, None)])
        self.feature_dim = emb_dims * 2

        self.model_init()
        # opt-in repo-convention init; default keeps the reference kaiming_normal_ (28 layers depend on it)
        if getattr(config, 'weight_init', 'kaiming') == 'trunc_normal':
            self.apply(self._init_weights)

        self._init_cls_loss(config, self.cls_dim)        # ce | focal
        self._init_feature_extraction()

        print_log(f'[DeepGCN] {block} block x {self.n_blocks} (k={k}, conv={conv}, '
                  f'dilation={use_dilation}, stochastic={stochastic}), '
                  f'fusion {fusion_dims}->{emb_dims}, {self.cls_dim} classes',
                  logger='DeepGCN')

    def model_init(self):
        """Reference init (architecture.py:63-70)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def load_model_from_ckpt(self, ckpt_path):
        if ckpt_path is None:
            print_log('Training from scratch!!!', logger='DeepGCN')
            return

        ckpt = torch.load(ckpt_path, map_location='cpu')
        if 'base_model' in ckpt:
            base_ckpt = {k.replace('module.', ''): v for k, v in ckpt['base_model'].items()}
        elif 'model' in ckpt:
            base_ckpt = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
        elif 'state_dict' in ckpt:                       # reference checkpoints
            base_ckpt = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
        else:
            base_ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}

        incompatible = self.load_state_dict(base_ckpt, strict=False)
        if incompatible.missing_keys:
            print_log('missing_keys', logger='DeepGCN')
            print_log(get_missing_parameters_message(incompatible.missing_keys), logger='DeepGCN')
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='DeepGCN')
            print_log(get_unexpected_parameters_message(incompatible.unexpected_keys), logger='DeepGCN')
        print_log(f'[DeepGCN] Successful Loading the ckpt from {ckpt_path}', logger='DeepGCN')

    def forward(self, pts):
        """pts: [B, N, 3] or [B, 3, N] -> logits [B, cls_dim]."""
        if pts.size(-1) != 3 and pts.size(1) == 3:
            pts = pts.transpose(1, 2)
        inputs = pts.transpose(1, 2).unsqueeze(-1).contiguous()   # reference works on [B, C, N, 1]

        feats = [self.head(inputs, self.knn(inputs[:, 0:3]))]
        for i in range(self.n_blocks - 1):
            feats.append(self.backbone[i](feats[-1]))

        feats = torch.cat(feats, dim=1)
        fusion = self.fusion_block(feats)
        x1 = F.adaptive_max_pool2d(fusion, 1)
        x2 = F.adaptive_avg_pool2d(fusion, 1)
        global_feat = torch.cat((x1, x2), dim=1)                  # [B, 2*emb, 1, 1]

        self._store_features(global_feat.squeeze(-1).squeeze(-1))
        return self.prediction(global_feat).squeeze(-1).squeeze(-1)
