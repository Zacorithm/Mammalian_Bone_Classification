"""KPConv — Flexible and Deformable Convolution for Point Clouds.
Thomas et al. ICCV 2019. doi:10.1109/ICCV.2019.00651
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.cls_losses import ClassificationLossMixin
from utils.feature_hooks import FeatureExtractionMixin
from utils.logger import print_log

# ----------------------------------------------------------------------------
#   Kernel point dispositions  (kernels/kernel_points.py)
# ----------------------------------------------------------------------------

KERNEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'kernel_dispositions')

def kernel_point_optimization(radius, num_points, num_kernels=1, dimension=3,
                              fixed='center', ratio=1.0, verbose=0):
    """Repulsion optimization of num_kernels candidate dispositions (numpy port of
    kernel_point_optimization_debug). Returns (points [num_kernels, num_points, dim], gradient_norms)."""

    # Radius used for optimization (points are rescaled afterwards)
    radius0 = 1
    diameter0 = 2

    # Factor multiplicating gradients for moving points (~learning rate)
    moving_factor = 1e-2
    continuous_moving_decay = 0.9995

    # Gradient threshold to stop optimization
    thresh = 1e-5

    # Gradient clipping value
    clip = 0.05 * radius0

    # Random kernel points
    kernel_points = np.random.rand(num_kernels * num_points - 1, dimension) * diameter0 - radius0
    while kernel_points.shape[0] < num_kernels * num_points:
        new_points = np.random.rand(num_kernels * num_points - 1, dimension) * diameter0 - radius0
        kernel_points = np.vstack((kernel_points, new_points))
        d2 = np.sum(np.power(kernel_points, 2), axis=1)
        kernel_points = kernel_points[d2 < 0.5 * radius0 * radius0, :]
    kernel_points = kernel_points[:num_kernels * num_points, :].reshape((num_kernels, num_points, -1))

    # Optional fixing
    if fixed == 'center':
        kernel_points[:, 0, :] *= 0
    if fixed == 'verticals':
        kernel_points[:, :3, :] *= 0
        kernel_points[:, 1, -1] += 2 * radius0 / 3
        kernel_points[:, 2, -1] -= 2 * radius0 / 3

    saved_gradient_norms = np.zeros((10000, num_kernels))
    old_gradient_norms = np.zeros((num_kernels, num_points))
    for iteration in range(10000):

        # Derivative of the sum of potentials of all points
        A = np.expand_dims(kernel_points, axis=2)
        B = np.expand_dims(kernel_points, axis=1)
        interd2 = np.sum(np.power(A - B, 2), axis=-1)
        inter_grads = (A - B) / (np.power(np.expand_dims(interd2, -1), 3 / 2) + 1e-6)
        inter_grads = np.sum(inter_grads, axis=1)

        # Derivative of the radius potential
        circle_grads = 10 * kernel_points

        # All gradients
        gradients = inter_grads + circle_grads

        if fixed == 'verticals':
            gradients[:, 1:3, :-1] = 0

        # Stop condition
        gradients_norms = np.sqrt(np.sum(np.power(gradients, 2), axis=-1))
        saved_gradient_norms[iteration, :] = np.max(gradients_norms, axis=1)

        if fixed == 'center' and np.max(np.abs(old_gradient_norms[:, 1:] - gradients_norms[:, 1:])) < thresh:
            break
        elif fixed == 'verticals' and np.max(np.abs(old_gradient_norms[:, 3:] - gradients_norms[:, 3:])) < thresh:
            break
        elif np.max(np.abs(old_gradient_norms - gradients_norms)) < thresh:
            break
        old_gradient_norms = gradients_norms

        # Clip gradient to get moving dists
        moving_dists = np.minimum(moving_factor * gradients_norms, clip)

        # Fix the points that have to stay in place
        if fixed == 'center':
            moving_dists[:, 0] = 0
        if fixed == 'verticals':
            moving_dists[:, 0] = 0

        # Move points
        kernel_points -= np.expand_dims(moving_dists, -1) * gradients / np.expand_dims(gradients_norms + 1e-6, -1)

        # moving factor decay
        moving_factor *= continuous_moving_decay

    # Rescale radius to fit the wanted ratio of radius
    r = np.sqrt(np.sum(np.power(kernel_points, 2), axis=-1))
    kernel_points *= ratio / np.mean(r[:, 1:])

    # Rescale kernels with real radius
    return kernel_points * radius, saved_gradient_norms

def load_kernels(radius, num_kpoints, num_kernels=1, dimension=3, fixed='center',
                 num_tries=100, cache_dir=KERNEL_CACHE_DIR):
    """Best of num_tries optimized unit dispositions, cached to .npy, then scaled to radius
    with a random rotation + small noise (once per layer, as in the reference)."""

    if dimension != 3:
        raise ValueError('Only 3D kernels are supported, got dimension=%d' % dimension)

    os.makedirs(cache_dir, exist_ok=True)
    kernel_file = os.path.join(cache_dir, 'k_{:03d}_{:s}.npy'.format(num_kpoints, fixed))

    if not os.path.exists(kernel_file):
        kernel_points, grad_norms = kernel_point_optimization(1.0,
                                                              num_kpoints,
                                                              num_kernels=num_tries,
                                                              dimension=dimension,
                                                              fixed=fixed,
                                                              verbose=0)
        best_k = np.argmin(grad_norms[-1, :])
        original_kernel = kernel_points[best_k, :, :]
        np.save(kernel_file, original_kernel.astype(np.float64))
    else:
        original_kernel = np.load(kernel_file)

    if fixed == 'verticals':
        # Rotations around the vertical axis only
        thetas = np.random.rand(num_kernels) * 2 * np.pi
        c, s = np.cos(thetas), np.sin(thetas)
        R = np.zeros((num_kernels, 3, 3), dtype=np.float32)
        R[:, 0, 0] = c
        R[:, 1, 1] = c
        R[:, 2, 2] = 1
        R[:, 0, 1] = s
        R[:, 1, 0] = -s

        original_kernel = radius * np.expand_dims(original_kernel, 0)
        return np.matmul(original_kernel, R)

    # Random 3D rotations built from two random orthogonal directions
    u = np.ones((num_kernels, 3))
    v = np.ones((num_kernels, 3))
    wrongs = np.abs(np.sum(u * v, axis=1)) > 0.99
    while np.any(wrongs):
        new_u = np.random.rand(num_kernels, 3) * 2 - 1
        new_u = new_u / np.expand_dims(np.linalg.norm(new_u, axis=1) + 1e-9, -1)
        u[wrongs, :] = new_u[wrongs, :]
        new_v = np.random.rand(num_kernels, 3) * 2 - 1
        new_v = new_v / np.expand_dims(np.linalg.norm(new_v, axis=1) + 1e-9, -1)
        v[wrongs, :] = new_v[wrongs, :]
        wrongs = np.abs(np.sum(u * v, axis=1)) > 0.99

    # Make v perpendicular to u
    v -= np.expand_dims(np.sum(u * v, axis=1), -1) * u
    v = v / np.expand_dims(np.linalg.norm(v, axis=1) + 1e-9, -1)

    # Last rotation vector
    w = np.cross(u, v)
    R = np.stack((u, v, w), axis=-1)

    original_kernel = radius * np.expand_dims(original_kernel, 0)
    kernels = np.matmul(original_kernel, R)
    kernels = kernels + np.random.normal(scale=radius * 0.01, size=kernels.shape)
    return kernels

# ----------------------------------------------------------------------------
#   Neighbourhoods, subsampling and masked batch norm
# ----------------------------------------------------------------------------

# Coordinate given to the shadow support point appended by the convolutions.
# `KPConv_ops` uses 1e6, `KPConv_deform_ops` uses 1e3 (it also compares squared
# distances against KP_extent**2, so a smaller value is enough there).
SHADOW_COORD_RIGID = 1e6
SHADOW_COORD_DEFORM = 1e3

def gather_neighbors(x, idx):
    """Gather per-neighbour rows: x [B, N, C] (shadow row included), idx [B, M, H] -> [B, M, H, C]."""
    B, N, C = x.shape
    M, H = idx.shape[1], idx.shape[2]
    flat = idx.reshape(B, M * H, 1).expand(-1, -1, C)
    return torch.gather(x, 1, flat).reshape(B, M, H, C)

def radius_neighbors(q_pts, s_pts, radius, max_neighbors, s_mask=None):
    """Neighbours of q_pts among s_pts inside radius -> [B, M, H] indices (H = min(max_neighbors, N));
    empty/padded slots hold the shadow index N. Radius search + big_neighborhood_filter in one step."""
    B, N, _ = s_pts.shape
    d2 = misc.square_distance(q_pts, s_pts)                       # [B, M, N]
    if s_mask is not None:
        d2 = d2.masked_fill(~s_mask.unsqueeze(1), float('inf'))
    H = min(max_neighbors, N)
    d2, idx = torch.topk(d2, H, dim=-1, largest=False, sorted=True)
    return torch.where(d2 <= radius * radius, idx, torch.full_like(idx, N))

def grid_subsample(points, mask, dl):
    """Voxel-grid barycentres (pure-torch grid_subsampling). points [B, N, 3], mask [B, N] or None, dl voxel size
    -> (sub_points [B, M, 3], sub_mask [B, M]); M = max occupied voxels, shorter clouds zero-padded."""
    B = points.shape[0]
    clouds = []
    for b in range(B):
        p = points[b] if mask is None else points[b][mask[b]]
        if p.shape[0] == 0:
            clouds.append(points.new_zeros((1, 3)))
            continue
        coords = torch.floor((p - p.min(dim=0, keepdim=True)[0]) / dl).long()
        # hash the voxel coordinates into a single int64 key
        n = coords.max().item() + 1
        keys = (coords[:, 0] * n + coords[:, 1]) * n + coords[:, 2]
        _, inverse = torch.unique(keys, return_inverse=True)
        n_vox = int(inverse.max().item()) + 1
        sums = p.new_zeros((n_vox, 3)).index_add_(0, inverse, p)
        counts = p.new_zeros(n_vox).index_add_(0, inverse, torch.ones_like(p[:, 0]))
        clouds.append(sums / counts.unsqueeze(1))

    M = max(c.shape[0] for c in clouds)
    sub_points = points.new_zeros((B, M, 3))
    sub_mask = torch.zeros((B, M), dtype=torch.bool, device=points.device)
    for b, c in enumerate(clouds):
        sub_points[b, :c.shape[0]] = c
        sub_mask[b, :c.shape[0]] = True
    return sub_points, sub_mask

class MaskedBatchNorm(nn.Module):
    """BatchNorm over the valid points (with subsampling: fps that's plain BatchNorm1d over [B*N, C]).
    use_batch_norm: false falls back to a bare bias term."""

    def __init__(self, num_features, use_batch_norm=True, momentum=0.98):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            # TF keeps `momentum` of the running stats, torch keeps (1 - momentum)
            self.bn = nn.BatchNorm1d(num_features, eps=1e-6, momentum=1.0 - momentum)
        else:
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mask=None):
        """x: [B, N, C] (or [B, C]) -> same shape."""
        if not self.use_batch_norm:
            return x + self.bias
        if x.dim() == 2:
            return self.bn(x)
        if mask is None:
            B, N, C = x.shape
            return self.bn(x.reshape(B * N, C)).reshape(B, N, C)
        out = torch.zeros_like(x)
        out[mask] = self.bn(x[mask])
        return out

def kp_weight_init(*shape):
    """The reference's `network_blocks.weight_variable`: a truncated normal of
    stddev `sqrt(2 / out_dim)`, rounded to 1e-3."""
    std = math.sqrt(2.0 / shape[-1])
    w = torch.empty(*shape)
    nn.init.trunc_normal_(w, mean=0.0, std=std, a=-2 * std, b=2 * std)
    return torch.round(w * 1000.0) / 1000.0

class UnaryConv(nn.Module):
    """`conv_ops.unary_convolution` — a 1x1 convolution, i.e. a bias-free
    matmul with the reference's weight layout `[in_fdim, out_fdim]`."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.weights = nn.Parameter(kp_weight_init(in_dim, out_dim))

    def forward(self, x):
        return torch.matmul(x, self.weights)

    def extra_repr(self):
        return 'in_dim={}, out_dim={}'.format(self.in_dim, self.out_dim)

def ind_max_pool(x, inds, mask=None):
    """`network_blocks.ind_max_pool`: max over each pooling cell, with a shadow
    row holding the per-cloud minimum so empty slots never win the max."""
    shadow = x.min(dim=1, keepdim=True)[0] if mask is None else \
        torch.where(mask.unsqueeze(-1), x, torch.full_like(x, float('inf'))).min(dim=1, keepdim=True)[0]
    x = torch.cat([x, shadow], dim=1)
    return gather_neighbors(x, inds).max(dim=2)[0]

# ----------------------------------------------------------------------------
#   The convolution itself  (kernels/convolution_ops.py)
# ----------------------------------------------------------------------------

def kernel_sq_distances(neighbors, k_points):
    """Squared distances neighbour -> kernel point. neighbors [B, M, H, 3], k_points [K, 3] or [B, M, K, 3]
    -> [B, M, H, K]. Uses |n|^2 - 2 n.k + |k|^2 (no [B, M, H, K, 3] difference tensor)."""
    n2 = (neighbors * neighbors).sum(-1, keepdim=True)                  # [B,M,H,1]
    if k_points.dim() == 2:
        nk = torch.matmul(neighbors, k_points.t())                      # [B,M,H,K]
        k2 = (k_points * k_points).sum(-1)                              # [K]
    else:
        nk = torch.einsum('bmhd,bmkd->bmhk', neighbors, k_points)
        k2 = (k_points * k_points).sum(-1).unsqueeze(2)                 # [B,M,1,K]
    return (n2 - 2.0 * nk + k2).clamp(min=0.0)

def kernel_influence(sq_distances, KP_extent, KP_influence, aggregation_mode):
    """Kernel point influences, `[B, M, H, K] -> [B, M, K, H]`."""
    if KP_influence == 'constant':
        all_weights = torch.ones_like(sq_distances)
    elif KP_influence == 'linear':
        # clamped sqrt: the reference differentiates through sqrt(0) whenever a
        # deformed kernel point lands exactly on a neighbour
        all_weights = (1.0 - torch.sqrt(sq_distances.clamp(min=1e-12)) / KP_extent).clamp(min=0.0)
    elif KP_influence == 'gaussian':
        sigma = KP_extent * 0.3
        all_weights = torch.exp(-sq_distances / (2 * sigma * sigma + 1e-9))
    else:
        raise ValueError('Unknown influence function type (KP_influence): %s' % KP_influence)

    if aggregation_mode == 'closest':
        neighbors_1nn = sq_distances.argmin(dim=-1)
        all_weights = all_weights * F.one_hot(neighbors_1nn, sq_distances.shape[-1]).to(all_weights.dtype)
    elif aggregation_mode != 'sum':
        raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")

    return all_weights.transpose(-1, -2)

class KPConvLayer(nn.Module):
    """Rigid KPConv — `conv_ops.KPConv` / `conv_ops.KPConv_ops`."""

    def __init__(self, in_dim, out_dim, num_kpoints, KP_extent, fixed='center',
                 KP_influence='linear', aggregation_mode='sum'):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.num_kpoints = num_kpoints
        self.KP_extent = KP_extent
        self.KP_influence = KP_influence
        self.aggregation_mode = aggregation_mode

        # Kernel disposition: radius 1.5 * KP_extent, not trainable (as in the
        # reference), registered as a buffer so it is saved in the checkpoint
        K_points = load_kernels(1.5 * KP_extent, num_kpoints, num_kernels=1,
                                dimension=3, fixed=fixed)
        self.register_buffer('K_points',
                             torch.from_numpy(K_points.reshape(num_kpoints, 3).astype(np.float32)))
        self.weights = nn.Parameter(kp_weight_init(num_kpoints, in_dim, out_dim))

    def forward(self, q_pts, s_pts, neighb_inds, x):
        """q_pts [B,M,3], s_pts [B,N,3], neighb_inds [B,M,H], x [B,N,Cin]."""
        # Shadow point / shadow feature row (index N in neighb_inds)
        s_pts = torch.cat([s_pts, torch.full_like(s_pts[:, :1, :], SHADOW_COORD_RIGID)], dim=1)
        x = torch.cat([x, torch.zeros_like(x[:, :1, :])], dim=1)

        neighbors = gather_neighbors(s_pts, neighb_inds) - q_pts.unsqueeze(2)   # [B,M,H,3]
        sq_distances = kernel_sq_distances(neighbors, self.K_points)            # [B,M,H,K]
        all_weights = kernel_influence(sq_distances, self.KP_extent,
                                       self.KP_influence, self.aggregation_mode)

        neighborhood_features = gather_neighbors(x, neighb_inds)                # [B,M,H,Cin]
        weighted_features = torch.matmul(all_weights, neighborhood_features)    # [B,M,K,Cin]
        return torch.einsum('bmkc,kco->bmo', weighted_features, self.weights)

    def extra_repr(self):
        return 'in_dim={}, out_dim={}, num_kpoints={}, KP_extent={:.4f}'.format(
            self.in_dim, self.out_dim, self.num_kpoints, self.KP_extent)

class KPConvDeformable(nn.Module):
    """Deformable KPConv: a first rigid KPConv (zero-init, no BN) predicts one offset per kernel point
    per query; the conv then runs with the deformed kernel. self.offset_loss holds the fitting/permissive penalty."""

    def __init__(self, in_dim, out_dim, num_kpoints, KP_extent, conv_radius,
                 fixed='center', KP_influence='linear', aggregation_mode='sum',
                 modulated=False, offsets_loss='fitting'):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.num_kpoints = num_kpoints
        self.KP_extent = KP_extent
        self.conv_radius = conv_radius
        self.KP_influence = KP_influence
        self.aggregation_mode = aggregation_mode
        self.modulated = modulated
        self.offsets_loss = offsets_loss
        self.offset_loss = None

        K_points = load_kernels(1.5 * KP_extent, num_kpoints, num_kernels=1,
                                dimension=3, fixed=fixed)
        self.register_buffer('K_points',
                             torch.from_numpy(K_points.reshape(num_kpoints, 3).astype(np.float32)))

        self.offset_dim = (3 + int(modulated)) * num_kpoints
        # `K_values0` / `b0` of the reference — both start at zero, so the first
        # forward is a rigid convolution
        self.offset_weights = nn.Parameter(torch.zeros(num_kpoints, in_dim, self.offset_dim))
        self.offset_bias = nn.Parameter(torch.zeros(self.offset_dim))
        self.weights = nn.Parameter(kp_weight_init(num_kpoints, in_dim, out_dim))

    def _offsets(self, q_pts, s_pts, neighb_inds, x):
        """The rigid KPConv that predicts the offsets (and the modulations)."""
        s_pts_ext = torch.cat([s_pts, torch.full_like(s_pts[:, :1, :], SHADOW_COORD_RIGID)], dim=1)
        x_ext = torch.cat([x, torch.zeros_like(x[:, :1, :])], dim=1)

        neighbors = gather_neighbors(s_pts_ext, neighb_inds) - q_pts.unsqueeze(2)
        sq_distances = kernel_sq_distances(neighbors, self.K_points)
        all_weights = kernel_influence(sq_distances, self.KP_extent,
                                       self.KP_influence, self.aggregation_mode)
        neighborhood_features = gather_neighbors(x_ext, neighb_inds)
        weighted_features = torch.matmul(all_weights, neighborhood_features)
        features0 = torch.einsum('bmkc,kco->bmo', weighted_features, self.offset_weights)
        features0 = features0 + self.offset_bias

        if self.modulated:
            offsets = features0[..., :3 * self.num_kpoints]
            offsets = offsets.reshape(*offsets.shape[:2], self.num_kpoints, 3)
            modulations = 2 * torch.sigmoid(features0[..., 3 * self.num_kpoints:])
        else:
            offsets = features0.reshape(*features0.shape[:2], self.num_kpoints, 3)
            modulations = None

        # Rescale offset for this layer
        return offsets * self.KP_extent, modulations

    def forward(self, q_pts, s_pts, neighb_inds, x):
        offsets, modulations = self._offsets(q_pts, s_pts, neighb_inds, x)

        B, N, _ = s_pts.shape
        shadow_ind = N
        s_pts = torch.cat([s_pts, torch.full_like(s_pts[:, :1, :], SHADOW_COORD_DEFORM)], dim=1)
        x = torch.cat([x, torch.zeros_like(x[:, :1, :])], dim=1)

        neighbors = gather_neighbors(s_pts, neighb_inds) - q_pts.unsqueeze(2)   # [B,M,H,3]
        deformed_K_points = offsets + self.K_points                             # [B,M,K,3]
        sq_distances = kernel_sq_distances(neighbors, deformed_K_points)        # [B,M,H,K]

        # Drop the neighbours that no deformed kernel point reaches, and shrink
        # the neighbourhood to the largest number of remaining neighbours
        in_range = (sq_distances < self.KP_extent ** 2).any(dim=-1)             # [B,M,H]
        new_max_neighb = int(in_range.sum(dim=-1).max().item())
        new_max_neighb = max(new_max_neighb, 1)
        new_neighb_bool, new_neighb_inds = torch.topk(in_range.to(sq_distances.dtype),
                                                      new_max_neighb, dim=-1)
        new_neighb_bool = new_neighb_bool > 0.5                                 # [B,M,H']

        new_neighbors_indices = torch.gather(neighb_inds, 2, new_neighb_inds)
        new_sq_distances = torch.gather(
            sq_distances, 2, new_neighb_inds.unsqueeze(-1).expand(-1, -1, -1, self.num_kpoints))

        # New shadow neighbors have to point to the last shadow point
        new_neighbors_indices = torch.where(new_neighb_bool, new_neighbors_indices,
                                            torch.full_like(new_neighbors_indices, shadow_ind))

        if self.KP_influence == 'constant':
            all_weights = (new_sq_distances < self.KP_extent ** 2).to(x.dtype).transpose(-1, -2)
        else:
            all_weights = kernel_influence(new_sq_distances, self.KP_extent,
                                           self.KP_influence, self.aggregation_mode)

        neighborhood_features = gather_neighbors(x, new_neighbors_indices)
        weighted_features = torch.matmul(all_weights, neighborhood_features)    # [B,M,K,Cin]
        if modulations is not None:
            weighted_features = weighted_features * modulations.unsqueeze(-1)

        self.offset_loss = self._compute_offset_loss(sq_distances, offsets)
        return torch.einsum('bmkc,kco->bmo', weighted_features, self.weights)

    def _compute_offset_loss(self, sq_distances, offsets):
        """Offset regularization: 'fitting' = square distance from each deformed kernel point to its
    closest input point (/ layer radius); 'permissive' = hinge pushing them back inside the radius."""
        if self.offsets_loss == 'none' or not self.training:
            return None
        if self.offsets_loss == 'fitting':
            # min over the neighbours -> [B, M, K]
            KP_min_d2 = sq_distances.min(dim=2)[0] / (self.conv_radius ** 2)
            return KP_min_d2.mean()
        if self.offsets_loss == 'permissive':
            KP_locs = (offsets + self.K_points) / self.conv_radius
            return (KP_locs.norm(dim=-1) - 1.0).clamp(min=0.0).mean()
        raise ValueError('Unknown offset loss: %s' % self.offsets_loss)

    def extra_repr(self):
        return 'in_dim={}, out_dim={}, num_kpoints={}, KP_extent={:.4f}, modulated={}'.format(
            self.in_dim, self.out_dim, self.num_kpoints, self.KP_extent, self.modulated)

# ----------------------------------------------------------------------------
#   Blocks  (models/network_blocks.py)
# ----------------------------------------------------------------------------

class BlockBase(nn.Module):
    """Common plumbing: which pyramid level a block reads and writes."""

    def __init__(self, layer_ind, strided):
        super().__init__()
        self.layer_ind = layer_ind
        self.strided = strided

    @property
    def out_layer(self):
        return self.layer_ind + 1 if self.strided else self.layer_ind

    def in_mask(self, pyr):
        return pyr['masks'][self.layer_ind]

    def out_mask(self, pyr):
        return pyr['masks'][self.out_layer]

    def conv_args(self, pyr):
        """(query points, support points, neighbour indices) for this block."""
        if self.strided:
            return (pyr['points'][self.layer_ind + 1],
                    pyr['points'][self.layer_ind],
                    pyr['pools'][self.layer_ind])
        return (pyr['points'][self.layer_ind],
                pyr['points'][self.layer_ind],
                pyr['neighbors'][self.layer_ind])

class UnaryBlock(BlockBase):
    """`unary_block` — 1x1 conv + BN + LeakyReLU."""

    def __init__(self, layer_ind, in_dim, out_dim, config):
        super().__init__(layer_ind, strided=False)
        self.conv = UnaryConv(in_dim, out_dim)
        self.bn = MaskedBatchNorm(out_dim, config['use_batch_norm'], config['batch_norm_momentum'])
        self.act = nn.LeakyReLU(0.2)
        self.out_dim = out_dim

    def forward(self, x, pyr):
        return self.act(self.bn(self.conv(x), self.out_mask(pyr)))

class SimpleBlock(BlockBase):
    """`simple_block` / `simple_strided_block` — one KPConv + BN + LeakyReLU."""

    def __init__(self, layer_ind, in_dim, out_dim, config, strided=False):
        super().__init__(layer_ind, strided)
        self.conv = make_kpconv(in_dim, out_dim, layer_ind, config, deformable=False)
        self.bn = MaskedBatchNorm(out_dim, config['use_batch_norm'], config['batch_norm_momentum'])
        self.act = nn.LeakyReLU(0.2)
        self.out_dim = out_dim

    def forward(self, x, pyr):
        q_pts, s_pts, neighb = self.conv_args(pyr)
        return self.act(self.bn(self.conv(q_pts, s_pts, neighb, x), self.out_mask(pyr)))

class ResnetBottleneckBlock(BlockBase):
    """resnetb_block (+ strided / deformable variants): 1x1 -> fdim//2, KPConv fdim//2, 1x1 -> 2*fdim,
    plus a shortcut (max-pooled when strided, projected on a width change)."""

    def __init__(self, layer_ind, in_dim, fdim, config, strided=False, deformable=False):
        super().__init__(layer_ind, strided)
        out_dim = 2 * fdim
        self.out_dim = out_dim
        use_bn, bn_mom = config['use_batch_norm'], config['batch_norm_momentum']

        self.conv1 = UnaryConv(in_dim, fdim // 2)
        self.bn1 = MaskedBatchNorm(fdim // 2, use_bn, bn_mom)
        self.conv2 = make_kpconv(fdim // 2, fdim // 2, layer_ind, config, deformable=deformable)
        self.bn2 = MaskedBatchNorm(fdim // 2, use_bn, bn_mom)
        self.conv3 = UnaryConv(fdim // 2, out_dim)
        self.bn3 = MaskedBatchNorm(out_dim, use_bn, bn_mom)

        self.project_shortcut = in_dim != out_dim
        if self.project_shortcut:
            self.shortcut_conv = UnaryConv(in_dim, out_dim)
            self.shortcut_bn = MaskedBatchNorm(out_dim, use_bn, bn_mom)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, pyr):
        q_pts, s_pts, neighb = self.conv_args(pyr)
        in_mask, out_mask = self.in_mask(pyr), self.out_mask(pyr)

        h = self.act(self.bn1(self.conv1(x), in_mask))
        h = self.act(self.bn2(self.conv2(q_pts, s_pts, neighb, h), out_mask))
        h = self.bn3(self.conv3(h), out_mask)

        if self.strided:
            # the reference pools the shortcut with ind_max_pool, not closest_pool
            shortcut = ind_max_pool(x, pyr['pools'][self.layer_ind], in_mask)
        else:
            shortcut = x
        if self.project_shortcut:
            shortcut = self.shortcut_bn(self.shortcut_conv(shortcut), out_mask)

        return self.act(h + shortcut)

class MaxPoolBlock(BlockBase):
    """`max_pool_block` — pooling only, no weights."""

    def __init__(self, layer_ind, in_dim):
        super().__init__(layer_ind, strided=True)
        self.out_dim = in_dim

    def forward(self, x, pyr):
        return ind_max_pool(x, pyr['pools'][self.layer_ind], self.in_mask(pyr))

class GlobalAverageBlock(BlockBase):
    """`global_average_block` — mean over the (valid) points of each cloud."""

    def __init__(self, layer_ind, in_dim):
        super().__init__(layer_ind, strided=False)
        self.out_dim = in_dim

    def forward(self, x, pyr):
        mask = self.in_mask(pyr)
        if mask is None:
            return x.mean(dim=1)
        w = mask.to(x.dtype).unsqueeze(-1)
        return (x * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)

def make_kpconv(in_dim, out_dim, layer_ind, config, deformable):
    """Build a block's KPConv with this layer's kernel extent
    (= KP_extent * first_subsampling_dl * 2**layer)."""
    extent = config['KP_extent'] * config['first_subsampling_dl'] * (2 ** layer_ind)
    if not deformable:
        return KPConvLayer(in_dim, out_dim, config['num_kernel_points'], extent,
                           fixed=config['fixed_kernel_points'],
                           KP_influence=config['KP_influence'],
                           aggregation_mode=config['convolution_mode'])
    # radius used to normalize the offset losses, `2**layer * dl * density`
    conv_radius = config['first_subsampling_dl'] * config['density_parameter'] * (2 ** layer_ind)
    return KPConvDeformable(in_dim, out_dim, config['num_kernel_points'], extent, conv_radius,
                            fixed=config['fixed_kernel_points'],
                            KP_influence=config['KP_influence'],
                            aggregation_mode=config['convolution_mode'],
                            modulated=config['modulated'],
                            offsets_loss=config['offsets_loss'])

# ----------------------------------------------------------------------------
#   Backbone  (assemble_CNN_blocks + tf_classification_inputs)
# ----------------------------------------------------------------------------

BLOCK_BUILDERS = ('unary', 'simple', 'simple_strided', 'resnetb', 'resnetb_strided',
                  'resnetb_deformable', 'resnetb_deformable_strided',
                  'max_pool', 'global_average')

class KPCNNBackbone(nn.Module):
    """The convolutional part of `KernelPointCNN`, up to `global_average`."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.architecture = list(cfg['architecture'])
        self._parse_layers()

        # ---- blocks, following `assemble_CNN_blocks` -------------------------
        layer, fdim = 0, cfg['first_features_dim']
        in_dim = cfg['in_features_dim']
        blocks = []
        for block in self.architecture:
            if 'upsample' in block:
                raise ValueError('Upsampling blocks belong to KPFCNN (segmentation), '
                                 'not to the classification network: %s' % block)
            blocks.append(self._make_block(block, layer, in_dim, fdim))
            in_dim = blocks[-1].out_dim
            if 'pool' in block or 'strided' in block:
                layer += 1
                fdim *= 2
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = in_dim

    def _make_block(self, block, layer, in_dim, fdim):
        cfg = self.cfg
        if block == 'unary':
            return UnaryBlock(layer, in_dim, fdim, cfg)
        if block == 'simple':
            return SimpleBlock(layer, in_dim, fdim, cfg, strided=False)
        if block == 'simple_strided':
            return SimpleBlock(layer, in_dim, fdim, cfg, strided=True)
        if block == 'resnetb':
            return ResnetBottleneckBlock(layer, in_dim, fdim, cfg, strided=False, deformable=False)
        if block == 'resnetb_strided':
            return ResnetBottleneckBlock(layer, in_dim, fdim, cfg, strided=True, deformable=False)
        if block == 'resnetb_deformable':
            return ResnetBottleneckBlock(layer, in_dim, fdim, cfg, strided=False, deformable=True)
        if block == 'resnetb_deformable_strided':
            return ResnetBottleneckBlock(layer, in_dim, fdim, cfg, strided=True, deformable=True)
        if block == 'max_pool' or block == 'max_pool_wide':
            return MaxPoolBlock(layer, in_dim)
        if block == 'global_average':
            return GlobalAverageBlock(layer, in_dim)
        raise ValueError('Unsupported block %r; this port implements %s' % (block, BLOCK_BUILDERS))

    def _parse_layers(self):
        """Per-layer search radii and subsampling sizes — the construction-time
        half of `common.tf_classification_inputs`."""
        cfg = self.cfg
        dl0, KP_extent = cfg['first_subsampling_dl'], cfg['KP_extent']
        density = cfg['density_parameter']
        r_normal = dl0 * KP_extent * 2.5
        deform_factor = density / (KP_extent * 2.5)

        self.conv_radius, self.pool_radius, self.pool_dl = [], [], []
        layer_blocks = []
        for block_i, block in enumerate(self.architecture):
            if 'global' in block:
                break
            if not ('pool' in block or 'strided' in block):
                layer_blocks += [block]
                if block_i < len(self.architecture) - 1 and \
                        not ('global' in self.architecture[block_i + 1]):
                    continue

            if layer_blocks:
                # the reference tests layer_blocks[:-1], so a layer whose only
                # convolution is deformable still searches with r_normal
                deform = any('deformable' in b for b in layer_blocks[:-1])
                self.conv_radius.append(r_normal * deform_factor if deform else r_normal)
            else:
                self.conv_radius.append(None)

            if 'pool' in block or 'strided' in block:
                self.pool_dl.append(2 * r_normal / (KP_extent * 2.5))
                self.pool_radius.append(r_normal * deform_factor if 'deformable' in block else r_normal)
            else:
                self.pool_dl.append(None)
                self.pool_radius.append(None)

            r_normal *= 2
            layer_blocks = []

        self.num_layers = len(self.conv_radius)
        max_nb = self.cfg['max_neighbors']
        if isinstance(max_nb, (list, tuple)):
            assert len(max_nb) >= self.num_layers, \
                'max_neighbors needs one entry per layer (%d)' % self.num_layers
            self.max_neighbors = list(max_nb)[:self.num_layers]
        else:
            self.max_neighbors = [int(max_nb)] * self.num_layers

    def build_pyramid(self, pts):
        """Points, neighbours and pooling indices of every layer (runtime half of tf_classification_inputs).
    masks replaces stack_lengths (None throughout with subsampling: fps)."""
        cfg = self.cfg
        points, masks, neighbors, pools = [pts], [None], [], []

        for layer in range(self.num_layers):
            if self.conv_radius[layer] is not None:
                neighbors.append(radius_neighbors(points[layer], points[layer],
                                                  self.conv_radius[layer],
                                                  self.max_neighbors[layer],
                                                  masks[layer]))
            else:
                neighbors.append(None)

            if self.pool_radius[layer] is None:
                pools.append(None)
                continue

            if cfg['subsampling'] == 'grid':
                sub_pts, sub_mask = grid_subsample(points[layer], masks[layer], self.pool_dl[layer])
            elif cfg['subsampling'] == 'fps':
                n_sub = max(1, points[layer].shape[1] // cfg['sampling_ratio'])
                sub_pts, sub_mask = misc.fps(points[layer].contiguous(), n_sub), None
            else:
                raise ValueError("subsampling must be 'fps' or 'grid', got %r" % cfg['subsampling'])

            pools.append(radius_neighbors(sub_pts, points[layer], self.pool_radius[layer],
                                          self.max_neighbors[layer], masks[layer]))
            points.append(sub_pts)
            masks.append(sub_mask)

        return {'points': points, 'masks': masks, 'neighbors': neighbors, 'pools': pools}

    def input_features(self, pts):
        """A column of ones, optionally followed by xyz. in_features_dim: 1 = ones only (ModelNet40 default);
    4 = ones ++ xyz (the reference's mis-named in_features_dim == 3). 5/7 need normals, which BoneDataset lacks."""
        ones = torch.ones_like(pts[..., :1])
        dim = self.cfg['in_features_dim']
        if dim == 1:
            return ones
        if dim == 4:
            return torch.cat([ones, pts], dim=-1)
        raise ValueError('in_features_dim must be 1 (ones) or 4 (ones ++ xyz) '
                         'on a dataset without normals, got %d' % dim)

    def forward(self, pts):
        pyr = self.build_pyramid(pts)
        x = self.input_features(pts)
        for block in self.blocks:
            x = block(x, pyr)
        return x

    def offsets_loss(self):
        """Sum of the last forward's deformable offset penalties. Modules are looked up per call
    (a cached list would point at the master's modules under nn.DataParallel)."""
        losses = [m.offset_loss for m in self.modules()
                  if isinstance(m, KPConvDeformable) and m.offset_loss is not None]
        if not losses:
            return None
        return torch.stack(losses).sum()

class _AddAuxLoss(torch.autograd.Function):
    """Return x unchanged but seed d(total)/d(aux) = 1 in backward (i.e. total = loss(x) + aux),
    so forward can keep its plain-logits signature."""

    @staticmethod
    def forward(ctx, x, aux):
        ctx.save_for_backward(aux)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_x):
        aux, = ctx.saved_tensors
        return grad_x, torch.ones_like(aux)

# ----------------------------------------------------------------------------
#   Classifier
# ----------------------------------------------------------------------------

DEFAULTS = {
    # Architecture (training_ModelNet40.py::Modelnet40Config)
    'architecture': ['simple',
                     'resnetb',
                     'resnetb_strided',
                     'resnetb',
                     'resnetb_strided',
                     'resnetb',
                     'resnetb_strided',
                     'resnetb_deformable',
                     'resnetb_deformable_strided',
                     'resnetb_deformable',
                     'global_average'],
    'num_kernel_points': 15,
    'first_subsampling_dl': 0.08,     # reference: 0.02 m on ModelNet40 meshes
    'first_features_dim': 64,
    'in_features_dim': 1,
    'density_parameter': 5.0,
    'KP_influence': 'linear',
    'KP_extent': 1.0,
    'convolution_mode': 'sum',
    'fixed_kernel_points': 'center',
    'modulated': False,
    'use_batch_norm': True,
    'batch_norm_momentum': 0.98,
    'offsets_loss': 'fitting',
    'offsets_decay': 0.1,
    'dropout': 0.5,
    'logits_init': 'reference',
    # Port-specific (see deviations 2 and 3 in the module docstring)
    'subsampling': 'fps',
    'sampling_ratio': 4,
    'max_neighbors': 32,
}

@MODELS.register_module()
class KPConv_Classification(nn.Module, ClassificationLossMixin, FeatureExtractionMixin):

    def __init__(self, config, **kwargs):
        super().__init__()
        print_log('[KPConv] Initializing KPConv model', logger='KPConv')

        self.config = config
        self.name = config.NAME
        self.cls_dim = config.cls_dim

        cfg = {k: getattr(config, k, v) for k, v in DEFAULTS.items()}
        # a config parsed from yaml hands back an EasyDict-ish list wrapper
        cfg['architecture'] = list(cfg['architecture'])
        self.kp_cfg = cfg
        self.offsets_decay = cfg['offsets_decay']
        self.offsets_loss_value = None

        self.backbone = KPCNNBackbone(cfg)
        self.feature_dim = self.backbone.out_channels

        # Classification head (`network_blocks.classification_head`): a single
        # 1024-wide fully connected layer with BN + LeakyReLU, dropout
        # (keep_prob 0.5 in the reference), then the logits layer with a bias.
        self.fc = UnaryConv(self.feature_dim, 1024)
        self.fc_bn = MaskedBatchNorm(1024, cfg['use_batch_norm'], cfg['batch_norm_momentum'])
        self.fc_act = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(cfg['dropout'])
        self.logits_layer = nn.Linear(1024, self.cls_dim)
        with torch.no_grad():
            if cfg['logits_init'] == 'reference':
                # reference init: stddev sqrt(2/num_classes) ~= 0.38 (14 classes) -- wide,
                # logits start around +-30. `logits_init: repo` uses trunc_normal_(std=.02) instead.
                self.logits_layer.weight.copy_(kp_weight_init(1024, self.cls_dim).t())
            elif cfg['logits_init'] == 'repo':
                nn.init.trunc_normal_(self.logits_layer.weight, std=.02)
            else:
                raise ValueError("logits_init must be 'reference' or 'repo', got %r"
                                 % cfg['logits_init'])
            self.logits_layer.bias.zero_()

        # ce | focal
        self._init_cls_loss(config, self.cls_dim)
        self._init_feature_extraction()

        print_log('[KPConv] {:d} layers, {:d} classes, kernel points {:d}, '
                  'first_subsampling_dl {:.3f}, subsampling {:s}'.format(
                      self.backbone.num_layers, self.cls_dim, cfg['num_kernel_points'],
                      cfg['first_subsampling_dl'], cfg['subsampling']), logger='KPConv')

    def load_model_from_ckpt(self, ckpt_path):
        """Load weights from a checkpoint (KPConv trains from scratch)."""
        if ckpt_path is None:
            print_log('Training from scratch!!!', logger='KPConv')
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
            print_log('missing_keys', logger='KPConv')
            print_log(get_missing_parameters_message(incompatible.missing_keys), logger='KPConv')
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='KPConv')
            print_log(get_unexpected_parameters_message(incompatible.unexpected_keys), logger='KPConv')
        print_log(f'[KPConv] Successful Loading the ckpt from {ckpt_path}', logger='KPConv')

    def forward(self, pts):
        """pts: [B, N, 3] (or [B, 3, N]) -> logits [B, cls_dim]."""
        if pts.dim() == 3 and pts.size(-1) != 3 and pts.size(1) == 3:
            pts = pts.transpose(2, 1)   # [B, 3, N] -> [B, N, 3]

        global_feature = self.backbone(pts)          # [B, C]
        self._store_features(global_feature)

        x = self.fc_act(self.fc_bn(self.fc(global_feature)))
        x = self.dropout(x)
        logits = self.logits_layer(x)

        # deformable offset regularization (see deviation 8)
        offsets_loss = self.backbone.offsets_loss()
        if offsets_loss is not None:
            offsets_loss = self.offsets_decay * offsets_loss
            self.offsets_loss_value = float(offsets_loss.detach())
            if logits.requires_grad:
                logits = _AddAuxLoss.apply(logits, offsets_loss)
        else:
            self.offsets_loss_value = None

        return logits
