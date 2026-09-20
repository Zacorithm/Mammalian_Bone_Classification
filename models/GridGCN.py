"""Grid-GCN — Grid-GCN for Fast and Scalable Point Cloud Learning.
Xu et al. CVPR 2020. doi:10.1109/CVPR42600.2020.00570
"""

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

_DIST_EPS = 1e-12
_WEIGHT_EPS = 1e-8
# geometric descriptors the original supports (§5.2); the value is also the
# channel count, except that anything <= 3 means "geo_vec only" (3 channels).
_DESCRIPTOR_DIMS = (0, 3, 4, 5, 10, 11, 12)

def descriptor_channels(fdim):
    """Channel count of the §5.2 descriptor selected by `fdim`."""
    assert fdim in _DESCRIPTOR_DIMS, \
        f'descriptor dim must be one of {_DESCRIPTOR_DIMS}, got {fdim}'
    if fdim == 0:
        return 0
    return 3 if fdim <= 3 else fdim

class MLP2d(nn.Sequential):
    """[Conv2d(1x1, bias=True) -> BN2d -> ReLU] * n over [B, C, O, P]. bias=True is deliberate (param counts)."""

    def __init__(self, in_channels, dims, bn_momentum=0.1, bn_eps=1e-3):
        layers = []
        for d in dims:
            layers += [
                nn.Conv2d(in_channels, d, 1, bias=True),
                nn.BatchNorm2d(d, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=True),
            ]
            in_channels = d
        super().__init__(*layers)
        self.out_channels = in_channels

class MLP1d(nn.Sequential):
    """The MLP2d chain over [B, C, O]."""

    def __init__(self, in_channels, dims, bn_momentum=0.1, bn_eps=1e-3):
        layers = []
        for d in dims:
            layers += [
                nn.Conv1d(in_channels, d, 1, bias=True),
                nn.BatchNorm1d(d, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=True),
            ]
            in_channels = d
        super().__init__(*layers)
        self.out_channels = in_channels

def _nearest_query(xyz, cent_xyz, max_p, radius=None):
    """Node points by nearest-neighbour, optionally radius-limited (square_distance + topk).
    Returns (idx [B, O, P] long, msk [B, O, P] float)."""
    B, N, _ = xyz.shape
    O = cent_xyz.shape[1]
    k = min(max_p, N)
    d2 = misc.square_distance(cent_xyz, xyz)                    # [B, O, N]
    d2, idx = torch.topk(d2, k, dim=-1, largest=False, sorted=True)

    if radius is None:
        msk = torch.ones_like(idx, dtype=xyz.dtype)
    else:
        inside = d2 <= radius * radius
        # ascending distances => `inside` is a prefix; pad the rest by
        # repeating the nearest node point, never a zero (§4.2.4)
        idx = torch.where(inside, idx, idx[:, :, :1].expand_as(idx))
        msk = inside.to(xyz.dtype)

    if k < max_p:  # fewer points than P: pad by repeating the nearest one
        pad = max_p - k
        idx = torch.cat([idx, idx[:, :, :1].expand(B, O, pad)], dim=-1)
        msk = torch.cat([msk, msk.new_zeros(B, O, pad)], dim=-1)
    return idx, msk

def _segment_sum(vals, seg_start, seg_end):
    """Per-slot sum of vals [B, N, D] via prefix sums over contiguous runs
    (not scatter_add_: its CUDA atomics aren't bitwise reproducible)."""
    pref = vals.cumsum(dim=1)
    d = vals.shape[-1]
    gs = seg_start.unsqueeze(-1).expand(-1, -1, d)
    ge = seg_end.unsqueeze(-1).expand(-1, -1, d)
    return pref.gather(1, ge) - pref.gather(1, gs) + vals.gather(1, gs)

def _cube_query(f_sorted, slot, slot_pnt, slot_cnt, occ, sel, bidx,
                grid_size, kernel_size, max_p, dtype):
    """Exact cube query: scan the k^3 voxel window around each centre and fill P node slots.
    O*P writes via offset/cumsum (no [B, O, k^3, P] tensor); still ~45 MB of [B, O, k^3] int64 at stage 0."""
    B, N = f_sorted.shape
    O = sel.shape[1]
    device = f_sorted.device
    g0, g1, g2 = (int(g) for g in grid_size)
    grid_vol = g0 * g1 * g2
    k = int(kernel_size)
    k3 = k ** 3
    half = (k - 1) // 2

    # dense voxel -> compact slot table (4 MB vs the original's 262 MB voxel->points table)
    coor2slot = torch.full((B, grid_vol), -1, dtype=torch.long, device=device)
    coor2slot[bidx[occ], f_sorted[occ]] = slot[occ]

    cf = f_sorted.gather(1, sel)                                 # [B, O]
    cf_ok = cf >= 0
    cfc = cf.clamp(min=0)
    cx = cfc % g0
    cy = (cfc // g0) % g1
    cz = cfc // (g0 * g1)

    # window offsets in the kernel's scan order (§4.2)
    nei = torch.arange(k3, device=device)
    dz = nei // (k * k) - half
    dy = (nei % (k * k)) // k - half
    dx = nei % k - half

    wx = cx.unsqueeze(-1) + dx
    wy = cy.unsqueeze(-1) + dy
    wz = cz.unsqueeze(-1) + dz
    inb = ((wx >= 0) & (wx < g0) & (wy >= 0) & (wy < g1) &
           (wz >= 0) & (wz < g2) & cf_ok.unsqueeze(-1))
    wflat = (wz * (g0 * g1) + wy * g0 + wx).clamp(0, grid_vol - 1)

    nb_slot = coor2slot.gather(1, wflat.reshape(B, -1)).reshape(B, O, k3)
    inb = inb & (nb_slot >= 0)
    nb_slot = nb_slot.clamp(min=0)

    cnt = slot_cnt.gather(1, nb_slot.reshape(B, -1)).reshape(B, O, k3)
    cnt = torch.where(inb, cnt.clamp(max=max_p), torch.zeros_like(cnt))

    # how many of each window voxel's points actually fit into the P slots
    excl = cnt.cumsum(-1) - cnt
    take = torch.minimum(cnt, (max_p - excl).clamp(min=0)).clamp(min=0)
    ctake = take.cumsum(-1)
    total = ctake[..., -1]

    p_ar = torch.arange(max_p, device=device)
    # which window voxel does output slot p come from
    j = torch.searchsorted(ctake.reshape(B * O, k3).contiguous(),
                           p_ar.expand(B * O, max_p).contiguous(),
                           right=True).reshape(B, O, max_p).clamp(max=k3 - 1)
    local = (p_ar.view(1, 1, -1) - (ctake.gather(-1, j) - take.gather(-1, j)))
    local = local.clamp(min=0, max=max_p - 1)
    src_slot = nb_slot.gather(-1, j)

    b3 = torch.arange(B, device=device).view(B, 1, 1).expand(B, O, max_p)
    idx = slot_pnt[b3, src_slot, local]
    msk = p_ar.view(1, 1, -1) < total.unsqueeze(-1)
    # unfilled slots repeat the first node point found (§4.2.4)
    idx = torch.where(msk, idx, idx[:, :, :1].expand_as(idx))
    return idx, msk.to(dtype)

@torch.no_grad()
def cagq(xyz, weight, n_valid, voxel_size, grid_size, coord_shift,
         max_o, max_p, kernel_size, loc=True, query='ball', radius=None,
         deterministic=False):
    """Coverage-Aware Grid Query: voxelize + RVS centre sampling + node query,
    fused into one non-differentiable index op.

    Returns:
        nebidx  [B, O, P] long   node-point indices into xyz
        nebmsk  [B, O, P] float  1 = real node point, 0 = padding
        cent    [B, O, 4] float  centre xyz + accumulated coverage weight
        centmsk [B, O]    float  1 = genuinely occupied centre
        n_cent  [B]       long   occupied voxel count, clamped to O (leading slots of the O axis)
    """
    B, N, _ = xyz.shape
    device, dtype = xyz.device, xyz.dtype
    vs = torch.as_tensor(voxel_size, dtype=dtype, device=device)
    gs = torch.as_tensor(grid_size, dtype=torch.long, device=device)
    cs = torch.as_tensor(coord_shift, dtype=dtype, device=device)
    g0, g1, g2 = (int(g) for g in gs)

    # never ask for more centers / node points than there are points; this
    # repo has been bitten before by an unguarded topk (§12.5)
    max_o = min(int(max_o), N)
    max_p = min(int(max_p), N)

    pos = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
    bidx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)

    # step 1: voxelize; points outside the grid are dropped
    c = torch.floor((xyz + cs) / vs).long()
    inside = ((c >= 0) & (c < gs)).all(dim=-1)
    alive = inside & (pos < n_valid.unsqueeze(1))
    flat = c[..., 2] * (g0 * g1) + c[..., 1] * g0 + c[..., 0]    # z-major
    flat = torch.where(alive, flat, torch.full_like(flat, -1))

    # step 2: group points by voxel into compact slots; randomize first so
    # "keep the first P of each voxel" matches the CUDA reservoir sampler
    if deterministic:
        perm = pos.clone()
    else:
        perm = torch.rand(B, N, device=device).argsort(dim=1)
    f = flat.gather(1, perm)
    order = f.argsort(dim=1, stable=True)
    f = f.gather(1, order)
    pid = perm.gather(1, order)                                  # point index
    valid = f >= 0

    xs = torch.gather(xyz, 1, pid.unsqueeze(-1).expand(-1, -1, 3))
    ws = torch.gather(weight.reshape(B, N), 1, pid)
    ws = torch.where(valid, ws, torch.zeros_like(ws))

    new = torch.ones(B, N, dtype=torch.bool, device=device)
    if N > 1:
        new[:, 1:] = f[:, 1:] != f[:, :-1]
    slot = new.long().cumsum(1) - 1                              # [B, N]
    start = torch.cummax(torch.where(new, pos, torch.zeros_like(pos)), dim=1).values
    rank = pos - start                                           # rank in voxel

    keep = valid & (rank < max_p)
    slot_pnt = torch.zeros(B, N, max_p, dtype=torch.long, device=device)
    slot_pnt[bidx[keep], slot[keep], rank[keep]] = pid[keep]
    occ = new & valid                        # one True per occupied voxel

    # first/last position of every slot's run, for exact prefix-sum reductions
    last = torch.ones(B, N, dtype=torch.bool, device=device)
    if N > 1:
        last[:, :-1] = new[:, 1:]
    seg_start = torch.zeros(B, N, dtype=torch.long, device=device)
    seg_start[bidx[new], slot[new]] = pos[new]
    seg_end = torch.zeros(B, N, dtype=torch.long, device=device)
    seg_end[bidx[last], slot[last]] = pos[last]
    slot_cnt = seg_end - seg_start + 1       # points per slot

    # step 3: RVS -- uniform random subset of the occupied voxels
    if deterministic:
        # first-touch order (what the CUDA keeps when the centre buffer doesn't overflow)
        key = torch.where(occ, -pos.to(dtype), torch.full_like(xyz[..., 0], -1e9))
    else:
        key = torch.where(occ, torch.rand(B, N, device=device, dtype=dtype),
                          torch.full((B, N), -1.0, device=device, dtype=dtype))
    sel = key.topk(max_o, dim=1).indices                         # [B, O]
    c_slot = slot.gather(1, sel)
    centmsk = occ.gather(1, sel).to(dtype)
    n_cent = occ.sum(1).clamp(max=max_o)

    # step 4: centre coordinates
    if loc:
        acc = _segment_sum(torch.cat([ws.unsqueeze(-1), xs * ws.unsqueeze(-1)], dim=-1),
                           seg_start, seg_end)                  # [B, N, 1+3]
        cent_xyz = acc[..., 1:] / acc[..., 0:1].clamp(min=_WEIGHT_EPS)
        cent_xyz = torch.gather(cent_xyz, 1, c_slot.unsqueeze(-1).expand(-1, -1, 3))
    else:
        cf = f.gather(1, sel).clamp(min=0)
        vox = torch.stack([cf % g0, (cf // g0) % g1, cf // (g0 * g1)], dim=-1)
        cent_xyz = (vox.to(dtype) + 0.5) * vs - cs

    # step 5: node-point query
    if query == 'cube':
        nebidx, nebmsk = _cube_query(f, slot, slot_pnt, slot_cnt, occ, sel, bidx,
                                     (g0, g1, g2), kernel_size, max_p, dtype)
    elif query == 'knn':
        nebidx, nebmsk = _nearest_query(xyz, cent_xyz, max_p, radius=None)
    elif query == 'ball':
        if radius is None:
            radius = float(3 ** 0.5 * int(kernel_size) * float(vs.max()) / 2.0)
        nebidx, nebmsk = _nearest_query(xyz, cent_xyz, max_p, radius=radius)
    else:
        raise ValueError(f"query must be 'ball', 'cube' or 'knn', got {query!r}")

    # step 6: the centre's coverage weight
    nb_w = torch.gather(weight.reshape(B, N), 1, nebidx.reshape(B, -1)).reshape_as(nebmsk)
    cent_w = (nb_w * nebmsk).sum(dim=-1, keepdim=True)
    cent = torch.cat([cent_xyz, cent_w], dim=-1)                 # [B, O, 4]
    return nebidx, nebmsk, cent, centmsk, n_cent

class GridConv(nn.Module):
    """One Grid-GCN stage: node tensor [B, C, O, P] -> updated centre features [B, out_C, O]."""

    def __init__(self, in_channels, pt_dims, att_dims, attfdim=4, localfdim=3,
                 use_context=True, cntxt_mlp=(), att_full='next', out_dims=(),
                 relu=True, agg='max_pooling', bn_momentum=0.1, bn_eps=1e-3):
        super().__init__()
        assert att_full in ('next', 'last', 'off'), \
            f"att_full must be 'next', 'last' or 'off', got {att_full!r}"
        # the shipped config only ever aggregates by max, and §5.7 skips the
        # neighbor mask in that case (padding repeats a real point, §4.2.4)
        assert agg in ('max', 'max_pooling'), \
            f"agg must be 'max' or 'max_pooling', got {agg!r}"
        # §11.2: the original mutates att_ele_lst[-1] in place to force this;
        # assert instead so the config stays explicit
        assert att_dims[-1] == pt_dims[-1], (
            f'att_ele_dim[-1] ({att_dims[-1]}) must equal pt_ele_dim[-1] '
            f'({pt_dims[-1]}): the attention gate is an elementwise product')

        self.att_full = att_full
        self.attfdim = attfdim
        self.localfdim = localfdim
        self.relu = relu
        self._needs_global_density = any(fd in (5, 11, 12) for fd in (attfdim, localfdim))
        self._needs_density = 12 in (attfdim, localfdim)
        self.has_feats = in_channels > 4
        feat_c = in_channels - 4

        local_c = descriptor_channels(localfdim)
        att_c = descriptor_channels(attfdim)
        assert att_c > 0, 'attfdim must select a non-empty descriptor'
        if not self.has_feats:
            assert local_c > 0, 'stage 0 has no input features; localfdim must be > 0'
            node_pre_c = local_c                                 # §5.3
        elif localfdim != 0:
            node_pre_c = local_c + feat_c
        else:
            node_pre_c = feat_c

        # §5.5 — grid context pooling: `cntxt_mlp_lst: [[],[],[]]` is falsy but
        # not None, so the feature is ON with no MLP (§11.4)
        self.context_mlp = None
        ctx_c = 0
        if use_context:
            if len(cntxt_mlp):
                self.context_mlp = MLP2d(node_pre_c, cntxt_mlp, bn_momentum, bn_eps)
                ctx_c = self.context_mlp.out_channels
            else:
                ctx_c = node_pre_c
        self.ctx_channels = ctx_c

        self.pt_mlp = MLP2d(node_pre_c, pt_dims, bn_momentum, bn_eps)     # §5.4
        self.att_head = MLP2d(att_c, [att_dims[0]], bn_momentum, bn_eps)  # §5.6

        cat_c = att_dims[0] + ctx_c
        if att_full == 'next':
            cat_c += self.pt_mlp.out_channels        # post-MLP node features
        elif att_full == 'last':
            cat_c += node_pre_c                      # pre-MLP node features
        self.att_tail = MLP2d(cat_c, att_dims[1:], bn_momentum, bn_eps)

        self.out_mlp = MLP1d(pt_dims[-1], out_dims, bn_momentum, bn_eps) if len(out_dims) else None
        self.out_channels = self.out_mlp.out_channels if self.out_mlp is not None else pt_dims[-1]

        # exposed for the shape test of §8
        self.node_pre_channels = node_pre_c
        self.att_concat_channels = cat_c

    def _descriptor(self, fdim, geo_vec, geo_dist, centers_xyz, nb_xyz,
                    geo_global_density, geo_density):
        """The descriptor table selected by fdim."""
        if fdim <= 3:
            return geo_vec
        if fdim == 4:
            return torch.cat([geo_dist, geo_vec], dim=1)
        if fdim == 5:
            return torch.cat([geo_dist, geo_vec, geo_global_density], dim=1)
        base = [geo_dist, geo_vec, centers_xyz.expand_as(nb_xyz), nb_xyz]
        if fdim == 10:
            return torch.cat(base, dim=1)
        if fdim == 11:
            return torch.cat(base + [geo_global_density], dim=1)
        return torch.cat(base + [geo_density, geo_global_density], dim=1)  # 12

    def forward(self, neighbors, centers_xyz, center_den, center_masks,
                neighbor_masks, num_points):
        """neighbors [B, C, O, P], centers_xyz [B, 3, O], center_den [B, 1, O],
        center_masks [B, O], neighbor_masks [B, O, P], num_points int -> [B, out_C, O]."""
        nb_xyz = neighbors[:, 0:3]
        geo_num = neighbors[:, 3:4]
        nb_feats = neighbors[:, 4:] if self.has_feats else None

        # §5.2 geometric descriptors
        centers_exp = centers_xyz.unsqueeze(3)                   # [B, 3, O, 1]
        geo_vec = nb_xyz - centers_exp
        geo_dist = torch.sqrt((geo_vec * geo_vec).sum(dim=1, keepdim=True) + _DIST_EPS)
        # density terms only exist for the descriptors that read them; with the
        # shipped attfdim=4 / localfdim=3 neither is needed (§5.2)
        geo_global_density = geo_num / float(num_points) if self._needs_global_density else None
        geo_density = (geo_num / center_den.unsqueeze(3).clamp(min=_WEIGHT_EPS)) \
            if self._needs_density else None

        geo_feats = None
        if self.localfdim != 0 or nb_feats is None:
            geo_feats = self._descriptor(self.localfdim, geo_vec, geo_dist, centers_exp,
                                         nb_xyz, geo_global_density, geo_density)

        # §5.3 node features
        if nb_feats is None:
            node_pre = geo_feats
        elif self.localfdim != 0:
            node_pre = torch.cat([geo_feats, nb_feats], dim=1)
        else:
            node_pre = nb_feats

        # §5.5 grid context pooling (max over P of the pre-MLP features)
        contextvec = None
        if self.ctx_channels > 0:
            ctx_src = node_pre if self.context_mlp is None else self.context_mlp(node_pre)
            contextvec = ctx_src.max(dim=3, keepdim=True).values.expand_as(ctx_src)

        # §5.4 node-feature MLP
        node = self.pt_mlp(node_pre)

        # §5.6 edge attention
        att_vec = self._descriptor(self.attfdim, geo_vec, geo_dist, centers_exp,
                                   nb_xyz, geo_global_density, geo_density)
        att = self.att_head(att_vec)
        parts = [att]
        if self.att_full == 'next':
            parts.append(node)
        elif self.att_full == 'last':
            parts.append(node_pre)
        if contextvec is not None:
            parts.append(contextvec)
        att = self.att_tail(torch.cat(parts, dim=1) if len(parts) > 1 else att)
        pair_feats = att * node

        # §5.7 aggregate + update + mask; neighbour mask deliberately NOT applied for max pooling
        agg = pair_feats.max(dim=3).values                       # [B, C, O]
        if self.relu:
            agg = F.relu(agg)
        if self.out_mlp is not None:
            agg = self.out_mlp(agg)
        return agg * center_masks.unsqueeze(1)

    def extra_repr(self):
        return (f'node_pre={self.node_pre_channels}, att_concat={self.att_concat_channels}, '
                f'out={self.out_channels}')

@MODELS.register_module()
class GridGCN_Classification(nn.Module, ClassificationLossMixin, FeatureExtractionMixin):
    """Grid-GCN whole-cloud classifier, scratch-only."""

    def __init__(self, config, **kwargs):
        super().__init__()
        print_log('[GridGCN] Initializing Grid-GCN model', logger='GridGCN')

        self.config = config
        self.name = config.NAME
        self.cls_dim = config.cls_dim

        # CAGQ / gridification defaults
        self.coord_shift = list(getattr(config, 'coord_shift', [1.0, 1.0, 1.0]))
        self.voxel_size_lst = [list(v) for v in getattr(
            config, 'voxel_size_lst', [[0.05] * 3, [0.25] * 3, [2.0] * 3])]
        self.grid_size_lst = [list(g) for g in getattr(
            config, 'grid_size_lst', [[40] * 3, [8] * 3, [1] * 3])]
        self.max_o_grid_lst = list(getattr(config, 'max_o_grid_lst', [1024, 128, 1]))
        self.max_p_grid_lst = list(getattr(config, 'max_p_grid_lst', [64, 64, 128]))
        self.kernel_size_lst = list(getattr(config, 'kernel_size_lst', [7, 3, 1]))
        self.query_type = getattr(config, 'query_type', 'ball')
        self.loc_within = getattr(config, 'loc_within', True)
        self.deterministic_eval = getattr(config, 'deterministic_eval', True)

        pt_ele_dim = [list(d) for d in getattr(
            config, 'pt_ele_dim', [[64, 64, 128], [128, 128, 256], [256, 256, 512]])]
        att_ele_dim = [list(d) for d in getattr(
            config, 'att_ele_dim', [[64, 128, 128], [128, 256, 256], [256, 512, 512]])]
        cntxt_mlp_lst = [list(d) for d in getattr(config, 'cntxt_mlp_lst', [[], [], []])]
        gcn_out_dim = [list(d) for d in getattr(config, 'gcn_outDim', [[], [], []])]
        self.attfdim = getattr(config, 'attfdim', 4)
        self.localfdim = getattr(config, 'localfdim', 3)
        self.use_context = getattr(config, 'use_context', True)
        self.att_full = getattr(config, 'att_full', 'next')
        self.agg = getattr(config, 'agg', 'max_pooling')
        self.stage_relu = getattr(config, 'relu', True)
        bn_momentum = getattr(config, 'bn_momentum', 0.1)   # == MXNet 0.9 (§11.1)
        bn_eps = getattr(config, 'bn_eps', 1e-3)            # MXNet BN default

        self.num_stages = len(self.max_o_grid_lst)
        for name, lst in (('voxel_size_lst', self.voxel_size_lst),
                          ('grid_size_lst', self.grid_size_lst),
                          ('max_p_grid_lst', self.max_p_grid_lst),
                          ('kernel_size_lst', self.kernel_size_lst),
                          ('pt_ele_dim', pt_ele_dim),
                          ('att_ele_dim', att_ele_dim),
                          ('cntxt_mlp_lst', cntxt_mlp_lst),
                          ('gcn_outDim', gcn_out_dim)):
            assert len(lst) == self.num_stages, \
                f'{name} has {len(lst)} entries, expected {self.num_stages}'

        # ball radius = circumsphere of the cube window (window always fully contained)
        radius_lst = getattr(config, 'radius_lst', None)
        self.radius_lst = list(radius_lst) if radius_lst else [
            3 ** 0.5 * k * max(v) / 2.0
            for k, v in zip(self.kernel_size_lst, self.voxel_size_lst)]

        stages = []
        in_channels = 4                                  # xyz + coverage weight
        for i in range(self.num_stages):
            stage = GridConv(
                in_channels=in_channels,
                pt_dims=pt_ele_dim[i],
                att_dims=att_ele_dim[i],
                attfdim=self.attfdim,
                localfdim=self.localfdim,
                use_context=self.use_context,
                cntxt_mlp=cntxt_mlp_lst[i],
                att_full=self.att_full,
                out_dims=gcn_out_dim[i],
                relu=self.stage_relu,
                agg=self.agg,
                bn_momentum=bn_momentum,
                bn_eps=bn_eps,
            )
            stages.append(stage)
            in_channels = 4 + stage.out_channels         # concat(centers, feats)
        self.stages = nn.ModuleList(stages)

        # head
        head_dims = list(getattr(config, 'head_dims', [512, 256]))
        self.dropout = getattr(config, 'dropout', 0.5)
        layers = []
        c = self.stages[-1].out_channels
        for d in head_dims:
            layers += [nn.Linear(c, d), nn.BatchNorm1d(d), nn.ReLU(inplace=True),
                       nn.Dropout(self.dropout)]
            c = d
        layers.append(nn.Linear(c, self.cls_dim))
        self.cls_head = nn.Sequential(*layers)

        # ce | focal
        self._init_cls_loss(config, self.cls_dim)
        self._init_feature_extraction()

        self.apply(self._init_weights)

        print_log(f'[GridGCN] {self.num_stages} stages '
                  f'(O={self.max_o_grid_lst}, P={self.max_p_grid_lst}, '
                  f'query={self.query_type}), {self.cls_dim} classes', logger='GridGCN')

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
            print_log('Training from scratch!!!', logger='GridGCN')
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
            print_log('missing_keys', logger='GridGCN')
            print_log(get_missing_parameters_message(incompatible.missing_keys), logger='GridGCN')
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='GridGCN')
            print_log(get_unexpected_parameters_message(incompatible.unexpected_keys), logger='GridGCN')
        print_log(f'[GridGCN] Successful Loading the ckpt from {ckpt_path}', logger='GridGCN')

    def forward(self, pts):
        """pts: [B, N, 3] or [B, 3, N] -> logits [B, cls_dim]."""
        if pts.size(-1) != 3 and pts.size(1) == 3:
            pts = pts.transpose(1, 2)
        pts = pts.contiguous()
        B, N, _ = pts.shape
        device = pts.device

        # channel 3 is the coverage weight: every raw point counts as 1 (§2)
        feat = torch.cat([pts, pts.new_ones(B, N, 1)], dim=-1)
        n_valid = torch.full((B,), N, dtype=torch.long, device=device)
        b_idx = torch.arange(B, device=device).view(B, 1, 1)
        deterministic = self.deterministic_eval and not self.training

        center_feats = None
        for i, stage in enumerate(self.stages):
            nebidx, nebmsk, cent, centmsk, n_cent = cagq(
                feat[..., :3], feat[..., 3:4], n_valid,
                voxel_size=self.voxel_size_lst[i],
                grid_size=self.grid_size_lst[i],
                coord_shift=self.coord_shift,
                max_o=self.max_o_grid_lst[i],
                max_p=self.max_p_grid_lst[i],
                kernel_size=self.kernel_size_lst[i],
                loc=self.loc_within,
                query=self.query_type,
                radius=self.radius_lst[i],
                deterministic=deterministic,
            )

            neighbors = feat[b_idx, nebidx]                      # [B, O, P, C]
            neighbors = neighbors.permute(0, 3, 1, 2).contiguous()  # [B, C, O, P]
            centers_xyz = cent[..., 0:3].transpose(1, 2).contiguous()
            center_den = cent[..., 3:4].transpose(1, 2).contiguous()

            center_feats = stage(neighbors, centers_xyz, center_den,
                                 centmsk, nebmsk, num_points=N)  # [B, C, O]

            # next stage consumes concat(centres, features); head consumes center_feats only
            feat = torch.cat([cent, center_feats.transpose(1, 2)], dim=-1)
            n_valid = n_cent

        # last stage has O = 1, so this is a squeeze; max() stays valid for larger O
        global_feat = center_feats.max(dim=-1).values            # [B, C]

        self._store_features(global_feat)
        return self.cls_head(global_feat)
