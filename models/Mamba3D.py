"""Mamba3D — Enhancing Local Features for 3D Point Cloud Analysis via State Space Model.
Han et al. ACM MM 2024. doi:10.1145/3664647.3681173
"""

import math
import numpy as np
import random
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .build import MODELS
from collections import namedtuple
from functools import partial
from timm.models.layers import DropPath, trunc_normal_
from typing import Optional
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from utils.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from utils.cls_losses import ClassificationLossMixin
from utils.feature_hooks import FeatureExtractionMixin

# Mamba import
from .bimamba_ssm.modules.mamba_simple import Mamba

try:
    from .bimamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        """point_groups [B, G, N, 3] -> feature_global [B, G, C]."""
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)

class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        """xyz [B, N, 3] -> grouped [B, G, M, 3], centers [B, G, 3]."""
        batch_size, num_points, _ = xyz.shape
        center = misc.fps(xyz, self.num_group)  # B G 3
        _, idx = misc.knn(xyz, center, self.group_size)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)  # [B * G * M]
        xyz_flat = xyz.view(batch_size * num_points, -1)  # [B*N, 3]
        neighborhood = xyz_flat[idx, :]  # [B*G*M, 3]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)  # B G M 3
        return neighborhood, center

class GroupFeature(nn.Module):  # FPS + KNN
    def __init__(self, group_size):
        super().__init__()
        self.group_size = group_size  # the first is the point itself

    def forward(self, xyz, feat):
        """xyz [B, N, 3], feat [B, N, C] -> neighbourhood [B, N, K, 3], feature [B, N, K, C]."""
        batch_size, num_points, _ = xyz.shape # B N 3 : 1 128 3
        C = feat.shape[-1]

        center = xyz
        _, idx = misc.knn(xyz, xyz, self.group_size) # B N K : get K idx for every center
        assert idx.size(1) == num_points # N center
        assert idx.size(2) == self.group_size # K knn group
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :] # B N K 3
        neighborhood = neighborhood.view(batch_size, num_points, self.group_size, 3).contiguous() # 1 128 8 3
        neighborhood_feat = feat.contiguous().view(-1, C)[idx, :] # BxNxK C 128x8 384   128*26*8
        assert neighborhood_feat.shape[-1] == feat.shape[-1]
        neighborhood_feat = neighborhood_feat.view(batch_size, num_points, self.group_size, feat.shape[-1]).contiguous() # 1 128 8 384
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        
        return neighborhood, neighborhood_feat

class Sine(nn.Module):
    def __init__(self, w0 = 30.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

class K_Norm(nn.Module):
    def __init__(self, out_dim, k_group_size, alpha, beta):
        super().__init__()
        self.group_feat = GroupFeature(k_group_size)
        self.affine_alpha_feat = nn.Parameter(torch.ones([1, 1, 1, out_dim]))
        self.affine_beta_feat = nn.Parameter(torch.zeros([1, 1, 1, out_dim]))
        self.eps = 1e-5

    def forward(self, lc_xyz, lc_x):
        knn_xyz, knn_x = self.group_feat(lc_xyz, lc_x) # B G K 3, B G K C

        # blueprint normalize: centre-feature subtraction + a single scalar std
        mean_x = lc_x.unsqueeze(dim=-2) # B G 1 C
        std_x = torch.std(knn_x - mean_x)

        mean_xyz = lc_xyz.unsqueeze(dim=-2)
        std_xyz = torch.std(knn_xyz - mean_xyz) # B G 1 3

        knn_x = (knn_x - mean_x) / (std_x + self.eps)
        knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + self.eps) # B G K 3

        B, G, K, C = knn_x.shape

        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1) # B G K 2C

        knn_x = self.affine_alpha_feat * knn_x + self.affine_beta_feat 
        
        knn_x_w = knn_x.permute(0, 3, 1, 2) # B 2C G K

        return knn_x_w

class K_Pool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, knn_x_w):
        # exp-weighted soft pooling over K: sum_k softmax(v)_k * v_k
        # (max-subtracted form of the blueprint's mean(v*e^v)/mean(e^v))
        max_val = knn_x_w.max(dim=-1, keepdim=True)[0]
        e_x = torch.exp(knn_x_w - max_val)
        lc_x = (knn_x_w * e_x).mean(-1) / e_x.mean(-1)
        return lc_x

class Post_ShareMLP(nn.Module):
    def __init__(self, in_dim, out_dim, permute=True):
        super().__init__()
        self.share_mlp = torch.nn.Conv1d(in_dim, out_dim, 1)
        self.permute = permute
    
    def forward(self, x):
        # x: B 2C G mlp-> B C G  permute-> B G C
        output = self.share_mlp(x)
            
        if self.permute:
            return output.permute(0, 2, 1)
        else:
            return output

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class LNPBlock(nn.Module):
    def __init__(self, lga_out_dim, k_group_size, alpha, beta, mlp_in_dim, mlp_out_dim, num_group=128, act_layer=nn.SiLU, drop_path=0., norm_layer=nn.LayerNorm,):
        super().__init__()
        """x -> (lga -> pool -> mlp -> act) -> x. lga_out_dim = mlp_in_dim = 2C, mlp_out_dim = C."""
        self.num_group = num_group
        self.lga_out_dim = lga_out_dim

        self.lga = K_Norm(self.lga_out_dim, k_group_size, alpha, beta)
        self.kpool = K_Pool()
        self.mlp = Post_ShareMLP(mlp_in_dim, mlp_out_dim)
        self.pre_norm_ft = norm_layer(self.lga_out_dim, eps=1e-6)  # Increased epsilon

        self.act = act_layer()

    def forward(self, center, feat):
        # feat: B G+1 C
        B, G, C = feat.shape
        cls_token = feat[:,0,:].view(B, 1, C)
        feat = feat[:,1:,:] # B G C
        lc_x_w = self.lga(center, feat) # B 2C G K 
        lc_x_w = self.kpool(lc_x_w) # B 2C G : 1 768 128

        # norm([2C])
        lc_x_w = self.pre_norm_ft(lc_x_w.permute(0, 2, 1)) #pre-norm B G 2C
        lc_x = self.mlp(lc_x_w.permute(0, 2, 1)) # B G C : 1 128 384
        
        lc_x = self.act(lc_x)
        
        lc_x = torch.cat((cls_token, lc_x), dim=1) # B G+1 C : 1 129 384
        return lc_x

class Mamba3DBlock(nn.Module):
    def __init__(self, 
                dim, 
                mlp_ratio=4., 
                drop=0., 
                drop_path=0., 
                act_layer=nn.SiLU, 
                norm_layer=nn.LayerNorm,
                k_group_size=8, 
                alpha=100, 
                beta=1000,
                num_group=128,
                bimamba_type="v2",
                ):
        super().__init__()
        self.norm1 = norm_layer(dim, eps=1e-6)  # Increased epsilon
        self.norm2 = norm_layer(dim, eps=1e-6)  # Increased epsilon

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.num_group = num_group
        self.k_group_size = k_group_size
                
        self.lfa = LNPBlock(lga_out_dim=dim*2, 
                    k_group_size=self.k_group_size, 
                    alpha=alpha, 
                    beta=beta, 
                    mlp_in_dim=dim*2, 
                    mlp_out_dim=dim, 
                    num_group=self.num_group,
                    act_layer=act_layer,
                    drop_path=drop_path,
                    norm_layer=norm_layer,
                    )

        self.mixer = Mamba(dim, bimamba_type=bimamba_type)

    def forward(self, center, x):   
        # x + norm(x)->lfa(x)->dropout
        residual = x
        lfa_out = self.lfa(center, self.norm1(x))
        x = residual + self.drop_path(lfa_out)
        
        # x + norm(x)->mamba(x)->dropout
        residual = x
        mamba_out = self.mixer(self.norm2(x))
            
        x = residual + self.drop_path(mamba_out)
        return x

class Mamba3DEncoder(nn.Module):
    def __init__(self, k_group_size=8, embed_dim=768, depth=4, drop_path_rate=0., num_group=128, bimamba_type="v2",):
        super().__init__()
        self.num_group = num_group
        self.k_group_size = k_group_size
        self.blocks = nn.ModuleList([
            Mamba3DBlock(
                dim=embed_dim, #
                k_group_size = self.k_group_size,
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate, #
                num_group=self.num_group,
                bimamba_type=bimamba_type,
                )
            for i in range(depth)])

    def forward(self, center, x, pos):
        """x [B, G+1, C], pos [B, G+1, C] -> x [B, G+1, C]. Positional encoding is re-added every block."""
        for i, block in enumerate(self.blocks):
            x = block(center, x + pos)
        return x

class Mamba3DDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, drop_path_rate=0.1, norm_layer=nn.LayerNorm, bimamba_type="v2"):
        super().__init__()
        self.blocks = nn.ModuleList([
            Mamba3DDecoderBlock(
                dim=embed_dim, 
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                bimamba_type=bimamba_type
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim, eps=1e-6)  # Increased epsilon
        self.head = nn.Identity()
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
        
        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
            
        return x

class Mamba3DDecoderBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.SiLU, 
                 norm_layer=nn.LayerNorm, bimamba_type="v2"):
        super().__init__()
        self.norm1 = norm_layer(dim, eps=1e-6)  # Increased epsilon
        self.norm2 = norm_layer(dim, eps=1e-6)  # Increased epsilon
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mixer = Mamba(dim, bimamba_type=bimamba_type)
        
    def forward(self, x):
        residual = x
        x = residual + self.drop_path(self.mixer(self.norm1(x)))
        
        residual = x
        x = residual + self.drop_path(self.mlp(self.norm2(x)))

        return x

class MaskMamba3D(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        
        self.mask_ratio = config.transformer_config.mask_ratio 
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth 
        self.drop_path_rate = config.transformer_config.drop_path_rate
        print_log(f'[args] {config.transformer_config}', logger='MaskMamba3D')
        
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        
        self.mask_type = config.transformer_config.mask_type
        self.k_group_size = getattr(config.transformer_config, 'center_local_k', 8)
        # bimamba_type: from transformer_config, else root config (blueprint default v4)
        self.bimamba_type = getattr(config.transformer_config, 'bimamba_type',
                                    getattr(config, 'bimamba_type', "v4"))
        
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.SiLU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = Mamba3DEncoder(
            embed_dim=self.trans_dim,
            k_group_size=self.k_group_size,
            depth=self.depth,
            drop_path_rate=dpr,
            bimamba_type=self.bimamba_type,
        )

        self.norm = nn.LayerNorm(self.trans_dim, eps=1e-6)

        # learnable CLS token / CLS pos
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))

        self.apply(self._init_weights)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        """center [B, G, 3] -> mask [B, G] bool."""
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G
        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        """center [B, G, 3] -> mask [B, G] bool."""
        B, G, _ = center.shape
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device) # B G

    def forward(self, neighborhood, center, noaug=False):
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug) # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)

        group_input_tokens = self.encoder(neighborhood)  #  B G C

        batch_size, seq_len, C = group_input_tokens.size()

        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_vis = torch.cat((cls_tokens, x_vis), dim=1)

        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)
        cls_pos = self.cls_pos.expand(batch_size, -1, -1)
        pos = torch.cat((cls_pos, pos), dim=1)

        x_vis = self.blocks(masked_center, x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis, bool_masked_pos

@MODELS.register_module()
class Mamba3D_Pretrain(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Mamba3D_Pretrain]', logger='Mamba3D_Pretrain')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskMamba3D(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        # bimamba_type: from transformer_config, else root config (blueprint default v4)
        self.bimamba_type = getattr(config.transformer_config, 'bimamba_type',
                                    getattr(config, 'bimamba_type', "v4"))
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.SiLU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = Mamba3DDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            bimamba_type=self.bimamba_type,
        )

        print_log(f'[Mamba3D_Pretrain] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger='Mamba3D_Pretrain')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = [ChamferDistanceL1()]
        elif loss_type =='cdl2':
            self.loss_func = [ChamferDistanceL2()]
        elif loss_type == 'cdl12':
            self.loss_func = [ChamferDistanceL1(), ChamferDistanceL2()]
        else:
            raise NotImplementedError

    def forward(self, pts, vis=False, **kwargs):
        loss = 0
        neighborhood, center = self.group_divider(pts)

        x_vis, mask = self.MAE_encoder(neighborhood, center)
        B, _, C = x_vis.shape # B VIS+1 C (includes cls token)

        # Remove cls token for position embedding calculation
        x_vis_no_cls = x_vis[:, 1:]  # Remove cls token
        
        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

        _, N, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis_no_cls, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        x_rec = self.MAE_decoder(x_full, pos_full, N)

        B, M, C = x_rec.shape
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)

        gt_points = neighborhood[mask].reshape(B*M, -1, 3)

        for l in self.loss_func:
            # FIX: was `loss = l(...)`, which discarded all but the last term
            # (with cdl12 only Chamfer-L2 was actually optimized)
            loss += l(rebuild_points, gt_points)

        if vis:  # visualization
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
            full_vis = vis_points + center[~mask].unsqueeze(1)
            full_rebuild = rebuild_points + center[mask].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0)
            full_center = torch.cat([center[mask], center[~mask]], dim=0)
            ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
            ret1 = full.reshape(-1, 3).unsqueeze(0)
            return ret1, ret2, full_center
        else:
            return loss

@MODELS.register_module()
class Mamba3D_Classification(nn.Module, ClassificationLossMixin, FeatureExtractionMixin):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.name = config.NAME
        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.cls_dim = config.cls_dim
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims
        self.freeze_encoder = getattr(config, 'freeze_encoder', False)

        self.drop_path_rate = config.drop_path_rate

        # self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.SiLU(),
            nn.Linear(128, self.trans_dim)
        )

        self.ordering = getattr(config, 'ordering', False)  # stored, unused (kept for config compatibility)
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        
        self.k_group_size = config.center_local_k # default=8

        self.bimamba_type = config.bimamba_type

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        #define the encoder
        self.blocks = Mamba3DEncoder(
            embed_dim=self.trans_dim,
            k_group_size=self.k_group_size,
            depth=self.depth,
            drop_path_rate=dpr,
            num_group=self.num_group,
            bimamba_type=self.bimamba_type,
        )
        #embed_dim=768, depth=4, drop_path_rate=0.

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )

        # ce | focal
        self._init_cls_loss(config, self.cls_dim)
        self._init_feature_extraction()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def freeze_encoder_weights(self):
        """Freeze encoder and transformer blocks if specified"""
        if self.freeze_encoder:
            print_log('Freezing encoder weights...', logger='Mamba3D_Classification')
            # Freeze the point encoder
            for param in self.encoder.parameters():
                param.requires_grad = False 
            # Freeze the transformer blocks
            for param in self.blocks.parameters():
                param.requires_grad = False
            # Freeze positional embeddings
            for param in self.pos_embed.parameters():
                param.requires_grad = False
            # Keep cls_token, cls_pos, norm, and classification head trainable
            print_log('PointMamba Encoder frozen. Only classification head and cls tokens remain trainable.', 
                     logger='Mamba3D_Classification')

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder') :
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Mamba3D_Classification')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Mamba3D_Classification')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Mamba3D_Classification'
                )

            print_log(f'[Mamba3D_Classification] Successful Loading the ckpt from {bert_ckpt_path}', logger='Mamba3D_Classification')
        else:
            print_log('Training from scratch!!!', logger='Mamba3D_Classification')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        neighborhood, center = self.group_divider(pts) # B G K 3
        group_input_tokens = self.encoder(neighborhood)  # B G C

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center) # B G C

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(center, x, pos) # enter transformer blocks
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0] + x[:, 1:].mean(1)[0]], dim=-1)

        self._store_features(concat_f)
        
        ret = self.cls_head_finetune(concat_f)
        return ret   
    
