"""Point-BERT — Pre-training 3D Point Cloud Transformers with Masked Point Modeling.
Yu et al. CVPR 2022. doi:10.1109/CVPR52688.2022.01871
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import timm
from timm.models.layers import DropPath, trunc_normal_

from .build import MODELS
from utils import misc
from utils.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from utils.cls_losses import ClassificationLossMixin
from utils.feature_hooks import FeatureExtractionMixin
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random

@torch.no_grad()
def concat_all_gather(tensor):
    """all_gather across ranks (no gradient)."""
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor
        
    tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class KNN:
    def __init__(self, k=1, transpose_mode=False):
        self.k = k
        self.transpose_mode = transpose_mode
    
    def __call__(self, support_points, query_points):
        return self.forward(support_points, query_points)
    
    def forward(self, support_points, query_points):
        """support_points, query_points (B, N, 3) -> (distances, indices), both (B, N, k)."""
        if self.transpose_mode:
            support_points = support_points.transpose(-2, -1).contiguous()
            query_points = query_points.transpose(-2, -1).contiguous()
        batch_size = support_points.size(0)
        num_support = support_points.size(1)
        num_query = query_points.size(1)
        
        device = support_points.device
        query_points = query_points.to(device)
        
        
        # broadcasting (memory-efficient for small clouds)
        if num_support * num_query < 1000000:  # Threshold to avoid memory issues
            support_expanded = support_points.unsqueeze(2)  # (B, M, 1, 3)
            query_expanded = query_points.unsqueeze(1)      # (B, 1, N, 3)
            
            squared_distances = torch.sum((support_expanded - query_expanded) ** 2, dim=-1)  # (B, M, N)
            
        else:
            # matrix form: ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
            support_norm = torch.sum(support_points ** 2, dim=-1, keepdim=True)  # (B, M, 1)
            query_norm = torch.sum(query_points ** 2, dim=-1, keepdim=True)      # (B, N, 1)
            
            # Batch matrix multiplication
            cross_term = torch.bmm(support_points, query_points.transpose(-2, -1))  # (B, M, N)
            squared_distances = support_norm + query_norm.transpose(-2, -1) - 2 * cross_term  # (B, M, N)
        
        squared_distances = squared_distances.transpose(-2, -1)  # (B, N, M)
        
        # Get k smallest distances and their indices
        distances, indices = torch.topk(squared_distances, k=self.k, dim=-1, largest=False)
        
        return distances, indices
    
knn = KNN(k=4, transpose_mode=True)

class DGCNN(nn.Module):
    def __init__(self, encoder_channel, output_channel):
        super().__init__()
        self.input_trans = nn.Conv1d(encoder_channel, 128, 1) 

        self.layer1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 256),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 512),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer3 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 512),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer4 = nn.Sequential(nn.Conv2d(1024, 1024, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 1024),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer5 = nn.Sequential(nn.Conv1d(2304, output_channel, kernel_size=1, bias=False),
                                nn.GroupNorm(4, output_channel),
                                nn.LeakyReLU(negative_slope=0.2)
                                )

    @staticmethod
    def get_graph_feature(coor_q, x_q, coor_k, x_k):

        # coor: bs, 3, np, x: bs, c, np

        k = 4
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            _, idx = knn(coor_k, coor_q)  # bs k np
            idx = idx.transpose(-2, -1).contiguous()
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, f, coor):
        # f: B G C
        # coor: B G 3 

        # bs 3 N   bs C N
        feature_list  = []
        coor = coor.transpose(1, 2).contiguous()         # B 3 N
        f = f.transpose(1, 2).contiguous()               # B C N
        f = self.input_trans(f)             # B 128 N

        f = self.get_graph_feature(coor, f, coor, f) # B 256 N k
        f = self.layer1(f)                           # B 256 N k
        f = f.max(dim=-1, keepdim=False)[0]          # B 256 N
        feature_list.append(f)

        f = self.get_graph_feature(coor, f, coor, f) # B 512 N k
        f = self.layer2(f)                           # B 512 N k
        f = f.max(dim=-1, keepdim=False)[0]          # B 512 N
        feature_list.append(f)

        f = self.get_graph_feature(coor, f, coor, f) # B 1024 N k
        f = self.layer3(f)                           # B 512 N k
        f = f.max(dim=-1, keepdim=False)[0]          # B 512 N
        feature_list.append(f)

        f = self.get_graph_feature(coor, f, coor, f) # B 1024 N k
        f = self.layer4(f)                           # B 1024 N k
        f = f.max(dim=-1, keepdim=False)[0]          # B 1024 N
        feature_list.append(f)

        f = torch.cat(feature_list, dim = 1)         # B 2304 N

        f = self.layer5(f)                           # B C' N
        
        f = f.transpose(-1, -2)

        return f   

class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        """xyz [B, N, 3] -> grouped [B, G, M, 3], centers [B, G, 3]."""
        batch_size, num_points, _ = xyz.shape
        center = misc.fps(xyz, self.num_group) # B G 3
        _, idx = misc.knn(xyz, center, self.group_size)
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center

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

class Decoder(nn.Module):
    def __init__(self, encoder_channel, num_fine):
        super().__init__()
        self.num_fine = num_fine
        self.grid_size = 2
        self.num_coarse = self.num_fine // 4
        assert num_fine % 4 == 0

        self.mlp = nn.Sequential(
            nn.Linear(encoder_channel, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )
        self.final_conv = nn.Sequential(
            nn.Conv1d(encoder_channel + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2) # 1 2 S

    def forward(self, feature_global):
        """feature_global [B, G, C] -> coarse [B, G, M, 3], fine [B, G, N, 3]."""
        bs, g, c = feature_global.shape
        feature_global = feature_global.reshape(bs * g, c)

        coarse = self.mlp(feature_global).reshape(bs * g, self.num_coarse, 3) # BG M 3

        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size**2, -1) # BG (M) S 3
        point_feat = point_feat.reshape(bs * g, self.num_fine, 3).transpose(2, 1) # BG 3 N

        seed = self.folding_seed.unsqueeze(2).expand(bs * g, -1, self.num_coarse, -1) # BG 2 M (S)
        seed = seed.reshape(bs * g, -1, self.num_fine).to(feature_global.device)  # BG 2 N

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_fine) # BG 1024 N
        feat = torch.cat([feature_global, seed, point_feat], dim=1) # BG C N
    
        center = coarse.unsqueeze(2).expand(-1, -1, self.grid_size**2, -1) # BG (M) S 3
        center = center.reshape(bs * g, self.num_fine, 3).transpose(2, 1) # BG 3 N

        fine = self.final_conv(feat) + center   # BG 3 N
        fine = fine.reshape(bs, g, 3, self.num_fine).transpose(-1, -2)
        coarse = coarse.reshape(bs, g, self.num_coarse, 3)
        return coarse, fine

@MODELS.register_module()
class PointBERT_DVAE_Pretrain(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims
        self.tokens_dims = config.tokens_dims

        self.decoder_dims = config.decoder_dims
        self.num_tokens = config.num_tokens

        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        self.dgcnn_1 = DGCNN(encoder_channel = self.encoder_dims, output_channel = self.num_tokens)
        self.codebook = nn.Parameter(torch.randn(self.num_tokens, self.tokens_dims))

        self.dgcnn_2 = DGCNN(encoder_channel = self.tokens_dims, output_channel = self.decoder_dims)
        self.decoder = Decoder(encoder_channel = self.decoder_dims, num_fine = self.group_size)
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

    def recon_loss(self, ret, gt):
        loss = 0
        whole_coarse, whole_fine, coarse, fine, group_gt, _ = ret

        bs, g, _, _ = coarse.shape

        coarse = coarse.reshape(bs*g, -1, 3).contiguous()
        fine = fine.reshape(bs*g, -1, 3).contiguous()
        group_gt = group_gt.reshape(bs*g, -1, 3).contiguous()

        for l in self.loss_func:
            loss += l(coarse, group_gt) # loss_coarse_block
            loss += l(fine, group_gt) # loss_fine_block

        return loss

    def get_loss(self, ret, gt):

        loss_recon = self.recon_loss(ret, gt)
        # KL divergence
        logits = ret[-1] # B G N
        softmax = F.softmax(logits, dim=-1)
        mean_softmax = softmax.mean(dim=1)
        log_qy = torch.log(mean_softmax)
        log_uniform = torch.log(torch.tensor([1. / self.num_tokens], device = gt.device))
        loss_klv = F.kl_div(log_qy, log_uniform.expand(log_qy.size(0), log_qy.size(1)), None, None, 'batchmean', log_target = True)

        return loss_recon, loss_klv

    def forward(self, inp, temperature = 1., hard = False, **kwargs):
        neighborhood, center = self.group_divider(inp)
        logits = self.encoder(neighborhood)   #  B G C
        logits = self.dgcnn_1(logits, center) #  B G N
        soft_one_hot = F.gumbel_softmax(logits, tau = temperature, dim = 2, hard = hard) # B G N
        sampled = torch.einsum('b g n, n c -> b g c', soft_one_hot, self.codebook) # B G C
        feature = self.dgcnn_2(sampled, center)
        coarse, fine = self.decoder(feature)

        with torch.no_grad():
            whole_fine = (fine + center.unsqueeze(2)).reshape(inp.size(0), -1, 3)
            whole_coarse = (coarse + center.unsqueeze(2)).reshape(inp.size(0), -1, 3)

        assert fine.size(2) == self.group_size
        ret = (whole_coarse, whole_fine, coarse, fine, neighborhood, logits)
        return ret

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

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TransformerEncoder(nn.Module):
    """Transformer encoder, no hierarchical structure."""
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x

class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.mask_ratio = config.transformer_config.mask_ratio 
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth 
        self.drop_path_rate = config.transformer_config.drop_path_rate 
        self.cls_dim = config.transformer_config.cls_dim 
        self.replace_pob = config.transformer_config.replace_pob
        self.num_heads = config.transformer_config.num_heads 
        print_log(f'[Transformer args] {config.transformer_config}', logger = 'dVAE BERT')
        self.encoder_dims =  config.dvae_config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        self.reduce_dim = nn.Linear(self.encoder_dims,  self.trans_dim)
        try:
            self.mask_rand = config.mask_rand
        except:
            self.mask_rand = False
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads
        )
        self.norm = nn.LayerNorm(self.trans_dim)
        self.num_tokens = config.dvae_config.num_tokens
        self.lm_head = nn.Linear(self.trans_dim, self.num_tokens)
        self.cls_head = nn.Sequential(
            nn.Linear(self.trans_dim, self.cls_dim),
            nn.GELU(),
            nn.Linear(self.cls_dim, self.cls_dim)
        )  
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        trunc_normal_(self.mask_token, std=.02)

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

    def _prepare_encoder(self, dvae_ckpt):
        ckpt = torch.load(dvae_ckpt, map_location='cpu')
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        encoder_ckpt = {k.replace("encoder.", ""): v for k, v in base_ckpt.items() if 'encoder' in k}

        self.encoder.load_state_dict(encoder_ckpt, strict=True)
        print_log(f'[Encoder] Successful Loading the ckpt for encoder from {dvae_ckpt}', logger = 'dVAE BERT')

    def _mask_center(self, center, noaug = False):
        """center [B, G, 3] -> mask [B, G] bool."""
        if noaug or self.mask_ratio[1] == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0) # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p =2 ,dim = -1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0] # G
            ratio = random.uniform(self.mask_ratio[0], self.mask_ratio[1])
            mask_num = int(ratio * len(idx))

            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())
        
        bool_masked_pos = torch.stack(mask_idx).to(center.device) # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug = False):
        """center [B, G, 3] -> mask [B, G] bool."""
        if noaug or self.mask_ratio[1] == 0:
            return torch.zeros(center.shape[:2]).bool()
        
        ratio = random.random() * (self.mask_ratio[1] - self.mask_ratio[0]) + self.mask_ratio[0]
        bool_masked_pos = (torch.rand(center.shape[:2]) < ratio).bool().to(center.device)

        return bool_masked_pos

    def _random_replace(self, group_input_tokens, bool_masked_pos, noaug = False):
        '''
            group_input_tokens : B G C
            bool_masked_pos : B G  
            -----------------
            replaced_group_input_tokens: B G C
        '''
        # skip replace
        if noaug or self.replace_pob == 0:
            return group_input_tokens, bool_masked_pos
        
        replace_mask = (torch.rand(group_input_tokens.shape[:2]) < self.replace_pob).to(bool_masked_pos.device).bool()
        replace_mask = (replace_mask & ~bool_masked_pos)  # do not replace the mask pos

        overall_mask = (replace_mask + bool_masked_pos).bool().to(bool_masked_pos.device)  #  True for flake input

        detached_group_input_tokens = group_input_tokens.detach()
        flatten_group_input_tokens = detached_group_input_tokens.reshape(detached_group_input_tokens.size(0) * detached_group_input_tokens.size(1), detached_group_input_tokens.size(2))
        idx = torch.randperm(flatten_group_input_tokens.shape[0])
        shuffled_group_input_tokens = flatten_group_input_tokens[idx].reshape(detached_group_input_tokens.size(0), detached_group_input_tokens.size(1), detached_group_input_tokens.size(2))

        replace_mask = replace_mask.unsqueeze(-1).type_as(detached_group_input_tokens)
        replaced_group_input_tokens = group_input_tokens * (1 - replace_mask) + shuffled_group_input_tokens * replace_mask
        return replaced_group_input_tokens, overall_mask

    def forward(self, neighborhood, center, return_all_tokens = False, only_cls_tokens = False, noaug = False):
        # generate mask
        if self.mask_rand:
            bool_masked_pos = self._mask_center_rand(center, noaug = noaug) # B G
        else:
            bool_masked_pos = self._mask_center(center, noaug = noaug) # B G
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  #  B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # replace the point
        replaced_group_input_tokens, overall_mask = self._random_replace(group_input_tokens, bool_masked_pos.clone(), noaug = noaug)
        batch_size, seq_len, _ = replaced_group_input_tokens.size()
        # prepare cls and mask
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  
        cls_pos = self.cls_pos.expand(batch_size, -1, -1)  
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)
        # mask the input tokens
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        maksed_group_input_tokens = replaced_group_input_tokens * (1 - w) + mask_token * w
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, maksed_group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        # only return the cls feature, for moco contrast
        if only_cls_tokens:
            return self.cls_head(x[:, 0])
        logits = self.lm_head(x[:, 1:])
        if return_all_tokens:
            return self.cls_head(x[:, 0]), logits
        else:
            # return the flake logits
            return self.cls_head(x[:, 0]), logits[~overall_mask], logits[overall_mask], overall_mask # reduce the Batch dim        

@MODELS.register_module()
class PointBERT_Pretrain(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_BERT] build dVAE_BERT ...', logger ='PointBERT_Pretrain')
        self.config = config
        self.m = config.m
        self.T = config.T
        self.K = config.K

        self.moco_loss = config.transformer_config.moco_loss
        self.dvae_loss = config.transformer_config.dvae_loss
        self.cutmix_loss = config.transformer_config.cutmix_loss

        self.return_all_tokens = config.transformer_config.return_all_tokens
        if self.return_all_tokens:
            print_log(f'[Point_BERT] Point_BERT calc the loss for all token ...', logger ='PointBERT_Pretrain')
        else:
            print_log(f'[Point_BERT] Point_BERT [NOT] calc the loss for all token ...', logger ='PointBERT_Pretrain')
        
        self.transformer_q = MaskTransformer(config)
        self.transformer_q._prepare_encoder(self.config.dvae_config.ckpt)
        
        self.transformer_k = MaskTransformer(config)
        for param_q, param_k in zip(self.transformer_q.parameters(), self.transformer_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.dvae = PointBERT_DVAE_Pretrain(config.dvae_config)
        self._prepare_dvae()

        for param in self.dvae.parameters():
            param.requires_grad = False

        self.group_size = config.dvae_config.group_size
        self.num_group = config.dvae_config.num_group

        print_log(f'[Point_BERT Group] cutmix_BERT divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='PointBERT_Pretrain')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        # create the queue
        self.register_buffer("queue", torch.randn(self.transformer_q.cls_dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # loss
        self.build_loss_func()

    def _prepare_dvae(self):
        dvae_ckpt = self.config.dvae_config.ckpt
        ckpt = torch.load(dvae_ckpt, map_location='cpu')
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        self.dvae.load_state_dict(base_ckpt, strict=True)
        print_log(f'[dVAE] Successful Loading the ckpt for dvae from {dvae_ckpt}', logger ='PointBERT_Pretrain')

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.transformer_q.parameters(), self.transformer_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # Only gather keys if distributed training is being used
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # Ensure we don't exceed queue size
        if self.K % batch_size != 0:
            # Handle case where batch size doesn't divide evenly into queue size
            remaining_space = self.K - ptr
            if batch_size <= remaining_space:
                self.queue[:, ptr:ptr + batch_size] = keys.T
                ptr = (ptr + batch_size) % self.K
            else:
                # Split the batch to fit in remaining space
                self.queue[:, ptr:] = keys[:remaining_space].T
                remaining_keys = keys[remaining_space:ptr + batch_size - self.K]
                if remaining_keys.size(0) > 0:
                    self.queue[:, :remaining_keys.size(0)] = remaining_keys.T
                ptr = remaining_keys.size(0)
        else:
            # Original logic when batch size divides evenly
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_ce_batch = nn.CrossEntropyLoss(reduction='none')

    def forward_eval(self, pts):
        with torch.no_grad():
            neighborhood, center = self.group_divider(pts)
            cls_feature = self.transformer_q(neighborhood, center, only_cls_tokens = True, noaug = True)
            return cls_feature

    def _mixup_pc(self, neighborhood, center, dvae_label):
        """neighborhood [B, G, M, 3], center [B, G, 3], dvae_label [B, G]. mixup_label = alpha*origin + (1-alpha)*flip."""
        mixup_ratio = torch.rand(neighborhood.size(0))
        mixup_mask = torch.rand(neighborhood.shape[:2]) < mixup_ratio.unsqueeze(-1)
        mixup_mask = mixup_mask.type_as(neighborhood)
        mixup_neighborhood = neighborhood * mixup_mask.unsqueeze(-1).unsqueeze(-1) + neighborhood.flip(0) * (1 - mixup_mask.unsqueeze(-1).unsqueeze(-1))
        mixup_center = center * mixup_mask.unsqueeze(-1) + center.flip(0) * (1 - mixup_mask.unsqueeze(-1))
        mixup_dvae_label = dvae_label * mixup_mask + dvae_label.flip(0) * (1 - mixup_mask)

        return mixup_ratio.to(neighborhood.device), mixup_neighborhood, mixup_center, mixup_dvae_label.long()

    def forward(self, pts, noaug = False, **kwargs):
        if noaug:
            return self.forward_eval(pts)
        else:
            # divide the point cloud in the same form. This is important
            neighborhood, center = self.group_divider(pts)
            # produce the gt point tokens
            with torch.no_grad():
                gt_logits = self.dvae.encoder(neighborhood) 
                gt_logits = self.dvae.dgcnn_1(gt_logits, center) #  B G N
                dvae_label = gt_logits.argmax(-1).long() # B G 
            # forward the query model in mask style 1.
            if self.return_all_tokens:
                q_cls_feature, logits = self.transformer_q(neighborhood, center, return_all_tokens = self.return_all_tokens) # logits :  N G C 
            else:
                q_cls_feature, real_logits, flake_logits, mask = self.transformer_q(neighborhood, center, return_all_tokens = self.return_all_tokens) # logits :  N' C where N' is the mask.sum() 
            q_cls_feature = nn.functional.normalize(q_cls_feature, dim=1)

            mixup_ratio, mixup_neighborhood, mixup_center, mix_dvae_label = self._mixup_pc(neighborhood, center, dvae_label)
            if self.return_all_tokens:
                mixup_cls_feature, mixup_logits = self.transformer_q(mixup_neighborhood, mixup_center, return_all_tokens = self.return_all_tokens) 
            else:
                mixup_cls_feature, mixup_real_logits, mixup_flake_logits, mixup_mask = self.transformer_q(mixup_neighborhood, mixup_center, return_all_tokens = self.return_all_tokens)
            mixup_cls_feature = nn.functional.normalize(mixup_cls_feature, dim=1)

            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder
                k_cls_feature = self.transformer_k(neighborhood, center, only_cls_tokens = True)  # keys: NxC
                k_cls_feature = nn.functional.normalize(k_cls_feature, dim=1)

            if self.moco_loss:
                # ce loss with moco contrast
                l_pos = torch.einsum('nc, nc->n', [q_cls_feature, k_cls_feature]).unsqueeze(-1) # n 1 
                l_neg = torch.einsum('nc, ck->nk', [q_cls_feature, self.queue.clone().detach()]) # n k
                ce_logits = torch.cat([l_pos, l_neg], dim=1)
                ce_logits /= self.T
                labels = torch.zeros(l_pos.shape[0], dtype=torch.long).to(pts.device)
                moco_loss = self.loss_ce(ce_logits, labels)
            else:
                moco_loss = torch.tensor(0.).to(pts.device)

            if self.dvae_loss:
                if self.return_all_tokens:
                    dvae_loss = self.loss_ce(logits.reshape(-1, logits.size(-1)), dvae_label.reshape(-1,)) + \
                             self.loss_ce(mixup_logits.reshape(-1, mixup_logits.size(-1)), mix_dvae_label.reshape(-1,))
                else:
                    dvae_loss = self.loss_ce(flake_logits, dvae_label[mask]) + \
                        self.loss_ce(mixup_flake_logits, mix_dvae_label[mixup_mask])
            else:
                dvae_loss = torch.tensor(0.).to(pts.device)

            if self.cutmix_loss:                
                l_pos = torch.einsum('nc, mc->nm', [mixup_cls_feature, k_cls_feature]) # n n
                l_neg = torch.einsum('nc, ck->nk', [mixup_cls_feature, self.queue.clone().detach()]) # n k
                ce_logits = torch.cat([l_pos, l_neg], dim=1)
                ce_logits /= self.T
                labels = torch.arange(l_pos.shape[0], dtype=torch.long).to(pts.device)
                cutmix_loss = (mixup_ratio * self.loss_ce_batch(ce_logits, labels) + (1 - mixup_ratio) * self.loss_ce_batch(ce_logits, labels.flip(0))).mean()
            else:
                cutmix_loss = torch.tensor(0.).to(pts.device)
            self._dequeue_and_enqueue(k_cls_feature)
            return moco_loss + dvae_loss, cutmix_loss

@MODELS.register_module()
class PointBERT_Classification(nn.Module, ClassificationLossMixin, FeatureExtractionMixin):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.name = config.NAME
        self.trans_dim = config.trans_dim
        self.depth = config.depth 
        self.drop_path_rate = config.drop_path_rate 
        self.cls_dim = config.cls_dim 
        self.num_heads = config.num_heads 

        self.group_size = config.group_size
        self.num_group = config.num_group

        self.freeze_encoder = getattr(config, 'freeze_encoder', False)
        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        self.encoder_dims =  config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        self.reduce_dim = nn.Linear(self.encoder_dims,  self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads
        )

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

    def freeze_encoder_weights(self):
        """Freeze encoder and transformer blocks if specified"""
        if self.freeze_encoder:
            print_log('Freezing encoder weights...', logger='PointBERT_Classification')
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
            print_log('PointBERT Encoder frozen. Only classification head and cls tokens remain trainable.', 
                     logger='PointBERT_Classification')

    def load_model_from_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]

        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print_log('missing_keys', logger = 'PointBERT_Classification')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger = 'PointBERT_Classification'
            )
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger = 'PointBERT_Classification')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger = 'PointBERT_Classification'
            )

        print_log(f'[PointBERT_Classification] Successful Loading the ckpt from {bert_ckpt_path}', logger = 'PointBERT_Classification')

    def forward(self, pts):
        # divide the point clo  ud in the same form. This is important
        neighborhood, center = self.group_divider(pts)
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  #  B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:,0], x[:, 1:].max(1)[0]], dim = -1)

        self._store_features(concat_f)

        ret = self.cls_head_finetune(concat_f)
        return ret