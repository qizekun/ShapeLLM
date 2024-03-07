import math
import timm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ReConV2.utils import misc
from ReConV2.utils.logger import *
from ReConV2.utils.knn import knn_point
from timm.layers import Mlp, DropPath
from typing import Optional, List


class PatchEmbedding(nn.Module):  # Embedding module
    def __init__(self, embed_dim, input_channel=3, large=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_channel = input_channel

        # embed_dim_list = [c * (embed_dim // 512 + 1) for c in [128, 256, 512]]
        #
        # self.first_conv = nn.Sequential(
        #     nn.Conv1d(self.input_channel, embed_dim_list[0], 1),
        #     nn.BatchNorm1d(embed_dim_list[0]),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(embed_dim_list[0], embed_dim_list[1], 1)
        # )
        # self.second_conv = nn.Sequential(
        #     nn.Conv1d(embed_dim_list[2], embed_dim_list[2], 1),
        #     nn.BatchNorm1d(embed_dim_list[2]),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(embed_dim_list[2], self.embed_dim, 1)
        # )

        if large:
            self.first_conv = nn.Sequential(
                nn.Conv1d(self.input_channel, 256, 1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Conv1d(256, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, 1024, 1)
            )
            self.second_conv = nn.Sequential(
                nn.Conv1d(2048, 2048, 1),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Conv1d(2048, embed_dim, 1)
            )
        else:
            self.first_conv = nn.Sequential(
                nn.Conv1d(self.input_channel, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 256, 1)
            )
            self.second_conv = nn.Sequential(
                nn.Conv1d(512, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, embed_dim, 1)
            )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3/6
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, self.input_channel)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.embed_dim)


class PositionEmbeddingCoordsSine(nn.Module):
    """Similar to transformer's position encoding, but generalizes it to
    arbitrary dimensions and continuous coordinates.

    Args:
        n_dim: Number of input dimensions, e.g. 2 for image coordinates.
        d_model: Number of dimensions to encode into
        temperature:
        scale:
    """

    def __init__(self, n_dim: int = 1, d_model: int = 256, temperature=1.0, scale=None):
        super().__init__()

        self.n_dim = n_dim
        self.num_pos_feats = d_model // n_dim // 2 * 2
        self.temperature = temperature
        self.padding = d_model - self.num_pos_feats * self.n_dim

        if scale is None:
            scale = 1.0
        self.scale = scale * 2 * math.pi

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: Point positions (*, d_in)

        Returns:
            pos_emb (*, d_out)
        """
        assert xyz.shape[-1] == self.n_dim

        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t,
                                                   2, rounding_mode='trunc') / self.num_pos_feats)

        xyz = xyz * self.scale
        pos_divided = xyz.unsqueeze(-1) / dim_t
        pos_sin = pos_divided[..., 0::2].sin()
        pos_cos = pos_divided[..., 1::2].cos()
        pos_emb = torch.stack([pos_sin, pos_cos], dim=-1).reshape(*xyz.shape[:-1], -1)

        # Pad unused dimensions with zeros
        pos_emb = F.pad(pos_emb, (0, self.padding))
        return pos_emb


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, pts):
        '''
            input: B N 3/6
            ---------------------------
            output: B G M 3/6
            center : B G 3
        '''
        xyz = pts[:, :, :3]
        c = pts.shape[2]
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        xyz = xyz.float()
        center = misc.fps(xyz.contiguous(), self.num_group)  # B G 3
        # knn to get the neighborhood
        idx = knn_point(self.group_size, xyz, center)
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = pts.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, c).contiguous()
        # normalize
        neighborhood[:, :, :, :3] = neighborhood[:, :, :, :3] - center.unsqueeze(2)
        return neighborhood, center


class ZGroup(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def simplied_morton_sorting(self, xyz, center):
        """
        Simplifying the Morton code sorting to iterate and set the nearest patch to the last patch as the next patch, we found this to be more efficient.
        """
        batch_size, num_points, _ = xyz.shape
        distances_batch = torch.cdist(center, center)
        distances_batch[:, torch.eye(self.num_group).bool()] = float("inf")
        idx_base = torch.arange(
            0, batch_size, device=xyz.device) * self.num_group
        sorted_indices_list = [idx_base]
        distances_batch = distances_batch.view(batch_size, self.num_group, self.num_group).transpose(
            1, 2).contiguous().view(batch_size * self.num_group, self.num_group)
        distances_batch[idx_base] = float("inf")
        distances_batch = distances_batch.view(
            batch_size, self.num_group, self.num_group).transpose(1, 2).contiguous()
        for i in range(self.num_group - 1):
            distances_batch = distances_batch.view(
                batch_size * self.num_group, self.num_group)
            distances_to_last_batch = distances_batch[sorted_indices_list[-1]]
            closest_point_idx = torch.argmin(distances_to_last_batch, dim=-1)
            closest_point_idx = closest_point_idx + idx_base
            sorted_indices_list.append(closest_point_idx)
            distances_batch = distances_batch.view(batch_size, self.num_group, self.num_group).transpose(
                1, 2).contiguous().view(batch_size * self.num_group, self.num_group)
            distances_batch[closest_point_idx] = float("inf")
            distances_batch = distances_batch.view(
                batch_size, self.num_group, self.num_group).transpose(1, 2).contiguous()
        sorted_indices = torch.stack(sorted_indices_list, dim=-1)
        sorted_indices = sorted_indices.view(-1)
        return sorted_indices

    def forward(self, pts):
        """
            input: B N 3/6
            ---------------------------
            output: B G M 3/6
            center : B G 3
        """
        xyz = pts[:, :, :3]
        c = pts.shape[2]
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        xyz = xyz.float()
        center = misc.fps(xyz.contiguous(), self.num_group)  # B G 3
        # knn to get the neighborhood
        idx = knn_point(self.group_size, xyz, center)
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = pts.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, c).contiguous()
        # normalize
        neighborhood[:, :, :, :3] = neighborhood[:, :, :, :3] - center.unsqueeze(2)

        # can utilize morton_sorting by choosing morton_sorting function
        sorted_indices = self.simplied_morton_sorting(xyz, center)

        neighborhood = neighborhood.view(
            batch_size * self.num_group, self.group_size, c)[sorted_indices, :, :]
        neighborhood = neighborhood.view(
            batch_size, self.num_group, self.group_size, c).contiguous()
        center = center.view(
            batch_size * self.num_group, 3)[sorted_indices, :]
        center = center.view(
            batch_size, self.num_group, 3).contiguous()

        return neighborhood, center


# Transformers
class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if mask is not None:
            attn = attn.masked_fill(mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = y.shape
        kv = self.kv(y).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        B, N, C = x.shape
        q = self.q(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if mask is not None:
            attn = attn.masked_fill(mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, attn_mask=None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class CrossBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            stop_grad: bool = False
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.stop_grad = stop_grad

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.stop_grad:
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), self.norm1(y.detach()))))
        else:
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), self.norm1(y))))

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class ReConBlocks(nn.Module):
    def __init__(
            self,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            proj_drop: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: List = [],
            norm_layer: nn.Module = nn.LayerNorm,
            act_layer: nn.Module = nn.GELU,
            stop_grad: bool = False,
            pretrained_model_name: str = 'vit_base_patch32_clip_224.openai',
            every_layer_add_pos: bool = True,
    ):
        super().__init__()

        self.depth = depth
        self.stop_grad = stop_grad
        self.pretrained_model_name = pretrained_model_name
        self.every_layer_add_pos = every_layer_add_pos
        if 'dino' in self.pretrained_model_name:
            init_values = 1e-5
        if 'giant' in self.pretrained_model_name:
            mlp_ratio = 48 / 11
        self.local_blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)])

        self.global_blocks = nn.Sequential(*[
            CrossBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                stop_grad=stop_grad
            )
            for i in range(depth)])

    def load_pretrained_timm_weights(self):
        model = timm.create_model(self.pretrained_model_name, pretrained=True)
        state_dict = model.blocks.state_dict()
        self.local_blocks.load_state_dict(state_dict, strict=True)

        cross_state_dict = {}
        for k, v in state_dict.items():
            if 'qkv' in k:
                cross_state_dict[k.replace('qkv', 'q')] = v[:int(v.shape[0] / 3)]
                cross_state_dict[k.replace('qkv', 'kv')] = v[int(v.shape[0] / 3):]
            else:
                cross_state_dict[k] = v
        self.global_blocks.load_state_dict(cross_state_dict, strict=True)

    def forward(self, x, pos, attn_mask=None, query=None):
        if self.every_layer_add_pos:
            for i in range(self.depth):
                x = self.local_blocks[i](x + pos, attn_mask)
                if query is not None:
                    query = self.global_blocks[i](query, x)
        else:
            x = x + pos
            for i in range(self.depth):
                x = self.local_blocks[i](x, attn_mask)
                if query is not None:
                    query = self.global_blocks[i](query, x)
        return x, query


class GPTExtractor(nn.Module):
    def __init__(
            self,
            embed_dim: int = 768,
            num_heads: int = 12,
            depth: int = 12,
            group_size: int = 32,
            drop_path_rate: float = 0.0,
            stop_grad: bool = False,
            pretrained_model_name: str = 'vit_base_patch32_clip_224.openai',
    ):
        super(GPTExtractor, self).__init__()

        self.embed_dim = embed_dim
        self.group_size = group_size

        # start of sequence token
        self.sos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.sos_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.sos)
        nn.init.normal_(self.sos_pos)

        drop_path_rate = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = ReConBlocks(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            drop_path_rate=drop_path_rate,
            stop_grad=stop_grad,
            pretrained_model_name=pretrained_model_name,
        )

        self.ln_f1 = nn.LayerNorm(embed_dim)
        self.ln_f2 = nn.LayerNorm(embed_dim)

    def forward(self, x, pos, attn_mask, query):
        """
        Expect input as shape [sequence len, batch]
        """

        batch, length, _ = x.shape

        # prepend sos token
        sos = self.sos.expand(batch, -1, -1)
        sos_pos = self.sos_pos.expand(batch, -1, -1)

        x = torch.cat([sos, x[:, :-1]], dim=1)
        pos = torch.cat([sos_pos, pos[:, :-1]], dim=1)

        # transformer
        x, query = self.blocks(x, pos, attn_mask, query)

        encoded_points = self.ln_f1(x)
        query = self.ln_f2(query)

        return encoded_points, query


class GPTGenerator(nn.Module):
    def __init__(
            self,
            embed_dim: int = 768,
            num_heads: int = 12,
            depth: int = 4,
            group_size: int = 32,
            drop_path_rate: float = 0.0,
            input_channel: int = 3,
    ):
        super(GPTGenerator, self).__init__()

        self.embed_dim = embed_dim
        self.input_channel = input_channel

        drop_path_rate = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, drop_path=drop_path_rate[i]) for i in range(depth)])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.increase_dim = nn.Sequential(
            nn.Conv1d(embed_dim, input_channel * group_size, 1)
        )

    def forward(self, x, pos, attn_mask):
        batch, length, C = x.shape

        # transformer
        for block in self.blocks:
            x = block(x + pos, attn_mask)

        x = self.ln_f(x)

        rebuild_points = self.increase_dim(x.transpose(1, 2)).transpose(
            1, 2).reshape(batch * length, -1, self.input_channel)

        return rebuild_points


class MAEExtractor(nn.Module):
    def __init__(
            self,
            embed_dim: int = 768,
            num_heads: int = 12,
            depth: int = 12,
            group_size: int = 32,
            drop_path_rate: float = 0.0,
            stop_grad: bool = False,
            pretrained_model_name: str = 'vit_base_patch32_clip_224.openai',
    ):
        super(MAEExtractor, self).__init__()

        self.embed_dim = embed_dim
        self.group_size = group_size

        drop_path_rate = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = ReConBlocks(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            drop_path_rate=drop_path_rate,
            stop_grad=stop_grad,
            pretrained_model_name=pretrained_model_name,
        )

        self.ln_f1 = nn.LayerNorm(embed_dim)
        self.ln_f2 = nn.LayerNorm(embed_dim)

    def forward(self, x, pos, mask=None, query=None):
        """
        Expect input as shape [sequence len, batch]
        """

        batch, length, C = x.shape
        if mask is not None:
            x_vis = x[~mask].reshape(batch, -1, C)
            pos_vis = pos[~mask].reshape(batch, -1, C)
        else:
            x_vis = x
            pos_vis = pos

        # transformer
        x_vis, query = self.blocks(x_vis, pos_vis, None, query)

        encoded_points = self.ln_f1(x_vis)
        query = self.ln_f2(query)

        return encoded_points, query


class MAEGenerator(nn.Module):
    def __init__(
            self,
            embed_dim: int = 768,
            num_heads: int = 12,
            depth: int = 4,
            group_size: int = 32,
            drop_path_rate: float = 0.0,
            input_channel: int = 3,
    ):
        super(MAEGenerator, self).__init__()

        self.embed_dim = embed_dim
        self.input_channel = input_channel
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        drop_path_rate = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, drop_path=drop_path_rate[i]) for i in range(depth)])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.increase_dim = nn.Sequential(
            nn.Conv1d(embed_dim, input_channel * group_size, 1)
        )

    def forward(self, x_vis, pos, mask):
        batch, length, C = x_vis.shape

        pos_vis = pos[~mask].reshape(batch, -1, C)
        pos_mask = pos[mask].reshape(batch, -1, C)
        pos_full = torch.cat([pos_vis, pos_mask], dim=1)
        mask_token = self.mask_token.expand(batch, pos_mask.shape[1], -1)
        x = torch.cat([x_vis, mask_token], dim=1)

        # transformer
        for block in self.blocks:
            x = block(x + pos_full)

        x = self.ln_f(x[:, -pos_mask.shape[1]:])

        rebuild_points = self.increase_dim(x.transpose(1, 2)).transpose(
            1, 2).reshape(batch * pos_mask.shape[1], -1, self.input_channel)

        return rebuild_points
