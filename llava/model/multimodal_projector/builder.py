import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, *args, **kwargs):
        return x, y

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class LinearProjector(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj1 = nn.Linear(in_channels, out_channels)
        self.proj2 = nn.Linear(in_channels, out_channels)
        self.proj3 = nn.Linear(in_channels, out_channels)

    def forward(self, x, y, z, *args, **kwargs):
        x = self.proj1(x)
        y = self.proj2(y)
        z = self.proj3(z)

        return torch.cat([x, y, z], dim=1)

    @property
    def config(self):
        return {"mm_projector_type": 'linear'}


class ReConProjector(nn.Module):
    def __init__(self, in_channels, out_channels, mlp_depth, prompt_token_num,
                 with_ape=True, with_local=True, with_global=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp_depth = mlp_depth
        self.hidden_size = [1024 * 2 ** i for i in range(mlp_depth)]
        self.prompt_token_num = prompt_token_num
        self.with_ape = with_ape
        self.with_local = with_local
        self.with_global = with_global

        if prompt_token_num > 0:
            self.prompt1 = nn.Parameter(torch.zeros(1, prompt_token_num, out_channels))
            self.prompt2 = nn.Parameter(torch.zeros(1, prompt_token_num, out_channels))
            self.prompt3 = nn.Parameter(torch.zeros(1, prompt_token_num, out_channels))

        self.proj1 = self.set_proj()
        self.proj2 = self.set_proj()
        self.proj3 = self.set_proj()

    def set_proj(self):
        modules = [nn.Linear(self.in_channels, self.hidden_size[0])]
        for i in range(1, self.mlp_depth):
            modules.append(nn.LayerNorm(self.hidden_size[i - 1]))
            modules.append(nn.GELU())
            modules.append(nn.Linear(self.hidden_size[i - 1], self.hidden_size[i]))
        modules.append(nn.LayerNorm(self.hidden_size[-1]))
        modules.append(nn.GELU())
        modules.append(nn.Linear(self.hidden_size[-1], self.out_channels))
        return nn.Sequential(*modules)

    def forward(self, pos_feat, local_feat, global_feat, *args, **kwargs):
        B = pos_feat.shape[0]
        pos_feat = self.proj1(pos_feat)
        local_feat = self.proj2(local_feat)
        global_feat = self.proj3(global_feat)

        if self.prompt_token_num > 0:
            pos_feat = torch.cat([pos_feat, self.prompt1.expand(B, -1, -1)], dim=1)
            local_feat = torch.cat([local_feat, self.prompt2.expand(B, -1, -1)], dim=1)
            global_feat = torch.cat([global_feat, self.prompt3.expand(B, -1, -1)], dim=1)

        pts_feat = [feat for feat, flag in [(pos_feat, self.with_ape), (local_feat, self.with_local), (global_feat, self.with_global)] if flag]
        pts_feat = torch.cat(pts_feat, dim=1)

        return pts_feat

    @property
    def config(self):
        return {"mm_projector_type": 'mlp'}


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return LinearProjector(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        return ReConProjector(config.mm_hidden_size, config.hidden_size, mlp_depth, config.prompt_token_num,
                              config.with_ape, config.with_local, config.with_global)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
