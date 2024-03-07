import torch
import torch.nn as nn
from ReConV2.models.ReCon import ReCon2
from ReConV2.utils.config import cfg_from_yaml_file


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.cfg_path = vision_tower
        self.vision_tower_path = args.vision_tower_path
        self.config = cfg_from_yaml_file(self.cfg_path)
        self.config.with_color = args.with_color
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        self.vision_tower = ReCon2(self.config.model)
        self.hidden_size = self.vision_tower.embed_dim
        self.global_query_num = self.vision_tower.global_query_num
        self.is_loaded = False

    def load_model(self):
        ckpt = torch.load(self.vision_tower_path, map_location='cpu')
        state_dict = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        self.vision_tower.load_state_dict(state_dict, strict=True)
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    @torch.no_grad()
    def forward(self, pts):

        if type(pts) is list:
            pos_features = []
            local_features = []
            global_features = []
            for pt in pts:
                pos_feature, local_feature, global_feature = self.vision_tower.model.inference(pt.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                pos_features.append(pos_feature.to(pts.dtype))
                local_features.append(local_feature.to(pts.dtype))
                global_features.append(global_feature.to(pts.dtype))
        else:
            pos_features, local_features, global_features = self.vision_tower.model.inference(pts.to(device=self.device, dtype=self.dtype))
            local_features = local_features.to(pts.dtype)
            global_features = global_features.to(pts.dtype)

        return pos_features, local_features, global_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def num_patches(self):
        return self.config.num_group
