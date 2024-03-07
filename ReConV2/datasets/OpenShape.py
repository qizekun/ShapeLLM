import os
import json
import torch
import random
import numpy as np
import torch.utils.data as data
from .build import DATASETS
from ReConV2.utils.logger import *
from ReConV2.utils.data import normalize_pc, augment_pc
from llava.mm_utils import occlusion, rotation


@DATASETS.register_module()
class OpenShape(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.ratio = config.ratio
        self.use_color = config.with_color
        self.num_points = config.npoints
        self.rgb_random_drop_prob = config.rgb_random_drop_prob

        if os.path.exists("/mnt/host0/openshape"):
            self.data_root = "/mnt/host0/openshape"

        self.img_queries = config.img_queries
        assert self.img_queries in [1, 13]
        self.text_queries = config.text_queries

        metadata_path = os.path.join(self.data_root, 'meta_data/split/train_all.json')
        metadata = json.load(open(metadata_path, 'r'))
        self.metadata = metadata[: int(len(metadata) * self.ratio)]

        gpt4_filtering_path = os.path.join(self.data_root, 'meta_data/gpt4_filtering.json')
        self.gpt4_filtering = json.load(open(gpt4_filtering_path, "r"))
        self.text_embed_version = "prompt_avg"

        self.occlusion = config.occlusion

        print_log(f'[DATASET] Using occlusion: {self.occlusion}', logger='OpenShape')
        print_log(f'[DATASET] Sample out {self.num_points} points', logger='OpenShape')
        print_log(f'[DATASET] Open file {metadata_path}', logger='OpenShape')
        print_log(f'[DATASET] Load ratio is {self.ratio}', logger='OpenShape')
        print_log(f'[DATASET] {len(self.metadata)} instances were loaded', logger='OpenShape')

    def get_objaverse(self, meta):
        dataset = meta['dataset']
        group = meta['group']
        uid = meta["id"]
        data_path = os.path.join(self.data_root, 'objaverse-processed/merged_for_training_final/' +
                                 dataset + '/' + group + '/' + uid + '.npy')
        data = np.load(data_path, allow_pickle=True).item()
        xyz = data['xyz']
        rgb = data['rgb']
        if self.occlusion and np.random.rand() < 0.5:
            xyz = normalize_pc(xyz)
            pts = np.concatenate([xyz, rgb], axis=1)
            pts = occlusion(pts, self.num_points, 90, False)
            xyz = pts[:, :3]
            rgb = pts[:, 3:]

        n = xyz.shape[0]
        if n > self.num_points:
            idx = random.sample(range(n), self.num_points)
            xyz = xyz[idx]
            rgb = rgb[idx]

        xyz = normalize_pc(xyz)
        xyz = augment_pc(xyz)

        if self.use_color:
            if np.random.rand() < self.rgb_random_drop_prob:
                rgb = np.ones_like(rgb) * 0.4
            pts = torch.from_numpy(np.concatenate([xyz, rgb], axis=1)).type(torch.float32)
        else:
            pts = torch.from_numpy(xyz).type(torch.float32)

        pts[:, :3] = rotation(pts[:, :3], [0, 0, -90])

        text_feat = []
        if self.gpt4_filtering[uid]["flag"] != "N":
            text_feat.append(data["text_feat"][0][self.text_embed_version][0])

        if np.random.rand() < 0.5:
            if len(data["blip_caption"]) > 0:
                text_feat.append(data["blip_caption_feat"][self.text_embed_version][0])
        else:
            if len(data["msft_caption"]) > 0:
                text_feat.append(data["msft_caption_feat"][self.text_embed_version][0])

        if len(data["retrieval_text"]) > 0:
            idx = np.random.randint(len(data["retrieval_text"]))
            text_feat.append(data["retrieval_text_feat"][idx]["original"][0])

        thumbnail_feat = np.expand_dims(data['thumbnail_feat'], axis=0)
        img_feat = np.concatenate([thumbnail_feat, data['image_feat']], axis=0)
        img_feat = random.choice(img_feat)
        img_feat = torch.from_numpy(img_feat).type(torch.float32)
        text_feat = torch.tensor(text_feat).type(torch.float32)

        if self.img_queries == 1:
            img_feat = random.choice(img_feat)
            img_feat = img_feat.unsqueeze(0)

        if self.text_queries == 1:
            text_feat = random.choice(text_feat)
            text_feat = text_feat.unsqueeze(0)

        return pts, img_feat, text_feat, uid

    def get_others(self, meta):
        dataset = meta['dataset']
        group = meta['group']
        uid = meta["id"]
        data_path = os.path.join(self.data_root, 'objaverse-processed/merged_for_training_final/' +
                                 dataset + '/' + group + '/' + uid + '.npy')
        data = np.load(data_path, allow_pickle=True).item()
        xyz = data['xyz']
        rgb = data['rgb']
        if self.occlusion and np.random.rand() < 0.5:
            xyz = normalize_pc(xyz)
            pts = np.concatenate([xyz, rgb], axis=1)
            pts = occlusion(pts, self.num_points, 90, False)
            xyz = pts[:, :3]
            rgb = pts[:, 3:]

        n = xyz.shape[0]
        if n > self.num_points:
            idx = random.sample(range(n), self.num_points)
            xyz = xyz[idx]
            rgb = rgb[idx]

        xyz = normalize_pc(xyz)
        xyz = augment_pc(xyz)

        if self.use_color:
            if np.random.rand() < self.rgb_random_drop_prob:
                rgb = np.ones_like(rgb) * 0.4
            pts = torch.from_numpy(np.concatenate([xyz, rgb], axis=1)).type(torch.float32)
        else:
            pts = torch.from_numpy(xyz).type(torch.float32)

        pts[:, :3] = rotation(pts[:, :3], [0, 0, -90])

        text_feat = []
        idx = np.random.randint(len(data["text"]))
        text_feat.append(data["text_feat"][idx][self.text_embed_version][0])

        if np.random.rand() < 0.5:
            if len(data["blip_caption"]) > 0:
                text_feat.append(data["blip_caption_feat"][self.text_embed_version][0])
        else:
            if len(data["msft_caption"]) > 0:
                text_feat.append(data["msft_caption_feat"][self.text_embed_version][0])

        if len(data["retrieval_text"]) > 0:
            idx = np.random.randint(len(data["retrieval_text"]))
            text_feat.append(
                data["retrieval_text_feat"][idx]["original"][0])  # no prompt engineering for retrieval text

        text_feat = torch.tensor(text_feat).type(torch.float32)
        img_feat = torch.from_numpy(data['image_feat']).type(torch.float32)

        if self.img_queries == 1:
            img_feat = random.choice(img_feat)
            img_feat = img_feat.unsqueeze(0)

        if self.text_queries == 1:
            text_feat = random.choice(text_feat)
            text_feat = text_feat.unsqueeze(0)

        return pts, img_feat, text_feat, uid

    def __getitem__(self, idx):
        meta = self.metadata[idx]
        if meta["dataset"] == "Objaverse":
            pts, img_feat, text_feat, uid = self.get_objaverse(meta)
        else:
            pts, img_feat, text_feat, uid = self.get_others(meta)

        img = torch.zeros((self.img_queries, img_feat.shape[-1]))
        img[:img_feat.shape[0], :] = img_feat
        text = torch.zeros((self.text_queries, text_feat.shape[-1]))
        text[:text_feat.shape[0], :] = text_feat

        return pts, img, text, uid

    def __len__(self):
        return len(self.metadata)
