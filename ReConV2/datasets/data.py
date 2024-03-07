import copy
import json
import torch
import random
import numpy as np

from ReConV2.utils.data import normalize_pc
from torch.utils.data import Dataset, DataLoader


class ModelNet40Test(Dataset):
    def __init__(self, config):
        self.split = json.load(open(config.modelnet40.test_split, "r"))
        self.pcs = np.load(config.modelnet40.test_pc, allow_pickle=True)
        self.num_points = config.npoints
        self.use_color = config.model.with_color
        clip_feat = np.load(config.modelnet40.clip_feat_path, allow_pickle=True).item()
        self.categories = list(clip_feat.keys())
        self.clip_cat_feat = []
        self.category2idx = {}
        for i, category in enumerate(self.categories):
            self.category2idx[category] = i
            self.clip_cat_feat.append(clip_feat[category]["open_clip_text_feat"])
        self.clip_cat_feat = np.concatenate(self.clip_cat_feat, axis=0)

    def __getitem__(self, index: int):
        pc = copy.deepcopy(self.pcs[index])
        n = pc['xyz'].shape[0]
        idx = random.sample(range(n), self.num_points)
        xyz = pc['xyz'][idx]
        rgb = pc['rgb'][idx]
        rgb = rgb / 255.0

        xyz = normalize_pc(xyz)

        if self.use_color:
            pcs = np.concatenate([xyz, rgb], axis=1)
        else:
            pcs = xyz

        assert not np.isnan(xyz).any()

        return {
            "pcs": torch.from_numpy(pcs).type(torch.float32),
            "name": self.split[index]["name"],
            "category": self.category2idx[self.split[index]["category"]],
        }

    def __len__(self):
        return len(self.split)


def make_modelnet40test(config):
    dataset = ModelNet40Test(config)
    data_loader = DataLoader(
        dataset,
        num_workers=config.modelnet40.num_workers,
        batch_size=config.modelnet40.batch_size,
        pin_memory=True,
        shuffle=False
    )
    return data_loader


class ObjaverseLVIS(Dataset):
    def __init__(self, config):
        self.split = json.load(open(config.objaverse_lvis.split, "r"))
        self.num_points = config.npoints
        self.use_color = config.model.with_color
        self.categories = sorted(np.unique([data['category'] for data in self.split]))
        self.category2idx = {self.categories[i]: i for i in range(len(self.categories))}
        self.clip_cat_feat = np.load(config.objaverse_lvis.clip_feat_path, allow_pickle=True)

    def __getitem__(self, index: int):
        data_path = self.split[index]['data_path'].replace('/mnt/data/', 'ReConV2/data/openshape/')
        data = np.load(data_path, allow_pickle=True).item()
        n = data['xyz'].shape[0]
        idx = random.sample(range(n), self.num_points)
        xyz = data['xyz'][idx]
        rgb = data['rgb'][idx]

        xyz = normalize_pc(xyz)
        if self.use_color:
            pcs = np.concatenate([xyz, rgb], axis=1)
        else:
            pcs = xyz

        return {
            "pcs": torch.from_numpy(pcs).type(torch.float32),
            "group": self.split[index]['group'],
            "name": self.split[index]['uid'],
            "category": self.category2idx[self.split[index]["category"]],
        }

    def __len__(self):
        return len(self.split)


def make_objaverse_lvis(config):
    return DataLoader(
        ObjaverseLVIS(config),
        num_workers=config.objaverse_lvis.num_workers,
        batch_size=config.objaverse_lvis.batch_size,
        pin_memory=True,
        shuffle=False
    )


class ScanObjectNNTest(Dataset):
    def __init__(self, config):
        self.data = np.load(config.scanobjectnn.data_path, allow_pickle=True).item()
        self.num_points = config.npoints
        self.use_color = config.model.with_color
        clip_feat = np.load(config.scanobjectnn.clip_feat_path, allow_pickle=True).item()
        self.categories = ["bag", "bin", "box", "cabinet", "chair", "desk", "display", "door", "shelf", "table", "bed",
                           "pillow", "sink", "sofa", "toilet"]
        self.clip_cat_feat = []
        self.category2idx = {}
        for i, category in enumerate(self.categories):
            self.category2idx[category] = i
            self.clip_cat_feat.append(clip_feat[category]["open_clip_text_feat"])
        self.clip_cat_feat = np.concatenate(self.clip_cat_feat, axis=0)

    def __getitem__(self, index: int):
        xyz = copy.deepcopy(self.data['xyz'][index])
        if 'rgb' not in self.data:
            rgb = np.ones_like(xyz) * 0.4
        else:
            rgb = self.data['rgb'][index]
        label = self.data['label'][index]
        n = xyz.shape[0]
        if n > self.num_points:
            idx = np.random.choice(n, self.num_points)
            xyz = xyz[idx]
            rgb = rgb[idx]

        xyz = normalize_pc(xyz)
        if self.use_color:
            pcs = np.concatenate([xyz, rgb], axis=1)
        else:
            pcs = xyz

        assert not np.isnan(xyz).any()
        return {
            "pcs": torch.from_numpy(pcs).type(torch.float32),
            "name": str(index),
            "category": label,
        }

    def __len__(self):
        return len(self.data['xyz'])


def make_scanobjectnntest(config):
    dataset = ScanObjectNNTest(config)
    data_loader = DataLoader(
        dataset,
        num_workers=config.scanobjectnn.num_workers,
        batch_size=config.scanobjectnn.batch_size,
        pin_memory=True,
        shuffle=False
    )
    return data_loader
