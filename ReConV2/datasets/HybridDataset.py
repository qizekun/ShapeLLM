import os
import torch
import random
import numpy as np
from PIL import Image
import torch.utils.data as data

from .io import IO
from .build import DATASETS
from ReConV2.utils.logger import *
from ReConV2.utils.data import normalize_pc, augment_pc
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, PILToTensor


class Hybrid_points(data.Dataset):
    def __init__(self, data_root, subset, sample_points_num):
        self.data_root = data_root
        self.subset = subset
        self.sample_points_num = sample_points_num

        self.data_list_file = os.path.join(self.data_root, f'pretrain/{self.subset}.txt')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger='Hybrid')

        lines = open(self.data_list_file).readlines()
        self.index_list = [x.strip() for x in lines]

        print_log(f'[DATASET] {len(self.index_list)} instances were loaded', logger='Hybrid')

    def random_sample(self, pc, num):
        permutation = np.arange(pc.shape[0])
        np.random.shuffle(permutation)
        pc = pc[permutation[:num]]
        return pc

    def __getitem__(self, idx):
        path = self.index_list[idx]

        pc_path = os.path.join(self.data_root, path)
        pc = IO.get(pc_path).astype(np.float32)
        pc = self.random_sample(pc, self.sample_points_num)
        pc = normalize_pc(pc)
        pc = augment_pc(pc)
        pc = torch.from_numpy(pc).float()

        return pc, path

    def __len__(self):
        return len(self.index_list)


class Hybrid_depth(data.Dataset):
    def __init__(self, data_root, subset, img_path):
        self.data_root = data_root
        self.subset = subset
        self.img_path = img_path
        self.data_list_file = os.path.join(self.data_root, f'pretrain/{self.subset}.txt')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger='Hybrid')

        lines = open(self.data_list_file).readlines()
        self.index_list = [x.strip() for x in lines]

        print_log(f'[DATASET] {len(self.index_list)} instances were loaded', logger='Hybrid')

    def __getitem__(self, idx):
        path = self.index_list[idx]
        id = path.replace("/", "-")[:-4]
        img_path = [os.path.join(self.img_path, id + f'-{i}.png') for i in range(10)]
        transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = [transform(Image.open(x)) for x in img_path]
        img = torch.stack(img, dim=0)

        return img, id

    def __len__(self):
        return len(self.index_list)


@DATASETS.register_module()
class Hybrid(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.img_feature_path = config.IMG_FEATURE_PATH
        self.ratio = config.ratio
        self.subset = config.subset

        self.img_queries = config.img_queries
        assert self.img_queries in [1, 10]

        self.data_list_file = os.path.join(self.data_root, f'pretrain/{self.subset}.txt')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger='Hybrid')

        lines = open(self.data_list_file).readlines()
        index_list = [x.strip() for x in lines]
        self.index_list = index_list[: int(len(index_list) * self.ratio)]
        self.sample_points_num = config.npoints

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger='Hybrid')
        print_log(f'[DATASET] load ratio is {self.ratio}', logger='Hybrid')
        print_log(f'[DATASET] {len(self.index_list)} instances were loaded', logger='Hybrid')

    def random_sample(self, pc, num):
        permutation = np.arange(pc.shape[0])
        np.random.shuffle(permutation)
        pc = pc[permutation[:num]]
        return pc

    def __getitem__(self, idx):
        path = self.index_list[idx]

        pc_path = os.path.join(self.data_root, path)
        pc = IO.get(pc_path).astype(np.float32)
        pc = self.random_sample(pc, self.sample_points_num)
        pc = normalize_pc(pc)
        pc = augment_pc(pc)
        pc = torch.from_numpy(pc).float()

        id = path.replace("/", "-")[:-4]
        img_path = id + '.pt'
        img_feat = torch.load(os.path.join(self.img_feature_path, img_path), map_location='cpu').detach().float()

        if self.img_queries == 1:
            img_feat = random.choice(img_feat)
            img_feat = img_feat.unsqueeze(0)

        return pc, img_feat, img_feat.mean(dim=0).unsqueeze(0), id

    def __len__(self):
        return len(self.index_list)


@DATASETS.register_module()
class HybridLabeled(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.ratio = config.ratio
        self.subset = config.subset

        self.data_list_file = os.path.join(self.data_root, f'post_pretrain/{self.subset}.txt')
        self.label_list_file = os.path.join(self.data_root, f'post_pretrain/{self.subset}_num.txt')
        test_data_list_file = os.path.join(self.data_root, 'post_pretrain/test.txt')
        test_label_list_file = os.path.join(self.data_root, 'post_pretrain/test_num.txt')

        print_log(f'[DATASET] Open file {self.data_list_file}', logger='Hybrid')
        self.whole = config.get('whole')

        lines = open(self.data_list_file).readlines()
        labels = open(self.label_list_file).readlines()
        if self.whole:
            test_lines = open(test_data_list_file).readlines()
            print_log(f'[DATASET] Open file {test_data_list_file}', logger='Hybrid')
            lines = test_lines + lines
            test_labels = open(test_label_list_file).readlines()
            print_log(f'[DATASET] Open file {test_label_list_file}', logger='Hybrid')
            labels = test_labels + labels
        assert len(lines) == len(labels)

        index_list = []
        for i in range(len(lines)):
            index_list.append(
                {
                    'index': lines[i].strip(),
                    'label': labels[i].strip()
                }
            )
        self.index_list = index_list[: int(len(index_list) * self.ratio)]
        self.sample_points_num = config.npoints

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger='Hybrid')
        print_log(f'[DATASET] load ratio is {self.ratio}', logger='Hybrid')
        print_log(f'[DATASET] {len(self.index_list)} instances were loaded', logger='Hybrid')

    def random_sample(self, pc, num):
        permutation = np.arange(pc.shape[0])
        np.random.shuffle(permutation)
        pc = pc[permutation[:num]]
        return pc

    def __getitem__(self, idx):
        index = self.index_list[idx]['index']
        label = self.index_list[idx]['label']

        pc_path = os.path.join(self.data_root, index)
        pc = IO.get(pc_path).astype(np.float32)
        pc = self.random_sample(pc, self.sample_points_num)
        pc = normalize_pc(pc)
        pc = torch.from_numpy(pc).float()

        return 'HyBrid', 'sample', (pc, label)

    def __len__(self):
        return len(self.index_list)
