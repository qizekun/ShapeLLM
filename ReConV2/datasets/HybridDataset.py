import os
import torch
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from ReConV2.utils.logger import *
from ReConV2.utils.transforms import get_transforms

import io
import refile
from joblib import Parallel, delayed


@DATASETS.register_module()
class Hybrid(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.img_path = config.IMG_PATH
        self.ratio = config.ratio
        self.subset = config.subset
        self.using_saved_features = config.using_saved_features

        self.img_views = config.img_views
        assert self.img_views in [1, 10]

        self.data_list_file = os.path.join(self.data_root, f'pretrain/{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'pretrain/test.txt')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger='Hybrid')
        self.whole = config.get('whole')

        lines = open(self.data_list_file).readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            print_log(f'[DATASET] Open file {test_data_list_file}', logger='Hybrid')
            lines = test_lines + lines
        index_list = [x.strip() for x in lines]
        self.index_list = index_list[: int(len(index_list) * self.ratio)]
        self.sample_points_num = config.npoints

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger='Hybrid')
        print_log(f'[DATASET] load ratio is {self.ratio}', logger='Hybrid')
        print_log(f'[DATASET] {len(self.index_list)} instances were loaded', logger='Hybrid')

    def pc_norm(self, pc):
        # normalize pc to [-1, 1]
        pc = pc - np.mean(pc, axis=0)
        if np.max(np.linalg.norm(pc, axis=1)) < 1e-6:
            pc = np.zeros_like(pc)
        else:
            pc = pc / np.max(np.linalg.norm(pc, axis=1))
        return pc

    def random_sample(self, pc, num):
        permutation = np.arange(pc.shape[0])
        np.random.shuffle(permutation)
        pc = pc[permutation[:num]]
        return pc

    def read_single_img(self, view_index, index):

        # img_index = f'{index.replace("/", "-")[:-4]}-{view_index}.jpg'
        # img_path = os.path.join(self.img_path, img_index)
        # img = Image.open(img_path).convert('RGB')
        # img = get_transforms()['train'](img)

        img_path = f's3://qzk/HybridDatasets/depth/{index}-{view_index}.jpg'
        while True:
            try:
                with refile.smart_open(img_path, "rb") as f:
                    bytes_data = f.read()
                break
            except:
                print('img_path', img_path)
                import time
                time.sleep(1)
        img = Image.open(io.BytesIO(bytes_data), "r").convert('RGB')
        img = get_transforms()['train'](img)
        return img

    def parallel_load_img(self, index, max_workers):
        img_list = Parallel(n_jobs=max_workers)(delayed(self.read_single_img)(view_index, index) for view_index in range(max_workers))
        return img_list

    def __getitem__(self, idx):
        index = self.index_list[idx]

        pc_path = os.path.join(self.data_root, index)
        pc = IO.get(pc_path).astype(np.float32)
        pc = self.random_sample(pc, self.sample_points_num)
        pc = self.pc_norm(pc)
        pc = torch.from_numpy(pc).float()

        index = index.replace("/", "-")[:-4]

        if self.img_views == 1:
            view_index = random.randint(0, 9)
            img = self.read_single_img(view_index, index)
        else:
            if self.using_saved_features:
                img_list = f's3://qzk/features/hybrid_clip_B32/{index}.pt'
                while True:
                    try:
                        with refile.smart_open(img_list, "rb") as f:
                            bytes_data = f.read()
                        break
                    except:
                        import time
                        print('img_list', img_list)
                        time.sleep(1)
                img = torch.load(io.BytesIO(bytes_data))
            else:
                img_list = self.parallel_load_img(index, max_workers=10)
                img = torch.stack(img_list, dim=0)

        text = ""

        return pc, img, text, index

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

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

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
        pc = self.pc_norm(pc)
        pc = torch.from_numpy(pc).float()

        return 'HyBrid', 'sample', (pc, label)

    def __len__(self):
        return len(self.index_list)
