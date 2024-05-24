import os
import json
import torch
import random
import numpy as np
from PIL import Image

from .io import IO
import pandas as pd
from ReConV2.utils.logger import *
from .build import DATASETS
import torch.utils.data as data
from ReConV2.utils.transforms import get_transforms

from joblib import Parallel, delayed


@DATASETS.register_module()
class ShapeNet(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.img_path = config.IMG_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.ratio = config.ratio
        self.using_saved_features = config.using_saved_features

        self.img_views = config.img_views
        assert self.img_views in [1, 12]

        self.index_list = {}
        self.text_list = {}
        for index, row in pd.read_json(config.CATEGORY_PATH).iterrows():
            self.index_list["0" + str(row['catalogue'])] = index
            self.text_list["0" + str(row['catalogue'])] = row['describe']

        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')

        self.text_query_file = os.path.join(config.TEXT_PATH)
        assert os.path.exists(self.text_query_file), "text query file not found"
        with open(self.text_query_file, 'r') as f:
            self.text_query = json.load(f)

        self.sample_points_num = config.npoints
        self.whole = config.get('whole')

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger='ShapeNet')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger='ShapeNet')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            print_log(f'[DATASET] Open file {test_data_list_file}', logger='ShapeNet')
            lines = test_lines + lines

        self.file_list = []
        for line in lines[: int(self.ratio * len(lines))]:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })

        print_log(f'[DATASET] load ratio is {self.ratio}', logger='ShapeNet')
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger='ShapeNet')

        self.permutation = np.arange(self.npoints)

    def pc_norm(self, pc):
        # normalize pc to [-1, 1]
        pc = pc - np.mean(pc, axis=0)
        if np.max(np.linalg.norm(pc, axis=1)) < 1e-6:
            pc = np.zeros_like(pc)
        else:
            pc = pc / np.max(np.linalg.norm(pc, axis=1))
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def read_single_img(self, sample, view_index):

        img_index = f"{sample['taxonomy_id']}/{sample['file_path'][9:-4]}-{view_index}.png"
        img = Image.open(os.path.join(self.img_path, img_index)).convert('RGB')
        img = get_transforms()['train'](img)
        return img

    def parallel_load_img(self, sample, max_workers):
        img_list = Parallel(n_jobs=max_workers)(delayed(self.read_single_img)(sample, view_index) for view_index in range(max_workers))
        return img_list

    def __getitem__(self, idx):
        sample = self.file_list[idx]

        pc = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        pc = self.random_sample(pc, self.sample_points_num)
        pc = self.pc_norm(pc)
        pc = torch.from_numpy(pc).float()

        index = sample['file_path'].split('.')[0]

        if self.img_views == 1:
            view_index = random.randint(0, 11)
            img = self.read_single_img(sample, view_index)
        else:
            if self.using_saved_features:
                pass
            else:
                img_list = self.parallel_load_img(sample, max_workers=12)
                img = torch.stack(img_list, dim=0)

        text = self.text_query[index]

        return pc, img, text, index

    def __len__(self):
        return len(self.file_list)