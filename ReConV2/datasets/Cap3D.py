import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch.utils.data as data
from .build import DATASETS
from ReConV2.utils.logger import *
from ReConV2.utils.transforms import get_transforms

import io
import refile
from joblib import Parallel, delayed


@DATASETS.register_module()
class Cap3D(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.img_path = config.IMG_PATH
        self.text_path = config.TEXT_PATH
        self.ratio = config.ratio
        self.use_color = config.USE_COLOR
        self.using_saved_features = config.using_saved_features

        self.img_views = config.img_views
        assert self.img_views in [1, 8]

        self.sample_points_num = config.npoints
        self.info_list = pd.read_csv(self.text_path, header=None)
        self.info_list = self.info_list.iloc[: int(len(self.info_list) * self.ratio)]

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger='Cap3D')
        print_log(f'[DATASET] Open file {self.text_path}', logger='Cap3D')
        print_log(f'[DATASET] load ratio is {self.ratio}', logger='Cap3D')
        print_log(f'[DATASET] {len(self.info_list)} instances were loaded', logger='Cap3D')

        self.permutation = np.arange(self.sample_points_num)

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

    def read_single_img(self, view_index, index):

        img_index = f'Cap3D_imgs_view{view_index}/{index}_{view_index}.jpeg'
        img_path = os.path.join(self.img_path, img_index)
        img = Image.open(img_path).convert('RGB')
        img = get_transforms()['train'](img)

        # img_path = f's3://qzk/Cap3D/RenderedImage/Cap3D_imgs_view{view_index}/{index}_{view_index}.jpeg'
        # while True:
        #     try:
        #         with refile.smart_open(img_path, "rb") as f:
        #             bytes_data = f.read()
        #         break
        #     except:
        #         print('img_path', img_path)
        #         import time
        #         time.sleep(1)
        # img = Image.open(io.BytesIO(bytes_data), "r").convert('RGB')
        # img = get_transforms()['train'](img)
        return img

    def parallel_load_img(self, index, max_workers):
        img_list = Parallel(n_jobs=max_workers)(delayed(self.read_single_img)(view_index, index) for view_index in range(max_workers))
        return img_list

    def __getitem__(self, idx):

        index, text = self.info_list.iloc[idx]

        # pc_path = os.path.join(self.pc_path, index + '.pt')
        pc_path = f'/mnt/host0/Cap3D/Cap3D_pcs_pt/{index}.pt'
        pc = torch.load(pc_path).permute(1, 0).numpy()
        if not self.use_color:
            pc = pc[:, :3]
        else:
            pc[:, 3:] = pc[:, 3:] / 255.0
        pc = self.random_sample(pc, self.sample_points_num)
        pc[:, :3] = self.pc_norm(pc[:, :3])
        pc = torch.from_numpy(pc).float()

        # pc_path = f's3://qzk/Cap3D/Cap3D_pcs_pt/{index}.pt'
        # while True:
        #     try:
        #         with refile.smart_open(pc_path, "rb") as f:
        #             bytes_data = f.read()
        #         break
        #     except:
        #         import time
        #         print('pc_path', pc_path)
        #         time.sleep(1)
        # pc = torch.load(io.BytesIO(bytes_data)).permute(1, 0).numpy()
        # if not self.use_color:
        #     pc = pc[:, :3]
        # else:
        #     pc[:, 3:] = pc[:, 3:] / 255.0
        # pc = self.random_sample(pc, self.sample_points_num)
        # pc[:, :3] = self.pc_norm(pc[:, :3])
        # pc = torch.from_numpy(pc).float()

        if self.img_views == 1:
            view_index = random.randint(0, 7)
            img = self.read_single_img(view_index, index)
        else:
            if self.using_saved_features:
                img_list = f'/mnt/host0/features/cap3d/{index}.pt'
                img = torch.load(img_list)
                # img_list = f's3://qzk/features/cap3d_clip_B32/{index}.pt'
                # while True:
                #     try:
                #         with refile.smart_open(img_list, "rb") as f:
                #             bytes_data = f.read()
                #         break
                #     except:
                #         import time
                #         print('img_list', img_list)
                #         time.sleep(1)
                # img = torch.load(io.BytesIO(bytes_data))
            else:
                img_list = self.parallel_load_img(index, max_workers=8)
                img = torch.stack(img_list, dim=0)

        return pc, img, text, index

    def __len__(self):
        return len(self.info_list)
