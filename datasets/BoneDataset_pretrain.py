import math
import numpy as np
import pandas as pd
import scipy.spatial
import torch
import warnings
import os
from .build import DATASETS

from plyfile import PlyData
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight

from .BoneDataset import compute_cv_fold_split

def read_ply_file(ply_file_path):
    with open(ply_file_path, 'rb') as f:
        plydata = PlyData.read(f)
        x = np.array(plydata['vertex']['x'])
        y = np.array(plydata['vertex']['y'])
        z = np.array(plydata['vertex']['z'])
        vertices = np.vstack((x, y, z)).T
    return vertices

@DATASETS.register_module()
class BoneDataset_pretrain(Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH if hasattr(config, 'DATA_PATH') else config.get('data_root', '')
        self.metadata_file_path = config.METADATA_PATH if hasattr(config, 'METADATA_PATH') else config.get('metadata_file_path', '')
        self.npoints = config.N_POINTS if hasattr(config, 'N_POINTS') else config.get('npoints', 1024)
        self.num_category = config.NUM_CATEGORY if hasattr(config, 'NUM_CATEGORY') else config.get('num_category', 10)
        self.sample_points_num = config.npoints if hasattr(config, 'npoints') else self.npoints
        self.subset = getattr(config, 'subset', 'train')
        # runtime K-fold cross-validation (injected by main.py for --train_mode cv)
        self.cv_fold = getattr(config, 'cv_fold', None)
        self.cv_folds = getattr(config, 'cv_folds', None)
        self.cv_seed = getattr(config, 'cv_seed', 0)

        if self.cv_fold is not None:
            # CV mode: restrict pretraining to this fold's rows (same shared
            # split function as BoneDataset, so the SSL stage sees exactly the
            # same data distribution as the finetune stage of this fold)
            train_df, valid_df = compute_cv_fold_split(
                self.metadata_file_path, self.cv_folds, self.cv_seed, self.cv_fold)
            data = train_df if self.subset == 'train' else valid_df
            print(f'[DATASET] CV fold {self.cv_fold}/{self.cv_folds - 1} '
                  f'(seed {self.cv_seed}): {len(train_df) + len(valid_df)} pooled '
                  f'samples -> {len(data)} for pretrain {self.subset}')
        else:
            data = pd.read_csv(self.metadata_file_path)

        self.file_list = []
        for idx, row in data.iterrows():
            bone_type = row['Bone']
            file_path = f"{self.data_root}/{row['File']}"
            model_id = os.path.basename(row['File']).split('.')[0]
            
            self.file_list.append({
                'taxonomy_id': bone_type,
                'model_id': model_id,
                'file_path': file_path,
            })
        
        print(f'[DATASET] {len(self.file_list)} instances were loaded')
        
        self.permutation = np.arange(self.npoints)

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        
    def random_sample(self, pc, num):
        if len(pc) <= num:
            return pc
        np.random.shuffle(self.permutation[:len(pc)])
        pc = pc[self.permutation[:num]]
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = np.array(read_ply_file(sample['file_path'])).astype(np.float32)
        
        data = self.random_sample(data, self.sample_points_num)
        data = self.pc_norm(data)
        data = torch.from_numpy(data).float()
        return sample['taxonomy_id'], sample['model_id'], data

    def __len__(self):
        return len(self.file_list)