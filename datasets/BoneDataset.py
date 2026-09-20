import os
import numpy as np
import warnings
import pandas as pd
import torch
import scipy.spatial
from torch.utils.data import Dataset
from .build import DATASETS
from utils.logger import *
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')

def read_ply_file(ply_file_path):
    from plyfile import PlyData
    with open(ply_file_path, 'rb') as f:
        plydata = PlyData.read(f)
        x = np.array(plydata['vertex']['x'])
        y = np.array(plydata['vertex']['y'])
        z = np.array(plydata['vertex']['z'])
        vertices = np.vstack((x, y, z)).T
    return vertices

def pc_normalize(pc):
    """
    Normalize point cloud using PyTorch.
    Input:
        pc: Tensor of shape [N, D] (typically D=3)
    Output:
        Normalized pc: Tensor of shape [N, D]
    """
    centroid = torch.mean(pc, dim=0, keepdim=True)  # [1, D]
    pc = pc - centroid                              # center the point cloud
    scale = torch.max(torch.norm(pc, dim=1))        # max distance to origin
    pc = pc / scale                                  # normalize to unit sphere
    return pc

def fps(points, num_samples):
    """
    Torch-based FPS implementation
    Input:
        points: torch tensor [N, D]
        num_samples: number of samples
    Return:
        sampled points: [num_samples, D]
    """
    device = points.device
    N, C = points.shape
    indices = torch.zeros(num_samples, dtype=torch.long, device=device)
    distances = torch.ones(N, device=device) * 1e10
    farthest = torch.randint(0, N, (1,), dtype=torch.long, device=device)
    
    for i in range(num_samples):
        indices[i] = farthest
        centroid = points[farthest].view(1, C)
        dist = torch.sum((points - centroid) ** 2, dim=-1)
        distances = torch.min(distances, dist)
        farthest = torch.argmax(distances)
    
    sampled_points = points[indices]
    return sampled_points

def modified_fps(points, num_samples):
    """
    Modified FPS that aligns object with Z-axis and samples strategically
    Input:
        points: numpy array [N, D]
        num_samples: number of samples
    Return:
        sampled points: [num_samples, D]
    """
    points_tensor = torch.tensor(points, dtype=torch.float32)
    
    points_cpu = points_tensor.cpu().numpy()
    hull = scipy.spatial.ConvexHull(points_cpu)
    hull_vertices = hull.vertices
    hull_points = points_tensor[hull_vertices]
    
    # Find the pair of points with maximum distance
    hull_points_expanded = hull_points.unsqueeze(1)
    distances = torch.norm(hull_points_expanded - hull_points, dim=2)
    max_idx = torch.argmax(distances)
    max_pair = torch.div(max_idx, distances.shape[1], rounding_mode='floor'), max_idx % distances.shape[1]
    
    point1 = hull_points[max_pair[0]]
    point2 = hull_points[max_pair[1]]
    
    direction = point2 - point1
    direction = direction / torch.norm(direction)
    
    z_axis = torch.tensor([0, 0, 1], dtype=points_tensor.dtype)
    v = torch.cross(direction, z_axis)
    s = torch.norm(v)
    c = torch.dot(direction, z_axis)
    
    if s != 0:
        vx = torch.tensor([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ], dtype=points_tensor.dtype)
        
        rotation_matrix = torch.eye(3, dtype=points_tensor.dtype) + vx + (vx @ vx) * ((1 - c) / (s**2))
        points_rotated = torch.matmul(points_tensor, rotation_matrix.T)
    else:
        points_rotated = points_tensor
    
    z_min, z_max = points_rotated[:, 2].min(), points_rotated[:, 2].max()
    z_range = z_max - z_min
    
    boundary_part1 = z_min + 0.2 * z_range
    boundary_part2 = z_min + 0.8 * z_range
    
    part1_points = points_rotated[points_rotated[:, 2] <= boundary_part1]
    part2_points = points_rotated[(points_rotated[:, 2] > boundary_part1) & (points_rotated[:, 2] < boundary_part2)]
    part3_points = points_rotated[points_rotated[:, 2] > boundary_part2]
    
    n1 = int(num_samples / 4)
    n2 = int((num_samples - n1) / 2)
    if n1 + 2 * n2 != num_samples:
        n1 += 1
    
    pc_parts = []
    if len(part1_points) > 0:
        n_samples_part1 = min(n2, len(part1_points))
        pc_1 = fps(part1_points, n_samples_part1)
        pc_parts.append(pc_1)
    
    if len(part2_points) > 0:
        n_samples_part2 = min(n1, len(part2_points))
        pc_center = fps(part2_points, n_samples_part2)
        pc_parts.append(pc_center)
    
    if len(part3_points) > 0:
        n_samples_part3 = min(n2, len(part3_points))
        pc_2 = fps(part3_points, n_samples_part3)
        pc_parts.append(pc_2)
    
    if pc_parts:
        pc = torch.cat(pc_parts, dim=0)
        # If we don't have enough points, pad with random sampling
        if pc.shape[0] < num_samples:
            remaining = num_samples - pc.shape[0]
            additional_indices = torch.randperm(points_tensor.shape[0])[:remaining]
            additional_points = points_tensor[additional_indices]
            pc = torch.cat([pc, additional_points], dim=0)
        elif pc.shape[0] > num_samples:
            pc = pc[:num_samples]
    else:
        pc = fps(points_tensor, num_samples)
    
    return pc

def compute_cv_fold_split(metadata_file_path, cv_folds, cv_seed, fold):
    """K-fold split of all non-test rows, stratified by the `Bone` label so
    every fold keeps the pool's class proportions. If the CSV has no `Split`
    column, every row is pooled.

    Shared by BoneDataset and main.py so both agree on the split.
    Returns (train_df, valid_df), copies of the original CSV rows.
    """
    from sklearn.model_selection import StratifiedKFold
    assert 0 <= fold < cv_folds, f'fold {fold} out of range for cv_folds {cv_folds}'
    full_data = pd.read_csv(metadata_file_path)
    if 'Split' in full_data.columns:
        split_col = full_data['Split'].str.strip().str.lower()
        pool = full_data[split_col != 'test'].reset_index(drop=True)
    else:
        pool = full_data.reset_index(drop=True)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=cv_seed)
    train_idx, valid_idx = list(skf.split(np.zeros(len(pool)), pool['Bone'].values))[fold]
    return pool.iloc[np.sort(train_idx)].copy(), pool.iloc[np.sort(valid_idx)].copy()

@DATASETS.register_module()
class BoneDataset(Dataset):
    def __init__(self, config):
        self.metadata_file_path = config.METADATA_FILE_PATH
        self.data_dir_path = config.DATA_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset
        self.num_category = config.NUM_CATEGORY
        self.sampling_algo = getattr(config, 'sampling_algo', 'fps')  # 'fps' or 'modified_fps'
        # runtime K-fold cross-validation (injected by main.py for --train_mode cv)
        self.cv_fold = getattr(config, 'cv_fold', None)
        self.cv_folds = getattr(config, 'cv_folds', None)
        self.cv_seed = getattr(config, 'cv_seed', 0)

        full_data = pd.read_csv(self.metadata_file_path)
        # Class mapping from the full CSV so label indices are identical across
        # subsets and folds even if one of them misses a rare class
        unique_classes = sorted(full_data['Bone'].unique())
        self.classes = {item: i for i, item in enumerate(unique_classes)}

        if self.cv_fold is not None and self.subset in ('train', 'valid'):
            # CV mode: ignore the CSV's train/valid assignment — see
            # instances, so the fold partition is consistent between them)
            train_df, valid_df = compute_cv_fold_split(
                self.metadata_file_path, self.cv_folds, self.cv_seed, self.cv_fold)
            data = train_df if self.subset == 'train' else valid_df
            print_log(
                f'CV fold {self.cv_fold}/{self.cv_folds - 1} (seed {self.cv_seed}): '
                f'{len(train_df) + len(valid_df)} pooled samples -> {len(data)} for {self.subset}',
                logger='BoneDataset')
        else:
            split_col = full_data['Split'].str.strip().str.lower()
            data = full_data[split_col == self.subset]

        data = data.copy()
        data['File'] = data.apply(lambda x: os.path.join(self.data_dir_path, x['File']), axis=1)
        self.datapath = list(data['File'])
        self.list_classes = list(data['Bone'])
        self.list_labels = [self.classes[label] for label in self.list_classes]
        
        print_log(f'The size of {self.subset} data is {len(self.datapath)}', logger='BoneDataset')
        print_log(f'Number of categories: {len(self.classes)}', logger='BoneDataset')
        print_log(f'Using sampling method: {self.sampling_algo}', logger='BoneDataset')

    def __len__(self):
        return len(self.datapath)

    def get_weights(self):
        try:
            class_weights = compute_class_weight(
                'balanced', 
                classes=np.unique(self.list_labels), 
                y=self.list_labels
            )            
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
            return weights_tensor
        except Exception as e:
            print(f"Error calculating class weights: {e}")
            return None

    def _get_item(self, index):
        path = self.datapath[index]
        label = self.list_labels[index]
        
        try:
            point_set = read_ply_file(path).astype(np.float32)
            point_set = torch.from_numpy(point_set).float()
            
            if point_set.shape[0] >= self.npoints:
                if self.sampling_algo == 'modified_fps':
                    try:
                        point_set = modified_fps(point_set, self.npoints)
                    except Exception as e:
                        print_log(f'Modified FPS failed for {path}, falling back to regular FPS: {e}', logger='BoneDataset')
                        point_set = fps(point_set, self.npoints)
                else:  # default to regular fps
                    point_set = fps(point_set, self.npoints)
            else:
                indices = torch.randint(0, point_set.shape[0], (self.npoints,))
                point_set = point_set[indices]
            
            point_set = pc_normalize(point_set)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            return None, None
        
        return point_set, label

    def __getitem__(self, index):
        points, label = self._get_item(index)  # points is now a torch.Tensor
    
        if points is None:
            print(f"Error: points is None for index {index}, path {self.datapath[index]}")
            # Return a dummy point cloud to avoid crashing
            points = torch.zeros((self.npoints, 3))
            
        if label is None:
            print(f"Error: label is None for index {index}, path {self.datapath[index]}")
            label = 0

        if self.subset == 'train':
            pt_idxs = torch.randperm(points.shape[0])
            points = points[pt_idxs]
            
        if points is None or label is None:
             print(f"CRITICAL ERROR: Returning None from __getitem__ index {index}")
             points = torch.zeros((self.npoints, 3))
             label = 0
        
        return 'BoneDataset', 'sample', (points.float(), label)

    def get_classes(self):
        return self.classes