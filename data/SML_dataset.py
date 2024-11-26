import numpy as np
from torch.utils.data import Dataset
import modules.midas.utils as utils
from PIL import Image
import pytorch_lightning as pl
from typing import Tuple
from torch.utils.data import DataLoader
import os

def load_input_image(input_image_fp):
    return utils.read_image(input_image_fp)

def load_sparse_depth(input_sparse_depth_fp, depth_scale):
    input_sparse_depth = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32) / depth_scale
    input_sparse_depth[input_sparse_depth <= 0] = 0.0
    return input_sparse_depth

class SML_dataset(Dataset):
    def __init__(self,
                 data_root,
                 mode="train",
                 depth_scale=1000.0,
                ):
        
        if mode == "train":
            with open(f"{data_root}/train_image.txt") as f: 
                self.image_paths = [data_root + "/" + line.rstrip() for line in f]
        elif mode == "val":
            with open(f"{data_root}/test_image.txt") as f:
                self.image_paths = [data_root + "/" + line.rstrip() for line in f]
        
        self.gt_depth_paths = [image_path.replace('image', 'ground_truth') for image_path in self.image_paths]
        self.sparse_paths = [image_path.replace('image', 'sparse_depth') for image_path in self.image_paths]
        self.pred_depth_paths = [image_path.replace('image', 'depth_infer_dpt') for image_path in self.image_paths]
        self.pred_depth_paths = [image_path.replace('.png', '.npy') for image_path in self.pred_depth_paths]
        self.n_samples = len(self.image_paths)
        self.depth_scale = depth_scale

        
    def __getitem__(self, index):
        image = load_input_image(self.image_paths[index])
        gt_depth = load_sparse_depth(self.gt_depth_paths[index], depth_scale=self.depth_scale)
        sparse_depth = load_sparse_depth(self.sparse_paths[index], depth_scale=self.depth_scale)
        depth_pred = np.load(self.pred_depth_paths[index])
        image, gt_depth, sparse_depth, depth_pred = [
            T.astype(np.float32) for T in [image, gt_depth, sparse_depth, depth_pred]
        ]
        
        return image, gt_depth, sparse_depth, depth_pred
    
    def __len__(self):
        return self.n_samples