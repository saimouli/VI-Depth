import numpy as np
from torch.utils.data import Dataset
import modules.midas.utils as utils
from PIL import Image
import pytorch_lightning as pl
from typing import Tuple
from torch.utils.data import DataLoader
import os
import modules.midas.transforms as transforms
import cv2
import torch

def load_input_image(input_image_fp):
    return utils.read_image(input_image_fp)

def load_depth(input_sparse_depth_fp, depth_scale):
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
        
        self.ga_depth_paths = [image_path.replace('image', 'ga_depth_inv') for image_path in self.image_paths]
        self.ga_depth_paths = [image_path.replace('.png', '.npy') for image_path in self.ga_depth_paths]

        self.interp_scale_paths = [image_path.replace('image', 'interp_scale') for image_path in self.image_paths]
        self.interp_scale_paths = [image_path.replace('.png', '.npy') for image_path in self.interp_scale_paths]
        
        self.n_samples = len(self.image_paths)
        self.depth_scale = depth_scale
        model_transforms = transforms.get_transforms("dpt_hybrid", "void", str(150))
        self.ScaleMapLearner_transform = model_transforms["sml_model"]

    def resize_with_aspect_ratio(self, image, target_width, ensure_multiple_of, interpolation=cv2.INTER_CUBIC):
        original_height, original_width = image.shape[:2]
        aspect_ratio = original_height / original_width

        # Calculate new dimensions
        new_width = target_width
        new_height = int(round(new_width * aspect_ratio))
        
        # Ensure height is a multiple of the given value
        new_height = (new_height // ensure_multiple_of) * ensure_multiple_of
        
        # Resize the image
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        return resized_image
        
    def __getitem__(self, index):
        image = load_input_image(self.image_paths[index])
        gt_depth = load_depth(self.gt_depth_paths[index], depth_scale=self.depth_scale)
        sparse_depth_inv = load_depth(self.sparse_paths[index], depth_scale=self.depth_scale)
        depth_pred_inv = np.load(self.pred_depth_paths[index])
        ga_depth_inv = np.load(self.ga_depth_paths[index])
        interp_scale = np.load(self.interp_scale_paths[index])

        gt_depth_resize = self.resize_with_aspect_ratio(gt_depth, target_width=384, ensure_multiple_of=32)
        gt_depth_resize[gt_depth_resize <= 0] = 0.0

        # target depth valid/mask
        mask = (gt_depth_resize < 8.0)
        mask *= (gt_depth_resize > 0.2)
        gt_depth_resize[~mask] = np.inf  # set invalid depth
        gt_depth_inv = 1.0 / gt_depth_resize

        image, gt_depth_inv, sparse_depth_inv, depth_pred_inv, ga_depth_inv, interp_scale = [
            T.astype(np.float32) for T in [image, gt_depth_inv, sparse_depth_inv, depth_pred_inv, ga_depth_inv, interp_scale]
        ]

        sample = {"image" : image, 
                  "int_depth" : ga_depth_inv, #LS aligned depth
                  "int_scales" : interp_scale, #interpolated scale
                  "int_depth_no_tf" : ga_depth_inv}
        sample = self.ScaleMapLearner_transform(sample)

        #resize gt depth to match output
        gt_depth_inv_resize = torch.from_numpy(gt_depth_inv).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return sample["image"], gt_depth_inv_resize, sparse_depth_inv, depth_pred_inv, sample["int_depth"], sample["int_scales"], mask
        #return image, gt_depth, sparse_depth, depth_pred, ga_depth, interp_scale
    
    def __len__(self):
        return self.n_samples