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
import matplotlib.pyplot as plt
import torch.nn.functional as F

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
        self.pose_path = [image_path.replace('image', 'absolute_pose') for image_path in self.image_paths]
        self.pose_path = [image_path.replace('.png', '.txt') for image_path in self.pose_path]

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
        pose_CtoG = np.loadtxt(self.pose_path[index])
        sparse_depth_inv = load_depth(self.sparse_paths[index], depth_scale=self.depth_scale)
        depth_pred_inv = np.load(self.pred_depth_paths[index])
        ga_depth_inv = np.load(self.ga_depth_paths[index])
        interp_scale = np.load(self.interp_scale_paths[index])

        gt_depth[gt_depth <= 0] = 0.0
        
        # new_width = 320
        # aspect_ratio = h / w
        # new_height = int(new_width * aspect_ratio)
        
        image, gt_depth, sparse_depth_inv, depth_pred_inv, ga_depth_inv, interp_scale = [
            T.astype(np.float32) for T in [image, gt_depth, sparse_depth_inv, depth_pred_inv, ga_depth_inv, interp_scale]
        ]

        # gt_depth = F.interpolate(torch.tensor(gt_depth).unsqueeze(0).unsqueeze(0), 
        #                                   size=(288, 384), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

        mask = (gt_depth < 8.0)
        mask *= (gt_depth > 0.2)
        gt_depth[~mask] = np.inf  # set invalid depth
        gt_depth_inv = 1.0 / gt_depth
        gt_depth_inv[gt_depth_inv == float("inf")] = 0
        gt_depth_inv = torch.from_numpy(gt_depth_inv).unsqueeze(0)

        # sample = {"image" : image, 
        #           "int_depth" : ga_depth_inv, #LS aligned depth
        #           "int_scales" : interp_scale, #interpolated scale
        #           "int_depth_no_tf" : ga_depth_inv}
        # sample = self.ScaleMapLearner_transform(sample)
        
        mask = torch.from_numpy(mask).unsqueeze(0)
        # plt.imshow((sample["image"].permute(1,2,0).numpy() * 255).astype('uint8'))
        # plt.show()

        return image, gt_depth_inv, sparse_depth_inv, depth_pred_inv, ga_depth_inv, interp_scale, mask, pose_CtoG
        #return image, gt_depth, sparse_depth, depth_pred, ga_depth, interp_scale
    
    def __len__(self):
        return self.n_samples