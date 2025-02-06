import torch
import numpy as np
import sys
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.SML_consistent_dataset import SML_consistent_dataset
from mhybrid_consistent_net import midasConsNet

# Define dataset
dataset = SML_consistent_dataset(data_root='/media/saimouli/Data6T/datasets/VOID_150', mode='val')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define model
model = midasConsNet(0.1, 8, 0.2, 5, 150, 
                      None, log_fn=None, isConvGRU=False)
model.to(device)  # Ensure the model is on the correct device

print("Model Parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

# Iterate over dataloader
for batch_data in dataloader:
    batch_data = tuple(
        [item.to(device) if isinstance(item, torch.Tensor) else 
         [subitem.to(device) if isinstance(subitem, torch.Tensor) else subitem for subitem in item]
         if isinstance(item, list) else item
         for item in batch_data]
    )

    # Unpacking batch_data (ensure these are correctly structured based on dataset output)
    tgt_img, tgt_gt_depth_inv, tgt_ga_depth, tgt_interp, tgt_sparse_depth, ref_img, \
    ref_ga_depth, ref_interp, ref_gt_depth, ref_sparse_depth, tgt_pose, ref_pose, intrinsics = batch_data
    
    # Forward pass through the model
    _, pred_inv_depth = model(tgt_img, ref_img, tgt_ga_depth, ref_ga_depth, tgt_interp,
                                ref_interp, tgt_pose, ref_pose, intrinsics)
    