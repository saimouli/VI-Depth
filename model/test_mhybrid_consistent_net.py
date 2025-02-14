import torch
import numpy as np
import sys
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.SML_consistent_dataset import SML_consistent_dataset
from mhybrid_consistent_net import midasConsNet
from utils.camera import Camera
from utils.pose import Pose
import cv2
from model.main import midasNetModule
import modules.midas.utils as utils

def compute_reproj_loss(depth_pred, depth_gt, tgt_pose_pred, tgt_pose_gt,
                            ref_pose_pred, ref_pose_gt, K):
    #Loss = ||π(T_pred * X_pred) - π(T_gt * X_gt)||
    #where X_pred = π^-1(x, D_pred), X_gt = π^-1(x, D_gt)
    valid_depth_mask = ((depth_gt > 0.2) & (depth_gt <= 5)).float().detach()
    device = depth_pred.device
    B, _, H, W = depth_pred.shape
    scale_factor = 1.0
        
    # Reconstruct 3D points using PREDICTED target pose and depth(global frame)
    tgt_cam_pred = Camera(K=K, Twc=tgt_pose_pred).scaled(scale_factor).to(device)
    points_pred_world = tgt_cam_pred.reconstruct(depth_pred*valid_depth_mask, frame='w')
        
    # Reconstruct GT points using GT target pose and depth (global frame)
    tgt_cam_gt = Camera(K=K, Twc=tgt_pose_gt).scaled(scale_factor).to(device)
    points_gt_world = tgt_cam_gt.reconstruct(depth_gt*valid_depth_mask, frame='w')
        
    # Project using PREDICTED reference poses (global frame)
    proj_pred = []
    for ref_pose in ref_pose_pred:
        ref_cam_pred = Camera(K=K, Twc=ref_pose)
        proj = ref_cam_pred.project(points_pred_world, frame='w', normalize=True)
        proj_pred.append(proj)
        
    # Project using GT reference poses and depth (global frame)
    proj_gt = []
    for ref_pose in ref_pose_gt:
        ref_cam_gt = Camera(K=K, Twc=ref_pose)
        proj = ref_cam_gt.project(points_gt_world, frame='w', normalize=True)
        proj_gt.append(proj)
        
    # Compute masked reprojection error
    loss = 0
    for p_pred, p_gt in zip(proj_pred, proj_gt):
        valid_mask = (p_pred.abs().max(dim=-1)[0] <= 1.0) & (p_gt.abs().max(dim=-1)[0] <= 1.0)
        valid_mask = valid_mask.float().detach() * valid_depth_mask.squeeze(1)
            
        error = torch.norm(p_pred - p_gt, dim=-1) * valid_mask
        loss += error.sum() / (valid_mask.sum() + 1e-6)
            
    return loss / len(proj_pred)
    
def draw_reprojection(img, proj_points):
    img = img.cpu().numpy().transpose(1, 2, 0) * 255  # Convert to HxWxC
    img = img.astype(np.uint8)
    
    proj_points = proj_points.detach().cpu().numpy()
    for x, y in proj_points.reshape(-1, 2):
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:  # Ensure points are in bounds
            cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), -1)
    
    cv2.imshow("Reprojected Points", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Define dataset
    dataset = SML_consistent_dataset(data_root='/home/sai/Documents/void_small', mode='val')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define model
    model = midasNetModule().load_from_checkpoint(checkpoint_path="/home/sai/Documents/VI-Depth/weights/total_loss=0.105.ckpt")
    model.eval()
    model.to(device)
    
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
        sml_depth_inv = model(tgt_sparse_depth, tgt_img, ref_ga_depth, tgt_interp, tgt_ga_depth)
        sml_depth_inv = sml_depth_inv.detach()
        
        loss = compute_reproj_loss(utils.inv2depth(sml_depth_inv), 
                                   utils.inv2depth(tgt_gt_depth_inv), 
                                   tgt_pose, tgt_pose,
                                   ref_pose, ref_pose, 
                                   intrinsics)
        print("Reprojection Loss: ", loss.item())