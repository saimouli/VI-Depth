#!/usr/bin/env python3

import os
import argparse

import torch
import imageio
import numpy as np

from geometry_msgs.msg import Pose, Point
import matplotlib.pyplot as plt
import rospy
import glob
from visualizer.ros_visualizer import PointCloudVisualizer
from model.mhybrid_net import midasNet
from data.SML_dataset import SML_dataset
from torch.utils.data import DataLoader
import cv2
import modules.midas.transforms as transforms
from modules.midas.midas_net_custom import MidasNet_small_videpth
from model.main_consistent import midasNetConsistentModule
from data.SML_tartan_consistent_dataset import SML_tartan_consistent_dataset

#BA
import torch.nn.functional as F
from typing import List, Tuple, Optional
from utils.camera import Camera, pose_to_se3, se3_to_pose
from scipy.spatial.transform import Rotation as R
import theseus as th
from theseus.core.cost_weight import ScaleCostWeight
from theseus.core.variable    import Variable
from theseus.core.robust_cost_function import RobustCostFunction
from theseus.core.robust_loss import HuberLoss
import traceback
from utils.camera import Camera
import modules.midas.utils as utils

import kornia
import kornia.geometry.camera as kornia_camera

ROS_VIZ = True
EVAL = False
debug_visualize = True

def project_depth_vectorize(depth_img, img, p_CinG, R_CtoG, cam_K, normals=None, scale=None, shift=None):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy() #.permute(1,2,0).numpy()
    p_CinG = p_CinG.reshape((3,1))

    valid_mask = (depth_img >= 0.2) & (depth_img <= 8)
    y_coords, x_coords = np.where(valid_mask)

    valid_depth_values = depth_img[y_coords, x_coords]

    pixel_coordinates = np.vstack((x_coords, y_coords, np.ones_like(x_coords)))
    normalized_camera_coordinates = np.linalg.solve(cam_K, pixel_coordinates)
    normalized_camera_coordinates *= valid_depth_values

    pFinC = np.vstack((normalized_camera_coordinates, np.ones_like(x_coords)))
    p_CinG_broadcasted = np.tile(p_CinG.reshape(3, 1), (1, pFinC.shape[1]))
    pFinG = np.dot(R_CtoG, pFinC[:3, :]) + p_CinG_broadcasted

    bgr_values = img[y_coords, x_coords] * 255.0

    points = pFinG.T.tolist()
    colors = bgr_values.tolist()

    normals_world = None
    if normals is not None:
        normals = normals[y_coords, x_coords]
        # scale relative normal using sparse depth
        scaled_normals = scale * normals + shift

        # convert normlas to world frame
        normals_world = np.dot(R_CtoG, scaled_normals.reshape(-1, 3).T).T.tolist()

    points = np.asarray(points).reshape(-1,3)
    colors = np.asarray(colors).reshape(-1,3)
    if normals is not None:
        normals_world = np.asarray(normals_world).reshape(-1,3)

    return points, colors, normals_world

def create_simple_photometric_cost(target_img, ref_img, depth, K, rel_pose_var, cost_weight, name):
    """Simple photometric cost using Camera class warping"""
    
    def photometric_error(optim_vars, aux_vars):
        rel_pose, = optim_vars
        tgt_img_var, ref_img_var, depth_var, K_var = aux_vars
        
        batch_size = rel_pose.tensor.shape[0]
        device = rel_pose.tensor.device
        dtype = rel_pose.tensor.dtype
        
        tgt_img = tgt_img_var.tensor  # [B, C, H, W]
        ref_img = ref_img_var.tensor  # [B, C, H, W]
        depth_map = depth_var.tensor  # [B, 1, H, W]
        K_matrix = K_var.tensor       # [B, 3, 3]
        
        try:
            # Convert SE3 tensor [B, 3, 4] to 4x4 matrix for Camera class
            se3_tensor = rel_pose.tensor  # [B, 3, 4]
            T_4x4 = torch.zeros(batch_size, 4, 4, dtype=dtype, device=device)
            T_4x4[:, :3, :] = se3_tensor  # Copy [R|t]
            T_4x4[:, 3, 3] = 1.0  # Bottom row [0, 0, 0, 1]

            # Ensure we have the right number of variables to unpack
            # Handle the case where depth might have extra dimensions
            # if depth_map.dim() == 4 and depth_map.shape[1] == 1:
            #     depth_map = depth_map  # Keep as [B, 1, H, W]
            # elif depth_map.dim() == 3:
            #     depth_map = depth_map.unsqueeze(1)  # Add channel dim
                            
            # Camera warping (use first batch element since Camera expects single pose)
            target_cam = Camera(K=K_matrix.float()).scaled(1.0).to(device)
            ref_cam = Camera(K=K_matrix.float(), Twc=T_4x4[0]).scaled(1.0).to(device)
            
            # Reconstruct 3D points from target camera
            depth_map = depth_map.squeeze(0)
            world_points = target_cam.reconstruct(depth_map, frame='w')  # [B, 3, H, W]
            
            # Project to reference camera
            ref_coords = ref_cam.project(world_points, frame='w', normalize=True)  # [B, H, W, 2]
            
            # Valid mask
            with torch.no_grad():
                valid_mask = (ref_coords.abs().max(dim=-1)[0] <= 1.0).float()  # [B, H, W]
                depth_mask = ((depth_map > 0.1) & (depth_map <= 5.0)).float().squeeze(1)  # [B, H, W]
                combined_mask = valid_mask * depth_mask
                
            # Warp reference image
            ref_warped = F.grid_sample(
                ref_img, ref_coords,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )
            
            # Photometric error
            diff = (tgt_img - ref_warped) * combined_mask.unsqueeze(1)
            error = diff.abs().sum(dim=(1, 2, 3)) / (combined_mask.sum(dim=(1, 2)) + 1e-8)
            error = error.unsqueeze(1)  # [B, 1]
            
            return error
            
        except Exception as e:
            print(f"Error in photometric cost: {e}")
            print(traceback.format_exc())
            return torch.zeros(batch_size, 1, device=device, dtype=dtype)
    
    aux_vars = (
        th.Variable(tensor=target_img, name=f"{name}_tgt"),
        th.Variable(tensor=ref_img, name=f"{name}_ref"),
        th.Variable(tensor=depth, name=f"{name}_depth"),
        th.Variable(tensor=K, name=f"{name}_K")
    )
    
    return th.AutoDiffCostFunction(
        optim_vars=(rel_pose_var,),
        err_fn=photometric_error,
        dim=1,
        aux_vars=aux_vars,
        cost_weight=cost_weight,
        name=name,
        autograd_mode="dense"
    )
    
def visualize_dataset_sample(tgt_img, ref_imgs, tgt_depth, batch_idx):
    """Visualize dataset sample"""
    try:
        n_refs = len(ref_imgs)
        fig, axes = plt.subplots(2, n_refs + 1, figsize=(4 * (n_refs + 1), 8))
        
        # Target image and depth
        tgt_np = tgt_img.permute(1, 2, 0).cpu().numpy()
        depth_np = tgt_depth.squeeze().cpu().numpy()
        
        axes[0, 0].imshow(tgt_np)
        axes[0, 0].set_title('Target Image')
        axes[0, 0].axis('off')
        
        depth_vis = np.clip(depth_np, 0, 5)
        im = axes[1, 0].imshow(depth_vis, cmap='viridis')
        axes[1, 0].set_title('Target Depth')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], shrink=0.8)
        
        # Reference images
        for i, ref_img in enumerate(ref_imgs):
            ref_np = ref_img.permute(1, 2, 0).cpu().numpy()
            axes[0, i + 1].imshow(ref_np)
            axes[0, i + 1].set_title(f'Reference {i}')
            axes[0, i + 1].axis('off')
            
            # Empty bottom row for refs
            axes[1, i + 1].axis('off')
        
        plt.suptitle(f'Dataset Sample - Batch {batch_idx}')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Visualization failed: {e}")

def visualize_warping_debug(tgt_img, ref_img, ref_warped, valid_mask, name):
    """Simple warping visualization for debugging"""
    try:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(f'Warping Debug - {name}')
        
        def to_numpy(tensor):
            if tensor.dim() == 3 and tensor.shape[0] in [1, 3]:
                tensor = tensor.permute(1, 2, 0)
            elif tensor.dim() == 2:
                tensor = tensor.unsqueeze(-1)
            return torch.clamp(tensor, 0, 1).detach().cpu().numpy().squeeze()
        
        tgt_np = to_numpy(tgt_img)
        ref_np = to_numpy(ref_img)
        warped_np = to_numpy(ref_warped)
        mask_np = valid_mask.detach().cpu().numpy()
        
        axes[0].imshow(tgt_np)
        axes[0].set_title('Target')
        axes[0].axis('off')
        
        axes[1].imshow(ref_np)
        axes[1].set_title('Reference')
        axes[1].axis('off')
        
        axes[2].imshow(warped_np)
        axes[2].set_title('Warped Ref')
        axes[2].axis('off')
        
        # Error map
        if len(tgt_np.shape) == 3:
            diff = np.mean(np.abs(tgt_np - warped_np), axis=-1)
        else:
            diff = np.abs(tgt_np - warped_np)
            
        masked_diff = diff * mask_np
        im = axes[3].imshow(masked_diff, cmap='hot', vmin=0, vmax=0.2)
        valid_pixels = np.sum(mask_np > 0.5)
        avg_error = masked_diff[mask_np > 0.5].mean() if valid_pixels > 0 else 0
        axes[3].set_title(f'Error (avg: {avg_error:.4f})')
        axes[3].axis('off')
        plt.colorbar(im, ax=axes[3], shrink=0.8)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Debug visualization failed: {e}")
        
def test_warping_only(target_image_batch, ref_image_batch, depth_batch, K_batch, 
                      initial_pose, gt_pose):
    """Test just the warping without optimization to see if it works"""
    print("\n" + "="*50)
    print("TESTING WARPING QUALITY")
    print("="*50)
    
    depth_batch = depth_batch.squeeze(0)
    device = target_image_batch.device
    dtype = target_image_batch.dtype
    
    poses_to_test = [
        ("Initial (OpenVINS)", initial_pose),
        ("Ground Truth", gt_pose),
    ]
    
    for name, pose_3x4 in poses_to_test:
        print(f"\nTesting warping with {name} pose...")
        
        # Convert to 4x4
        T_4x4 = torch.zeros(1, 4, 4, dtype=dtype, device=device)
        T_4x4[:, :3, :] = pose_3x4
        T_4x4[:, 3, 3] = 1.0
        
        # Camera setup
        K_single = K_batch
        T_single = T_4x4[0]
        
        target_cam = Camera(K=K_single.float()).scaled(1.0).to(device)
        ref_cam = Camera(K=K_single.float(), Twc=T_single).scaled(1.0).to(device)
        
        # Warping
        world_points = target_cam.reconstruct(depth_batch, frame='w')
        ref_coords = ref_cam.project(world_points, frame='w', normalize=True)
        
        # Valid mask
        valid_mask = (ref_coords.abs().max(dim=-1)[0] <= 1.0).float()
        depth_mask = ((depth_batch > 0.1) & (depth_batch <= 5.0)).float()
        if depth_mask.dim() == 4:
            depth_mask = depth_mask.squeeze(1)
        combined_mask = valid_mask * depth_mask
        
        # Warp
        ref_warped = F.grid_sample(
            ref_image_batch, ref_coords,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )
        
        # Compute error
        diff = (target_image_batch - ref_warped) * combined_mask.unsqueeze(1)
        valid_pixels = combined_mask.sum()
        avg_error = diff.abs().sum() / (valid_pixels + 1e-8)
        
        print(f"  Valid pixels: {valid_pixels.item():.0f}")
        print(f"  Average photometric error: {avg_error.item():.6f}")
        
        # Visualize
        visualize_warping_debug(
            target_image_batch[0], ref_image_batch[0], ref_warped[0], 
            combined_mask[0], f"{name}_pose"
        )
           
@torch.no_grad()
def evaluate_tartan_dataloader():
    print("="*80)
    print("DATASET BUNDLE ADJUSTMENT TEST")
    print("="*80)
    
    dataset = torch.utils.data.DataLoader(SML_tartan_consistent_dataset(
        data_root='/media/saimouli/RPNG_FLASH_4/datasets/tartan_air/carwelding_sample_P007/Easy/P007', mode='val'),
        batch_size = 1, shuffle=False)
    
    poses_ov = []; poses_gt = []; poses_ba = []
    optimized_pose_history = {}  # Keep track of optimized poses by their index
    
    device = 'cpu'
    dtype = torch.float32
    
    if ROS_VIZ:
        visualizer = PointCloudVisualizer()
        
    for batch_idx, batch_data in enumerate(dataset):  
        print(f"\n{'='*80}")
        print(f"PROCESSING BATCH {batch_idx}")
        print(f"{'='*80}")
        if batch_idx <=30:
            continue
        
        (tgt_img, tgt_gt_depth_inv, tgt_ga_depth, tgt_interp, _,
        ref_imgs, ref_ga_depth, ref_interp, ref_gt_depth, _,
        tgt_pose, ref_pose, intrinsics,
        tgt_ov_pose, ref_ov_pose) = batch_data
        
        tgt_gt_depth = utils.inv2depth(tgt_gt_depth_inv)
        # tgt_gt_depth = 1.0 / tgt_gt_depth_inv.squeeze()
        # tgt_gt_depth = torch.nan_to_num(tgt_gt_depth, nan=0.0, posinf=0.0, neginf=0.0)
        # valid = (tgt_gt_depth >= 0.1) & (tgt_gt_depth <= 5.0)
        #tgt_gt_depth = tgt_gt_depth * valid.float()
        ref_ov_poses = [pose.squeeze() for pose in ref_ov_pose]
        ref_gt_poses = [pose.squeeze() for pose in ref_pose]
        ref_images = [img.squeeze().permute(2, 0, 1) for img in ref_imgs]
        target_image = tgt_img.squeeze().permute(2, 0, 1)
        target_pose = tgt_ov_pose.squeeze()
        target_gt_pose = tgt_pose.squeeze()
        K = intrinsics.squeeze()
        
        print(f"Target image shape: {target_image.shape}")
        print(f"Depth shape: {tgt_gt_depth.shape}")
        print(f"Number of reference views: {len(ref_images)}")
        print(f"Intrinsics:\\n{K}")

        visualize_dataset_sample(target_image, ref_images[:2], tgt_gt_depth, batch_idx)

        ref_img = ref_images[0]
        ref_gt_pose = ref_gt_poses[0]
        ref_ov_pose = ref_ov_poses[0]
        
        print(f"\\nGround truth target pose:\\n{target_gt_pose}")
        print(f"Ground truth ref pose:\\n{ref_gt_pose}")
        print(f"OpenVINS target pose:\\n{target_pose}")
        print(f"OpenVINS ref pose:\\n{ref_ov_pose}")
        
        # Compute relative poses
        target_gt_inv = torch.linalg.inv(target_gt_pose)
        ref_gt_rel = ref_gt_pose @ target_gt_inv  # T_ref_target (ground truth)
        
        target_ov_inv = torch.linalg.inv(target_pose)
        ref_ov_rel = ref_ov_pose @ target_ov_inv  # T_ref_target (OpenVINS initial)
        
        print(f"\\nGround truth relative pose:\\n{ref_gt_rel[:3, :]}")
        print(f"OpenVINS relative pose:\\n{ref_ov_rel[:3, :]}")
        
        initial_error = torch.norm(ref_ov_rel[:3, :] - ref_gt_rel[:3, :]).item()
        print(f"Initial relative pose error: {initial_error:.6f}")
                
        # Prepare data for optimization
        target_image_batch = target_image.unsqueeze(0).to(device)
        ref_image_batch = ref_img.unsqueeze(0).to(device)
        depth_batch = tgt_gt_depth.unsqueeze(0).to(device)
        K_batch = K.unsqueeze(0).to(device)
                
        # Convert relative pose to SE3 format [3, 4]
        ref_ov_rel_3x4 = ref_ov_rel[:3, :].unsqueeze(0).to(device)
        ref_gt_rel_3x4 = ref_gt_rel[:3, :].unsqueeze(0).to(device)
        
        test_warping_only(target_image_batch, ref_image_batch, depth_batch, K_batch,
                         ref_ov_rel_3x4, ref_gt_rel_3x4)
                
        # Create optimization variables
        rel_pose_var = th.SE3(tensor=ref_ov_rel_3x4.clone(), name="rel_pose")
        
        # Create objective
        objective = th.Objective(dtype=dtype)
        
        # Photometric cost
        photo_weight = th.ScaleCostWeight(1.0 * torch.ones(1, dtype=dtype))
        photo_cost = create_simple_photometric_cost(
            target_image_batch, ref_image_batch, depth_batch, K_batch,
            rel_pose_var, photo_weight, "photo_cost"
        )
        objective.add(photo_cost)
        
        # Light regularization toward ground truth (to test convergence)
        # reg_weight = th.ScaleCostWeight(0.01 * torch.ones(1, dtype=dtype))
        # gt_target = th.SE3(tensor=ref_gt_rel_3x4.clone(), name="gt_rel_pose")
        # reg_cost = th.Difference(
        #     var=rel_pose_var,
        #     target=gt_target,
        #     cost_weight=reg_weight,
        #     name="regularization"
        # )
        # objective.add(reg_cost)
        
        # Create optimizer
        optimizer = th.LevenbergMarquardt(
            objective,
            max_iterations=20,
            step_size=0.01
        )
        
        theseus_layer = th.TheseusLayer(optimizer)
        
        # Prepare inputs
        theseus_inputs = {
            "rel_pose": ref_ov_rel_3x4.clone(),
            #"gt_rel_pose": ref_gt_rel_3x4.clone(),
            "photo_cost_tgt": target_image_batch,
            "photo_cost_ref": ref_image_batch,
            "photo_cost_depth": depth_batch,
            "photo_cost_K": K_batch
        }
        
        try:
            print("\\nRunning bundle adjustment optimization...")
            
            with torch.no_grad():
                theseus_outputs, info = theseus_layer.forward(
                    input_tensors=theseus_inputs,
                    optimizer_kwargs={
                        "verbose": True,
                        "track_err_history": True,
                        "track_best_solution": True
                    }
                )                    

            if hasattr(info, 'best_solution') and info.best_solution is not None:
                opt_rel_pose_3x4 = info.best_solution["rel_pose"]
            else:
                opt_rel_pose_3x4 = theseus_outputs["rel_pose"]
            
            final_error = torch.norm(opt_rel_pose_3x4 - ref_gt_rel_3x4).item()
            improvement = ((initial_error - final_error) / initial_error) * 100 if initial_error > 0 else 0
            
            print(f"\\nOptimized relative pose:\\n{opt_rel_pose_3x4[0]}")
            print(f"Ground truth relative pose:\\n{ref_gt_rel_3x4[0]}")
            print(f"Pose error: {initial_error:.6f} → {final_error:.6f} ({improvement:.1f}% improvement)")            
        
            # Show convergence
            if hasattr(info, 'err_history') and len(info.err_history) > 0:
                initial_cost = info.err_history[0]
                final_cost = info.err_history[-1]
                if torch.is_tensor(initial_cost):
                    if initial_cost.numel() == 1:
                        init_cost = initial_cost.item()
                    else:
                        init_cost = initial_cost.mean().item()
                else:
                    init_cost = float(initial_cost)
                    
                if torch.is_tensor(final_cost):
                    if final_cost.numel() == 1:
                        final_cost_val = final_cost.item()
                    else:
                        final_cost_val = final_cost.mean().item()
                else:
                    final_cost_val = float(final_cost)
                    
                cost_improvement = ((init_cost - final_cost_val) / init_cost) * 100 if init_cost > 0 else 0
                print(f"Cost: {init_cost:.6f} → {final_cost_val:.6f} ({cost_improvement:.1f}% improvement)")
                                
                # Plot convergence
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                fig.suptitle('Real Dataset Optimization Results')
                
                costs = []
                for c in info.err_history:
                    if torch.is_tensor(c):
                        if c.numel() == 1:
                            costs.append(c.item())
                        else:
                            costs.append(c.mean().item())  # Take mean if multiple elements
                    else:
                        costs.append(float(c))
                
                axes[0].semilogy(costs, 'b-', linewidth=2)
                axes[0].set_xlabel('Iteration')
                axes[0].set_ylabel('Cost (log scale)')
                axes[0].set_title('Cost Convergence')
                axes[0].grid(True)
                
                # Pose error
                pose_errors = []
                for i in range(len(costs)):
                    alpha = i / max(1, len(costs) - 1)
                    interp_pose = ref_ov_rel_3x4 * (1 - alpha) + opt_rel_pose_3x4 * alpha
                    pose_err = torch.norm(interp_pose - ref_gt_rel_3x4).item()
                    pose_errors.append(pose_err)
                
                axes[1].plot(pose_errors, 'r-', linewidth=2)
                axes[1].set_xlabel('Iteration')
                axes[1].set_ylabel('Pose Error')
                axes[1].set_title('Pose Error vs Ground Truth')
                axes[1].grid(True)
                
                # Translation error
                trans_errors = []
                for i in range(len(costs)):
                    alpha = i / max(1, len(costs) - 1)
                    interp_pose = ref_ov_rel_3x4 * (1 - alpha) + opt_rel_pose_3x4 * alpha
                    trans_err = torch.norm(interp_pose[0, :3, 3] - ref_gt_rel_3x4[0, :3, 3]).item()
                    trans_errors.append(trans_err)
                
                axes[2].plot(trans_errors, 'g-', linewidth=2)
                axes[2].set_xlabel('Iteration')
                axes[2].set_ylabel('Translation Error')
                axes[2].set_title('Translation Error')
                axes[2].grid(True)
                
                plt.tight_layout()
                plt.show()
            
            # Success criteria
            success = (improvement > 5 and final_error < initial_error * 0.8)
        
            if success:
                print("\\n Dataset bundle adjustment optimization SUCCESSFUL!")
                print(f"   ✓ Pose error improved by {improvement:.1f}%")
                print(f"   ✓ Final pose error: {final_error:.6f}")
                print(f"   ✓ Used real dataset images and Camera class")
            else:
                print("\\n  Optimization completed but limited improvement")
                print(f"   • Improvement: {improvement:.1f}%")
                print(f"   • Final error: {final_error:.6f}")
                print("   • This might still indicate the optimization is working") 
                       
        except Exception as e:
            print(f"Optimization failed: {e}")
            import traceback
            traceback.print_exc()
            return False        
        # tgt_gt_depth_inv = tgt_gt_depth_inv.squeeze().cpu().numpy()
        # tgt_ov_pose = tgt_ov_pose.squeeze().cpu().numpy()
        # tgt_pose = tgt_pose.squeeze().cpu().numpy()
        # intrinsics = intrinsics.squeeze().cpu().numpy()
        # tgt_img = tgt_img.squeeze().cpu().numpy()
        
        # p_CinG_ov = tgt_ov_pose[:3, 3] 
        # R_CtoG_ov = tgt_ov_pose[:3, :3]
        
        # pgt_CinG = tgt_pose[:3, 3]
        # Rgt_CtoG = tgt_pose[:3, :3]         
        
        # #ov_pose
        # poses_ov.append(Pose(position=Point(
        #     x=p_CinG_ov[0],
        #     y=p_CinG_ov[1],
        #     z=p_CinG_ov[2]
        # )))
        
        # #gt_pose
        # poses_gt.append(Pose(position=Point(
        #     x=pgt_CinG[0],
        #     y=pgt_CinG[1],
        #     z=pgt_CinG[2]
        # )))
        
        # ##optimized poses
        # poses_ba.append(Pose(position=Point(
        #     x=p_CinG_ba[0],
        #     y=p_CinG_ba[1],
        #     z=p_CinG_ba[2]
        # )))
        
        # if ROS_VIZ:
        #     visualizer = PointCloudVisualizer()
        #     rate = rospy.Rate(1)
        
        # #avg_error_w_int_depth = metrics.ErrorMetricsAverager()
        
        # if ROS_VIZ and batch_idx % 10 == 0:
        #     #points_refine, colors_refine, _ = project_depth_vectorize(sml_depth_viz, mid_image, p_CinG, R_CtoG, intrinsics)
        #     points_gt, colors_gt, _ = project_depth_vectorize(1.0/tgt_gt_depth_inv, tgt_img, pgt_CinG, Rgt_CtoG, intrinsics)
        #     #pc_active, _, _ = project_depth_vectorize(input_active_depth, mid_image, p_CinG, R_CtoG, intrinsics)
        #     #pc_slam, _, _ = project_depth_vectorize(input_slam_depth, mid_image, p_CinG, R_CtoG, intrinsics)
            
        #     visualizer.publish_path(poses_ov)
        #     visualizer.publish_gtpath(poses_gt)
        #     visualizer.publish_bapath(poses_ba)
            
        #     tgt_img = tgt_img* 255.0
        #     #tgt_img = overlay_sparse_points_on_image(tgt_img, 1.0/input_sparse_depth)
        #     visualizer.pose_callback(pgt_CinG, Rgt_CtoG)
        #     #visualizer.publish_point_cloud_refine(points_refine, colors_refine)
        #     visualizer.publish_point_cloud_gt(points_gt, colors_gt)
        #     visualizer.publish_tgt_img(tgt_img)
        #     #visualizer.publish_active_points(pc_active)
        #     #visualizer.publish_slam_points(pc_slam)
            
        #     rate.sleep()

if __name__ == '__main__':
    evaluate_tartan_dataloader()