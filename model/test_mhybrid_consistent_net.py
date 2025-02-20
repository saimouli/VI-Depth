import torch
import numpy as np
import sys
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.SML_consistent_dataset import SML_consistent_dataset
from mhybrid_consistent_net import midasConsNet
from utils.camera import Camera, pose_to_se3, se3_to_pose, se3_update
from utils.pose import Pose
import cv2
import matplotlib.pyplot as plt
from model.main import midasNetModule
import modules.midas.utils as utils
import torch.nn.functional as F

def visualize_reproj_error(
    tgt_img,          # Target image [H, W, 3] (numpy array, RGB, 0-255)
    proj_pred,        # Predicted reprojected points [B, H, W, 2]
    proj_gt,          # Ground truth reprojected points [B, H, W, 2]
    valid_mask,       # Valid mask [B, H, W]
    sample_points=100  # Number of points to sample for visualization
):
    """
    Visualize reprojection error by overlaying predicted and GT points on the target image.
    """
    # Convert tensors to numpy arrays
    if isinstance(tgt_img, torch.Tensor):
        tgt_img = tgt_img.squeeze(0).cpu().numpy()
    if isinstance(proj_pred, torch.Tensor):
        proj_pred = proj_pred.cpu().numpy()
    if isinstance(proj_gt, torch.Tensor):
        proj_gt = proj_gt.cpu().numpy()
    if isinstance(valid_mask, torch.Tensor):
        valid_mask = valid_mask.cpu().numpy()

    # Ensure target image is in [0, 255] range
    if tgt_img.max() <= 1.0:
        tgt_img = (tgt_img * 255).astype(np.uint8)

    # Sample random points for visualization
    B, H, W = valid_mask.shape
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    y_coords = y_coords.flatten()
    x_coords = x_coords.flatten()

    # Randomly sample points
    sample_indices = np.random.choice(H * W, size=sample_points, replace=False)
    y_samples = y_coords[sample_indices]
    x_samples = x_coords[sample_indices]

    # Plot the target image
    plt.figure(figsize=(10, 5))
    plt.imshow(tgt_img)
    plt.title("Reprojection Error Visualization")

    # Plot GT and predicted points for each reference view
    #for ref_idx in range(1):
    # Extract valid points
    valid_samples = valid_mask[0, y_samples, x_samples]
    pred_samples = proj_pred[0, y_samples, x_samples][valid_samples == 1]
    gt_samples = proj_gt[0, y_samples, x_samples][valid_samples == 1]

    # Convert normalized coordinates to pixel coordinates
    pred_px = (pred_samples + 1) * np.array([W, H]) / 2  # [-1, 1] → [0, W] and [0, H]
    gt_px = (gt_samples + 1) * np.array([W, H]) / 2

    # Plot GT points (green)
    plt.scatter(gt_px[:, 0], gt_px[:, 1], c='g', s=10, label=f'GT (Ref {0})')

    # Plot predicted points (red)
    plt.scatter(pred_px[:, 0], pred_px[:, 1], c='r', s=10, label=f'Pred (Ref {0})')

    # Draw lines between GT and predicted points
    for gt, pred in zip(gt_px, pred_px):
        plt.plot([gt[0], pred[0]], [gt[1], pred[1]], 'b-', linewidth=0.5)
    
    #also plot valid mask
    plt.imshow(valid_mask[0], alpha=0.5, cmap='gray')

    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
def compute_reproj_loss(depth_pred, depth_gt, tgt_pose_pred, tgt_pose_gt,
                        ref_pose_pred, ref_pose_gt, K,ref_img):
    #Loss = ||π(T_pred * X_pred) - π(T_gt * X_gt)||
    #where X_pred = π^-1(x, D_pred), X_gt = π^-1(x, D_gt)
    device = depth_pred.device
    B, _, H, W = depth_pred.shape
    scale_factor = H / depth_pred.shape[2] 
    N_ref = len(ref_pose_pred)
        
    depth_pred = torch.clamp(depth_pred, min=0.2, max=5.0)
    depth_gt = torch.clamp(depth_gt, min=0.2, max=5.0)
    valid_depth_mask = ((depth_gt > 0.2) & (depth_gt <= 5.0)).bool().detach()
        
    # Reconstruct 3D points in world coordinates
    def reconstruct_points(pose, depth):
        cam = Camera(K=K, Twc=pose).scaled(scale_factor).to(device)
        return cam.reconstruct(depth, frame='w')  # [B, 3, H, W]

    points_pred = reconstruct_points(tgt_pose_pred, depth_pred)
    points_gt = reconstruct_points(tgt_pose_gt, depth_gt)

    # Projection function
    def project(pose, points):
        cam = Camera(K=K, Twc=pose).scaled(scale_factor).to(device)
        return cam.project(points, frame='w', normalize=True)  # [B, H, W, 2]

    total_loss = 0
    valid_count = 1e-6  # Avoid division by zero

    for ref_idx in range(N_ref):
        # Get reference poses for current view
        ref_pose_p = ref_pose_pred[ref_idx]  # [1, 4, 4]
        ref_pose_g = ref_pose_gt[ref_idx]    # [1, 4, 4]

        # Project points
        proj_pred = project(ref_pose_p, points_pred)  # [1, H, W, 2]
        proj_gt = project(ref_pose_g, points_gt)     # [1, H, W, 2]

        # Calculate valid projections
        valid_proj = proj_pred.abs().max(dim=-1)[0] <= 1.0
        valid_proj &= (proj_gt.abs().max(dim=-1)[0] <= 1.0)    
        valid = valid_depth_mask.squeeze(1) & valid_proj  # [1, H, W]

        # Calculate Huber loss
        error = F.huber_loss(proj_pred, proj_gt, reduction='none').mean(-1)
        total_loss += (error * valid).sum()
        valid_count += valid.sum()

        # Visualize first sample in batch and first reference view
        if ref_idx == 0:
            visualize_reproj_error(
                ref_img[ref_idx],
                proj_pred.detach().cpu().numpy(),
                proj_gt.detach().cpu().numpy(),
                valid.bool().cpu().numpy()
            )

    return total_loss / valid_count

def compute_rpe(true_poses, perturbed_poses):
    """Compute Relative Pose Error (RPE) between true and perturbed absolute poses"""
    rpe_trans = []
    rpe_rot = []
    for true_pose, perturbed_pose in zip(true_poses, perturbed_poses):
        # Compute relative transformation: T_rel = T_true^{-1} @ T_perturbed
        rel_transform = torch.inverse(true_pose) @ perturbed_pose
        
        # Translation error
        trans_error = torch.norm(rel_transform[:3, 3])
        rpe_trans.append(trans_error.item())
        
        # Rotation error (angle in degrees)
        trace = torch.trace(rel_transform[:3, :3])
        rot_error = torch.rad2deg(torch.acos((trace - 1) / 2))
        rpe_rot.append(rot_error.item())
    
    return np.array(rpe_trans), np.array(rpe_rot)

def compute_ate(true_poses, perturbed_poses):
    """Compute Absolute Trajectory Error (ATE) between true and perturbed absolute poses"""
    # Extract positions
    true_pos = true_poses[:, :3, 3].cpu().numpy()
    perturbed_pos = perturbed_poses[:, :3, 3].cpu().numpy()
    
    # Compute ATE as the Euclidean distance between corresponding positions
    ate = np.linalg.norm(true_pos - perturbed_pos, axis=1)
    return ate

def plot_trajectories(true_poses, perturbed_poses):
    """Plot true and perturbed trajectories in 3D"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract positions
    true_pos = true_poses[:, :3, 3].cpu().numpy()
    perturbed_pos = perturbed_poses[:, :3, 3].cpu().numpy()
    
    # Plot trajectories
    ax.plot(true_pos[:, 0], true_pos[:, 1], true_pos[:, 2], label='True Trajectory', color='b')
    ax.plot(perturbed_pos[:, 0], perturbed_pos[:, 1], perturbed_pos[:, 2], label='Perturbed Trajectory', color='r')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

    
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

def test_trajectory(true_poses, perturbed_poses):
    # Convert lists to tensors
    true_poses = torch.stack(true_poses)
    perturbed_poses = torch.stack(perturbed_poses)
    
    rpe_trans, rpe_rot = compute_rpe(true_poses, perturbed_poses)
    ate = compute_ate(true_poses, perturbed_poses)
    
    print("Relative Pose Error (RPE) Statistics:")
    print(f"Translation RPE (mean ± std): {np.mean(rpe_trans):.4f} ± {np.std(rpe_trans):.4f} meters")
    print(f"Rotation RPE (mean ± std): {np.mean(rpe_rot):.4f} ± {np.std(rpe_rot):.4f} degrees")
    
    print("\nAbsolute Trajectory Error (ATE) Statistics:")
    print(f"ATE (mean ± std): {np.mean(ate):.4f} ± {np.std(ate):.4f} meters")
    
    plot_trajectories(true_poses, perturbed_poses)

@torch.no_grad()
def test_cost_warping(K, poseC2W, tgt_poseC2W, depth, img_ref, tgt_img, scale_factor=1.0): 
    cam = Camera(K=K.float(), Twc=tgt_poseC2W).scaled(scale_factor).to(device) # tcw = Identity
    ref_cam = Camera(K=K.float(), Twc=poseC2W).scaled(scale_factor).to(device)
        
    # Reconstruct world points from target_camera
    world_points = cam.reconstruct(depth, frame='w')
    # Project world points onto reference camera
    ref_coords = ref_cam.project(world_points, frame='w', normalize=True) #(b, h, w,2)
           
    warped_ref1 = F.grid_sample(img_ref.permute(0, 3, 1, 2), ref_coords, 
                                mode='bilinear', padding_mode='zeros', align_corners=True) # (b, c, h, w)
        
    #visualize images
    tgt_img_np = (tgt_img[0].cpu().numpy() * 255).astype(np.uint8)
    warped_ref1_np = (warped_ref1.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
    tgt_img_bgr = cv2.cvtColor(tgt_img_np, cv2.COLOR_RGB2BGR)
    warped_ref1_bgr = cv2.cvtColor(warped_ref1_np, cv2.COLOR_RGB2BGR)
        
    overlay_ref1 = cv2.addWeighted(tgt_img_bgr, 0.5, warped_ref1_bgr, 0.5, 0)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(tgt_img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Target Image")
    axes[0].axis("off")

    # Target + Warped Ref Image 1
    axes[1].imshow(cv2.cvtColor(overlay_ref1, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Overlay: Target + Warped Ref Image 1")
    axes[1].axis("off")
        
    plt.tight_layout()
    plt.show()
    

def test_cost_warping_with_poses(K, tgt_poseC2W, ref_poseC2W, perturbed_tgt_poseC2W, 
                                 perturbed_ref_poseC2W, depth, img_ref, tgt_img, scale_factor=1.0):
    device = depth.device

    def visualize_warp(cam, ref_cam, title):
        # Reconstruct world points from target camera
        world_points = cam.reconstruct(depth, frame='w')
        # Project world points onto reference camera
        ref_coords = ref_cam.project(world_points, frame='w', normalize=True)
        
        # Warp the reference image
        warped_ref = F.grid_sample(img_ref.permute(0, 3, 1, 2), ref_coords, 
                                   mode='bilinear', padding_mode='zeros', align_corners=True)
        
        # Convert to numpy for visualization
        tgt_img_np = (tgt_img[0].cpu().numpy() * 255).astype(np.uint8)
        warped_ref_np = (warped_ref.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        # Convert to BGR for OpenCV operations
        tgt_img_bgr = cv2.cvtColor(tgt_img_np, cv2.COLOR_RGB2BGR)
        warped_ref_bgr = cv2.cvtColor(warped_ref_np, cv2.COLOR_RGB2BGR)
        
        overlay = cv2.addWeighted(tgt_img_bgr, 0.5, warped_ref_bgr, 0.5, 0)
        
        return tgt_img_bgr, warped_ref_bgr, overlay

    # GT poses
    cam_gt = Camera(K=K.float(), Twc=tgt_poseC2W).scaled(scale_factor).to(device)
    ref_cam_gt = Camera(K=K.float(), Twc=ref_poseC2W).scaled(scale_factor).to(device)
    tgt_img_bgr_gt, warped_ref_bgr_gt, overlay_gt = visualize_warp(cam_gt, ref_cam_gt, "GT Poses")

    # Perturbed poses
    cam_perturbed = Camera(K=K.float(), Twc=perturbed_tgt_poseC2W).scaled(scale_factor).to(device)
    ref_cam_perturbed = Camera(K=K.float(), Twc=perturbed_ref_poseC2W).scaled(scale_factor).to(device)
    tgt_img_bgr_perturbed, warped_ref_bgr_perturbed, overlay_perturbed = visualize_warp(cam_perturbed, ref_cam_perturbed, "Perturbed Poses")

    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    
    # GT Results
    axes[0, 0].imshow(cv2.cvtColor(tgt_img_bgr_gt, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Target Image (GT)")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(cv2.cvtColor(warped_ref_bgr_gt, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Warped Ref Image (GT)")
    axes[0, 1].axis("off")
    
    axes[0, 2].imshow(cv2.cvtColor(overlay_gt, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("Overlay (GT)")
    axes[0, 2].axis("off")

    # Perturbed Results
    axes[1, 0].imshow(cv2.cvtColor(tgt_img_bgr_perturbed, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Target Image (Perturbed)")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(cv2.cvtColor(warped_ref_bgr_perturbed, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Warped Ref Image (Perturbed)")
    axes[1, 1].axis("off")
    
    axes[1, 2].imshow(cv2.cvtColor(overlay_perturbed, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("Overlay (Perturbed)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()
        
if __name__ == "__main__":
    # Define dataset
    dataset = SML_consistent_dataset(data_root='/media/saimouli/Data6T/datasets/VOID_150_small', mode='val')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define model
    model = midasNetModule().load_from_checkpoint(checkpoint_path="/home/saimouli/Documents/github/VI_Depth_sai/weights/total_loss=0.095.ckpt")
    model.eval()
    model.to(device)
    
    print("Model Parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    true_poses = []
    perturbed_poses = []

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
        ref_ga_depth, ref_interp, ref_gt_depth, ref_sparse_depth, tgt_pose, ref_poses, intrinsics, \
            tgt_pose_perturbed, ref_pose_perturbed = batch_data
        
        # Forward pass through the model
        true_poses.append(tgt_pose[0].cpu())
        perturbed_poses.append(tgt_pose_perturbed[0].cpu())
        
        sml_depth_inv = model(tgt_sparse_depth, tgt_img, ref_ga_depth, tgt_interp, tgt_ga_depth)
        sml_depth_inv = sml_depth_inv.detach()
        
        #K, poseC2W, tgt_poseC2W, depth, img_ref, tgt_img, scale_factor=1.0
        refined_rel_poses = [pose_to_se3(tgt_pose.inverse() @ ref_p) for ref_p in ref_poses]
        pose_ref = tgt_pose @ se3_to_pose(refined_rel_poses[0])

        test_cost_warping_with_poses(intrinsics, tgt_pose, ref_poses[0], 
                                     tgt_pose, ref_pose_perturbed[0], 
                                     utils.inv2depth(tgt_gt_depth_inv), 
                                     ref_img[0], tgt_img)
        
        compute_reproj_loss(utils.inv2depth(tgt_gt_depth_inv), utils.inv2depth(tgt_gt_depth_inv),
                            tgt_pose_perturbed, tgt_pose,
                            ref_pose_perturbed, ref_poses, intrinsics,
                            ref_img)
        
    # test_trajectory(true_poses, perturbed_poses)