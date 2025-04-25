import cv2
import numpy as np
import torch
from data.SML_consistent_dataset import SML_consistent_dataset

# Inputs
dataset = SML_consistent_dataset(data_root='/media/saimouli/Data6T/datasets/VOID_150_small', mode='val')
dataloader = torch.utils.data.DataLoader(dataset)
device = "cuda"

for batch_data in dataloader:
    batch_data = tuple(
        [
            item.to(device) if isinstance(item, torch.Tensor) else
            [
                subitem.to(device) if isinstance(subitem, torch.Tensor) else subitem
                for subitem in item
            ] if isinstance(item, list) else item
            for item in batch_data
        ]
    )
    
    # Unpack batch data
    tgt_img, tgt_gt_depth_inv, tgt_ga_depth, tgt_interp, tgt_sparse_depth, ref_img, \
        ref_ga_depth, ref_interp, ref_gt_depth, ref_sparse_depth, tgt_pose, ref_gt_pose, \
            intrinsics, _, ref_pose_perturbed, tgt_depth_pred, _ = batch_data #dataset[idx] #Cam2Wld poses (R_ctoG, p_CinG)
    
    tgt_img = torch.to_numpy(tgt_img)
    
    images = [cv2.imread(f"keyframe_{i}.png", 0) for i in range(3)]  # Grayscale, 640x480
    poses = [np.load(f"pose_{i}.npy") for i in range(3)]  # 4x4 matrices
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # Intrinsics
    sparse_points = np.load("sparse_points.npy")  # Nx3 points

    # Step 1: Project sparse points to target frame (index 0)
    sparse_2d = []
    sparse_depths = []
    target_pose_inv = np.linalg.inv(poses[0])
    for pt in sparse_points:
        pt_cam = (target_pose_inv @ np.append(pt, 1))[:3]
        if pt_cam[2] > 0:
            uv = K @ pt_cam
            uv = uv[:2] / uv[2]
            if 0 <= uv[0] < images[0].shape[1] and 0 <= uv[1] < images[0].shape[0]:
                sparse_2d.append(uv)
                sparse_depths.append(pt_cam[2])
    sparse_2d = np.array(sparse_2d)
    sparse_depths = np.array(sparse_depths)

    # Step 2: Compute disparity range
    focal_length = K[0, 0]
    baseline = np.mean([np.linalg.norm(poses[i][:3, 3] - poses[0][:3, 3]) for i in [1, 2]])
    z_min, z_max = np.min(sparse_depths), np.max(sparse_depths)
    disp_min = focal_length * baseline / z_max
    disp_max = focal_length * baseline / z_min
    disp_range = np.linspace(disp_min, disp_max, num=50)

    # Step 3: Initialize disparity map
    h, w = images[0].shape
    disp_map = np.random.uniform(disp_min, disp_max, (h, w))
    cost_map = np.full((h, w), np.inf)
    # Seed with sparse points
    for uv, z in zip(sparse_2d, sparse_depths):
        i, j = int(uv[1]), int(uv[0])
        if 0 <= i < h and 0 <= j < w:
            disp_map[i, j] = focal_length * baseline / z
            cost_map[i, j] = 0  # Low cost for sparse points

    # Step 4: PatchMatch iteration
    def compute_cost(i, j, disp, ref_img, view_imgs, poses, K):
        if not (0 <= i < h and 0 <= j < w):
            return np.inf
        Z = focal_length * baseline / disp
        pt_3d = np.linalg.inv(K) @ np.array([j, i, 1]) * Z
        pt_3d = np.append(pt_3d, 1) @ poses[0].T
        cost = 0
        patch_ref = ref_img[max(0, i-2):i+3, max(0, j-2):j+3]
        for view_idx in [1, 2]:
            pt_2d = K @ (poses[view_idx] @ pt_3d)[:3]
            if pt_2d[2] <= 0:
                return np.inf
            pt_2d = pt_2d[:2] / pt_2d[2]
            vi, vj = int(pt_2d[1]), int(pt_2d[0])
            if 0 <= vi < h and 0 <= vj < w:
                patch_view = view_imgs[view_idx][max(0, vi-2):vi+3, max(0, vj-2):vj+3]
                if patch_ref.shape == patch_view.shape:
                    cost += cv2.norm(patch_ref, patch_view, cv2.NORM_L2)
        return cost

    num_iterations = 3
    for iter in range(num_iterations):
        # Forward and backward pass
        for di in [1, -1]:
            for dj in [1, -1]:
                for i in range(0, h, 1 if di > 0 else -1):
                    for j in range(0, w, 1 if dj > 0 else -1):
                        # Current disparity and cost
                        curr_disp = disp_map[i, j]
                        curr_cost = cost_map[i, j]
                        # Try neighbor's disparity
                        ni, nj = i - di, j - dj
                        if 0 <= ni < h and 0 <= nj < w:
                            neighbor_disp = disp_map[ni, nj]
                            neighbor_cost = compute_cost(i, j, neighbor_disp, images[0], images, poses, K)
                            if neighbor_cost < curr_cost:
                                disp_map[i, j] = neighbor_disp
                                cost_map[i, j] = neighbor_cost
                        # Try random disparities
                        for _ in range(3):  # Test 3 random disparities
                            rand_disp = np.random.uniform(disp_min, disp_max)
                            rand_cost = compute_cost(i, j, rand_disp, images[0], images, K)
                            if rand_cost < curr_cost:
                                disp_map[i, j] = rand_disp
                                cost_map[i, j] = rand_cost

    # Step 5: Convert disparity to depth
    depth_map = np.zeros_like(disp_map)
    valid = (disp_map > disp_min) & (cost_map < np.inf)
    depth_map[valid] = focal_length * baseline / disp_map[valid]
    depth_map[~valid] = 0

    # Step 6: Save and visualize
    #np.save("dense_depth.npy", depth_map)
    import matplotlib.pyplot as plt
    plt.imshow(depth_map, cmap='jet')
    plt.colorbar()
    plt.show()