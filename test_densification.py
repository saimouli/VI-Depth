import cv2
import numpy as np
import torch
from data.SML_consistent_dataset import SML_consistent_dataset
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import modules.midas.utils as utils
from accelerated_features.modules.xfeat import XFeat

# --- Helper Functions ---

def project_sparse_points(sparse_depth_map, K):
    """Projects sparse depth points into a list of (u, v, depth)."""
    h, w = sparse_depth_map.shape
    ys, xs = np.nonzero(sparse_depth_map > 0)
    depths = sparse_depth_map[ys, xs]
    uv_depths = np.stack([xs, ys, depths], axis=-1)
    return uv_depths # Shape: (N, 3)

def calculate_disparity_range(sparse_uv_depths, K, baseline, min_depth=0.1, max_depth=10.0):
    """Calculates the min/max disparity based on sparse points."""
    if len(sparse_uv_depths) == 0:
        # Default range if no sparse points
        focal_length = K[0, 0]
        disp_min = focal_length * baseline / max_depth
        disp_max = focal_length * baseline / min_depth
        return disp_min, disp_max

    depths = sparse_uv_depths[:, 2]
    valid_depths = depths[(depths >= min_depth) & (depths <= max_depth)]
    if len(valid_depths) == 0:
        focal_length = K[0, 0]
        disp_min = focal_length * baseline / max_depth
        disp_max = focal_length * baseline / min_depth
        return disp_min, disp_max

    z_min, z_max = np.min(valid_depths), np.max(valid_depths)
    focal_length = K[0, 0] # Use fx
    disp_min = focal_length * baseline / z_max
    disp_max = focal_length * baseline / min_depth # Use min_depth for max disparity
    print(f"Calculated Disparity Range: min={disp_min:.2f}, max={disp_max:.2f} (from Z_min={z_min:.2f}, Z_max={z_max:.2f})")
    # Clamp range slightly to avoid extreme values if needed
    disp_min = max(0.1, disp_min)
    disp_max = min(200, disp_max) # Adjust max disparity limit if needed
    return disp_min, disp_max

def compute_patchmatch_cost(y, x, disp, tgt_img_gray, ref_imgs_gray, tgt_pose, ref_poses, K, baseline, patch_size=5):
    """Computes the photometric cost (SSD) for a given disparity."""
    h, w = tgt_img_gray.shape
    half_patch = patch_size // 2

    # Boundary check for patch extraction
    if not (half_patch <= y < h - half_patch and half_patch <= x < w - half_patch):
        return np.inf

    # --- Calculate 3D point in target camera frame ---
    focal_length = K[0, 0]
    if disp <= 1e-6: # Avoid division by zero or negative disparity
        return np.inf
    Z = focal_length * baseline / disp
    X = (x - K[0, 2]) * Z / K[0, 0]
    Y = (y - K[1, 2]) * Z / K[1, 1]
    pt_3d_tgt = np.array([X, Y, Z, 1.0]) # Homogeneous coordinates

    # --- Extract target patch ---
    patch_tgt = tgt_img_gray[y - half_patch : y + half_patch + 1,
                             x - half_patch : x + half_patch + 1]

    total_cost = 0
    valid_views = 0

    # --- Project to each reference view and compute cost ---
    for ref_idx, ref_img_gray in enumerate(ref_imgs_gray):
        # Relative pose from target to reference view k
        # T_refk_wld = ref_poses[ref_idx]
        # T_wld_tgt = np.linalg.inv(tgt_pose)
        # T_refk_tgt = T_refk_wld @ T_wld_tgt # Pose of target frame in ref frame k's coordinate system
        T_tgt_refk = np.linalg.inv(ref_poses[ref_idx]) @ tgt_pose # Pose of ref frame k in target frame's coordinate system

        # Transform 3D point from target frame to reference frame k
        pt_3d_refk = T_tgt_refk @ pt_3d_tgt
        pt_3d_refk = pt_3d_refk[:3] / pt_3d_refk[3] # Back to non-homogeneous

        # Project 3D point in ref frame k onto its image plane
        if pt_3d_refk[2] <= 0: # Point is behind the camera
            continue

        uv_refk_hom = K @ pt_3d_refk
        if uv_refk_hom[2] <= 1e-6: # Avoid division by zero
             continue
        uv_refk = uv_refk_hom[:2] / uv_refk_hom[2]

        # Check if projected point is within reference image bounds
        ref_x, ref_y = uv_refk[0], uv_refk[1]
        if not (half_patch <= ref_y < h - half_patch and half_patch <= ref_x < w - half_patch):
            continue

        # --- Extract reference patch using bilinear interpolation ---
        # (More robust than direct indexing for subpixel accuracy)
        ref_y_floor, ref_x_floor = int(ref_y), int(ref_x)
        ref_y_ceil, ref_x_ceil = ref_y_floor + 1, ref_x_floor + 1

        # Check bounds again after floor/ceil
        if not (half_patch <= ref_y_floor and ref_y_ceil < h - half_patch and \
                half_patch <= ref_x_floor and ref_x_ceil < w - half_patch):
            continue

        patch_ref = ref_img_gray[ref_y_floor - half_patch : ref_y_ceil + half_patch,
                                 ref_x_floor - half_patch : ref_x_ceil + half_patch]

        # Simple nearest neighbor patch extraction for now (can be improved with interpolation)
        # patch_ref = ref_img_gray[int(ref_y) - half_patch : int(ref_y) + half_patch + 1,
        #                          int(ref_x) - half_patch : int(ref_x) + half_patch + 1]

        # --- Compute cost (SSD) ---
        if patch_tgt.shape == patch_ref.shape:
            cost = np.sum((patch_tgt.astype(np.float32) - patch_ref.astype(np.float32))**2)
            total_cost += cost
            valid_views += 1
        # else:
        #     print(f"Shape mismatch: tgt {patch_tgt.shape}, ref {patch_ref.shape} at ({ref_y:.1f}, {ref_x:.1f})")


    if valid_views == 0:
        return np.inf

    return total_cost / valid_views

def patchmatch_stereo(tgt_img_gray, ref_imgs_gray, tgt_pose, ref_poses, K, sparse_uv_depths,
                      patch_size=5, num_iterations=3, num_random_samples=5):
    """Performs PatchMatch stereo."""
    h, w = tgt_img_gray.shape
    baseline = np.mean([np.linalg.norm(ref_poses[i][:3, 3] - tgt_pose[:3, 3]) for i in range(len(ref_imgs_gray))])
    if baseline < 1e-6:
        print("Warning: Baseline is near zero. Using a small default.")
        baseline = 0.1 # Assign a small default baseline

    disp_min, disp_max = calculate_disparity_range(sparse_uv_depths, K, baseline)
    print(f"Using baseline: {baseline:.4f}, Disparity range: [{disp_min:.4f}, {disp_max:.4f}]")

    # Initialize disparity map and cost map
    disp_map = np.random.uniform(disp_min, disp_max, (h, w)).astype(np.float32)
    cost_map = np.full((h, w), np.inf, dtype=np.float32)

    # Seed disparity map with sparse points
    for u, v, depth in sparse_uv_depths:
        i, j = int(v), int(u) # row (y), col (x)
        if 0 <= i < h and 0 <= j < w and depth > 0:
            disp = K[0, 0] * baseline / depth
            if disp_min <= disp <= disp_max:
                disp_map[i, j] = disp
                # Compute initial cost for sparse points
                cost_map[i, j] = compute_patchmatch_cost(i, j, disp, tgt_img_gray, ref_imgs_gray, tgt_pose, ref_poses, K, baseline, patch_size)
            # else:
            #     print(f"Warning: Sparse point disparity {disp:.2f} out of range [{disp_min:.2f}, {disp_max:.2f}] at ({i},{j})")


    # --- PatchMatch iterations ---
    for iter_num in tqdm(range(num_iterations), desc="PatchMatch Iterations"):
        # Determine scanline order (alternating directions)
        if iter_num % 2 == 0:
            y_range = range(h)
            x_range = range(w)
            neighbors = [(-1, 0), (0, -1)] # Up, Left
        else:
            y_range = range(h - 1, -1, -1)
            x_range = range(w - 1, -1, -1)
            neighbors = [(1, 0), (0, 1)] # Down, Right

        for y in y_range:
            for x in x_range:
                current_disp = disp_map[y, x]
                current_cost = cost_map[y, x]

                # --- 1. Spatial Propagation ---
                best_disp_prop = current_disp
                min_cost_prop = current_cost

                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        neighbor_disp = disp_map[ny, nx]
                        cost_prop = compute_patchmatch_cost(y, x, neighbor_disp, tgt_img_gray, ref_imgs_gray, tgt_pose, ref_poses, K, baseline, patch_size)

                        if cost_prop < min_cost_prop:
                            min_cost_prop = cost_prop
                            best_disp_prop = neighbor_disp

                # --- 2. Random Refinement ---
                best_disp_rand = best_disp_prop
                min_cost_rand = min_cost_prop

                # Exponentially decreasing search range (optional, can be fixed)
                # current_range = (disp_max - disp_min) * (0.5 ** iter_num)
                current_range = (disp_max - disp_min) * 0.1 # Fixed smaller range

                for _ in range(num_random_samples):
                    # Sample around the current best disparity
                    rand_disp_offset = (np.random.rand() - 0.5) * current_range
                    rand_disp = np.clip(best_disp_rand + rand_disp_offset, disp_min, disp_max)

                    cost_rand = compute_patchmatch_cost(y, x, rand_disp, tgt_img_gray, ref_imgs_gray, tgt_pose, ref_poses, K, baseline, patch_size)

                    if cost_rand < min_cost_rand:
                        min_cost_rand = cost_rand
                        best_disp_rand = rand_disp

                # --- Update pixel's disparity and cost ---
                disp_map[y, x] = best_disp_rand
                cost_map[y, x] = min_cost_rand

    # --- Convert final disparity map to depth ---
    depth_map = np.zeros_like(disp_map)
    valid_disp = (disp_map > 1e-6) & (cost_map < np.inf) # Check for valid disparity
    depth_map[valid_disp] = K[0, 0] * baseline / disp_map[valid_disp]
    depth_map[~valid_disp] = 0 # Set invalid depths to 0

    return depth_map, disp_map, cost_map


def createPlaneSweepStereo(target_idx, images, poses, K, depth_range=(0.5, 10.0), 
                           num_depths=64, sparse_depths=None, window_size=5):
    """
    Faster and more accurate plane sweep stereo implementation
    
    Args:
        target_idx (int): Index of the target frame
        images (list): List of keyframe images
        poses (list): List of 4x4 camera poses (world to camera)
        K (np.ndarray): 3x3 camera intrinsic matrix
        depth_range (tuple): (min_depth, max_depth) for depth hypotheses
        num_depths (int): Number of depth planes to sweep
        sparse_depths (dict, optional): Sparse depth points to use as priors
        window_size (int): Window size for NCC computation
    
    Returns:
        np.ndarray: Dense depth map and confidence map
    """
    # Convert images to grayscale if necessary
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img for img in images]
    
    height, width = images[target_idx].shape
    min_depth, max_depth = depth_range
    
    # Adaptive depth sampling (more samples at closer depths)
    depth_planes = min_depth + (max_depth - min_depth) * (np.arange(num_depths) / (num_depths - 1)) ** 2
    
    # Initialize cost volume and confidence maps
    cost_volume = np.ones((height, width, num_depths), dtype=np.float32) * np.inf
    
    # Camera intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Target pose and its inverse
    T_target = poses[target_idx]
    T_target_inv = np.linalg.inv(T_target)
    
    # Precompute pixel grid
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    pixels = np.stack([x_grid.flatten(), y_grid.flatten(), np.ones_like(x_grid.flatten())], axis=1)
    
    # Target image with padding for efficient window operations
    pad = window_size // 2
    tgt_padded = np.pad(images[target_idx], pad, mode='reflect')
    
    # Pre-compute normalized patches for target image (for NCC)
    tgt_patches = np.zeros((height, width, window_size*window_size), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            patch = tgt_padded[y:y+window_size, x:x+window_size].flatten()
            patch = patch - np.mean(patch)
            norm = np.linalg.norm(patch)
            if norm > 0:
                patch = patch / norm
            tgt_patches[y, x] = patch
    
    # For each source view
    for src_idx, (src_img, src_pose) in enumerate(zip(images, poses)):
        if src_idx == target_idx:
            continue
        
        # Relative pose (source to target)
        T_rel = src_pose @ T_target_inv
        
        # Pad source image
        src_padded = np.pad(src_img, pad, mode='reflect')
        
        # For each depth plane
        for d_idx, depth in enumerate(depth_planes):
            # Back-project target pixels to 3D at current depth
            points_3d = np.zeros((len(pixels), 4))
            points_3d[:, 0] = (pixels[:, 0] - cx) * depth / fx
            points_3d[:, 1] = (pixels[:, 1] - cy) * depth / fy
            points_3d[:, 2] = depth
            points_3d[:, 3] = 1.0
            
            # Project to source view
            points_src = (T_rel @ points_3d.T).T
            
            # Get source image coordinates
            z_src = points_src[:, 2]
            valid = z_src > 0.1
            
            u_src = (points_src[valid, 0] / z_src[valid] * fx + cx)
            v_src = (points_src[valid, 1] / z_src[valid] * fy + cy)
            
            # Check bounds
            in_bounds = (u_src >= pad) & (u_src < width - pad) & (v_src >= pad) & (v_src < height - pad)
            
            if not np.any(in_bounds):
                continue
                
            u_src = u_src[in_bounds]
            v_src = v_src[in_bounds]
            valid_indices = np.where(valid)[0][in_bounds]
            
            # Compute NCC for valid pixels
            for i, idx in enumerate(valid_indices):
                x, y = pixels[idx, 0], pixels[idx, 1]
                
                # Get source patch
                u_s, v_s = int(u_src[i]), int(v_src[i])
                src_patch = src_padded[v_s:v_s+window_size, u_s:u_s+window_size].flatten()
                src_patch = src_patch - np.mean(src_patch)
                src_norm = np.linalg.norm(src_patch)
                
                if src_norm > 0:
                    src_patch = src_patch / src_norm
                    # Compute NCC (dot product of normalized patches)
                    ncc = np.sum(tgt_patches[y, x] * src_patch)
                    # Convert NCC to cost (lower is better)
                    cost = 1.0 - ncc
                    
                    # Update cost volume if better match found
                    if cost < cost_volume[y, x, d_idx]:
                        cost_volume[y, x, d_idx] = cost
    
    # Find minimum cost and corresponding depth
    min_cost = np.min(cost_volume, axis=2)
    min_cost_idx = np.argmin(cost_volume, axis=2)
    depth_map = depth_planes[min_cost_idx]
    
    # Subpixel refinement
    refined_depth = np.copy(depth_map)
    for y in range(height):
        for x in range(width):
            idx = min_cost_idx[y, x]
            if idx > 0 and idx < num_depths - 1:
                c0 = cost_volume[y, x, idx-1]
                c1 = cost_volume[y, x, idx]
                c2 = cost_volume[y, x, idx+1]
                
                # Fit parabola
                if c0 != np.inf and c1 != np.inf and c2 != np.inf:
                    offset = 0.5 * (c0 - c2) / (c0 - 2*c1 + c2 + 1e-10)
                    if abs(offset) < 1:
                        refined_depth[y, x] = depth_planes[idx] + offset * (depth_planes[idx+1] - depth_planes[idx])
    
    # Confidence map based on cost curve shape
    confidence = np.zeros_like(depth_map)
    valid_mask = min_cost != np.inf
    
    # Calculate confidence: sharper cost curves = higher confidence
    for y in range(height):
        for x in range(width):
            if valid_mask[y, x]:
                # Use variance around minimum as confidence measure
                idx = min_cost_idx[y, x]
                start_idx = max(0, idx - 2)
                end_idx = min(num_depths, idx + 3)
                local_costs = cost_volume[y, x, start_idx:end_idx]
                valid_costs = local_costs[local_costs != np.inf]
                if len(valid_costs) > 1:
                    confidence[y, x] = 1.0 / (np.std(valid_costs) + 1e-5)
    
    # Normalize confidence
    if np.max(confidence) > 0:
        confidence = confidence / np.max(confidence)
    
    # Incorporate sparse depth points as strong priors
    if sparse_depths is not None:
        for u, v, d in sparse_depths:
            u, v = int(u), int(v)
            if 0 <= v < height and 0 <= u < width:
                refined_depth[v, u] = d
                confidence[v, u] = 1.0  # Highest confidence
    
    # Filter the depth map to reduce noise while preserving edges
    target_gray = images[target_idx].astype(np.float32) / 255.0
    
    # Edge-aware bilateral filter
    filtered_depth = cv2.bilateralFilter(
        (refined_depth * valid_mask).astype(np.float32), 
        d=7,  # Diameter of each pixel neighborhood
        sigmaColor=0.1,  # Filter sigma in the color space
        sigmaSpace=5.0   # Filter sigma in the coordinate space
    )
    
    # Keep known good depths (from sparse points)
    if sparse_depths is not None:
        for u, v, d in sparse_depths:
            u, v = int(u), int(v)
            if 0 <= v < height and 0 <= u < width:
                filtered_depth[v, u] = d
    
    return filtered_depth, confidence


def get_refined_depth_hypotheses(u, v, sparse_points, default_range=(0.2, 5.0), num_samples=30):
    """
    Generate refined depth hypotheses based on spatial and depth relationships
    
    Args:
        u, v: Coordinates of the point we want to estimate depth for
        sparse_points: List of existing VIO points [(u, v, depth), ...]
        default_range: Default depth range if no nearby points found
        num_samples: Number of depth samples to generate
        
    Returns:
        np.ndarray: Array of depth hypotheses
    """
    if not sparse_points:
        # No points at all, use inverse depth sampling in default range
        min_inv_depth = default_range[1]
        max_inv_depth = default_range[0]
        inv_depths = np.linspace(min_inv_depth, max_inv_depth, num_samples)
        return 1.0 / inv_depths
    
    # Calculate distances and collect depths
    distances = []
    depths = []
    for pu, pv, pd in sparse_points:
        dist = np.sqrt((u - pu)**2 + (v - pv)**2)
        distances.append(dist)
        depths.append(pd)
    
    # Convert to numpy arrays
    distances = np.array(distances)
    depths = np.array(depths)
    
    # Weight nearby points more heavily
    max_dist = 100  # Maximum distance to consider
    nearby_indices = distances < max_dist
    
    if not np.any(nearby_indices):
        # No nearby points, use global statistics
        mean_depth = np.mean(depths)
        std_depth = np.std(depths)
        range_min = max(0.1, mean_depth - 3*std_depth)
        range_max = mean_depth + 3*std_depth
        
        # Use inverse depth sampling for better near-depth resolution
        min_inv_depth = range_max
        max_inv_depth = range_min
        inv_depths = np.linspace(min_inv_depth, max_inv_depth, num_samples)
        return inv_depths
    
    # Get nearby depths
    nearby_depths = depths[nearby_indices]
    nearby_distances = distances[nearby_indices]
    
    # Weight by inverse squared distance - closer points have more influence
    weights = 1.0 / (nearby_distances**2 + 1e-6)
    weights = weights / np.sum(weights)  # Normalize weights
    
    # Calculate weighted mean and std
    weighted_mean = np.sum(nearby_depths * weights)
    weighted_var = np.sum(weights * (nearby_depths - weighted_mean)**2)
    weighted_std = np.sqrt(weighted_var)
    
    # Define range with adaptive width based on confidence
    confidence = min(1.0, 10.0 / len(nearby_depths))  # More points = more confidence
    range_width = max(0.3, weighted_std * (3.0 + 2.0 * confidence))
    
    range_min = max(0.1, weighted_mean - range_width)
    range_max = weighted_mean + range_width
    
    # Create hybrid sampling - concentrated + uniform
    # 1. Concentrated samples around weighted mean
    concentrated_count = int(num_samples * 0.7)
    concentrated_std = weighted_std * 0.5  # Tighter distribution
    concentrated_samples = np.random.normal(
        weighted_mean, 
        concentrated_std, 
        concentrated_count
    )
    concentrated_samples = np.clip(concentrated_samples, range_min, range_max)
    
    # 2. Uniform samples across range
    uniform_count = num_samples - concentrated_count
    uniform_samples = np.linspace(range_min, range_max, uniform_count)
    
    # 3. Create a small number of samples at exact depths of very nearby points
    exact_count = min(3, len(nearby_depths))
    if exact_count > 0:
        # Use depths from closest points
        closest_indices = np.argsort(nearby_distances)[:exact_count]
        exact_samples = nearby_depths[closest_indices]
        
        # Combine all samples
        all_samples = np.concatenate([
            concentrated_samples, 
            uniform_samples,
            exact_samples
        ])
    else:
        all_samples = np.concatenate([concentrated_samples, uniform_samples])
    
    # Sort and remove duplicates
    all_samples = np.sort(np.unique(all_samples))
    
    # Add fine-grained samples around most likely depth
    fine_count = min(5, num_samples // 10)
    if fine_count > 0 and len(nearby_depths) > 0:
        # Find closest point's depth
        closest_idx = np.argmin(nearby_distances)
        closest_depth = nearby_depths[closest_idx]
        
        # Add fine samples around it
        fine_range = closest_depth * 0.05  # 5% range
        fine_samples = np.linspace(
            closest_depth - fine_range,
            closest_depth + fine_range,
            fine_count
        )
        
        # Add to collection
        all_samples = np.sort(np.concatenate([all_samples, fine_samples]))
    
    # If we have too many samples, downsample while preserving distribution shape
    if len(all_samples) > num_samples:
        # Keep endpoints and sample the middle part
        indices = np.linspace(0, len(all_samples) - 1, num_samples).astype(int)
        all_samples = all_samples[indices]
    
    return all_samples

def densify_sparse_vio_points(target_img, ref_imgs, target_pose, ref_poses, K, sparse_points, num_new_points=200, patch_size=7):
    height, width = target_img.shape

    # Extract existing points
    existing_points = []
    for u, v, d in sparse_points:
        existing_points.append((int(u), int(v)))

    # Identify regions that need more points (using grid-based approach)
    grid_size = 16  # Size of grid cell
    grid_rows = height // grid_size
    grid_cols = width // grid_size

    # Create grid occupancy map
    grid_map = np.zeros((grid_rows, grid_cols), dtype=np.int32)

    for u, v in existing_points:
        grid_r = min(v // grid_size, grid_rows -1)
        grid_c = min(u // grid_size, grid_cols -1)
        grid_map[grid_r, grid_c] += 1

    # Find empty or low-density cells
    empty_cells = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            if grid_map[r, c] == 0:
                # Center of the cell
                cell_center_v = r * grid_size + grid_size // 2
                cell_center_u = c * grid_size + grid_size // 2
                if 0 <= cell_center_v < height and 0 <= cell_center_u < width:
                    empty_cells.append((cell_center_u, cell_center_v))

    # If we have more potential points than needed, prioritize
    # cells with high gradient (likely to have good features)
    if len(empty_cells) > num_new_points:
        # Compute image gradients
        grad_x = cv2.Sobel(target_img, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(target_img, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Score each empty cell based on gradient magnitude
        cell_scores = []
        for u, v in empty_cells:
            # Get gradient in a small window around point
            window_size = 3
            r_min = max(0, v - window_size//2)
            r_max = min(height, v + window_size//2 + 1)
            c_min = max(0, u - window_size//2)
            c_max = min(width, u + window_size//2 + 1)

            window_grad = grad_mag[r_min:r_max, c_min:c_max]
            score = np.mean(window_grad) if window_grad.size > 0 else 0
            cell_scores.append((score, (u,v)))
        
        #sort by score and take top N
        cell_scores.sort(reverse=True)
        candidate_points = [pt for _, pt in cell_scores[:num_new_points]]
    else:
        candidate_points = empty_cells

    # Camera intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Target pose inverse
    T_target_inv = np.linalg.inv(target_pose)

    # Prepare reference images with padding
    pad = patch_size // 2
    ref_imgs_padded = [np.pad(img, pad, mode='reflect') for img in ref_imgs]
    target_img_padded = np.pad(target_img, pad, mode='reflect')
        
    new_points = []
    for u, v in candidate_points:
        # Skip points too close to the border for patch extraction
        if u < pad or u >= width - pad or v < pad or v >= height - pad:
            continue
        
        # Get target patch
        target_patch = target_img_padded[v:v+patch_size, u:u+patch_size].flatten()
        target_patch_norm = target_patch - np.mean(target_patch)
        target_norm = np.linalg.norm(target_patch_norm)
        if target_norm < 1e-6:
            continue
        target_patch_norm = target_patch_norm / target_norm

        # Define depth hypotheses - try only a few depths for speed
        # Use the depths of nearby existing points as guide
        # nearby_depths = []
        # for pu, pv, pd in sparse_points:
        #     dist = np.sqrt((u - pu)**2 + (v - pv)**2)
        #     if dist < 50:  # Consider points within 50 pixels
        #         nearby_depths.append(pd)
        
        # if not nearby_depths:
        #     # If no nearby points, use a default range
        #     depth_hypotheses = np.linspace(0.2, 0, 5)
        # else:
        #     # Use nearby depths as guidance
        #     mean_depth = np.mean(nearby_depths)
        #     std_depth = max(0.2, np.std(nearby_depths))
        #     depth_hypotheses = np.linspace(
        #         max(0.1, mean_depth - 2*std_depth),
        #         mean_depth + 2*std_depth,
        #         10)
        depth_hypotheses = get_refined_depth_hypotheses(
            u, v, sparse_points, 
            default_range=(0.2, 5.0), 
            num_samples=500  # Use more samples for better precision
        )

        best_depth = None
        best_cost = float('inf')
        # For each depth hypothesis
        for depth in depth_hypotheses:
            # Back-project to 3D
            X = (u - cx) * depth / fx
            Y = (v - cy) * depth / fy
            Z = depth
            point_3d = np.array([X, Y, Z, 1.0])
            
            # Project to reference views
            costs = []
            
            for ref_idx, (ref_img_padded, ref_pose) in enumerate(zip(ref_imgs_padded, ref_poses)):
                # Transform point to reference frame
                T_ref = ref_pose
                T_rel = T_ref @ T_target_inv
                point_ref = T_rel @ point_3d
                
                # Skip if point is behind camera
                if point_ref[2] <= 0.1:
                    continue
                
                # Project to reference image
                u_ref = point_ref[0] / point_ref[2] * fx + cx
                v_ref = point_ref[1] / point_ref[2] * fy + cy
                
                # Skip if outside image bounds (accounting for padding)
                if (u_ref < pad or u_ref >= width - pad or 
                    v_ref < pad or v_ref >= height - pad):
                    continue
                
                # Extract reference patch with bilinear interpolation
                u0, v0 = int(u_ref), int(v_ref)
                wu, wv = u_ref - u0, v_ref - v0
                
                patch = np.zeros(patch_size * patch_size)
                for i in range(patch_size):
                    for j in range(patch_size):
                        y = v0 + i
                        x = u0 + j
                        p00 = ref_img_padded[y, x]
                        p01 = ref_img_padded[y, x+1]
                        p10 = ref_img_padded[y+1, x]
                        p11 = ref_img_padded[y+1, x+1]
                        
                        interp = (1-wu)*(1-wv)*p00 + wu*(1-wv)*p01 + \
                                 (1-wu)*wv*p10 + wu*wv*p11
                        
                        patch[i*patch_size + j] = interp
                
                # Normalize patch
                patch_norm = patch - np.mean(patch)
                ref_norm = np.linalg.norm(patch_norm)
                if ref_norm < 1e-6:
                    continue
                
                patch_norm = patch_norm / ref_norm
                
                # Compute NCC
                ncc = np.sum(target_patch_norm * patch_norm)
                cost = 1.0 - ncc
                
                costs.append(cost)
            
            # Aggregate costs
            if costs:
                avg_cost = np.mean(costs)
                if avg_cost < best_cost:
                    best_cost = avg_cost
                    best_depth = depth
        
        # Add point if match is good enough
        if best_depth is not None and best_cost < 0.3:  # Threshold for good matches
            new_points.append((u, v, best_depth))
            
            # Stop if we've reached the desired number of new points
            if len(new_points) >= num_new_points:
                break
    
    # Combine original points with new points
    return sparse_points,  new_points


def densify_vio_with_xfeat_triangulation(ref1_img, target_img, ref2_img, ref1_pose, target_pose, ref2_pose, K, 
                                         vio_points, xfeat, num_new_points=200):
    """
    Densify sparse VIO points by matching features across ref1-target-ref2 and triangulating
    Uses camera-to-world poses (inverts them for triangulation)
    
    Args:
        ref1_img: First reference frame
        target_img: Target frame where we want to densify depth
        ref2_img: Second reference frame
        ref1_pose: Camera-to-world pose for first reference frame (4x4)
        target_pose: Camera-to-world pose for target frame (4x4)
        ref2_pose: Camera-to-world pose for second reference frame (4x4)
        K: Camera intrinsics matrix (3x3)
        vio_points: List of existing VIO points in target frame [(u, v, depth), ...]
        xfeat: XFeat feature matching module
        num_new_points: Number of new points to add
        
    Returns:
        List of densified points in target frame [(u, v, depth), ...]

    Perform
    - Multi-view triangulation for common points
    - two-frame triangulation for non-common points
    - non linear refinement for all triangulated points
    """
    # Ensure images are in the format expected by XFeat
    if isinstance(ref1_img, torch.Tensor):
        ref1_tensor = ref1_img.permute(2, 0, 1) if ref1_img.dim() == 3 else ref1_img
    else:
        ref1_tensor = torch.from_numpy(ref1_img).permute(2, 0, 1)
        
    if isinstance(target_img, torch.Tensor):
        target_tensor = target_img.permute(2, 0, 1) if target_img.dim() == 3 else target_img
    else:
        target_tensor = torch.from_numpy(target_img).permute(2, 0, 1)
        
    if isinstance(ref2_img, torch.Tensor):
        ref2_tensor = ref2_img.permute(2, 0, 1) if ref2_img.dim() == 3 else ref2_img
    else:
        ref2_tensor = torch.from_numpy(ref2_img).permute(2, 0, 1)
    
    # Create a mask for target image to avoid detecting features near existing VIO points
    height, width = target_tensor.shape[1:3]
    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # Mark areas around existing VIO points as invalid for new features
    for u, v, _ in vio_points:
        if 0 <= int(v) < height and 0 <= int(u) < width:
            cv2.circle(mask, (int(u), int(v)), 15, 0, -1)  # 15 pixel radius
    
    # Match features: ref1 to target, and ref2 to target
    print("Matching features between reference frames and target...")
    
    # Match ref1 to target
    mkpts_ref1, mkpts_target1 = xfeat.match_xfeat(ref1_tensor, target_tensor, top_k=num_new_points*3)
    
    # Apply MAGSAC to filter matches
    H1, inlier_mask1 = cv2.findHomography(
        mkpts_ref1, mkpts_target1, 
        cv2.USAC_MAGSAC, 
        3.5, 
        maxIters=1000, 
        confidence=0.999
    )
    inlier_mask1 = inlier_mask1.flatten().astype(bool)
    
    # Filter to inliers
    mkpts_ref1 = mkpts_ref1[inlier_mask1]
    mkpts_target1 = mkpts_target1[inlier_mask1]
    
    # Match ref2 to target
    mkpts_ref2, mkpts_target2 = xfeat.match_xfeat(ref2_tensor, target_tensor, top_k=num_new_points*3)
    
    # Apply MAGSAC to filter matches
    H2, inlier_mask2 = cv2.findHomography(
        mkpts_ref2, mkpts_target2, 
        cv2.USAC_MAGSAC, 
        3.5, 
        maxIters=1000, 
        confidence=0.999
    )
    inlier_mask2 = inlier_mask2.flatten().astype(bool)
    
    # Filter to inliers
    mkpts_ref2 = mkpts_ref2[inlier_mask2]
    mkpts_target2 = mkpts_target2[inlier_mask2]
    
    print(f"Found {len(mkpts_target1)} inlier matches between ref1 and target")
    print(f"Found {len(mkpts_target2)} inlier matches between ref2 and target")
    
    # Filter out matches with target points in the mask (near existing VIO points)
    valid_matches1 = []
    for i, (_, _) in enumerate(mkpts_ref1):
        u, v = mkpts_target1[i]
        if 0 <= int(v) < height and 0 <= int(u) < width and mask[int(v), int(u)] > 0:
            valid_matches1.append(i)
    
    mkpts_ref1 = mkpts_ref1[valid_matches1]
    mkpts_target1 = mkpts_target1[valid_matches1]
    
    valid_matches2 = []
    for i, (_, _) in enumerate(mkpts_ref2):
        u, v = mkpts_target2[i]
        if 0 <= int(v) < height and 0 <= int(u) < width and mask[int(v), int(u)] > 0:
            valid_matches2.append(i)
    
    mkpts_ref2 = mkpts_ref2[valid_matches2]
    mkpts_target2 = mkpts_target2[valid_matches2]
    
    print(f"After mask filtering: {len(mkpts_target1)} ref1 matches, {len(mkpts_target2)} ref2 matches")
    
    # Find common target points between the two sets of matches
    common_points = {} # {point_id: (u_target, v_target, u_ref1, v_ref1, u_ref2, v_ref2)}
    ref1_only_points = {} # {point_id: (u_target, v_target, u_ref1, v_ref1)}
    ref2_only_points = {} # {point_id: (u_target, v_target, u_ref2, v_ref2)}
    next_point_id = 0
    
    for i, (u_target1, v_target1) in enumerate(mkpts_target1):
        u_ref1, v_ref1 = mkpts_ref1[i]
        
        # Look for a match in the second set
        for j, (u_target2, v_target2) in enumerate(mkpts_target2):
            # Check if it's the same target point (within a small threshold)
            if np.sqrt((u_target1 - u_target2)**2 + (v_target1 - v_target2)**2) < 1.0:
                u_ref2, v_ref2 = mkpts_ref2[j]
                
                # Use average position in target for more stability
                u_target = (u_target1 + u_target2) / 2
                v_target = (v_target1 + v_target2) / 2
                
                common_points[next_point_id] = (u_target, v_target, u_ref1, v_ref1, u_ref2, v_ref2)
                next_point_id += 1
                break
    
    # Then identify points visible only in ref1+target
    for i, (u_target, v_target) in enumerate(mkpts_target1):
        # Skip if this target point is already in common_points
        already_common = False
        for u, v, _, _, _, _ in common_points.values():
            if np.sqrt((u_target - u)**2 + (v_target - v)**2) < 2.0:
                already_common = True
                break
        
        if not already_common:
            u_ref1, v_ref1 = mkpts_ref1[i]
            ref1_only_points[len(ref1_only_points)] = (u_target, v_target, u_ref1, v_ref1)
    
    # Then identify points visible only in ref2+target
    for i, (u_target, v_target) in enumerate(mkpts_target2):
        # Skip if this target point is already in common_points or ref1_only_points
        already_common = False
        for u, v, _, _, _, _ in common_points.values():
            if np.sqrt((u_target - u)**2 + (v_target - v)**2) < 2.0:
                already_common = True
                break
        
        if not already_common:
            u_ref2, v_ref2 = mkpts_ref2[i]
            ref2_only_points[len(ref2_only_points)] = (u_target, v_target, u_ref2, v_ref2)

    print(f"Found {len(common_points)} common points visible in all three frames")
    print(f"Found {len(ref1_only_points)} points visible only in ref1+target")
    print(f"Found {len(ref2_only_points)} points visible only in ref2+target")

    #visualize common points
    vis_img_tgt = target_img.numpy() * 255.0
    vis_img_ref1 = ref1_img.numpy() * 255.0
    vis_img_ref2 = ref2_img.numpy() * 255.0
    vis_img_tgt = vis_img_tgt.astype(np.uint8)
    vis_img_ref1 = vis_img_ref1.astype(np.uint8)
    vis_img_ref2 = vis_img_ref2.astype(np.uint8)
    for u, v, u_ref1, v_ref1, u_ref2, v_ref2 in common_points.values():
        cv2.circle(vis_img_tgt, (int(u), int(v)), 3, (0, 255, 0), -1)
        cv2.circle(vis_img_ref1, (int(u_ref1), int(v_ref1)), 3, (0, 0, 255), -1)
        cv2.circle(vis_img_ref2, (int(u_ref2), int(v_ref2)), 3, (0, 0, 255), -1)
    cv2.imshow('Common Points tgt', vis_img_tgt)
    cv2.imshow('Common Points ref1', vis_img_ref1)
    cv2.imshow('Common Points ref2', vis_img_ref2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Convert camera-to-world poses to world-to-camera poses for triangulation
    target_pose_inv = np.linalg.inv(target_pose)
    ref1_pose_inv = np.linalg.inv(ref1_pose)
    ref2_pose_inv = np.linalg.inv(ref2_pose)

    # Prepare RT matrices for triangulation
    target_R = target_pose_inv[:3, :3]
    target_t = target_pose_inv[:3, 3].reshape(3, 1)
    target_RT = np.hstack((target_R, target_t))

    ref1_R = ref1_pose_inv[:3, :3]
    ref1_t = ref1_pose_inv[:3, 3].reshape(3, 1)
    ref1_RT = np.hstack((ref1_R, ref1_t))

    ref2_R = ref2_pose_inv[:3, :3]
    ref2_t = ref2_pose_inv[:3, 3].reshape(3, 1)
    ref2_RT = np.hstack((ref2_R, ref2_t))

    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]

    #Traingulate common points with multi-view approach
    new_points = []
    for point_id, (u_target, v_target, u_ref1, v_ref1, u_ref2, v_ref2) in common_points.items():
        # Convert to normalized coordinates
        x_target = (u_target - cx) / fx
        y_target = (v_target - cy) / fy

        x_ref1 = (u_ref1 - cx) / fx
        y_ref1 = (v_ref1 - cy) / fy

        x_ref2 = (u_ref2 - cx) / fx
        y_ref2 = (v_ref2 - cy) / fy
        
        # Create observation vectors for all 3 views
        observations = np.array([
            [x_target, y_target],
            [x_ref1, y_ref1],
            [x_ref2, y_ref2]
        ])

        proj_matrices = [
            K @ target_RT,
            K @ ref1_RT,
            K @ ref2_RT
        ]

        # Multi-view triangulation using DLT (Direct Linear Transform)
        A = np.zeros((3*2, 4))
        for i in range(3):  # 3 views
            x, y = observations[i]
            P = proj_matrices[i]

            A[i*2] = x * P[2] - P[0]
            A[i*2+1] = y * P[2] - P[1]
        
        #solve for SVD
        _, _, Vt = np.linalg.svd(A)
        point_3d = Vt[-1, :3] / Vt[-1, 3]

        #Get depth in target frame
        point_camera = target_R @ point_3d + target_t.flatten()
        depth = point_camera[2]

        #  Calculate reprojection errors in all views
        pt_proj_target = K @ (target_R @ point_3d + target_t.flatten())
        pt_proj_target = pt_proj_target / pt_proj_target[2]
        error_target = np.sqrt((pt_proj_target[0] - u_target)**2 + (pt_proj_target[1] - v_target)**2)

        pt_proj_ref1 = K @ (ref1_R @ point_3d + ref1_t.flatten())
        pt_proj_ref1 = pt_proj_ref1 / pt_proj_ref1[2]
        error_ref1 = np.sqrt((pt_proj_ref1[0] - u_ref1)**2 + (pt_proj_ref1[1] - v_ref1)**2)

        pt_proj_ref2 = K @ (ref2_R @ point_3d + ref2_t.flatten())
        pt_proj_ref2 = pt_proj_ref2 / pt_proj_ref2[2]
        error_ref2 = np.sqrt((pt_proj_ref2[0] - u_ref2)**2 + (pt_proj_ref2[1] - v_ref2)**2)
        
        avg_error = (error_target + error_ref1 + error_ref2) / 3

        # Add to list if it's a good triangulation
        if 0.1 < depth < 5.0 and avg_error < 1.0:
            new_points.append((u_target, v_target, depth, avg_error, "common", point_3d))

    
    # # 2. Triangulate two-frame-only points
    # # For ref1-target pairs
    # for point_id, (u_target, v_target, u_ref1, v_ref1) in ref1_only_points.items():
    #     # Convert to normalized coordinates
    #     x_target = (u_target - cx) / fx
    #     y_target = (v_target - cy) / fy
    #     target_point = np.array([[x_target, y_target]]).T

    #     x_ref1 = (u_ref1 - cx) / fx
    #     y_ref1 = (v_ref1 - cy) / fy
    #     ref1_point = np.array([[x_ref1, y_ref1]]).T

    #     # Triangulate using OpenCV
    #     points_4d = cv2.triangulatePoints(ref1_RT, target_RT, ref1_point, target_point)
        
    #     # Convert to 3D coordinates
    #     point_3d = points_4d[:3] / points_4d[3]

    #     # Get depth in target frame
    #     point_camera = target_R @ point_3d.flatten() + target_t.flatten()
    #     depth = point_camera[2]

    #     # Calculate reprojection errors
    #     pt_proj_target = K @ (target_R @ point_3d.flatten() + target_t.flatten())
    #     pt_proj_target = pt_proj_target / pt_proj_target[2]
    #     error_target = np.sqrt((pt_proj_target[0] - u_target)**2 + (pt_proj_target[1] - v_target)**2)

    #     pt_proj_ref1 = K @ (ref1_R @ point_3d.flatten() + ref1_t.flatten())
    #     pt_proj_ref1 = pt_proj_ref1 / pt_proj_ref1[2]
    #     error_ref1 = np.sqrt((pt_proj_ref1[0] - u_ref1)**2 + (pt_proj_ref1[1] - v_ref1)**2)

    #     avg_error = (error_target + error_ref1) / 2

    #     if 0.1 < depth < 5.0 and avg_error < 1.0:
    #         new_points.append((u_target, v_target, depth, avg_error, "ref1", point_3d))
        
    # # For ref2-target pairs
    # for point_id, (u_target, v_target, u_ref2, v_ref2) in ref2_only_points.items():
    #     # Convert to normalized coordinates
    #     x_target = (u_target - cx) / fx
    #     y_target = (v_target - cy) / fy
    #     target_point = np.array([[x_target, y_target]]).T

    #     x_ref2 = (u_ref2 - cx) / fx
    #     y_ref2 = (v_ref2 - cy) / fy
    #     ref2_point = np.array([[x_ref2, y_ref2]]).T

    #     # Triangulate using OpenCV
    #     points_4d = cv2.triangulatePoints(ref2_RT, target_RT, ref2_point, target_point)
        
    #     # Convert to 3D coordinates
    #     point_3d = points_4d[:3] / points_4d[3]

    #     point_camera = target_R @ point_3d.flatten() + target_t.flatten()
    #     depth = point_camera[2]

    #     # Calculate reprojection errors
    #     pt_proj_target = K @ (target_R @ point_3d.flatten() + target_t.flatten())
    #     pt_proj_target = pt_proj_target / pt_proj_target[2]
    #     error_target = np.sqrt((pt_proj_target[0] - u_target)**2 + (pt_proj_target[1] - v_target)**2)

    #     pt_proj_ref2 = K @ (ref2_R @ point_3d.flatten() + ref2_t.flatten())
    #     pt_proj_ref2 = pt_proj_ref2 / pt_proj_ref2[2]
    #     error_ref2 = np.sqrt((pt_proj_ref2[0] - u_ref2)**2 + (pt_proj_ref2[1] - v_ref2)**2)

    #     avg_error = (error_target + error_ref2) / 2

    #     if 0.1 < depth < 5.0 and avg_error < 1.0:
    #         new_points.append((u_target, v_target, depth, avg_error, "ref2", point_3d))

    # print(f"Successfully triangulated {len(new_points)} points before refinement")

    # 3. Non-linear refinement of all triangulated points
    from scipy.optimize import minimize
    def project_point(point_3d, pose, K):
        """Project 3D point to image coordinates"""
        R = pose[:3, :3]
        t = pose[:3, 3].reshape(3, 1)
        
        # Transform point to camera coordinates
        point_camera = R @ point_3d + t.flatten()
        
        # Project to image
        u = fx * point_camera[0] / point_camera[2] + cx
        v = fy * point_camera[1] / point_camera[2] + cy
        
        return u, v
    
    def reprojection_error(point_3d, observations, poses, K):
        """Calculate reprojection error across all views"""
        total_error = 0
        
        for i, (obs_u, obs_v) in enumerate(observations):
            proj_u, proj_v = project_point(point_3d, poses[i], K)
            error = (proj_u - obs_u)**2 + (proj_v - obs_v)**2
            total_error += error
            
        return total_error
    
    refined_points = []
    for u, v, depth, error, point_type, point_3d in new_points:
        if point_type == "common":
            # Get the original observations from the common_points dictionary
            matched_point = None
            for point_data in common_points.values():
                if abs(point_data[0] - u) < 0.1 and abs(point_data[1] - v) < 0.1:
                    matched_point = point_data
                    break
            
            if matched_point:
                u_target, v_target, u_ref1, v_ref1, u_ref2, v_ref2 = matched_point
                observations = [(u_target, v_target), (u_ref1, v_ref1), (u_ref2, v_ref2)]
                poses = [target_pose_inv, ref1_pose_inv, ref2_pose_inv]
            else:
                # Skip if we can't find the original observations
                continue
        
        elif point_type == "ref1_only":
            # Get the original observations from the ref1_only_points dictionary
            matched_point = None
            for point_data in ref1_only_points.values():
                if abs(point_data[0] - u) < 0.1 and abs(point_data[1] - v) < 0.1:
                    matched_point = point_data
                    break
            
            if matched_point:
                u_target, v_target, u_ref1, v_ref1 = matched_point
                observations = [(u_target, v_target), (u_ref1, v_ref1)]
                poses = [target_pose_inv, ref1_pose_inv]
            else:
                continue
        
        else: # ref2_only
            matched_point = None
            for point_data in ref2_only_points.values():
                if abs(point_data[0] - u) < 0.1 and abs(point_data[1] - v) < 0.1:
                    matched_point = point_data
                    break
            
            if matched_point:
                u_target, v_target, u_ref2, v_ref2 = matched_point
                observations = [(u_target, v_target), (u_ref2, v_ref2)]
                poses = [target_pose_inv, ref2_pose_inv]
            else:
                continue
        
        def objective(x):
            return reprojection_error(x, observations, poses, K)
        
        # Optimize using Powell method (derivative-free)
        result = minimize(objective, point_3d, method='Powell')
        refined_pt = result.x

        # Calculate refined depth
        point_camera = target_R @ refined_pt.flatten() + target_t.flatten()
        refined_depth = point_camera[2]

        # calculate final reprojection error
        final_error = result.fun / len(observations)  # Average error per observation

        if 0.1 < refined_depth < 5.0 and final_error < 1.0:
            refined_points.append((u, v, refined_depth, final_error))

    print(f"Successfully refined {len(refined_points)} points")
    # Sort by reprojection error (ascending) to get the best points
    refined_points.sort(key=lambda x: x[3])

    # Take top num_new_points points
    final_points = [(u, v, d) for u, v, d, _ in refined_points[:num_new_points]]
    print(f"Final selection: {len(final_points)} new points")

    # Visualize results
    if len(final_points) > 0:
        if isinstance(target_img, torch.Tensor):
            vis_img = target_img.numpy() * 255.0
            vis_img = vis_img.astype(np.uint8)
        else:
            vis_img = target_img.copy() * 255.0
            vis_img = vis_img.astype(np.uint8)
        
        # Draw original VIO points in red
        for u, v, _ in vio_points:
            cv2.circle(vis_img, (int(u), int(v)), 3, (0, 0, 255), -1)
        
        # Draw new points in green
        for u, v, _ in final_points:
            cv2.circle(vis_img, (int(u), int(v)), 3, (0, 255, 0), -1)
        
        cv2.imshow('Densified Points', vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Return combined points
    return vio_points , final_points

# --- Main Loop --
def draw_point_matches(imgA, ptsA, imgB, ptsB, max_matches=200, vio_points=None):
    """
    Draws matches between imgA and imgB.
    - imgA, imgB: H×W (gray) or H×W×3 (BGR) images
    - ptsA, ptsB: arrays of shape (N,2) with (u,v) coords
    - max_matches: how many of the first matches to draw
    Returns a BGR image showing imgA on the left, imgB on the right,
    with circles and lines connecting each match.
    """
    # Make both images BGR
    def to_bgr(img):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img.copy()
    
    A = to_bgr(imgA)
    B = to_bgr(imgB)
    hA, wA = A.shape[:2]
    hB, wB = B.shape[:2]
    
    # Composite canvas
    H = max(hA, hB)
    W = wA + wB
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:hA, :wA] = A
    canvas[:hB, wA:wA+wB] = B
    
    # Draw matches
    #write num matches 
    num_matches = min(len(ptsA), len(ptsB), max_matches)
    cv2.putText(canvas, f"Num matches: {num_matches}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    #draw vio points on the target image
    for u, v, _ in vio_points:
        cv2.circle(canvas, (int(u) + wA, int(v)), 3, (255, 80, 255), -1)

    num = min(len(ptsA), len(ptsB), max_matches)
    for i in range(num):
        uA, vA = map(int, ptsA[i])
        uB, vB = map(int, ptsB[i])
        ptA = (uA, vA)
        ptB = (uB + wA, vB)
        # circle on each image
        cv2.circle(canvas, ptA, 3, (0,0,255), -1)
        cv2.circle(canvas, ptB, 3, (0,255,0), -1)
        # connecting line
        cv2.line(canvas, ptA, ptB, (255,0,0), 1)
    
    return canvas

# not helping the traingulated points have 20cm error not useful
def densify_vio_with_triangulation_and_ba(ref1_img, target_img, ref2_img, ref1_pose, target_pose, ref2_pose, K, 
                                         vio_points, xfeat, gt_depth=None, num_new_points=200):
    """
    Densify sparse VIO points by matching features across ref1-target-ref2, triangulating and refining with BA
    
    Args:
        ref1_img: First reference frame
        target_img: Target frame where we want to densify depth
        ref2_img: Second reference frame
        ref1_pose: Camera-to-world pose for first reference frame (4x4)
        target_pose: Camera-to-world pose for target frame (4x4)
        ref2_pose: Camera-to-world pose for second reference frame (4x4)
        K: Camera intrinsics matrix (3x3)
        vio_points: List of existing VIO points in target frame [(u, v, depth), ...]
        xfeat: XFeat feature matching module
        gt_depth: Ground truth depth map for evaluation (optional)
        num_new_points: Number of new points to add
        
    Returns:
        List of densified points in target frame [(u, v, depth), ...]
    """
    # ==== 1. Prepare images and match features ====
    
    # Ensure images are in the format expected by XFeat
    if isinstance(ref1_img, torch.Tensor):
        ref1_tensor = ref1_img.permute(2, 0, 1) if ref1_img.dim() == 3 else ref1_img
    else:
        ref1_tensor = torch.from_numpy(ref1_img).permute(2, 0, 1)
        
    if isinstance(target_img, torch.Tensor):
        target_tensor = target_img.permute(2, 0, 1) if target_img.dim() == 3 else target_img
    else:
        target_tensor = torch.from_numpy(target_img).permute(2, 0, 1)
        
    if isinstance(ref2_img, torch.Tensor):
        ref2_tensor = ref2_img.permute(2, 0, 1) if ref2_img.dim() == 3 else ref2_img
    else:
        ref2_tensor = torch.from_numpy(ref2_img).permute(2, 0, 1)
    
    # Create a mask to avoid detecting features near existing VIO points
    height, width = target_tensor.shape[1:3]
    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    for u, v, _ in vio_points:
        if 0 <= int(v) < height and 0 <= int(u) < width:
            cv2.circle(mask, (int(u), int(v)), 3, 0, -1)  # 3 pixel radius
    
    # Match features between frames
    #print("Matching features between reference frames and target...")
    
    # Match ref1 to target
    mkpts_ref1, mkpts_target1 = xfeat.match_xfeat(ref1_tensor, target_tensor, top_k=num_new_points*3)
    mkpts_ref2, mkpts_target2 = xfeat.match_xfeat(ref2_tensor, target_tensor, top_k=num_new_points*3)
    
    # Apply subpixel refinement to feature locations
    def refine_keypoints_subpixel(img, kpts, window_size=11):
        img_np = img.numpy() if isinstance(img, torch.Tensor) else img
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if img_np.shape[-1] == 3 else img_np
        
        # Convert to cv2 keypoints format
        cv_kpts = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kpts]
        
        # Refine with cornerSubPix
        points = np.array([kp.pt for kp in cv_kpts], dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        cv2.cornerSubPix(img_gray, points, (window_size, window_size), (-1, -1), criteria)
        
        return points
    
    # Refine all keypoints to subpixel accuracy
    mkpts_ref1 = refine_keypoints_subpixel(ref1_img, mkpts_ref1)
    mkpts_target1 = refine_keypoints_subpixel(target_img, mkpts_target1)
    mkpts_ref2 = refine_keypoints_subpixel(ref2_img, mkpts_ref2)
    mkpts_target2 = refine_keypoints_subpixel(target_img, mkpts_target2)

    # Apply MAGSAC to filter matches
    H1, inlier_mask1 = cv2.findHomography(
        mkpts_ref1, mkpts_target1, 
        cv2.USAC_MAGSAC, 
        1.0, 
        maxIters=1000, 
        confidence=0.999
    )
    inlier_mask1 = inlier_mask1.flatten().astype(bool)
    
    # Filter to inliers
    mkpts_ref1 = mkpts_ref1[inlier_mask1]
    mkpts_target1 = mkpts_target1[inlier_mask1]
    
    
    # Apply MAGSAC to filter matches
    H2, inlier_mask2 = cv2.findHomography(
        mkpts_ref2, mkpts_target2, 
        cv2.USAC_MAGSAC, 
        1.0, 
        maxIters=1000, 
        confidence=0.999
    )
    inlier_mask2 = inlier_mask2.flatten().astype(bool)
    
    # Filter to inliers
    mkpts_ref2 = mkpts_ref2[inlier_mask2]
    mkpts_target2 = mkpts_target2[inlier_mask2]
    
    print(f"Found {len(mkpts_target1)} inlier matches between ref1 and target")
    print(f"Found {len(mkpts_target2)} inlier matches between ref2 and target")

    # Filter out matches near VIO points using the mask
    valid_matches1 = []
    for i, (_, _) in enumerate(mkpts_ref1):
        u, v = mkpts_target1[i]
        if 0 <= int(v) < height and 0 <= int(u) < width and mask[int(v), int(u)] > 0:
            valid_matches1.append(i)
    
    mkpts_ref1 = mkpts_ref1[valid_matches1]
    mkpts_target1 = mkpts_target1[valid_matches1]
    
    valid_matches2 = []
    for i, (_, _) in enumerate(mkpts_ref2):
        u, v = mkpts_target2[i]
        if 0 <= int(v) < height and 0 <= int(u) < width and mask[int(v), int(u)] > 0:
            valid_matches2.append(i)
    
    mkpts_ref2 = mkpts_ref2[valid_matches2]
    mkpts_target2 = mkpts_target2[valid_matches2]
    
    print(f"After mask filtering: {len(mkpts_target1)} ref1 matches, {len(mkpts_target2)} ref2 matches")


    # ##visualize matches between ref1 and target and ref2 and target draw the matches between them
    # vis1 = draw_point_matches(
    #     (ref1_img*255.0).numpy() if isinstance(ref1_img, torch.Tensor) else ref1_img*255.0,
    #     mkpts_ref1,
    #     (target_img*255.0).numpy() if isinstance(target_img, torch.Tensor) else target_img*255.0,
    #     mkpts_target1,
    #     max_matches=len(mkpts_target1),
    #     vio_points=vio_points
    # )

    # vis2 = draw_point_matches(
    #     (ref2_img*255.0).numpy() if isinstance(ref2_img, torch.Tensor) else ref2_img*255.0,
    #     mkpts_ref2,
    #     (target_img*255.0).numpy() if isinstance(target_img, torch.Tensor) else target_img*255.0,
    #     mkpts_target2,
    #     max_matches=len(mkpts_target2),
    #     vio_points=vio_points
    # )

    # cv2.imshow('Matches between ref1 and target', vis1)
    # cv2.imshow('Matches between ref2 and target', vis2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    #from mkpts_target1 and mkpts_target2 get the gt points and append to new_points
    # new_points = []
    # for i in range(len(mkpts_target1)):
    #     u, v = mkpts_target1[i]
    #     if gt_depth[int(v), int(u)] > 0 and (u, v) not in new_points:
    #         #add small noise to the depth
    #         depth = gt_depth[int(v), int(u)] + np.random.uniform(-0.03, 0.02) #tolerance 1-5cm beyond this is bad
    #         new_points.append((u, v, depth))
    
    # for i in range(len(mkpts_target2)):
    #     u, v = mkpts_target2[i]
    #     if gt_depth[int(v), int(u)] > 0 and (u, v) not in new_points:
    #         #add small noise to the depth
    #         depth = gt_depth[int(v), int(u)] + np.random.uniform(-0.02, 0.02)
    #         new_points.append((u, v, depth))

    ##==== 2. Setup for triangulation ====
    # Convert camera-to-world poses to world-to-camera poses
    target_pose_inv = np.linalg.inv(target_pose)
    ref1_pose_inv = np.linalg.inv(ref1_pose)
    ref2_pose_inv = np.linalg.inv(ref2_pose)

    # Prepare camera projection matrices
    target_R = target_pose_inv[:3, :3]
    target_t = target_pose_inv[:3, 3].reshape(3, 1)
    target_RT = np.hstack((target_R, target_t))
    target_P = K @ target_RT

    ref1_R = ref1_pose_inv[:3, :3]
    ref1_t = ref1_pose_inv[:3, 3].reshape(3, 1)
    ref1_RT = np.hstack((ref1_R, ref1_t))
    ref1_P = K @ ref1_RT

    ref2_R = ref2_pose_inv[:3, :3]
    ref2_t = ref2_pose_inv[:3, 3].reshape(3, 1)
    ref2_RT = np.hstack((ref2_R, ref2_t))
    ref2_P = K @ ref2_RT

    # Get camera intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # ==== 3. Triangulate using two-view matches with refinement ====
    from scipy.optimize import least_squares
    triangulated_points = []

    # Define projection and reprojection functions
    def project_3d_point(X, P):
        X_h = np.append(X, 1)  # Homogeneous
        p = P @ X_h
        return p[:2] / p[2]  # Normalized image coordinates
    
    def reprojection_error_2view(X, u1, v1, u2, v2, P1, P2):
        # Project to both views
        p1 = project_3d_point(X, P1)
        p2 = project_3d_point(X, P2)
        
        # Calculate reprojection errors
        err1 = np.array([p1[0] - u1, p1[1] - v1])
        err2 = np.array([p2[0] - u2, p2[1] - v2])
        
        return np.concatenate([err1, err2])
    
    # Function to compute parallax angle between rays
    def compute_parallax(u1, v1, u2, v2, P1, P2, X):
        # Get camera centers in world coordinates
        C1 = -np.linalg.inv(P1[:3, :3]) @ P1[:3, 3]
        C2 = -np.linalg.inv(P2[:3, :3]) @ P2[:3, 3]
        
        # Compute vectors from cameras to 3D point
        v1 = X - C1
        v2 = X - C2
        
        # Normalize vectors
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Compute angle between vectors
        cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        
        return np.degrees(angle_rad)

    # First try ref1-target pair (might have better baseline/parallax)
    print("Triangulating points from ref1-target pair...")
    for i in range(len(mkpts_target1)):
        u_target, v_target = mkpts_target1[i]
        u_ref1, v_ref1 = mkpts_ref1[i]
        
        # Convert to normalized coordinates
        x_target = (u_target - cx) / fx
        y_target = (v_target - cy) / fy
        target_point = np.array([[x_target, y_target]]).T
        
        x_ref1 = (u_ref1 - cx) / fx
        y_ref1 = (v_ref1 - cy) / fy
        ref1_point = np.array([[x_ref1, y_ref1]]).T
        
        # Triangulate using OpenCV
        points_4d = cv2.triangulatePoints(ref1_RT, target_RT, ref1_point, target_point)
        X_init = points_4d[:3] / points_4d[3]  # Initial 3D point
        
        # Refine with BA
        refine_fn = lambda X: reprojection_error_2view(X, u_target, v_target, u_ref1, v_ref1, target_P, ref1_P)
        result = least_squares(refine_fn, X_init.flatten(), method='lm', loss='linear')
        X_refined = result.x
        
        # Calculate depth in target frame
        X_camera = target_R @ X_refined + target_t.flatten()
        depth = X_camera[2]
        
        # Calculate final reprojection error
        final_error = np.sqrt(np.mean(refine_fn(X_refined)**2))
        # Calculate parallax angle for quality assessment
        parallax = compute_parallax(u_target, v_target, u_ref1, v_ref1, target_P, ref1_P, X_refined)

        if 0.1 < depth < 5.0 and final_error < 0.8 and parallax > 3.0:
            triangulated_points.append((u_target, v_target, depth, final_error, parallax, "ref1"))
            print("triangulated depth[v,u]: ", depth, "[", v_target, ",", u_target, "]")
            print("gt depth[v,u]: ", gt_depth[int(v_target), int(u_target)], "[", v_target, ",", u_target, "]")

    # Then try ref2-target pair
    print("Triangulating points from ref2-target pair...")
    for i in range(len(mkpts_target2)):
        u_target, v_target = mkpts_target2[i]
        u_ref2, v_ref2 = mkpts_ref2[i]
        
        # Check if this target point is already in our triangulated set
        already_triangulated = False
        for pt in triangulated_points:
            if np.sqrt((pt[0] - u_target)**2 + (pt[1] - v_target)**2) < 2.0:
                already_triangulated = True
                break
        
        if already_triangulated:
            continue
            
        # Convert to normalized coordinates
        x_target = (u_target - cx) / fx
        y_target = (v_target - cy) / fy
        target_point = np.array([[x_target, y_target]]).T
        
        x_ref2 = (u_ref2 - cx) / fx
        y_ref2 = (v_ref2 - cy) / fy
        ref2_point = np.array([[x_ref2, y_ref2]]).T
        
        # Triangulate using OpenCV
        points_4d = cv2.triangulatePoints(ref2_RT, target_RT, ref2_point, target_point)
        X_init = points_4d[:3] / points_4d[3]  # Initial 3D point
        
        # Refine with BA
        refine_fn = lambda X: reprojection_error_2view(X, u_target, v_target, u_ref2, v_ref2, target_P, ref2_P)
        result = least_squares(refine_fn, X_init.flatten(), method='lm', loss='linear')
        X_refined = result.x
        
        # Calculate depth in target frame
        X_camera = target_R @ X_refined + target_t.flatten()
        depth = X_camera[2]
        
        # Calculate final reprojection error
        final_error = np.sqrt(np.mean(refine_fn(X_refined)**2))
        
        # Calculate parallax angle for quality assessment
        parallax = compute_parallax(u_target, v_target, u_ref2, v_ref2, target_P, ref2_P, X_refined)
        
        # Check if depth is valid, error is small, and parallax is sufficient
        if 0.1 < depth < 5.0 and final_error < 0.8 and parallax > 3.0:
            triangulated_points.append((u_target, v_target, depth, final_error, parallax, "ref2"))
    
    print(f"Successfully triangulated {len(triangulated_points)} points")

    # ==== 4. Select best points based on multiple criteria ====
    
    # Normalize scores for balanced weighting
    if triangulated_points:
        errors = np.array([p[3] for p in triangulated_points])
        parallaxes = np.array([p[4] for p in triangulated_points])
        
        # Compute normalized scores (lower is better)
        error_scores = errors / np.max(errors) if np.max(errors) > 0 else errors
        parallax_scores = 1.0 - (parallaxes / np.max(parallaxes)) if np.max(parallaxes) > 0 else parallaxes
        
        # Compute combined score (lower is better)
        combined_scores = 0.4 * error_scores + 0.6 * parallax_scores
        
        # Sort by combined score (ascending)
        sorted_indices = np.argsort(combined_scores)
        triangulated_points = [triangulated_points[i] for i in sorted_indices]
    
    # Take the best points
    final_points = [(u, v, d) for u, v, d, _, _, _ in triangulated_points[:num_new_points]]

    ##print final points and gt depth 
    # for u, v, d in final_points:
    #     print("final depth[v,u]: ", d, "[", v, ",", u, "]")
    #     print("gt depth[v,u]: ", gt_depth[int(v), int(u)], "[", v, ",", u, "]")
    
    # ==== 5. Visualize results ====
    # if len(final_points) > 0:
    #     if isinstance(target_img, torch.Tensor):
    #         vis_img = target_img.numpy() * 255.0
    #         vis_img = vis_img.astype(np.uint8)
    #     else:
    #         vis_img = target_img.copy() * 255.0
    #         vis_img = vis_img.astype(np.uint8)
        
    #     # Draw original VIO points in red
    #     for u, v, _ in vio_points:
    #         cv2.circle(vis_img, (int(u), int(v)), 3, (0, 0, 255), -1)
        
    #     # Draw new points in green
    #     for u, v, _ in final_points:
    #         cv2.circle(vis_img, (int(u), int(v)), 3, (0, 255, 0), -1)
        
    #     cv2.imshow('Densified Points', vis_img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    
    

    # vis_img_tgt = target_img.numpy() * 255.0 if isinstance(target_img, torch.Tensor) else target_img.copy() * 255.0
    # vis_img_ref1 = ref1_img.numpy() * 255.0 if isinstance(ref1_img, torch.Tensor) else ref1_img.copy() * 255.0
    # vis_img_ref2 = ref2_img.numpy() * 255.0 if isinstance(ref2_img, torch.Tensor) else ref2_img.copy() * 255.0
    
    # vis_img_tgt = vis_img_tgt.astype(np.uint8)
    # vis_img_ref1 = vis_img_ref1.astype(np.uint8)
    # vis_img_ref2 = vis_img_ref2.astype(np.uint8)
    
    # for u, v, u_ref1, v_ref1, u_ref2, v_ref2 in common_points:
    #     cv2.circle(vis_img_tgt, (int(u), int(v)), 3, (0, 255, 0), -1)
    #     cv2.circle(vis_img_ref1, (int(u_ref1), int(v_ref1)), 3, (0, 0, 255), -1)
    #     cv2.circle(vis_img_ref2, (int(u_ref2), int(v_ref2)), 3, (0, 0, 255), -1)
    
    # cv2.imshow('Common Points tgt', vis_img_tgt)
    # cv2.imshow('Common Points ref1', vis_img_ref1)
    # cv2.imshow('Common Points ref2', vis_img_ref2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    return vio_points, final_points


def main():
    # Inputs
    dataset = SML_consistent_dataset(data_root='/media/saimouli/Data6T/datasets/VOID_150_small', mode='val')
    # Use only a subset for faster testing
    subset_indices = range(0, len(dataset), 1) # Take every 10th sample
    subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
    dataloader = torch.utils.data.DataLoader(subset_dataset, batch_size=1) # Batch size must be 1

    device = "cpu" 
    xfeat = XFeat()

    for batch_data in dataloader:
        # --- 1. Load Data ---

        # Move tensors to CPU and convert to NumPy
        tgt_img_tensor = batch_data[0].squeeze(0).cpu() # Remove batch dim
        tgt_sparse_depth_tensor = batch_data[4].squeeze(0).cpu()
        tgt_gt_depth_tensor_inv = batch_data[1].squeeze(0).cpu()
        ref_imgs_tensor = [img.squeeze(0).cpu() for img in batch_data[5]] # List of [C, H, W]
        tgt_pose_tensor = batch_data[10].squeeze(0).cpu()
        ref_poses_tensor = [pose.squeeze(0).cpu() for pose in batch_data[11]]
        intrinsics_tensor = batch_data[12].squeeze(0).cpu()

        # Convert to NumPy format expected by functions
        # Images need to be HWC, uint8 for visualization, float32 grayscale for cost
        tgt_img_np_uint8 = (tgt_img_tensor.numpy() * 255).astype(np.uint8)
        tgt_img_gray = cv2.cvtColor(tgt_img_np_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        ref_imgs_np_uint8 = [(img.numpy() * 255).astype(np.uint8) for img in ref_imgs_tensor]
        ref_imgs_gray = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0 for img in ref_imgs_np_uint8]
        
        tgt_gt_depth_np = utils.inv2depth(tgt_gt_depth_tensor_inv).squeeze(0).numpy()
        tgt_sparse_depth_np = tgt_sparse_depth_tensor.numpy()
        tgt_pose_np = tgt_pose_tensor.numpy()
        ref_poses_np = [pose.numpy() for pose in ref_poses_tensor]
        K_np = intrinsics_tensor.numpy()
        
        sparse_points = []
        for u in range(tgt_sparse_depth_np.shape[1]):
            for v in range(tgt_sparse_depth_np.shape[0]):
                if tgt_sparse_depth_np[v, u] > 0:
                    depth = tgt_sparse_depth_np[v, u]
                    #add small noise to the depth
                    #depth += np.random.uniform(-0.50, 0.50)
                    sparse_points.append((u, v, depth))

        #relative pose between ref1 and target
        target_pos = tgt_pose_np[:3, 3]
        ref1_pos = ref_poses_np[0][:3, 3]
        ref2_pos = ref_poses_np[1][:3, 3]
        dist_ref1 = np.linalg.norm(target_pos - ref1_pos)
        dist_ref2 = np.linalg.norm(target_pos - ref2_pos)
        #print("dist_ref1: ", dist_ref1)
        #print("dist_ref2: ", dist_ref2)

        # sparse_pts, new_pts = densify_vio_with_xfeat_triangulation(ref_imgs_tensor[0], tgt_img_tensor, ref_imgs_tensor[1], 
        #                                                            ref_poses_np[0], tgt_pose_np, ref_poses_np[1], K_np, 
        #                                                            sparse_points, xfeat, num_new_points=800)

        # sparse_pts, new_pts = densify_vio_with_triangulation_and_ba(ref_imgs_tensor[0], tgt_img_tensor, ref_imgs_tensor[1], 
        #                                                         ref_poses_np[0], tgt_pose_np, ref_poses_np[1], K_np, 
        #                                                         sparse_points, xfeat, gt_depth=tgt_gt_depth_np,
        #                                                         num_new_points=2000)
        
        #test_xfeat_densification(tgt_img_tensor, ref_imgs_tensor, tgt_pose_np, ref_poses_np, K_np, sparse_points, xfeat, num_new_points=4500)
        # sparse_pts, new_pts = densify_sparse_vio_points(
        #     target_img=tgt_img_gray,
        #     ref_imgs=ref_imgs_gray,
        #     target_pose=tgt_pose_np,
        #     ref_poses=ref_poses_np,
        #     K=K_np,
        #     sparse_points=sparse_points,
        #     num_new_points=400, #add 200 more points
        #     patch_size=11
        # )
        
        # vis_img = tgt_img_np_uint8.copy() #cv2.cvtColor(tgt_img_np_uint8.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        # print("sparse_pts: ", len(sparse_pts))
        # print("new_pts: ", len(new_pts))

        # #check new_pts depth with gt depth
        # for u, v, d in new_pts[:10]:
        #     gt_depth = tgt_gt_depth_np[int(v), int(u)]
        #     print(f"gt_depth: {gt_depth}, new_pts depth: {d}")
        
        # #check vio points depth with gt depth
        # for u, v, d in sparse_pts[:10]:
        #     gt_depth = tgt_gt_depth_np[int(v), int(u)]
        #     print(f"gt_depth: {gt_depth}, sparse_pts depth: {d}")

        # # Draw original points in red
        # for i in range(len(sparse_pts)):
        #     u, v, _ = sparse_points[i]
        #     cv2.circle(vis_img, (int(u), int(v)), 3, (0, 0, 255), -1)

        # # Draw new points in green
        # for i in range(len(new_pts)):
        #     u, v, _ = new_pts[i]
        #     cv2.circle(vis_img, (int(u), int(v)), 3, (0, 255, 0), -1)

        # cv2.imshow('Densified Points', vis_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    