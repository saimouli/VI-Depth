import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.ndimage import gaussian_filter
from PIL import Image
import os
from tqdm import tqdm
import pipeline
from utils_eval import compute_ls_solution
import modules.midas.utils as utils
import cv2
from modules.interpolator import Interpolator2D
from test_scaffolding import plot_surface_normals

def load_sparse_depth(input_sparse_depth_fp):
    input_sparse_depth = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32) / 256.0
    input_sparse_depth[input_sparse_depth <= 0] = 0.0
    return input_sparse_depth

def generate_synthetic_normals(width=200, height=200, noise_level=0.05):
    """Generate synthetic normal map with planar regions and a sphere for testing."""
    normals = np.zeros((height, width, 3), dtype=np.float64)  # Ensure float64 dtype
    
    # Create different planar regions
    for y in range(height):
        for x in range(width):
            # Left side plane
            if x < width/3:
                normal = np.array([0.2, 0, 0.98], dtype=np.float64) # Ensure float64 dtype
            # Middle plane
            elif x < 2*width/3:
                normal = np.array([0, 0, 1], dtype=np.float64) # Ensure float64 dtype
            # Right side plane
            else:
                normal = np.array([-0.2, 0, 0.98], dtype=np.float64) # Ensure float64 dtype
            
            # Add a sphere in the center
            center_x, center_y = width/2, height/2
            radius = min(width, height) / 5
            dist_to_center = np.sqrt((x-center_x)**2 + (y-center_y)**2)
            
            if dist_to_center < radius:
                # Create sphere normals
                nx = (x - center_x) / radius
                ny = (y - center_y) / radius
                nz = np.sqrt(1 - nx*nx - ny*ny)
                normal = np.array([nx, ny, nz], dtype=np.float64) # Ensure float64 dtype
            
            # Add noise
            normal += (np.random.random(3) - 0.5) * noise_level
            normal = normal / np.linalg.norm(normal)  # Normalize
            normals[y, x] = normal
    
    return normals


def visualize_normal_map(normals, title="Normal Map"):
    """Visualize normal map as RGB image."""
    # Convert normal vectors to RGB (x → R, y → G, z → B)
    rgb_normals = (normals + 1) * 0.5  # Convert from [-1,1] to [0,1]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_normals)
    plt.title(title)
    plt.axis('off')
    
    return rgb_normals


def bilateral_clustering(normals, kernel_size=5, similarity_threshold=0.9, max_iterations=5):
    """Cluster normals using bilateral filtering approach."""
    height, width, _ = normals.shape
    clusters = np.zeros((height, width), dtype=int)
    next_cluster_id = 1
    
    # First pass: assign initial cluster IDs
    for y in range(height):
        for x in range(width):
            if clusters[y, x] == 0:
                # Start a new cluster
                center_normal = normals[y, x]
                clusters[y, x] = next_cluster_id
                
                # Process kernel around this pixel
                half_kernel = kernel_size // 2
                for ky in range(max(0, y - half_kernel), min(height, y + half_kernel + 1)):
                    for kx in range(max(0, x - half_kernel), min(width, x + half_kernel + 1)):
                        if clusters[ky, kx] == 0:
                            neighbor_normal = normals[ky, kx]
                            
                            # Calculate dot product for similarity
                            similarity = np.dot(center_normal, neighbor_normal)
                            
                            if similarity > similarity_threshold:
                                clusters[ky, kx] = next_cluster_id
                
                next_cluster_id += 1
    
    # Second pass: refine clusters (optional)
    changed = True
    iterations = 0
    
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        
        for y in range(height):
            for x in range(width):
                center_normal = normals[y, x]
                current_cluster = clusters[y, x]
                
                # Look at neighbors
                best_cluster = current_cluster
                best_similarity = 0
                
                for ky in range(max(0, y - 1), min(height, y + 2)):
                    for kx in range(max(0, x - 1), min(width, x + 2)):
                        if kx == x and ky == y:
                            continue
                        
                        neighbor_normal = normals[ky, kx]
                        neighbor_cluster = clusters[ky, kx]
                        
                        # Calculate similarity
                        similarity = np.dot(center_normal, neighbor_normal)
                        
                        if similarity > similarity_threshold and similarity > best_similarity:
                            best_similarity = similarity
                            best_cluster = neighbor_cluster
                
                # Update cluster if better match found
                if best_cluster != current_cluster:
                    clusters[y, x] = best_cluster
                    changed = True
    
    return clusters


def region_growing_clustering(normals, similarity_threshold=0.95):
    """Cluster normals using region growing approach."""
    height, width, _ = normals.shape
    clusters = np.zeros((height, width), dtype=int)
    visited = np.zeros((height, width), dtype=bool)
    next_cluster_id = 1
    
    # Neighbor offsets (8-connectivity)
    #neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    neighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    
    for y in range(height):
        for x in range(width):
            if not visited[y, x]:
                # Start a new region
                seed_normal = normals[y, x]
                seed_points = [(y, x)]
                visited[y, x] = True
                clusters[y, x] = next_cluster_id
                
                # Region growing process
                while seed_points:
                    cy, cx = seed_points.pop(0)
                    
                    for dy, dx in neighbors:
                        ny, nx = cy + dy, cx + dx
                        
                        if (0 <= ny < height and 0 <= nx < width and 
                            not visited[ny, nx]):
                            
                            neighbor_normal = normals[ny, nx]
                            similarity = np.dot(seed_normal, neighbor_normal)
                            
                            if similarity > similarity_threshold:
                                clusters[ny, nx] = next_cluster_id
                                visited[ny, nx] = True
                                seed_points.append((ny, nx))
                
                next_cluster_id += 1
    
    return clusters


def kmeans_clustering(normals, n_clusters=8):
    """Cluster normals using K-means."""
    height, width, _ = normals.shape
    
    # Reshape for clustering
    normals_flat = normals.reshape(-1, 3)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(normals_flat)
    
    # Reshape back to image
    clusters = labels.reshape(height, width)
    
    return clusters


def dbscan_clustering(normals, eps=0.1, min_samples=10):
    """Cluster normals using DBSCAN."""
    height, width, _ = normals.shape
    
    # Reshape for clustering
    normals_flat = normals.reshape(-1, 3)
    
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(normals_flat)
    
    # Reshape back to image
    clusters = labels.reshape(height, width)
    
    return clusters


def visualize_clusters(clusters, title="Clustered Normals"):
    """Visualize cluster assignments with random colors."""
    # Get unique cluster IDs
    unique_clusters = np.unique(clusters)
    
    # Generate random colors for each cluster
    colormap = np.random.random((len(unique_clusters), 3))
    
    # Handle negative cluster IDs (like -1 from DBSCAN for noise)
    if np.any(unique_clusters < 0):
        colormap[0] = [0, 0, 0]  # Black for noise points
    
    # Create colored image
    height, width = clusters.shape
    cluster_image = np.zeros((height, width, 3))
    
    for i, cluster_id in enumerate(unique_clusters):
        mask = clusters == cluster_id
        for c in range(3):
            cluster_image[:, :, c][mask] = colormap[i, c]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cluster_image)
    plt.title(f"{title} - {len(unique_clusters)} clusters")
    plt.axis('off')
    
    return cluster_image


def depth_propagation_bilateral(normals, depth_seed=None, iterations=10):
    """
    Propagate depth using bilateral filtering on normals.
    If depth_seed is None, creates a seed at the center.
    """
    height, width, _ = normals.shape
    
    # Initialize depth map
    if depth_seed is None:
        # Create a simple initial depth seed at the center
        depth = np.zeros((height, width))
        center_y, center_x = height // 2, width // 2
        radius = min(height, width) // 10
        
        for y in range(max(0, center_y - radius), min(height, center_y + radius)):
            for x in range(max(0, center_x - radius), min(width, center_x + radius)):
                if (y - center_y)**2 + (x - center_x)**2 <= radius**2:
                    depth[y, x] = 1.0  # Initial seed value
    else:
        depth = depth_seed.copy()
    
    # Initialize confidence map (0 = unknown, 1 = confident)
    confidence = np.zeros((height, width))
    confidence[depth > 0] = 1.0
    
    # Parameters
    kernel_size = 5
    half_kernel = kernel_size // 2
    spatial_sigma = 2.0
    normal_sigma = 0.1
    
    # Precompute spatial kernel weights
    spatial_weights = np.zeros((kernel_size, kernel_size))
    for y in range(kernel_size):
        for x in range(kernel_size):
            spatial_weights[y, x] = np.exp(-((y-half_kernel)**2 + (x-half_kernel)**2) / (2 * spatial_sigma**2))
    
    # Propagation loop
    for _ in range(iterations):
        new_depth = depth.copy()
        new_confidence = confidence.copy()
        
        for y in range(height):
            for x in range(width):
                # Skip if already confident
                if confidence[y, x] >= 0.9:
                    continue
                
                center_normal = normals[y, x]
                
                total_weight = 0
                depth_sum = 0
                
                # Process kernel
                for ky in range(max(0, y - half_kernel), min(height, y + half_kernel + 1)):
                    kernel_y = ky - (y - half_kernel)
                    for kx in range(max(0, x - half_kernel), min(width, x + half_kernel + 1)):
                        kernel_x = kx - (x - half_kernel)
                        
                        # Skip if neighbor has no depth or low confidence
                        if confidence[ky, kx] < 0.5:
                            continue
                        
                        neighbor_normal = normals[ky, kx]
                        
                        # Normal similarity weight
                        normal_similarity = np.dot(center_normal, neighbor_normal)
                        normal_weight = np.exp(-(1 - normal_similarity) / normal_sigma)
                        
                        # Final weight combines spatial and normal weights
                        weight = spatial_weights[kernel_y, kernel_x] * normal_weight * confidence[ky, kx]
                        
                        # For depth propagation, we need to adjust for surface orientation
                        # Simple version: project depth change using normal orientation
                        dx, dy = kx - x, ky - y
                        if dx == 0 and dy == 0:
                            depth_adjustment = 0
                        else:
                            dist = np.sqrt(dx**2 + dy**2)
                            dir_vec = np.array([dx/dist, dy/dist, 0])
                            # Approximate depth change using normal projection
                            depth_adjustment = np.dot(center_normal, dir_vec) * dist
                        
                        adjusted_depth = depth[ky, kx] + depth_adjustment
                        
                        depth_sum += adjusted_depth * weight
                        total_weight += weight
                
                # Update depth if we got valid contributions
                if total_weight > 0:
                    new_depth[y, x] = depth_sum / total_weight
                    new_confidence[y, x] = min(0.95, confidence[y, x] + 0.2)  # Gradually increase confidence
        
        depth = new_depth
        confidence = new_confidence
    
    return depth, confidence


def depth_propagation_fast(normals, depth_seed=None, iterations=10):
    """
    Faster depth propagation using normal integration.
    Uses a more efficient gradient-based approach.
    """
    height, width, _ = normals.shape
    
    # Extract normal components
    nx = normals[:, :, 0]
    ny = normals[:, :, 1]
    nz = normals[:, :, 2]
    
    # Handle zeros in nz to avoid division by zero
    nz[np.abs(nz) < 1e-10] = 1e-10
    
    # Convert normals to gradients
    gx = -nx / nz
    gy = -ny / nz
    
    # Initialize depth map
    if depth_seed is None:
        # Create a simple initial depth seed at the center
        depth = np.zeros((height, width))
        center_y, center_x = height // 2, width // 2
        radius = min(height, width) // 10
        
        for y in range(max(0, center_y - radius), min(height, center_y + radius)):
            for x in range(max(0, center_x - radius), min(width, center_x + radius)):
                if (y - center_y)**2 + (x - center_x)**2 <= radius**2:
                    depth[y, x] = 1.0  # Initial seed value
    else:
        depth = depth_seed.copy()
    
    # Create initial mask of known depths
    mask = depth > 0
    
    # Propagation loop
    for _ in range(iterations):
        # Pad depth for gradient calculation
        depth_padded = np.pad(depth, ((1, 1), (1, 1)), mode='edge')
        
        # Calculate depth gradients in x and y directions
        dx = (depth_padded[1:-1, 2:] - depth_padded[1:-1, :-2]) / 2
        dy = (depth_padded[2:, 1:-1] - depth_padded[:-2, 1:-1]) / 2
        
        # Update depth using normal information
        for y in range(1, height-1):
            for x in range(1, width-1):
                if mask[y, x]:
                    continue  # Skip known depths
                
                # Count valid neighbors
                valid_count = 0
                depth_sum = 0
                
                # Check each neighbor for propagation
                if mask[y, x-1]:  # Left
                    depth_sum += depth[y, x-1] + gx[y, x-1]
                    valid_count += 1
                    
                if mask[y, x+1]:  # Right
                    depth_sum += depth[y, x+1] - gx[y, x+1]
                    valid_count += 1
                    
                if mask[y-1, x]:  # Top
                    depth_sum += depth[y-1, x] + gy[y-1, x]
                    valid_count += 1
                    
                if mask[y+1, x]:  # Bottom
                    depth_sum += depth[y+1, x] - gy[y+1, x]
                    valid_count += 1
                
                # Update depth if we got valid neighbors
                if valid_count > 0:
                    depth[y, x] = depth_sum / valid_count
                    mask[y, x] = True
        
        # Smooth the result slightly
        depth = gaussian_filter(depth, sigma=0.5)
    
    return depth, mask


def visualize_depth_map(depth, title="Depth Map"):
    """Visualize depth map."""
    plt.figure(figsize=(10, 8))
    plt.imshow(depth, cmap='viridis')
    plt.colorbar(label='Depth')
    plt.title(title)
    plt.axis('off')


def benchmark_propagation_methods(normals, methods=['bilateral', 'fast'], iterations=10):
    """Benchmark different propagation methods."""
    results = {}
    
    for method in methods:
        start_time = time.time()
        
        if method == 'bilateral':
            depth, confidence = depth_propagation_bilateral(normals, iterations=iterations)
        elif method == 'fast':
            depth, mask = depth_propagation_fast(normals, iterations=iterations)
            confidence = mask.astype(float)
        
        elapsed_time = time.time() - start_time
        
        results[method] = {
            'depth': depth,
            'confidence': confidence,
            'time': elapsed_time
        }
        
        print(f"Method: {method}, Time: {elapsed_time:.3f} seconds")
    
    return results

def detect_normals(depth_image, input_sparse_depth, input_image, display=False):
    depth_image_ = depth_image.astype(np.float32)
    rows, cols = depth_image_.shape
    viz_img = input_image.copy()
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    # Calculate the partial derivatives of depth with respect to x and y
    blurred_image = cv2.GaussianBlur(depth_image_, (5, 5), 0)
    dx = cv2.Sobel(blurred_image, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(blurred_image, cv2.CV_32F, 0, 1)

    #compute normal vector for each pixel
    normal = np.dstack((-dx, -dy, np.ones((rows, cols))))
    norm = np.sqrt(np.sum(normal**2, axis=2, keepdims=True))
    normal = np.divide(normal, norm, out=np.zeros_like(normal), where=norm != 0)

    return normal

def segment_planes_from_normals_dual_flood(normals, seed_spacing=(40, 40), 
                                          loDiff1=(0.01, 0.01, 0.01), upDiff1=(0.01, 0.01, 0.01),
                                          loDiff2=(0.1, 0.1, 0.1), upDiff2=(0.1, 0.1, 0.1),
                                          min_region_size=3000):

    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normal_map = normals / (norm + 1e-6)

    normal_map = (normal_map + 1) / 2  # Convert from -1:1 to 0:1
    height, width = normal_map.shape[:2]

    seed_y_n = height // seed_spacing[0]
    seed_x_n = width // seed_spacing[1]

    xs = np.linspace(0, width-1, seed_x_n+2, dtype=np.int32)[1:-1]
    ys = np.linspace(0, height-1, seed_y_n+2, dtype=np.int32)[1:-1]
    xx, yy = np.meshgrid(xs, ys)
    seed_pts = np.dstack((xx, yy)).reshape(-1,2)

    normal_map = normal_map.astype(np.float32)
    label_arr = np.zeros((height, width), dtype=np.int32)
    label_n=1    
    newVal = (1, 1, 1)

    for pt in seed_pts:
        x, y = pt
        if label_arr[y,x] !=0 or np.isnan(normal_map[y, x, 0]):
            continue

        mask1 = np.zeros((height+2, width+2), np.uint8)
        mask2 = np.zeros((height+2, width+2), np.uint8)

        retval1 = cv2.floodFill(normal_map.copy(), mask1, (x, y), newVal, loDiff1, upDiff1, flags=8)
        retval2 = cv2.floodFill(normal_map.copy(), mask2, (x, y), newVal, loDiff2, upDiff2, 
                              flags=cv2.FLOODFILL_FIXED_RANGE)
        
        # Check if both regions are large enough
        if retval1[0] > min_region_size and retval2[0] > min_region_size:
            combined_mask = mask1 * mask2

            label_arr[(combined_mask==1)[1:-1, 1:-1]] = label_n
            label_n += 1

    return label_arr

def visualize_depth_completion_process(input_image, input_sparse_depth, plane_masks, densified_depth, probability_threshold=0.7):
    """Visualize the depth completion process with stacked plane masks"""
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Get the original sparse points
    valid_mask_original = input_sparse_depth > 0.2
    y_orig, x_orig = np.where(valid_mask_original)
    
    # Get the newly added points
    valid_mask_new = densified_depth > 0.2
    new_points_mask = valid_mask_new & (~valid_mask_original)
    y_new, x_new = np.where(new_points_mask)
    
    # Figure 1: Original image with sparse points
    axes[0].imshow(input_image)
    axes[0].scatter(x_orig, y_orig, color='pink', s=5, alpha=0.8)
    axes[0].set_title(f'Original Image with Sparse Points ({len(x_orig)})')
    
    # Figure 2: Plane masks on image
    # Create a visualization showing the most probable plane for each pixel
    most_probable_plane = np.argmax(plane_masks, axis=0)
    max_probability = np.max(plane_masks, axis=0)
    valid_plane_mask = max_probability > probability_threshold
    
    # Create a colormap for visualization
    cmap = plt.cm.get_cmap('tab20', plane_masks.shape[0])
    # Create a colored visualization
    colored_mask = cmap(most_probable_plane)
    # Set alpha channel based on probability and validity
    colored_mask[..., 3] = np.where(valid_plane_mask, 0.7, 0)
    
    axes[1].imshow(input_image)
    axes[1].imshow(colored_mask)
    axes[1].set_title(f'Plane Masks ({plane_masks.shape[0]} planes)')
    
    # Figure 3: Image with newly added points
    axes[2].imshow(input_image)
    # Plot original points
    axes[2].scatter(x_orig, y_orig, color='pink', s=5, alpha=0.8, label='Original')
    # Plot new points
    axes[2].scatter(x_new, y_new, color='cyan', s=5, alpha=0.8, label='Added')
    axes[2].set_title(f'Added Points ({len(x_new)} new, {len(x_new) + len(x_orig)} total)')
    axes[2].legend()
    
    plt.tight_layout()
    return fig

def strategic_scaffold_filling(input_sparse_depth, plane_masks, min_points_per_plane = 1, fill_ratio=0.005):
    height, width = input_sparse_depth.shape
    valid_mask = input_sparse_depth > 0.2
    densified_depth = input_sparse_depth.copy()

    num_planes = plane_masks.shape[0]
    print("Number of planes: ", num_planes)

    for plane_idx in range(num_planes):
        # Create binary mask for this plane
        plane_mask = plane_masks[plane_idx] > 0.7
        
        # Skip empty planes
        if not np.any(plane_mask):
            continue
            
        # Find sparse points in this plane
        points_in_plane = plane_mask & valid_mask
        n_points = np.sum(points_in_plane)
        
        if n_points >= min_points_per_plane:
            # Calculate plane size and points to add
            plane_size = np.sum(plane_mask)
            n_points_to_add = min(int(plane_size * fill_ratio), plane_size - n_points)
            
            if n_points_to_add > 0:
                # Extract sparse point coordinates and depths
                y_coords, x_coords = np.where(points_in_plane)
                depths = input_sparse_depth[points_in_plane]
                
                # Find empty pixels in this plane
                empty_pixels = plane_mask & (~valid_mask)
                y_empty, x_empty = np.where(empty_pixels)
                
                if len(y_empty) > n_points_to_add:
                    indices = np.random.choice(len(y_empty), n_points_to_add, replace=False)
                    y_sampled = y_empty[indices]
                    x_sampled = x_empty[indices]
                else:
                    y_sampled = y_empty
                    x_sampled = x_empty
                
                # For each sampled point, assign depth from nearest sparse point
                for i in range(len(y_sampled)):
                    if len(y_coords) > 0:
                        distances = np.sqrt((y_sampled[i] - y_coords)**2 + 
                                           (x_sampled[i] - x_coords)**2)
                        nearest_idx = np.argmin(distances)
                        densified_depth[y_sampled[i], x_sampled[i]] = depths[nearest_idx]
                    
    return densified_depth

def main():
    data_dir = "/media/saimouli/Data6T/datasets/VOID_150_small/testing" #"/media/saimouli/RPNG_FLASH_4/datasets/VOID_150/training"
    # save_priors(data_dir)

    device = "cuda"; nsamples = 150; sml_model_path = ""
    depth_predictor = "dpt_hybrid"
    
    folders = [f for f in os.listdir(data_dir) if not f.endswith('.txt')]
    print("Folder length: ", len(folders))
    
    for folder in folders[-1:]:
        print("Folder: ", folder)

        image_folder = os.path.join(data_dir, folder, "image")
        intrinsics = np.genfromtxt(image_folder.replace('image', 'K.txt')).astype(np.float32).reshape((3, 3))
        # get list of images in the image folder
        images = np.sort([f for f in os.listdir(image_folder) if f.endswith('.png')])
        images_path = np.sort([os.path.join(image_folder, f) for f in images])

        sparse_folder = os.path.join(data_dir, folder, "sparse_depth")
        sprase_depth_path = np.sort([os.path.join(sparse_folder, f) for f in images])

        normal_mask_folder = os.path.join(data_dir, folder, "normal_masks/inference")
        normal_masks = sorted([f for f in os.listdir(normal_mask_folder) if f.endswith('.npy') and 'masks' in f], 
                            key=lambda x: int(x.split('_')[0]))
        normal_masks = [os.path.join(normal_mask_folder, f) for f in normal_masks]

        min_depth, max_depth = 0.1, 5.0
        min_pred, max_pred = 0.1, 8.0
        
        method = pipeline.VIDepth(
            depth_predictor, nsamples, sml_model_path, 
            min_pred, max_pred, min_depth, max_depth, device
        )
        
        for i in tqdm(range(len(images))):
            input_image_fp = images_path[i]
            input_sparse_depth_fp = sprase_depth_path[i]
            input_normal_mask_fp = normal_masks[i]
            input_image = utils.read_image(input_image_fp)
            input_sparse_depth = load_sparse_depth(input_sparse_depth_fp)
            normal_mask = np.load(input_normal_mask_fp)
            gt_depth_fp = input_image_fp.replace("image", "ground_truth")
            gt_depth = load_sparse_depth(gt_depth_fp)
            mask = (gt_depth < max_depth)
            mask *= (gt_depth > min_depth)
            gt_depth[~mask] = np.inf
            gt_depth_inv = 1.0 / gt_depth

            validity_map_fp = input_image_fp.replace("image", "validity_map")
            validity_map = np.array(Image.open(validity_map_fp), dtype=np.float32)
            assert(np.all(np.unique(validity_map) == [0, 256]))
            validity_map[validity_map > 0] = 1
            
            depth_infer_inv = method.infer_depth(input_image)
            #resize
            
            input_sparse_depth_valid = (input_sparse_depth < max_depth) * (input_sparse_depth > min_depth)

            input_sparse_depth_valid = input_sparse_depth_valid.astype(bool)
            
            print("Before Pts: ", np.count_nonzero(validity_map))
            reduce_pts = int(np.count_nonzero(validity_map) * 0.90)
            nonzero_indices = np.argwhere(validity_map == 1)
            remove_indices = np.random.choice(len(nonzero_indices), size=reduce_pts, replace=False)
            points_to_remove = nonzero_indices[remove_indices]
            for x, y in points_to_remove:
                validity_map[x, y] = 0
                input_sparse_depth[x, y] = 0
            print("After Pts: ", np.count_nonzero(validity_map))
            input_sparse_depth_valid = (validity_map == 1) * (input_sparse_depth < max_depth) * (input_sparse_depth > min_depth)
            input_sparse_depth_valid = input_sparse_depth_valid.astype(bool)
            #input_sparse_depth[~input_sparse_depth_valid] = np.inf # set invalid depth
            #input_sparse_depth_inv = 1.0 / input_sparse_depth
    
            # Visualize normal map
            normals = detect_normals(depth_infer_inv, None, input_image, False)
            #normals_ga_depth = detect_normals(ga_depth, input_sparse_depth, input_image, False)
            normals_gt_depth = detect_normals(gt_depth, None, input_image, False)

            # Resize the normals to 288x384
            resized_normals = cv2.resize(normals, (384, 288), interpolation=cv2.INTER_LINEAR)
            #resized_normals_gt_depth = cv2.resize(normals_gt_depth, (384, 288), interpolation=cv2.INTER_LINEAR)
            input_image_resize = cv2.resize(input_image, (384, 288), interpolation=cv2.INTER_LINEAR)
            
            # fig, axes = plt.subplots(1,1, figsize=(10, 8))
            # y_idx, x_idx = np.where(validity_map)
            # axes.scatter(x_idx, y_idx, color='pink', s=5, alpha=0.8)
            # axes.imshow(input_image_resize)
            #visualize_normal_map(resized_normals)
            #visualize_normal_map(resized_normals_gt_depth)
            #plot_surface_normals(input_image_resize, resized_normals, axes)
            print("normal mask: ", normal_mask.shape)
            densify_depth = strategic_scaffold_filling(input_sparse_depth, normal_mask, fill_ratio=0.0005)
            fig = visualize_depth_completion_process(input_image, input_sparse_depth, normal_mask, densify_depth)

            # segments_gt = segment_planes_from_normals_dual_flood(
            #     resized_normals_gt_depth, 
            #     seed_spacing=(30, 30),  # Adjust based on your image size
            #     min_region_size=1000    # Adjust based on your expected plane sizes
            # )
            # import time 
            # start_time = time.time()
            # segments = segment_planes_from_normals_dual_flood(
            #     resized_normals, 
            #     seed_spacing=(20, 20),  # Adjust based on your image size
            #     min_region_size=1000    # Adjust based on your expected plane sizes
            # )
            # end_time = time.time()
            # print(f"Time taken for segment_planes_from_normals_dual_flood: {end_time - start_time:.3f} seconds")

            # #plot both segments and segments_gt 
            # num_segments = np.max(segments)
            # #num_segments_gt = np.max(segments_gt)
            # colors = np.random.randint(0, 255, size=(num_segments+1, 3))
            # #colors_gt = np.random.randint(0, 255, size=(num_segments_gt+1, 3))
            # colors[0] = [0, 0, 0]
            # #colors_gt[0] = [0, 0, 0]
            # segment_vis = np.zeros_like(input_image_resize)
            # #segment_vis_gt = np.zeros_like(input_image_resize)
            # for i in range(1, num_segments+1):
            #     mask = (segments == i)
            #     segment_vis[mask] = colors[i]
            # # for i in range(1, num_segments_gt+1):
            # #     mask_gt = (segments_gt == i)
            # #     segment_vis_gt[mask_gt] = colors_gt[i]

            # overlay = cv2.addWeighted(input_image_resize, 0.7, segment_vis, 0.3, 0)
            # #overlay_gt = cv2.addWeighted(input_image_resize, 0.7, segment_vis_gt, 0.3, 0)

            #fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            #axes[0].imshow(input_image_resize)
            #axes[0].set_title('Original Image')

            # axes[1].imshow(segments_gt, cmap='tab20')
            # axes[1].set_title(f'Segmented GT Planes: {num_segments_gt} found')

            #axes[1].imshow(segments, cmap='tab20')
            #axes[1].set_title(f'Segmented Planes: {num_segments} found')

            #axes[2].imshow(overlay)
            #axes[2].set_title('Overlay')

            # axes[4].imshow(overlay_gt)
            # axes[4].set_title('Overlay GT')

            #plt.tight_layout()
            # plt.show()
            
            
            
            # Create a new figure for the segmented image
            
            
            # Compare different clustering methods
            #print("Clustering normals...")
            
            # 1. Bilateral clustering
            # start_time = time.time()
            # bilateral_clusters = bilateral_clustering(normals, similarity_threshold=0.95)
            # print(f"Bilateral clustering time: {time.time() - start_time:.3f} seconds")
            # visualize_clusters(bilateral_clusters, "Bilateral Clustering")
            
            # 2. Region growing
            # start_time = time.time()
            # region_clusters = region_growing_clustering(normals, similarity_threshold=0.95)
            # print(f"Region growing time: {time.time() - start_time:.3f} seconds")
            # visualize_clusters(region_clusters, "Region Growing")
            
            # # 3. K-means
            # start_time = time.time()
            # kmeans_clusters = kmeans_clustering(normals, n_clusters=10)
            # print(f"K-means clustering time: {time.time() - start_time:.3f} seconds")
            # visualize_clusters(kmeans_clusters, "K-means Clustering")
            
            # 4. DBSCAN
            # start_time = time.time()
            # dbscan_clusters = dbscan_clustering(normals, eps=0.1, min_samples=10)
            # print(f"DBSCAN clustering time: {time.time() - start_time:.3f} seconds")
            # visualize_clusters(dbscan_clusters, "DBSCAN Clustering")
            
            # Benchmark propagation methods
            # print("\nBenchmarking propagation methods...")
            # results = benchmark_propagation_methods(normals)
            
            # # Visualize depth maps
            # for method, result in results.items():
            #     visualize_depth_map(result['depth'], f"Depth Map - {method}")
            
            plt.show()


if __name__ == "__main__":
    main()