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

def main():
    data_dir = "/media/saimouli/Data6T/datasets/VOID_150_test/testing" #"/media/saimouli/RPNG_FLASH_4/datasets/VOID_150/training"
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
        images = [f for f in os.listdir(image_folder) if f.endswith('.png')]
        images_path = [os.path.join(image_folder, f) for f in images]

        sparse_folder = os.path.join(data_dir, folder, "sparse_depth")
        sprase_depth_path = [os.path.join(sparse_folder, f) for f in images]

        min_depth, max_depth = 0.1, 5.0
        min_pred, max_pred = 0.1, 8.0
        
        method = pipeline.VIDepth(
            depth_predictor, nsamples, sml_model_path, 
            min_pred, max_pred, min_depth, max_depth, device
        )
        
        for i in tqdm(range(len(images))):
            input_image_fp = images_path[i]
            input_sparse_depth_fp = sprase_depth_path[i]
            input_image = utils.read_image(input_image_fp)
            input_sparse_depth = load_sparse_depth(input_sparse_depth_fp)
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
            input_sparse_depth[~input_sparse_depth_valid] = np.inf # set invalid depth
            input_sparse_depth_inv = 1.0 / input_sparse_depth
            
            print("Before Pts: ", np.count_nonzero(validity_map))
            reduce_pts = int(np.count_nonzero(validity_map) * 0.90)
            nonzero_indices = np.argwhere(validity_map == 1)
            remove_indices = np.random.choice(len(nonzero_indices), size=reduce_pts, replace=False)
            points_to_remove = nonzero_indices[remove_indices]
            for x, y in points_to_remove:
                validity_map[x, y] = 0
            print("After Pts: ", np.count_nonzero(validity_map))
            input_sparse_depth_valid = (validity_map == 1) * (input_sparse_depth < max_depth) * (input_sparse_depth > min_depth)
            input_sparse_depth_valid = input_sparse_depth_valid.astype(bool)
            input_sparse_depth[~input_sparse_depth_valid] = np.inf # set invalid depth
            input_sparse_depth_inv = 1.0 / input_sparse_depth
    
            # Visualize normal map
            normals = detect_normals(depth_infer_inv, None, input_image, False)
            
            fig, axes = plt.subplots(1,1, figsize=(10, 8))
            y_idx, x_idx = np.where(validity_map)
            axes.scatter(x_idx, y_idx, color='red', s=5, alpha=0.8)
            axes.imshow(input_image)
            visualize_normal_map(normals)
            
            # Compare different clustering methods
            print("Clustering normals...")
            
            # 1. Bilateral clustering
            # start_time = time.time()
            # bilateral_clusters = bilateral_clustering(normals, similarity_threshold=0.95)
            # print(f"Bilateral clustering time: {time.time() - start_time:.3f} seconds")
            # visualize_clusters(bilateral_clusters, "Bilateral Clustering")
            
            # 2. Region growing
            start_time = time.time()
            region_clusters = region_growing_clustering(normals, similarity_threshold=0.90)
            print(f"Region growing time: {time.time() - start_time:.3f} seconds")
            visualize_clusters(region_clusters, "Region Growing")
            
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