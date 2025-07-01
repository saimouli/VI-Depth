import torch
import numpy as np
import pipeline
from tqdm import tqdm
import os
from modules.interpolator import Interpolator2D
import modules.midas.utils as utils
from utils_eval import compute_ls_solution
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
import open3d as o3d
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap

class ScalePropagator:
    def __init__(self, rgb_image, sparse_depth, pred_inv_depth, valid_mask, normals=None, poses=None):
        """
        Initialize the scale propagator with visualization capabilities.
        """
        self.rgb_image = rgb_image
        self.sparse_depth = sparse_depth
        self.pred_inv_depth = pred_inv_depth
        self.valid_mask = valid_mask.astype(bool)
        self.poses = poses
        
        self.height, self.width = pred_inv_depth.shape
        
        # Compute sparse point scales
        self.sparse_scales = np.zeros_like(pred_inv_depth)
        valid_indices = np.where(self.valid_mask)
        self.sparse_scales[valid_indices] = sparse_depth[valid_indices] / pred_inv_depth[valid_indices]
        
        # Store coordinates of valid points for faster lookup
        self.valid_points = list(zip(valid_indices[0], valid_indices[1]))
        self.num_valid = len(self.valid_points)
        
        # Compute or use provided normals
        if normals is None:
            self.normals = self._compute_normals(pred_inv_depth)
        else:
            self.normals = normals
            
        # Compute image gradients
        #self.gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY) if len(rgb_image.shape) == 3 else rgb_image
        #self.gray_image = self.gray_image.astype(np.float32) / 255.0
        self.gray_image = self._convert_to_grayscale(rgb_image)
        self.image_gradients = self._compute_image_gradients()
        
        # Initialize propagated scale map and confidence
        self.propagated_scales = np.ones_like(pred_inv_depth)
        self.propagated_scales[self.valid_mask] = self.sparse_scales[self.valid_mask]
        self.confidence_map = np.zeros_like(pred_inv_depth)
        self.confidence_map[self.valid_mask] = 1.0
        
        # Initialize normal similarity map (for visualization)
        self.normal_similarity_map = np.zeros_like(pred_inv_depth)
        
        # For very sparse points, create a large search window
        self.search_radius = min(100, max(30, 400 // max(1, self.num_valid)))
        print(f"Using search radius of {self.search_radius} pixels (based on {self.num_valid} valid points)")
        
        # Create a visualization figure
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 10))
        self.fig.suptitle(f"Scale Propagation based on {self.num_valid} points", fontsize=16)

    def _convert_to_grayscale(self, rgb_image):
        """Convert RGB image to grayscale, handling different input types."""
        if len(rgb_image.shape) == 3:
            # Ensure the image is in the correct format for cv2.cvtColor
            if rgb_image.dtype != np.uint8:
                rgb_image = (rgb_image * 255.0).astype(np.uint8)
            gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = rgb_image
        return gray_image.astype(np.float32) / 255.0
    
    def _compute_image_gradients(self):
        """Compute image gradients for edge awareness"""
        gray_blurred = cv2.GaussianBlur(self.gray_image, (3, 3), 0.5)
        grad_x = cv2.Sobel(gray_blurred, cv2.CV_32F, 1, 0)
        grad_y = cv2.Sobel(gray_blurred, cv2.CV_32F, 0, 1)
        return np.sqrt(grad_x**2 + grad_y**2)
    
    def _compute_normals(self, depth_map):
        depth_map = depth_map.astype(np.float32)
        blurred_depth = cv2.GaussianBlur(depth_map, (5, 5), 0)
        dx = cv2.Sobel(blurred_depth, cv2.CV_32F, 1, 0)
        dy = cv2.Sobel(blurred_depth, cv2.CV_32F, 0, 1)
        normal = np.dstack((-dx, -dy, np.ones((self.height, self.width))))
        norm = np.sqrt(np.sum(normal**2, axis=2, keepdims=True))
        normal = np.divide(normal, norm, out=np.zeros_like(normal), where=norm != 0)
        
        return normal

    def _compute_normal_similarity(self, reference_point=None):
        """
        Compute similarity of each pixel's normal to a reference point or all sparse points.
        If reference_point is None, use the average normal of valid points.
        """
        if reference_point is None:
            # Use average normal from all valid points
            avg_normal = np.zeros(3)
            for y, x in self.valid_points:
                avg_normal += self.normals[y, x]
            
            if len(self.valid_points) > 0:
                avg_normal /= len(self.valid_points)
                avg_normal /= np.linalg.norm(avg_normal)
            else:
                avg_normal = np.array([0, 0, 1])  # Default if no valid points
            
            reference_normal = avg_normal
        else:
            y, x = reference_point
            reference_normal = self.normals[y, x]
        
        # Compute similarity for each pixel
        similarity_map = np.zeros((self.height, self.width))
        for i in range(self.height):
            for j in range(self.width):
                similarity = np.dot(self.normals[i, j], reference_normal)
                similarity_map[i, j] = (similarity + 1) / 2  # Map from [-1,1] to [0,1]
        
        return similarity_map
    def _compute_normal_clusters(self, n_clusters=5):
        """
        Cluster normals to visualize similar surface orientations.
        This helps visualize which areas have similar normals.
        """
        # Reshape normals for clustering
        normals_reshaped = self.normals.reshape(-1, 3)
        
        # Only use a subset of pixels for efficiency
        indices = np.random.choice(normals_reshaped.shape[0], min(10000, normals_reshaped.shape[0]), replace=False)
        normals_sample = normals_reshaped[indices]
        
        # Basic clustering using k-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        _, labels, centers = cv2.kmeans(normals_sample.astype(np.float32), n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Assign each pixel to the nearest cluster
        cluster_map = np.zeros((self.height, self.width), dtype=np.int32)
        for i in range(self.height):
            for j in range(self.width):
                normal = self.normals[i, j]
                best_match = 0
                best_similarity = -1
                for k in range(n_clusters):
                    similarity = np.dot(normal, centers[k])
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = k
                cluster_map[i, j] = best_match
        
        return cluster_map, centers

    def _compute_normal_similarity_to_sparse(self):
        """
        Compute for each pixel, the similarity to the most similar valid point's normal.
        This visualizes how similar each pixel is to the closest sparse point.
        """
        similarity_map = np.zeros((self.height, self.width))
        
        if len(self.valid_points) == 0:
            return similarity_map
            
        for i in range(self.height):
            for j in range(self.width):
                max_similarity = -1
                pixel_normal = self.normals[i, j]
                
                for y, x in self.valid_points:
                    sparse_normal = self.normals[y, x]
                    similarity = np.dot(pixel_normal, sparse_normal)
                    if similarity > max_similarity:
                        max_similarity = similarity
                
                similarity_map[i, j] = (max_similarity + 1) / 2  # Map from [-1,1] to [0,1]
        
        return similarity_map
    
    def _compute_scale_for_pixel(self, i, j, normal_weight_factor=5.0):
        """Compute scale for a pixel using mostly normal-based propagation"""
        # Skip if already a valid point
        if self.valid_mask[i, j]:
            return self.sparse_scales[i, j], 1.0
        
        if self.num_valid == 0:
            return 1.0, 0.0
        
        # Find valid points within search radius
        nearest_points = []
        
        # For extremely sparse inputs, search all valid points
        if self.num_valid < 20:
            for y, x in self.valid_points:
                dist = np.sqrt((j - x)**2 + (i - y)**2)
                if dist < self.search_radius:
                    nearest_points.append((y, x, dist))
        else:
            # For more dense inputs, use a window-based approach
            min_i = max(0, i - self.search_radius)
            max_i = min(self.height - 1, i + self.search_radius)
            min_j = max(0, j - self.search_radius)
            max_j = min(self.width - 1, j + self.search_radius)
            
            for y in range(min_i, max_i + 1):
                for x in range(min_j, max_j + 1):
                    if self.valid_mask[y, x]:
                        dist = np.sqrt((j - x)**2 + (i - y)**2)
                        nearest_points.append((y, x, dist))
        
        if not nearest_points:
            return 1.0, 0.0  # No valid points found within radius
        
        # This pixel's normal
        normal_current = self.normals[i, j]
        
        # Now compute weighted average from all points
        sum_weighted_scale = 0.0
        sum_weights = 0.0
        max_normal_similarity = -1
        
        for y, x, dist in nearest_points:
            # Normal similarity weight (this is the dominant factor)
            normal_point = self.normals[y, x]
            dot_product = np.clip(np.dot(normal_current, normal_point), -1.0, 1.0)
            normal_weight = np.exp(-normal_weight_factor * (1.0 - dot_product))
            
            # Spatial weight (lower importance)
            spatial_weight = np.exp(-dist**2 / (2 * (self.search_radius/3)**2))
            
            # Combined weight (normal similarity dominates)
            total_weight = normal_weight * spatial_weight
            
            # Track maximum normal similarity for visualization
            similarity = (dot_product + 1) / 2  # Map to [0,1]
            if similarity > max_normal_similarity:
                max_normal_similarity = similarity
            
            # Accumulate weighted scale
            sum_weighted_scale += self.sparse_scales[y, x] * total_weight
            sum_weights += total_weight
        
        # Store normal similarity for visualization
        self.normal_similarity_map[i, j] = max_normal_similarity
        
        if sum_weights > 0:
            propagated_scale = sum_weighted_scale / sum_weights
            confidence = min(1.0, max_normal_similarity)  # Confidence based on normal similarity
            return propagated_scale, confidence
        else:
            return 1.0, 0.0
    
    def visualize_step(self, step, total_steps):
        """
        Visualize the current state of propagation.
        """
        # Clear previous plots
        for ax in self.axes.flatten():
            ax.clear()
            
        # RGB image with sparse points
        self.axes[0, 0].imshow(self.rgb_image)
        y_coords, x_coords = np.where(self.valid_mask)
        self.axes[0, 0].scatter(x_coords, y_coords, c='r', s=5)
        self.axes[0, 0].set_title(f'RGB Image with {len(x_coords)} Sparse Points')
        self.axes[0, 0].axis('off')
        
        # Normal visualization (using HSV colormap)
        normal_viz = np.zeros((self.height, self.width, 3))
        for i in range(self.height):
            for j in range(self.width):
                # Map x,y components to hue and saturation
                normal = self.normals[i, j]
                h = (np.arctan2(normal[1], normal[0]) / (2*np.pi) + 0.5) % 1.0  # Map to [0,1]
                s = np.sqrt(normal[0]**2 + normal[1]**2)  # Saturation from x,y magnitude
                v = 0.8  # Constant value
                
                # Convert HSV to RGB
                normal_viz[i, j] = plt.cm.hsv(h)[:3]
        
        self.axes[0, 1].imshow(normal_viz)
        self.axes[0, 1].set_title('Surface Normals (HSV encoding)')
        self.axes[0, 1].axis('off')
        
        # Normal similarity to sparse points
        self.axes[0, 2].imshow(self.normal_similarity_map, cmap='viridis', vmin=0, vmax=1)
        self.axes[0, 2].set_title('Normal Similarity to Sparse Points')
        self.axes[0, 2].axis('off')
        
        # Current propagated scales
        normalized_scales = (self.propagated_scales - np.min(self.propagated_scales)) / (np.max(self.propagated_scales) - np.min(self.propagated_scales) + 1e-8)
        self.axes[1, 0].imshow(normalized_scales, cmap='plasma')
        self.axes[1, 0].set_title(f'Propagated Scales (Step {step}/{total_steps})')
        self.axes[1, 0].axis('off')
        
        # Confidence map
        self.axes[1, 1].imshow(self.confidence_map, cmap='hot', vmin=0, vmax=1)
        self.axes[1, 1].set_title('Confidence Map')
        self.axes[1, 1].axis('off')
        
        # Normal clusters
        cluster_map, _ = self._compute_normal_clusters(n_clusters=6)
        cmap = plt.cm.get_cmap('tab10', 6)
        self.axes[1, 2].imshow(cluster_map, cmap=cmap, vmin=0, vmax=5)
        self.axes[1, 2].set_title('Normal Clusters (Similar Orientation)')
        self.axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.pause(0.1)  # Short pause to update display
        
    def propagate_scales_visualize(self, total_steps=10):
        """
        Propagate scales with step-by-step visualization.
        Focus on normal-based propagation.
        
        Args:
            total_steps: Number of visualization steps
            
        Returns:
            propagated_scales: Raw propagated scales
            normalized_scales: Normalized scales (0-1 range)
            confidence_map: Confidence in propagated scales
        """
        # Compute normal similarity to sparse points (for visualization)
        self.normal_similarity_map = self._compute_normal_similarity_to_sparse()
        
        # Process pixels in batches for visualization
        pixel_coords = []
        for i in range(self.height):
            for j in range(self.width):
                if not self.valid_mask[i, j]:
                    pixel_coords.append((i, j))
        
        total_pixels = len(pixel_coords)
        pixels_per_step = total_pixels // total_steps
        
        # Process in steps for visualization
        for step in range(total_steps):
            print(f"Step {step+1}/{total_steps}")
            
            start_idx = step * pixels_per_step
            end_idx = min(total_pixels, (step + 1) * pixels_per_step)
            
            # Process this batch of pixels
            for idx in range(start_idx, end_idx):
                i, j = pixel_coords[idx]
                
                scale, confidence = self._compute_scale_for_pixel(i, j, normal_weight_factor=5.0)
                self.propagated_scales[i, j] = scale
                self.confidence_map[i, j] = confidence
            
            # Visualize current state
            self.visualize_step(step + 1, total_steps)
        
        # Apply median filter to remove outliers while preserving edges
        self.propagated_scales = cv2.medianBlur(self.propagated_scales.astype(np.float32), 3)
        
        # Final visualization
        self.visualize_step(total_steps, total_steps)
        
        # Normalize for visualization and network input
        min_scale = np.min(self.propagated_scales)
        max_scale = np.max(self.propagated_scales)
        normalized_scales = (self.propagated_scales - min_scale) / (max_scale - min_scale + 1e-8)
        
        return self.propagated_scales, normalized_scales, self.confidence_map

def propagate_scales_with_visualization(rgb_image, sparse_depth, pred_inv_depth, valid_mask, normals=None, poses=None):
    """
    Propagate scale values with visualization, focusing on normal-based propagation.
    
    Args:
        rgb_image: RGB image (HxWx3)
        sparse_depth: Sparse depth measurements (HxW)
        pred_inv_depth: Predicted inverse depth from model (HxW)
        valid_mask: Boolean mask of valid sparse points (HxW)
        normals: Surface normals (HxWx3) or None to compute them
        poses: Camera poses from VIO or None
        
    Returns:
        propagated_scales: Raw propagated scales
        normalized_scales: Normalized scales (0-1 range)
        confidence_map: Confidence in propagated scales
    """
    propagator = ScalePropagator(
        rgb_image=rgb_image,
        sparse_depth=sparse_depth,
        pred_inv_depth=pred_inv_depth,
        valid_mask=valid_mask,
        normals=normals,
        poses=poses
    )
    
    # Show initial state
    propagator.visualize_step(0, 10)
    
    # Propagate with visualization
    propagated_scales, normalized_scales, confidence_map = propagator.propagate_scales_visualize(total_steps=10)
    
    plt.show()  # Keep the final visualization
    
    return propagated_scales, normalized_scales, confidence_map

def load_sparse_depth(input_sparse_depth_fp):
    input_sparse_depth = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32) / 256.0
    input_sparse_depth[input_sparse_depth <= 0] = 0.0
    return input_sparse_depth

def plot_surface_normals(rgb_image, normals, ax, scale=5):
    """Plot surface normals as arrows on the given axis."""
    ax.imshow(rgb_image)

    # Plot surface normals as arrows
    for y in range(0, normals.shape[0], 10):
        for x in range(0, normals.shape[1], 10):
            if np.isnan(normals[y, x]).any():
                continue
            ax.arrow(x, y, normals[y, x, 0] * scale, normals[y, x, 1] * scale,
                     head_width=2, head_length=2, fc='r', ec='r')
    
    ax.axis('off')

def comprehensive_visualization(image, depth_infer_inv, gt_depth, sparse_valid, ga_depth, interpolated_scale, save_path=None):
    """
    Create a comprehensive visualization of the depth processing pipeline.
    
    Args:
        image: Original RGB image [H, W, 3]
        depth_infer_inv: Inferred inverse depth map [H, W]
        gt_depth: Ground truth depth map [H, W]
        sparse_valid: Binary mask of valid sparse points [H, W]
        ga_depth: Globally aligned depth [H, W]
        interpolated_scale: Interpolated scale map [H, W]
        save_path: Optional path to save the visualization
    """
    # Compute normals
    normals = detect_normals(depth_infer_inv, None, image, False)  # [H, W, 3]
    normals_gt = detect_normals(gt_depth, None, image, False)      # [H, W, 3]
    
    # Create figure with 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Plot original image with sparse points overlaid
    axes[0].imshow(image)
    y_idx, x_idx = np.where(sparse_valid)
    axes[0].scatter(x_idx, y_idx, color='red', s=5, alpha=0.8)
    axes[0].set_title(f"Original Image with {len(y_idx)} Sparse Points")
    axes[0].axis('off')
    
    # Plot GA depth with sparse points
    ga_plot = axes[1].imshow(ga_depth, cmap='plasma')
    axes[1].scatter(x_idx, y_idx, color='red', s=5, alpha=0.8)
    axes[1].set_title("Globally Aligned Depth")
    axes[1].axis('off')
    fig.colorbar(ga_plot, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot inferred normals with direction vectors
    plot_surface_normals(image, normals, axes[2], scale=10)
    axes[2].scatter(x_idx, y_idx, color='yellow', s=5, alpha=0.8)
    axes[2].set_title("Inferred Surface Normals")
    
    # Plot GT normals with direction vectors
    plot_surface_normals(image, normals_gt, axes[3], scale=10)
    axes[3].scatter(x_idx, y_idx, color='yellow', s=5, alpha=0.8)
    axes[3].set_title("GT Surface Normals")
    
    # Plot interpolated scale with sparse points
    scale_plot = axes[4].imshow(interpolated_scale, cmap='viridis')
    axes[4].scatter(x_idx, y_idx, color='red', s=5, alpha=0.8)
    axes[4].set_title("Interpolated Scale")
    axes[4].axis('off')
    fig.colorbar(scale_plot, ax=axes[4], fraction=0.046, pad=0.04)
    
    # Empty subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

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
    
    if display:
        # Map the normal vectors to the [0, 255] range and convert to uint8
        normal_viz = (normal + 1) * 127.5
        normal_viz = normal_viz.clip(0, 255).astype(np.uint8)
        normal_bgr = cv2.cvtColor(normal_viz, cv2.COLOR_RGB2BGR)
        
        bmsk = (np.sqrt(dx**2 + dy**2) + 1) * 127.5
        bmsk = bmsk.clip(0, 255).astype(np.uint8)
        
        cv2.imshow("Normal x", normal_viz[:, :, 0])
        cv2.imshow("Normal y", normal_viz[:, :, 1])
        cv2.imshow("Normal z", normal_viz[:, :, 2])
        cv2.imshow("Sig grad", bmsk)
        #add normals in all three directions
        normal_viz_xyz = np.abs(normal_viz[:, :, 0]) + np.abs(normal_viz[:, :, 1])
        cv2.imshow("Normal xyz", normal_viz_xyz)
        cv2.waitKey(0)

    return normal

def simple_scale_propagation(rgb_image, sparse_depth, pred_inv_depth, valid_mask, normals):
    """
    Simple propagation of scale values from sparse points, respecting normal directions and edges.
    
    Args:
        rgb_image: RGB image (HxWx3)
        sparse_depth: Sparse depth measurements (HxW)
        pred_inv_depth: Predicted inverse depth from model (HxW) 
        valid_mask: Boolean mask of valid sparse points (HxW)
        normals: Surface normals (HxWx3)
        
    Returns:
        propagated_scales: Propagated scale values
    """
    # Setup
    height, width = pred_inv_depth.shape
    valid_mask = valid_mask.astype(bool)
    
    # Compute sparse scales
    sparse_scales = np.ones_like(pred_inv_depth)
    sparse_scales[valid_mask] = sparse_depth[valid_mask] / pred_inv_depth[valid_mask]
    
    # Compute edges from RGB image
    #gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    rgb_image = (rgb_image * 255.0).astype(np.uint8)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    edges = cv2.Canny(np.uint8(gray_image * 255), 100, 200)
    dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    
    # Create propagation map
    propagated_scales = np.ones_like(pred_inv_depth)
    propagated_scales[valid_mask] = sparse_scales[valid_mask]
    
    # Create visited mask
    visited = np.zeros_like(valid_mask)
    visited[valid_mask] = True
    
    # Search radius - adjust as needed
    search_radius = 30
    
    # Create a visualization figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Scale Propagation using Normals and Edges", fontsize=16)
    
    # Initialize the visualization
    axes[0, 0].imshow(rgb_image)
    axes[0, 0].set_title("RGB Image")
    
    axes[0, 1].imshow(dilated_edges, cmap='gray')
    axes[0, 1].set_title("Edge Boundaries")
    
    # Setup normal visualization
    axes[0, 2].imshow(rgb_image)
    for y in range(0, height, 10):
        for x in range(0, width, 10):
            if not np.isnan(normals[y, x]).any():
                axes[0, 2].arrow(x, y, normals[y, x, 0] * 5, normals[y, x, 1] * 5,
                              head_width=1, head_length=1, fc='r', ec='r')
    axes[0, 2].set_title("Surface Normals")
    
    # Setup the sparse points visualization
    sparse_vis = np.zeros_like(pred_inv_depth)
    y_idx, x_idx = np.where(valid_mask)
    #sparse_vis[valid_mask] = sparse_scales[valid_mask]
    axes[1, 0].scatter(x_idx, y_idx, color='red', s=5, alpha=0.8)
    #axes[1, 0].imshow(sparse_vis, cmap='viridis')
    axes[1, 0].set_title("Sparse Scales")
    
    # Setup propagated scale visualization
    prop_im = axes[1, 1].imshow(propagated_scales, cmap='viridis')
    axes[1, 1].set_title("Propagated Scales (Initial)")
    
    # Setup confidence visualization
    conf_vis = np.zeros_like(pred_inv_depth)
    conf_vis[valid_mask] = 1.0
    conf_im = axes[1, 2].imshow(conf_vis, cmap='hot', vmin=0, vmax=1)
    axes[1, 2].set_title("Confidence Map")
    
    plt.tight_layout()
    plt.pause(1.0)
    
    # Initialize queue with valid points
    queue = []
    valid_coords = np.where(valid_mask)
    for i, j in zip(valid_coords[0], valid_coords[1]):
        queue.append((i, j))
    
    # Neighbor offsets (4-connectivity)
    offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    
    # Propagation loop
    step = 0
    max_steps = 150
    while queue and step < max_steps:
        step += 1
        new_queue = []
        
        for i, j in queue:
            for di, dj in offsets:
                ni, nj = i + di, j + dj
                
                # Skip if out of bounds
                if not (0 <= ni < height and 0 <= nj < width):
                    continue
                
                # Skip if already visited
                if visited[ni, nj]:
                    continue
                
                # Skip if hitting an edge
                if dilated_edges[ni, nj] > 0:
                    continue
                
                # Check normal similarity
                dot_product = np.dot(normals[i, j], normals[ni, nj])
                if dot_product < 0.7:  # Normal diverges too much
                    continue
                
                # Propagate scale
                propagated_scales[ni, nj] = propagated_scales[i, j]
                
                # Mark as visited and add to new queue
                visited[ni, nj] = True
                new_queue.append((ni, nj))
        
        # Update visualization
        prop_im.set_data(propagated_scales)
        conf_vis = visited.astype(float)
        conf_im.set_data(conf_vis)
        axes[1, 1].set_title(f"Propagated Scales (Step {step}/{max_steps})")
        plt.pause(0.5)  # Longer pause to see the propagation
        
        queue = new_queue
        print(f"Step {step}/{max_steps}: {len(queue)} new points")
        
        if not queue:
            print("Queue empty, propagation complete")
            break
    
    # Final visualization - hold until closed
    axes[1, 1].set_title(f"Propagated Scales (Final)")
    plt.show()
    
    return propagated_scales


def propagate_scales(rgb_image, sparse_depth, pred_inv_depth, valid_mask, normals=None, poses=None):
    # propagator = ScalePropagator(
    #     rgb_image=rgb_image,
    #     sparse_depth=sparse_depth,
    #     pred_inv_depth=pred_inv_depth,
    #     valid_mask=valid_mask,
    #     normals=normals,
    #     poses=None
    # )
    
    # propagated_scales, normalized_scales, confidence_map = propagator.propagate_scales_visualize(total_steps=10)
    # #fig = propagator.visualize_results()
    # plt.show()
    
    propagated_scales = simple_scale_propagation(rgb_image, sparse_depth, pred_inv_depth, valid_mask, normals)

    return propagated_scales

 
def create_scaffolding(depth_pred, input_sparse_depth_valid, input_sparse_depth_inv):
    assert (np.sum(input_sparse_depth_valid) >= 3), "not enough valid sparse points"
    int_depth,_,_ = compute_ls_solution(depth_pred, input_sparse_depth_inv, input_sparse_depth_valid, min_pred, max_pred)
    ScaleMapInterpolator = Interpolator2D(
        pred_inv = int_depth,
        sparse_depth_inv = input_sparse_depth_inv,
        valid = input_sparse_depth_valid,
    )
    ScaleMapInterpolator.generate_interpolated_scale_map(
        interpolate_method='linear', 
        fill_corners=False
    )
    int_scales = ScaleMapInterpolator.interpolated_scale_map.astype(np.float32)
    int_scales = utils.normalize_unit_range(int_scales)
    
    return int_depth, int_scales

if __name__ == "__main__":
    data_dir = "/media/saimouli/Data6T/datasets/VOID_150_test/testing" #"/media/saimouli/RPNG_FLASH_4/datasets/VOID_150/training"
    # save_priors(data_dir)

    device = "cuda"; nsamples = 150; sml_model_path = ""
    depth_predictor = "dpt_hybrid"
    
    folders = [f for f in os.listdir(data_dir) if not f.endswith('.txt')]
    print("Folder length: ", len(folders))
    
    for folder in folders:
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
            input_sparse_depth_valid = (input_sparse_depth < max_depth) * (input_sparse_depth > min_depth)

            input_sparse_depth_valid = input_sparse_depth_valid.astype(bool)
            input_sparse_depth[~input_sparse_depth_valid] = np.inf # set invalid depth
            input_sparse_depth_inv = 1.0 / input_sparse_depth
            
            # print("Before Pts: ", np.count_nonzero(validity_map))
            # reduce_pts = int(np.count_nonzero(validity_map) * 0.90)
            # nonzero_indices = np.argwhere(validity_map == 1)
            # remove_indices = np.random.choice(len(nonzero_indices), size=reduce_pts, replace=False)
            # points_to_remove = nonzero_indices[remove_indices]
            # for x, y in points_to_remove:
            #     validity_map[x, y] = 0
            # print("After Pts: ", np.count_nonzero(validity_map))
            # input_sparse_depth_valid = (validity_map == 1) * (input_sparse_depth < max_depth) * (input_sparse_depth > min_depth)
            # input_sparse_depth_valid = input_sparse_depth_valid.astype(bool)
            # input_sparse_depth[~input_sparse_depth_valid] = np.inf # set invalid depth
            # input_sparse_depth_inv = 1.0 / input_sparse_depth
            
            #ga_depth_inv, int_scale = create_scaffolding(depth_infer_inv, input_sparse_depth_valid, input_sparse_depth_inv)
            
            #visualize_3d_normals(torch.from_numpy(depth_infer_inv), input_image)
            #ax = plt.gca()
            normals_cam = detect_normals(depth_infer_inv, input_sparse_depth_inv, input_image, False)
            #plot_surface_normals(input_image, normals_cam, ax, scale=10)
            #plt.show()
            
            

            #normals = compute_normals_from_depth(gt_depth)
            #plot_surface_normals(input_image, normals)
            #comprehensive_visualization(input_image, depth_infer_inv, 1.0/gt_depth_inv, input_sparse_depth_valid, 1.0/ga_depth_inv, int_scale)

            
            prop_scales = propagate_scales(input_image, input_sparse_depth, 
                                                                depth_infer_inv, input_sparse_depth_valid, normals=normals_cam)