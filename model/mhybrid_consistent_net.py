import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if module_path not in sys.path:
    sys.path.append(module_path)
from modules.midas.midas_net_cons_custom import MidasNet_small_cons_videpth, ResNetEncoder, ResNetEncoder_orig
import numpy as np
import modules.midas.utils as utils
import modules.midas.transforms as transforms
from utils.camera import Camera, pose_to_se3, se3_to_pose, se3_update
#import pypose as pp 
from utils.pose import Pose
from functools import partial
#from pytorch3d.transforms import se3_exp_map, se3_log_map
import torchvision
import math
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cv2

#from modules.midas.blocks import OutputConv

class SparsityAwarePooling(nn.Module):
    def __init__(self, kernel_size, stride, padding=0, use_max_pool=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_max_pool = use_max_pool
        #if use_max_pool:
        #    self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        #else:
        #    self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding, count_include_pad=False)
        #self.use_max_pool = use_max_pool
    
    def forward(self, sparse_depth):
        valid_mask = (sparse_depth > 0).float()
        sparse_depth_valid = sparse_depth * valid_mask
        
        if self.use_max_pool:
            x_mod = sparse_depth.clone()
            x_mod[x_mod == 0] = -1e9
            
            # Apply max pooling
            pooled = F.max_pool2d(
                x_mod,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding
            )
            # Reset negative values to zero
            pooled[pooled < 0] = 0
            return pooled
        
        else:
            x_sum = F.avg_pool2d(
                sparse_depth,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ) * (self.kernel_size * self.kernel_size)
            
            # Count valid elements per window
            count = F.avg_pool2d(
                valid_mask,
                kernel_size=self.kernel_size,
                stride=self.stride, 
                padding=self.padding
            ) * (self.kernel_size * self.kernel_size)
            
            count = torch.clamp(count, min=1e-6)
            pooled = x_sum / count
            pooled = pooled * (count > 0).float()      
            return pooled

@torch.no_grad()
def visualize_sparse_depth(sparse_depth_inv, save_dir='debug_vis'):
    """
    Visualize sparse depth at original and different scales for debugging
    
    Args:
        sparse_depth_inv: The inverse sparse depth tensor [B, 1, H, W]
        save_dir: Directory to save visualization files
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create pooling layers
    pool_s2 = SparsityAwarePooling(kernel_size=2, stride=2)
    pool_s4 = SparsityAwarePooling(kernel_size=4, stride=4)
    pool_s8 = SparsityAwarePooling(kernel_size=8, stride=8)
    
    # Get pooled versions
    with torch.no_grad():
        depth_s2 = pool_s2(sparse_depth_inv)
        depth_s4 = pool_s4(sparse_depth_inv)
        depth_s8 = pool_s8(sparse_depth_inv)
    
    # Only process the first batch item for simplicity
    batch_idx = 0
    
    # Extract sparse points for each scale
    def extract_sparse_points(depth):
        valid_mask = (depth[batch_idx, 0] > 0)
        indices = torch.nonzero(valid_mask, as_tuple=True)
        points = [(y.item(), x.item()) for y, x in zip(*indices)]
        depths = depth[batch_idx, 0][valid_mask].cpu().numpy()
        return points, depths
    
    sparse_points_orig, depths_orig = extract_sparse_points(sparse_depth_inv)
    sparse_points_s2, depths_s2 = extract_sparse_points(depth_s2)
    sparse_points_s4, depths_s4 = extract_sparse_points(depth_s4)
    sparse_points_s8, depths_s8 = extract_sparse_points(depth_s8)
    
    # Original resolution
    H, W = sparse_depth_inv.shape[2:]
    # Sizes at each scale
    H_s2, W_s2 = depth_s2.shape[2:]
    H_s4, W_s4 = depth_s4.shape[2:]
    H_s8, W_s8 = depth_s8.shape[2:]
    
    # Create a colormap for depths
    all_depths = np.concatenate([depths_orig, depths_s2, depths_s4, depths_s8])
    vmin, vmax = np.min(all_depths), np.max(all_depths)
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Helper function to create scatter plot
    def plot_sparse_points(points, depths, h, w, title, filename):
        plt.figure(figsize=(10, 8))
        plt.scatter([p[1] for p in points], [p[0] for p in points], 
                   c=depths, cmap='viridis', norm=norm, alpha=0.7)
        plt.colorbar(label='Depth')
        plt.xlim(0, w)
        plt.ylim(h, 0)  # Invert y-axis to match image coordinates
        plt.title(f"{title} - {len(points)} points")
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()
    
    # Create density maps
    def create_density_map(points, h, w, title, filename):
        density_map = np.zeros((h, w), dtype=np.float32)
        for y, x in points:
            if 0 <= y < h and 0 <= x < w:
                density_map[y, x] = 1.0
        
        # Apply Gaussian blur to make the visualization clearer
        density_map = cv2.GaussianBlur(density_map, (7, 7), 0)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(density_map, cmap='hot')
        plt.colorbar(label='Density')
        plt.title(f"{title} - {len(points)} points")
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()
        
        return density_map
    
    # Plot scatter and density for each resolution
    plot_sparse_points(sparse_points_orig, depths_orig, H, W, 
                     "Original Resolution Sparse Depth", "orig_scatter.png")
    plot_sparse_points(sparse_points_s2, depths_s2, H_s2, W_s2, 
                     "1/2 Scale Sparse Depth", "s2_scatter.png")
    plot_sparse_points(sparse_points_s4, depths_s4, H_s4, W_s4, 
                     "1/4 Scale Sparse Depth", "s4_scatter.png")
    plot_sparse_points(sparse_points_s8, depths_s8, H_s8, W_s8, 
                     "1/8 Scale Sparse Depth", "s8_scatter.png")
    
    density_orig = create_density_map(sparse_points_orig, H, W, 
                                    "Original Resolution Density", "orig_density.png")
    density_s2 = create_density_map(sparse_points_s2, H_s2, W_s2, 
                                  "1/2 Scale Density", "s2_density.png")
    density_s4 = create_density_map(sparse_points_s4, H_s4, W_s4, 
                                  "1/4 Scale Density", "s4_density.png")
    density_s8 = create_density_map(sparse_points_s8, H_s8, W_s8, 
                                  "1/8 Scale Density", "s8_density.png")
    
    # Create a combined visualization
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    
    # First row: scatter plots
    axs[0, 0].scatter([p[1] for p in sparse_points_orig], [p[0] for p in sparse_points_orig], 
                    c=depths_orig, cmap='viridis', norm=norm, alpha=0.7, s=2)
    axs[0, 0].set_title(f"Original - {len(sparse_points_orig)} points")
    axs[0, 0].set_xlim(0, W)
    axs[0, 0].set_ylim(H, 0)
    
    axs[0, 1].scatter([p[1] for p in sparse_points_s2], [p[0] for p in sparse_points_s2], 
                    c=depths_s2, cmap='viridis', norm=norm, alpha=0.7, s=2)
    axs[0, 1].set_title(f"1/2 Scale - {len(sparse_points_s2)} points")
    axs[0, 1].set_xlim(0, W_s2)
    axs[0, 1].set_ylim(H_s2, 0)
    
    axs[0, 2].scatter([p[1] for p in sparse_points_s4], [p[0] for p in sparse_points_s4], 
                    c=depths_s4, cmap='viridis', norm=norm, alpha=0.7, s=2)
    axs[0, 2].set_title(f"1/4 Scale - {len(sparse_points_s4)} points")
    axs[0, 2].set_xlim(0, W_s4)
    axs[0, 2].set_ylim(H_s4, 0)
    
    axs[0, 3].scatter([p[1] for p in sparse_points_s8], [p[0] for p in sparse_points_s8], 
                    c=depths_s8, cmap='viridis', norm=norm, alpha=0.7, s=2)
    axs[0, 3].set_title(f"1/8 Scale - {len(sparse_points_s8)} points")
    axs[0, 3].set_xlim(0, W_s8)
    axs[0, 3].set_ylim(H_s8, 0)
    
    # Second row: density maps
    axs[1, 0].imshow(density_orig, cmap='hot')
    axs[1, 0].set_title("Original Density")
    
    axs[1, 1].imshow(density_s2, cmap='hot')
    axs[1, 1].set_title("1/2 Scale Density")
    
    axs[1, 2].imshow(density_s4, cmap='hot')
    axs[1, 2].set_title("1/4 Scale Density")
    
    axs[1, 3].imshow(density_s8, cmap='hot')
    axs[1, 3].set_title("1/8 Scale Density")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "combined_visualization.png"))
    plt.close()
    
    # Create relative density analysis
    original_density = len(sparse_points_orig) / (H * W)
    s2_density = len(sparse_points_s2) / (H_s2 * W_s2)
    s4_density = len(sparse_points_s4) / (H_s4 * W_s4)
    s8_density = len(sparse_points_s8) / (H_s8 * W_s8)
    
    density_increase = {
        "1/2 scale": s2_density / original_density,
        "1/4 scale": s4_density / original_density,
        "1/8 scale": s8_density / original_density
    }
    
    # Plot density analysis
    plt.figure(figsize=(10, 6))
    scales = ["Original", "1/2 Scale", "1/4 Scale", "1/8 Scale"]
    densities = [original_density, s2_density, s4_density, s8_density]
    plt.bar(scales, densities)
    plt.title("Sparse Point Density at Different Scales")
    plt.ylabel("Points per pixel")
    plt.savefig(os.path.join(save_dir, "density_analysis.png"))
    plt.close()
    
    # Write statistics to file
    with open(os.path.join(save_dir, "sparse_depth_stats.txt"), "w") as f:
        f.write(f"Original resolution: {H}x{W}, {len(sparse_points_orig)} points, {original_density:.6f} points/pixel\n")
        f.write(f"1/2 scale: {H_s2}x{W_s2}, {len(sparse_points_s2)} points, {s2_density:.6f} points/pixel, {density_increase['1/2 scale']:.2f}x increase\n")
        f.write(f"1/4 scale: {H_s4}x{W_s4}, {len(sparse_points_s4)} points, {s4_density:.6f} points/pixel, {density_increase['1/4 scale']:.2f}x increase\n")
        f.write(f"1/8 scale: {H_s8}x{W_s8}, {len(sparse_points_s8)} points, {s8_density:.6f} points/pixel, {density_increase['1/8 scale']:.2f}x increase\n")
    
    print(f"Visualization saved to {save_dir}")
    
#Implement multi-scale affinity propagation
class MultiScaleAffinityPropagation(nn.Module):
    def __init__(self, feature_dim=32, scales=[8,4,2], num_layers=18):
        super().__init__()
        self.scales = scales
        
        self.encoder = ResNetEncoder(
            num_layers=num_layers, 
            num_input_images=1, 
            pretrained=True, 
            out_chs=feature_dim 
            #stride=4  # Set to smallest stride to get highest resolution features
        )
        
        self.initial_depth_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_dim // 2, 1, kernel_size=1)
            ) for _ in scales
        ])
        
        # # Affinity modules for each level
        # self.affinity_modules = nn.ModuleList([
        #     AffinityPropagation(feature_dim=feature_dim, chunk_size=1024 * (2**i))
        #     for i, _ in enumerate(scales)
        # ])
        
        # Shift correction modules
        self.shift_correction_modules = nn.ModuleList([
            ShiftCorrectionModule(feature_dim) for _ in scales
        ])
        
        # Cross-scale fusion modules
        self.cross_scale_modules = nn.ModuleList([
            nn.Conv2d(feature_dim + 1, feature_dim, kernel_size=3, padding=1)
            for _ in range(len(scales) - 1)
        ])
        
        # Pooling for sparse depth
        self.sparse_pooling = nn.ModuleList([
            SparsityAwarePooling(kernel_size=scale, stride=scale)
            for scale in scales
        ])
        
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ) for _ in scales
        ])
        
        # Final refinement layer
        self.final_refine = nn.Sequential(
            nn.Conv2d(feature_dim + 1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )
    
    def forward(self, image, sparse_depth_inv, normals=None):
        B, _, H, W = sparse_depth_inv.shape
        
        # Get encoder features at multiple scales
        feats_s2, feats_s4, feats_s8 = self.encoder(image)
        encoder_features = [feats_s8, feats_s4, feats_s2]
        
        outputs = {
            'initial_depths': {},
            'propagated_depths': {}
        }
        current_features = None
        current_depth=None
        
        # Process each scale from coarsest to finest
        for i, scale_idx in enumerate(range(len(self.scales))):
            scale = self.scales[scale_idx]
            scale_name = f's{scale}'
            
            # Get features for this scale
            features = encoder_features[scale_idx]
            
            pooled_sparse_depth = self.sparse_pooling[scale_idx](sparse_depth_inv)
            sparse_points = self._extract_sparse_points(pooled_sparse_depth)
            
            # Fuse previous scale features if available
            if current_features is not None and current_depth is not None:
                # Upsample previous features and depth
                upsampled_depth = F.interpolate(current_depth,
                                                size=features.shape[2:], 
                                                mode='bilinear',
                                                align_corners=False)
                #concatenate and process
                fused = self.cross_scale_modules[i-1](
                    torch.cat([features, upsampled_depth], dim=1)
                )
                features = features + fused
            
            # Process decoder for this scale
            decoded_features = self.decoders[scale_idx](features)
            
            # Get initial depth prediction
            initial_depth = self.initial_depth_predictors[scale_idx](decoded_features)
            outputs['initial_depths'][scale_name] = utils.inv2depth(initial_depth)
            
            # Skip affinity correction for coarsest scale if first decoder stage
            if scale_idx == 0 and current_features is None:
                # For coarsest scale with no prior info, just use initial depth
                current_features = decoded_features
                current_depth = initial_depth
                outputs['propagated_depths'][scale_name] = utils.inv2depth(initial_depth)
                continue
            
            corrected_depth, point_features = self.shift_correction_modules[scale_idx](
                decoded_features, initial_depth, sparse_points
            )
            
            outputs['propagated_depths'][scale_name] = utils.inv2depth(corrected_depth)
            
            if point_features is not None:
                current_features = decoded_features + point_features
            else:
                current_features = decoded_features
            
            current_depth = corrected_depth
        
        return current_depth, outputs
    
    def _extract_sparse_points(self, depth):
        B, _, H, W = depth.shape
        valid_mask = (depth > 0).float()
        sparse_points_idx = []
        sparse_depth_values = []
        
        for b in range(B):
            indices = torch.where(valid_mask[b, 0] > 0)
            points = [(y.item(), x.item()) for y, x in zip(indices[0], indices[1])]
            depths = depth[b, 0][indices].detach()
            sparse_points_idx.append(points)
            sparse_depth_values.append(depths)
        
        return sparse_points_idx, sparse_depth_values
    
class ShiftCorrectionModule(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        # Transformer for depth point features
        self.transformer = nn.TransformerEncoderLayer(
            d_model=feature_dim + 1,  # +1 for depth value
            nhead=3,
            dim_feedforward=256,
            batch_first=True
        )
        # Projections for cross-attention
        self.query_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.key_proj = nn.Linear(feature_dim + 1, feature_dim)
        
        # Point feature processing
        self.point_feature_proj = nn.Linear(feature_dim + 1, feature_dim)
        
        self.confidence = nn.Sequential(
            nn.Conv2d(feature_dim + 2 + 1, 64, kernel_size=3, padding=1),  # +2 for initial and shifted depths, +1 for dist field
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.scale = math.sqrt(feature_dim)
        
    def forward(self, features, initial_depth, sparse_points, full_res=None):
        B, C, H, W = features.shape
        device = features.device
        
        sparse_points_idx, sparse_depth_values = sparse_points
        N_max = max([len(points) for points in sparse_points_idx] or [0])
        
        if N_max == 0:
            return initial_depth, None
        
        # Add positional embeddings to features
        pos_embeddings = self._get_positional_embeddings(H, W, C, device)
        features_with_pos = features + pos_embeddings
        
        # Sample features and depths at sparse point locations
        sparse_features = []
        valid_mask = torch.zeros(B, N_max, device=device, dtype=torch.bool)
        initial_depth_at_points = torch.zeros(B, N_max, device=device)
        
        for b in range(B):
            points = sparse_points_idx[b]
            depths = sparse_depth_values[b]
            
            if len(points) == 0:
                sparse_features.append(torch.zeros(N_max, C + 1, device=device))
                continue
            
            # Get features at point locations
            point_features = torch.zeros(len(points), C, device=device)
            for i, (y, x) in enumerate(points):
                if 0 <= y < H and 0 <= x < W:
                    point_features[i] = features_with_pos[b, :, y, x]
                    initial_depth_at_points[b, i] = initial_depth[b, 0, y, x]
            
            #combine with depth values
            point_features_with_depth = torch.cat([
                point_features,
                depths.view(-1,1)
            ], dim=-1)
            
            valid_mask[b, :len(points)] = True
            
            #pad if necessary
            if len(points) < N_max:
                padding = torch.zeros(N_max - len(points), C + 1, device=device)
                point_features_with_depth = torch.cat([point_features_with_depth, padding], dim=0)
                
            sparse_features.append(point_features_with_depth)
        
        sparse_features = torch.stack(sparse_features, dim=0)
        
        # Process point features with transformer
        key_padding_mask = ~valid_mask
        sparse_features_transformed = self.transformer(
            sparse_features,
            src_key_padding_mask=key_padding_mask
        )
        
        # Compute cross-attention between image and point features
        q_features = self.query_proj(features).view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
        k_features = self.key_proj(sparse_features_transformed)  # [B, N_max, C]
        
        # Compute attention affinity
        affinity = torch.bmm(q_features, k_features.transpose(1, 2)) / self.scale
        
        #Mask invalid points and apply softmax
        affinity.masked_fill_(~valid_mask.unsqueeze(1), -1e9)
        affinity = F.softmax(affinity, dim=-1)  # [B, H*W, N_max]
        
        # Get input depths at sparse point locations
        input_depths = torch.zeros(B, N_max, 1, device=device)
        for b in range(B):
            if len(sparse_depth_values[b]) > 0:
                input_depths[b, :len(sparse_depth_values[b]), 0] = sparse_depth_values[b]
        
        # Calculate depth errors at sparse point locations
        # (d_j - D_initial_j)
        # D_shift_i = sum_j A_ij(D_initial_i - D_initial_j) + sum_j A_ij d_j
        # First term: Compute the relative depth structure
        initial_depth_flat = initial_depth.view(B, 1, H*W).permute(0, 2, 1)  # [B, H*W, 1]
        
        # For each pixel, we need its depth difference with respect to each sparse point
        initial_depth_at_points = initial_depth_at_points.unsqueeze(-1)  # [B, N_max, 1]
        # Compute relative depth: D_initial_i - D_initial_j
        relative_depth = initial_depth_flat - initial_depth_at_points.transpose(1, 2) 
        
        # This computes sum_j A_ij(D_initial_i - D_initial_j)
        weighted_relative_depth = (affinity * relative_depth).sum(dim=2, keepdim=True) 
        weighted_input_depth = torch.bmm(affinity, input_depths)  # [B, H*W, 1]
        shift_corrected_depth = (weighted_relative_depth + weighted_input_depth).view(B, H, W, 1).permute(0, 3, 1, 2)
        
        # Generate distance field for confidence prediction
        distance_field = self._compute_distance_field(B, H, W, sparse_points_idx, device)
        
        # Create feature map from point features for fusion in the decoder
        point_features_proj = self.point_feature_proj(sparse_features_transformed)
        point_feature_map = torch.bmm(affinity, point_features_proj).view(B, H, W, C).permute(0, 3, 1, 2)
        
        # Predict confidence for blending initial and corrected depths
        confidence_input = torch.cat([
            features,
            initial_depth,
            shift_corrected_depth,
            distance_field
        ], dim=1)
        
        blend_weight = self.confidence(confidence_input)
        
        # Blend between initial and shift-corrected depth
        final_depth = blend_weight * shift_corrected_depth + (1 - blend_weight) * initial_depth
        
        return final_depth, point_feature_map
    
    def _compute_distance_field(self, B, H, W, sparse_points_idx, device):
        """Compute normalized distance field from sparse points for each batch"""
        distance_fields = torch.ones(B, 1, H, W, device=device)
        
        for b in range(B):
            points = sparse_points_idx[b]
            if not points:
                distance_fields[b] = 1.0
                continue
                
            y_grid, x_grid = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )
            
            field = torch.ones(H, W, device=device) * float('inf')
            
            for y, x in points:
                if 0 <= y < H and 0 <= x < W:
                    dist = torch.sqrt((y_grid - y).float()**2 + (x_grid - x).float()**2)
                    field = torch.minimum(field, dist)
            
            # Normalize distance field
            max_possible_dist = math.sqrt(H*H + W*W)
            field = torch.clamp(field / (max_possible_dist * 0.1), 0.0, 1.0)
            distance_fields[b, 0] = field
            
        return distance_fields
    
    def _get_positional_embeddings(self, H, W, C, device):
        """Create sinusoidal positional embeddings"""
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        
        dim_t = C // 4
        pos_embed = torch.zeros(1, C, H, W, device=device)
        div_term = torch.exp(torch.arange(0, dim_t, 2, device=device) * (-math.log(10000.0) / dim_t))
        
        pos_y = y_grid.unsqueeze(0).unsqueeze(1).expand(1, dim_t, H, W)
        pos_embed[:, 0:dim_t:2] = torch.sin(pos_y[:, ::2] * div_term.view(1, -1, 1, 1))
        pos_embed[:, 1:dim_t:2] = torch.cos(pos_y[:, 1::2] * div_term.view(1, -1, 1, 1))
        
        pos_x = x_grid.unsqueeze(0).unsqueeze(1).expand(1, dim_t, H, W)
        pos_embed[:, dim_t:2*dim_t:2] = torch.sin(pos_x[:, ::2] * div_term.view(1, -1, 1, 1))
        pos_embed[:, dim_t+1:2*dim_t:2] = torch.cos(pos_x[:, 1::2] * div_term.view(1, -1, 1, 1))
        
        return pos_embed
    
#propagate scale based on normals
# class AffinityPropagation(nn.Module):
#     def __init__(self, feature_dim, hidden_dim=64, chunk_size=1024):
#         super().__init__()
#         self.chunk_size = chunk_size
#         self.scale = math.sqrt(feature_dim)
        
#         # Transformer layer for sparse point features
#         #self.transformer_layer = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4)
#         # Projection for feature-based affinity
#         self.query_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
#         self.key_proj = nn.Linear(feature_dim + 1, feature_dim) #for sparse features
        
#         # Point feature processing
#         self.point_feature_proj = nn.Linear(feature_dim + 1, feature_dim)
        
#         # For creating point features from image features and depth
#         self.transformer_encoder = nn.TransformerEncoderLayer(
#             d_model=feature_dim + 1,  # +1 for depth value
#             nhead=3,
#             dim_feedforward=256,
#             batch_first=True
#         )
        
#         self.softmax = nn.Softmax(dim=-1)
        
#     def forward(self, features, sparse_points_idx, sparse_depth_values, full_res=(288, 384)):
#         """
#         Args:
#             features: Feature map from decoder stage [B, C, H, W]
#             sparse_points_idx: List of lists of (y, x) coordinates of sparse points [B]
#             sparse_scales: List of tensors of sparse scale values [B]
#             relative_normals: Relative normals from MiDaS [B, 3, H, W]
#             full_res: Tuple (H_full, W_full) indicating the full resolution of sparse points and normals
        
#         Returns:
#             affinity_map: Affinity weights [B, H*W, N_max]
#         """
#         B, C, H, W = features.shape
#         device = features.device
#         # Handle case with no valid points
#         N_max = max([len(points) for points in sparse_points_idx] or [0])
#         if N_max == 0:
#             return torch.zeros(B, H * W, 1, device=device)
        
#         #positional embeddings
#         pos_embeddings = self._get_positional_embeddings(H, W, C, device)
#         features_with_pos = features + pos_embeddings
        
#         # Sample features at sparse point locations and create point features
#         sparse_features = []
#         valid_mask = torch.zeros(B, N_max, device=device, dtype=torch.bool)
        
#         for b in range(B):
#             points = sparse_points_idx[b]
#             depths = sparse_depth_values[b]
            
#             if len(points) == 0:
#                 # No points for this batch
#                 sparse_features.append(torch.zeros(N_max, C + 1, device=device))
#                 continue
            
#             #Get features at point locations
#             point_features = torch.zeros(len(points), C, device=device)
#             for i, (y, x) in enumerate(points):
#                 if 0 <= y < H and 0 <= x < W:
#                     point_features[i] = features_with_pos[b, :, y, x]
            
#             # Combine with depth values
#             point_features_with_depth = torch.cat([
#                 point_features,
#                 depths.view(-1,1)
#             ], dim=-1)  # [N, C + 1]
            
#             # Create valid mask
#             valid_mask[b, :len(points)] = True
            
#             #pad if necessary
#             if len(points) < N_max:
#                 padding = torch.zeros(N_max - len(points), C + 1, device=device)
#                 point_features_with_depth = torch.cat([point_features_with_depth, padding], dim=0)
            
#             sparse_features.append(point_features_with_depth)
        
#         sparse_features = torch.stack(sparse_features, dim=0)
        
#         #Apply transformer to point features
#         key_padding_mask = ~valid_mask
#         sparse_features_transformed = self.transformer_encoder(sparse_features, src_key_padding_mask=key_padding_mask)
        
#         #Compute affinities
#         q_features = self.query_proj(features).view(B, -1, H * W).permute(0, 2, 1)
#         k_features = self.key_proj(sparse_features_transformed)  # [B, N_max, C]
        
#         # Compute affinities in chunks to save memory
#         affinity_map = torch.zeros(B, H * W, N_max, device=device)
        
#         for i in range(0, H * W, self.chunk_size):
#             end = min(i + self.chunk_size, H * W)
#             q_chunk = q_features[:, i:end, :]
            
#             # Compute dot product similarity
#             chunk_affinities = torch.bmm(q_chunk, k_features.transpose(1, 2))
#             chunk_affinities = chunk_affinities / self.scale
            
#             chunk_affinities.masked_fill_(~valid_mask.unsqueeze(1), -1e9)
#             affinity_map[:, i:end, :] = self.softmax(chunk_affinities)
            
#         return affinity_map

#     def _get_positional_embeddings(self, H, W, C, device):
#         # Create position encodings
#         y_grid, x_grid = torch.meshgrid(
#             torch.linspace(-1, 1, H, device=device),
#             torch.linspace(-1, 1, W, device=device),
#             indexing='ij'
#         )
        
#         dim_t = C // 4
#         pos_embed = torch.zeros(1, C, H, W, device=device)
#         div_term = torch.exp(torch.arange(0, dim_t, 2, device=device) * (-math.log(10000.0) / dim_t))
        
#         pos_y = y_grid.unsqueeze(0).unsqueeze(1).expand(1, dim_t, H, W)
#         pos_embed[:, 0:dim_t:2] = torch.sin(pos_y[:, ::2] * div_term.view(1, -1, 1, 1))
#         pos_embed[:, 1:dim_t:2] = torch.cos(pos_y[:, 1::2] * div_term.view(1, -1, 1, 1))
        
#         pos_x = x_grid.unsqueeze(0).unsqueeze(1).expand(1, dim_t, H, W)
#         pos_embed[:, dim_t:2*dim_t:2] = torch.sin(pos_x[:, ::2] * div_term.view(1, -1, 1, 1))
#         pos_embed[:, dim_t+1:2*dim_t:2] = torch.cos(pos_x[:, 1::2] * div_term.view(1, -1, 1, 1))
        
#         if C > 2*dim_t:
#             pos_embed[:, 2*dim_t:] = pos_embed[:, :C-2*dim_t]
            
#         return pos_embed
    
class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class ProjectionInputDepth(nn.Module):
    def __init__(self, cost_dim, hidden_dim, out_chs, downsample_ratio=2):
        super().__init__()
        self.out_chs = out_chs
        self.convc1 = nn.Conv2d(cost_dim, hidden_dim, 1, padding=0)
        self.convc2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        
        self.convd1 = nn.Conv2d(1, hidden_dim, 7, padding=3)
        self.convd2 = nn.Conv2d(hidden_dim, 64, 3, padding=1)
        
        self.convd = nn.Conv2d(64+hidden_dim, out_chs - 1, 3, padding=1)
        
        # Downsample layer
        if downsample_ratio > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(out_chs, out_chs, kernel_size=3, stride=downsample_ratio, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.downsample = nn.Identity()
        
    def forward(self, depth, cost):
        #cost -> shape [B, cost_dim, H, W]
        cor = F.relu(self.convc1(cost))
        cor = F.relu(self.convc2(cor))

        # depth -> shape [B, 1, H, W]
        dept = F.relu(self.convd1(depth))
        dept = F.relu(self.convd2(dept))
        cor_dfm = torch.cat([cor, dept], dim=1)
        
        out_d = F.relu(self.convd(cor_dfm))
        # final shape -> [B, out_chs, H, W]
        out = torch.cat([out_d, depth], dim=1)
        return self.downsample(out)

class ProjectionInputPose(nn.Module):
    def __init__(self, cost_dim, hidden_dim, out_chs, downsample_ratio=8):
        super().__init__()
        self.out_chs = out_chs
        self.convc1 = nn.Conv2d(cost_dim, hidden_dim, 1, padding=0)
        self.convc2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
               
        self.convp1 = nn.Conv2d(6, hidden_dim, 7, padding=3)
        self.convp2 = nn.Conv2d(hidden_dim, 64, 3, padding=1)
        
        self.convp = nn.Conv2d(64+hidden_dim, out_chs - 6, 3, padding=1)
        # Downsample layer
        if downsample_ratio > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(out_chs, out_chs, kernel_size=3, stride=downsample_ratio, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.downsample = nn.Identity()
            
    def forward(self, pose, cost):
        bs, _, h, w = cost.shape
        cor = F.relu(self.convc1(cost))
        cor = F.relu(self.convc2(cor))
        
        pose = pose.view(bs, 6, 1, 1).repeat(1, 1, h, w)
        pfm = F.relu(self.convp1(pose))
        pfm = F.relu(self.convp2(pfm))
        cor_pfm = torch.cat([cor, pfm], dim=1)
                
        out_p = F.relu(self.convp(cor_pfm))
        out= torch.cat([out_p, pose], dim=1)
        return self.downsample(out)
    
#Limitations: 
# limited receptive field: (1,5) and (5,1) convolutions separable
# lack of global context
class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

class OutputScaleConv(nn.Module):
    """Output conv block.
    """
    def __init__(self, features, groups, activation, non_negative):

        super(OutputScaleConv, self).__init__()

        self.output_conv = nn.Sequential(
            nn.Conv2d(features, features//2, kernel_size=3, stride=1, padding=1, groups=groups),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(features//2, 32, kernel_size=3, stride=1, padding=1),
            activation,
            
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            activation,
            
            # Final 1x1 conv to get single-channel scale map
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            # Ensure positive scale factors
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

    def forward(self, x):        
        return self.output_conv(x)
    
class BasicUpdateBlockDepth(nn.Module):
    def __init__(self, hidden_dim=128, cost_dim=64, ratio=3, context_dim=64, min_pred=None, max_pred=None, log_fn=None):
        super(BasicUpdateBlockDepth, self).__init__()
                
        self.encoder = ProjectionInputDepth(cost_dim=cost_dim, hidden_dim=hidden_dim, out_chs=hidden_dim, downsample_ratio=1)
        self.depth_gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=self.encoder.out_chs+context_dim)
        self.depth_head = OutputScaleConv(features=hidden_dim, groups=1, activation=nn.ReLU(False), non_negative=False)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim*2, ratio*ratio*9, 1, padding=0))
        self.min_pred = min_pred
        self.max_pred = max_pred
        self.log_fn = log_fn

    def forward(self, hidden, cost_func, inv_depth, context, seq_len=4):
        inv_depth_list = [] 
        mask_list = []
        cost_means = []
        for i in range(seq_len):
            cost, _ = cost_func(inv_depth)
            #print("cost {}: mean{}".format(i, cost.mean()))
            
            # If necessary, downsample inv_depth and context to match the cost resolution.
            if cost.shape[-2:] != inv_depth.shape[-2:]:
                inv_depth_low = F.interpolate(inv_depth, size=cost.shape[-2:], mode='bilinear', align_corners=True)
            else:
                inv_depth_low = inv_depth
                
            if self.log_fn:
                with torch.no_grad():
                    self.log_fn(f"UpdateBlock/Iteration_{i}/cost_mean", cost.mean().item(), on_step=True, logger=True)
                    cost_means.append(cost.mean().item())
                
            input_features = self.encoder(inv_depth_low, cost) #(b,1,72,96),(b,32,72,96) 
            inp_i = torch.cat([context, input_features], dim=1)#(2,32,72,96),(2,128,288,384) 

            #if self.log_fn:
            #    self.log_fn(f"UpdateBlock/Iteration_{i}/hidden_state_norm_before", hidden.norm().item(), on_step=True, logger=True)
                
            hidden = self.depth_gru(hidden, inp_i)
            
            #if self.log_fn:
            #    self.log_fn(f"UpdateBlock/Iteration_{i}/hidden_state_norm_after", hidden.norm().item(), on_step=True, logger=True)
                
            delta_scales = self.depth_head(hidden)
            delta_scales = 1.0 + 0.5 * torch.tanh(delta_scales)  # [0.5, 1.5] range #ensure positive scale
            #print(f"inv_depth mean step {i}: {inv_depth.mean().item()}")
            #print("delta_scales mean step", i, delta_scales.mean().item())
            
            if self.log_fn:
                self.log_fn(f"UpdateBlock/Iteration_{i}/delta_scales_mean", delta_scales.mean().item(), on_step=True, logger=True)
                
            inv_depth_pred = inv_depth * delta_scales

            # clamp pred to min and max
            if self.min_pred is not None and self.max_pred is not None:
                inv_depth_pred = torch.clamp(
                    inv_depth_pred, 
                    min=1.0 / self.max_pred, 
                    max=1.0 / self.min_pred
                )   
            
             # *** Update for next iteration ***
            inv_depth = inv_depth_pred
            # scale mask to balence gradients
            mask = 0.25 * self.mask(hidden) #helps numerical stability
            
            inv_depth_list.append(inv_depth_pred)
            mask_list.append(mask)
            
        return hidden, mask_list, inv_depth_list, cost_means
    
class PoseHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128):
        super(PoseHead, self).__init__()
        
        self.conv1_pose = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2_pose = nn.Conv2d(hidden_dim, 6, 3, padding=1)

        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x_p):
        out = self.conv2_pose(self.relu(self.conv1_pose(x_p))).mean(3).mean(2)
        return torch.cat([out[:, :3], 0.1 * out[:, 3:]], dim=1)
    
class BasicUpdateBlockPose(nn.Module):
    def __init__(self, hidden_dim=128, cost_dim=64, ratio=3, context_dim=64, log_fn=None):
        super(BasicUpdateBlockPose, self).__init__()
        self.encoder = ProjectionInputPose(cost_dim=cost_dim, hidden_dim=hidden_dim, out_chs=hidden_dim, downsample_ratio=2)
        self.pose_gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=self.encoder.out_chs+context_dim)
        self.pose_head = PoseHead(hidden_dim, hidden_dim=hidden_dim)
        
    def forward(self, hidden_p, cost_func, ref_pose, inp, seq_len=4, depth_only=False):
        pose_list = []
        #convert to ref_pose from 4x4 to 6x1
        ref_pose_init = pose_to_se3(ref_pose)
        for i in range(seq_len):
            res = cost_func(poseC2W=ref_pose)
            cost = res['cost']
            input_features = self.encoder(ref_pose_init, cost)
            inp_i = torch.cat([inp, input_features], dim=1) #(b, 32, 36, 48)
                
            hidden_p = self.pose_gru(hidden_p, inp_i)
            delta_pose = self.pose_head(hidden_p)
            
            if depth_only:
                pose = ref_pose
            else:
                pose = ref_pose @ se3_to_pose(delta_pose)
            pose_list.append(pose)
        return hidden_p, pose_list
class UpMaskNet(nn.Module):
    def __init__(self, hidden_dim=128, ratio=8):
        super(UpMaskNet, self).__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim*2, ratio*ratio*9, 1, padding=0))

    def forward(self, feat):
        # scale mask to balence gradients
        mask = .25 * self.mask(feat)
        return mask
    
class midasConsNet(nn.Module):
    def __init__(self, min_pred, max_pred, min_depth, max_depth, nsamples, sml_model_path, 
                 is_train=True, log_fn=None, isConvGRU=False):
        super(midasConsNet, self).__init__()
        self.is_train = is_train
        self.log_fn = log_fn  # Logger function from Lightning module
        self.UseConvGRU = isConvGRU
        
        self.min_pred, self.max_pred = min_pred, max_pred
        self.min_depth, self.max_depth = min_depth, max_depth

        model_transforms = transforms.get_transforms("dpt_hybrid", "void", str(nsamples))
        self.ScaleMapLearner_transform = model_transforms["sml_model"]

        self.hidden_dim = 128 #96
        self.cost_dim = 32 #64 
        self.iter_steps = 0
        self.seq_len = 0
        
        #self.ap = AffinityPropagation(feature_dim=self.cost_dim, hidden_dim=self.hidden_dim)
        self.depth_prop = MultiScaleAffinityPropagation(feature_dim=self.cost_dim)
        
        #self.fnet = ResNetEncoder(out_chs=self.cost_dim, stride=4)
        #self.fnet_midas = MidasNet_small_cons_videpth(features=32, in_channels=3)
        #self.cnet_depth_affinity = ResNetEncoder(out_chs=self.hidden_dim + self.cost_dim-1, stride=4, context_num=1, pretrained=False)
        #self.cnet_depth = ResNetEncoder_orig(out_chs=self.hidden_dim + self.cost_dim, stride=4, context_num=2, pretrained=False)
        
        #self.upsample_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # self.up = nn.Sequential(
        #     nn.ConvTranspose2d(self.hidden_dim + self.cost_dim-1, self.hidden_dim + self.cost_dim-1, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(self.hidden_dim + self.cost_dim-1),
        #     nn.ReLU(inplace=True),

        #     nn.ConvTranspose2d(self.hidden_dim + self.cost_dim-1, self.hidden_dim + self.cost_dim-1, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(self.hidden_dim + self.cost_dim-1),
        #     nn.ReLU(inplace=True),
        # )
        
        if self.UseConvGRU:
            self.contextPose = ResNetEncoder(out_chs=self.hidden_dim+self.cost_dim, 
                                           num_input_images=2)#pose
            
            self.update_block_depth = BasicUpdateBlockDepth(hidden_dim=self.hidden_dim, 
                                                            cost_dim=self.cost_dim,
                                                            ratio=3, 
                                                            context_dim=self.cost_dim,
                                                            min_pred=self.min_pred,
                                                            max_pred=self.max_pred,
                                                            log_fn=self.log_fn)
            
            self.update_block_pose = BasicUpdateBlockPose(hidden_dim=self.hidden_dim,
                                                          cost_dim=self.cost_dim,
                                                          ratio=3,
                                                          context_dim=self.cost_dim,
                                                          )
            self.inter_sup = False
            
        else:
            pass
            #self.refine_net = DepthPoseRefineNet(hidden_dim=self.hidden_dim)
            
        self.scaleOutput = OutputScaleConv(features=self.hidden_dim + self.cost_dim, groups=1, 
                                            activation=nn.ReLU(False), non_negative=False)

    def upsample_depth(self, depth, mask, ratio):
        """ Upsample depth field [H/ratio, W/ratio, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = depth.shape

        mask = mask.view(N, 1, 9, ratio, ratio, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(depth, [3,3], padding=1)
        up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 1, ratio*H, ratio*W)

    def clam_depth(self, inv_depth_pred):
        if self.min_pred is not None and self.max_pred is not None:
            inv_depth_pred = torch.clamp(
                inv_depth_pred, 
                min=1.0 / self.max_pred, 
                max=1.0 / self.min_pred
            )
        return inv_depth_pred

    def get_cost_each(self, tgt_poseC2W, poseC2W, fmap, fmap_ref, depth, K, scale_factor):
        """
            ga_depth: (b, 1, h, w)
            fmap, fmap_ref: (b, c, h, w)
        """
        device = depth.device
        ref_cam = Camera(K=K.float(), Twc=poseC2W).scaled(scale_factor).to(device)
        cam = Camera(K=K.float(), Twc=tgt_poseC2W).scaled(scale_factor).to(device) # tcw = Identity
        #resize depth to same size as the feature map
        depth_small = F.interpolate(depth, size=(fmap.shape[2], fmap.shape[3]), mode='bilinear', align_corners=False)
        # Reconstruct world points from target_camera
        world_points = cam.reconstruct(depth_small, frame='w')
        # Project world points onto reference camera
        ref_coords = ref_cam.project(world_points, frame='w', normalize=True) #(b, h, w,2)
        with torch.no_grad():
           coord_valid = (ref_coords.abs().max(dim=-1)[0] <= 1)  # [B, H, W]
           #depth_valid = (world_points[..., 2] > 0.1) & (world_points[..., 2] < 5.0)
           valid_mask = coord_valid #(coord_valid & depth_valid).float()
           
        fmap_warped = F.grid_sample(fmap_ref, ref_coords, 
                                    mode='bilinear', padding_mode='zeros', align_corners=True) # (b, c, h, w)
        
        cost = (fmap - fmap_warped)**2 * valid_mask.unsqueeze(1) #cost = (fmap * fmap_warped).sum(dim=1, keepdim=True) #try correlation
        #cost = cost.mean(dim=1, keepdim=True)
        
        # fmap_norm = F.normalize(fmap, p=2, dim=1)
        # fmap_warped_norm = F.normalize(fmap_warped, p=2, dim=1)
        # correlation = (fmap_norm * fmap_warped_norm).sum(dim=1, keepdim=True)
        # cost = (1.0 - correlation) * valid_mask.unsqueeze(1)

        return {
            'cost': cost,
            'fmap': fmap,
            'fmap_warped': fmap_warped,
            'valid_mask': valid_mask
        }
    
    def depth_cost_calc(self, inv_depth, fmap, fmaps_ref, pose_list, tgt_pose, K, scale_factor):
        cost_list = []
        warping_vis = []
        for idx, (pose, fmap_r) in enumerate(zip(pose_list, fmaps_ref)):
            result = self.get_cost_each(tgt_pose, pose, fmap, fmap_r, 
                                      utils.inv2depth(inv_depth), K, scale_factor)
            
            if idx == 0 and self.is_train:
                warping_vis.append({
                    'src_feat': result['fmap'][0].detach(),
                    'warped_feat': result['fmap_warped'][0].detach(),
                    'valid_mask': result['valid_mask'][0].detach(),
                    'cost': result['cost'][0].detach()
                })

            cost_list.append(result['cost'])  # (b, c,h, w) (1,64,144,192)
        
        cost = torch.stack(cost_list, dim=1).mean(dim=1)
        return cost, warping_vis

    def preprocess_batch(self, *args, keys, device=None):
        assert len(args) == len(keys)
        
        batch_size = args[0].shape[0]
        batch_inputs = []
        device = device or args[0].device

        for i in range(batch_size):
            sample = {key: arg[i].squeeze().cpu().numpy() for key, arg in zip(keys, args)}

            sample = self.ScaleMapLearner_transform(sample)
            
            x = torch.cat([torch.tensor(sample[key]) for key in keys], 0)
            x = x.to(device)
            batch_inputs.append(x)
        
        return torch.stack(batch_inputs, dim=0)
    
    def visualize_sparse_scales(self, sparse_scales, valid_mask, tgt_interp, sparse_scales_pre, valid_mask_pre, int_interp_pre):
        """
        Visualize and compare sparse scales before and after preprocessing, including interpolated values.
        
        Args:
            sparse_scales: Original sparse scales [B, H, W]
            valid_mask: Original valid mask [B, H, W]
            tgt_interp: Original interpolated values [B, H, W] or [B, 1, H, W]
            sparse_scales_pre: Preprocessed sparse scales [B, 1, H, W]
            valid_mask_pre: Preprocessed valid mask [B, 1, H, W]
            int_interp_pre: Preprocessed interpolated values [B, 1, H, W]
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Convert to numpy for visualization
        batch_idx = 0  # Visualize first batch
        
        # Original scales at valid locations
        orig_scales = sparse_scales[batch_idx][valid_mask[batch_idx] > 0].detach().cpu().numpy()
        
        # Preprocessed scales at valid locations
        pre_scales = sparse_scales_pre[batch_idx, 0][valid_mask_pre[batch_idx, 0] > 0].detach().cpu().numpy()
        
        # Check if tgt_interp is already batched with channel dimension
        if len(tgt_interp.shape) == 3:  # [B, H, W]
            orig_interp = tgt_interp[batch_idx].detach().cpu().numpy()
        else:  # [B, C, H, W]
            orig_interp = tgt_interp[batch_idx, 0].detach().cpu().numpy()
        
        # Preprocessed interpolated values
        pre_interp = int_interp_pre[batch_idx, 0].detach().cpu().numpy()
        
        # Create figure with multiple subplots - 2x3 grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Histogram of original scales
        if len(orig_scales) > 0:
            axes[0, 0].hist(orig_scales, bins=50, alpha=0.7, color='blue')
            axes[0, 0].set_title('Original Sparse Scales Distribution')
            axes[0, 0].set_xlabel('Scale Value')
            axes[0, 0].set_ylabel('Frequency')
            
            # Add stats text
            stats_text = f"Mean: {orig_scales.mean():.4f}\nStd: {orig_scales.std():.4f}\n"
            stats_text += f"Min: {orig_scales.min():.4f}\nMax: {orig_scales.max():.4f}\n"
            stats_text += f"Valid points: {len(orig_scales)}"
            
            axes[0, 0].text(0.05, 0.95, stats_text, transform=axes[0, 0].transAxes,
                            fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        else:
            axes[0, 0].text(0.5, 0.5, "No valid original scales", ha='center', va='center')
            axes[0, 0].set_title('Original Sparse Scales (Empty)')
        
        # 2. Histogram of preprocessed scales
        if len(pre_scales) > 0:
            axes[0, 1].hist(pre_scales, bins=50, alpha=0.7, color='green')
            axes[0, 1].set_title('Preprocessed Sparse Scales Distribution')
            axes[0, 1].set_xlabel('Scale Value')
            axes[0, 1].set_ylabel('Frequency')
            
            # Add stats text
            stats_text = f"Mean: {pre_scales.mean():.4f}\nStd: {pre_scales.std():.4f}\n"
            stats_text += f"Min: {pre_scales.min():.4f}\nMax: {pre_scales.max():.4f}\n"
            stats_text += f"Valid points: {len(pre_scales)}"
            
            axes[0, 1].text(0.05, 0.95, stats_text, transform=axes[0, 1].transAxes,
                            fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        else:
            axes[0, 1].text(0.5, 0.5, "No valid preprocessed scales", ha='center', va='center')
            axes[0, 1].set_title('Preprocessed Sparse Scales (Empty)')
        
        # 3. Original sparse scales visualization
        orig_vis = np.zeros((sparse_scales.shape[1], sparse_scales.shape[2]))
        if len(orig_scales) > 0:
            orig_vis[valid_mask[batch_idx].detach().cpu().numpy() > 0] = orig_scales
        im0 = axes[1, 0].imshow(orig_vis, cmap='viridis')
        axes[1, 0].set_title('Original Sparse Scales (Valid Locations)')
        plt.colorbar(im0, ax=axes[1, 0])
        
        # 4. Preprocessed sparse scales visualization
        pre_vis = np.zeros((sparse_scales_pre.shape[2], sparse_scales_pre.shape[3]))
        if len(pre_scales) > 0:
            pre_vis[valid_mask_pre[batch_idx, 0].detach().cpu().numpy() > 0] = pre_scales
        im1 = axes[1, 1].imshow(pre_vis, cmap='viridis')
        axes[1, 1].set_title('Preprocessed Sparse Scales (Valid Locations)')
        plt.colorbar(im1, ax=axes[1, 1])
        
        # 5. Original interpolated values (dense map)
        im2 = axes[0, 2].imshow(orig_interp, cmap='plasma')
        axes[0, 2].set_title('Original Interpolated Scale Map')
        plt.colorbar(im2, ax=axes[0, 2])
        
        # Add stats text for original interp
        stats_text = f"Mean: {orig_interp.mean():.4f}\nStd: {orig_interp.std():.4f}\n"
        stats_text += f"Min: {orig_interp.min():.4f}\nMax: {orig_interp.max():.4f}"
        
        axes[0, 2].text(0.05, 0.95, stats_text, transform=axes[0, 2].transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # 6. Preprocessed interpolated values (dense map)
        im3 = axes[1, 2].imshow(pre_interp, cmap='plasma')
        axes[1, 2].set_title('Preprocessed Interpolated Scale Map')
        plt.colorbar(im3, ax=axes[1, 2])
        
        # Add stats text for preprocessed interp
        stats_text = f"Mean: {pre_interp.mean():.4f}\nStd: {pre_interp.std():.4f}\n"
        stats_text += f"Min: {pre_interp.min():.4f}\nMax: {pre_interp.max():.4f}"
        
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Add global title
        plt.suptitle("Sparse Scales and Interpolation Comparison", fontsize=16)
        
        plt.tight_layout()
        plt.show()
        
        # Print additional statistics
        print("\n===== Scale Statistics =====")
        
        # Count of valid points
        print(f"Original valid points: {(valid_mask[batch_idx]>0).sum().item()}")
        print(f"Preprocessed valid points: {(valid_mask_pre[batch_idx, 0]>0).sum().item()}")
        
        # Stats for sparse scales
        if len(orig_scales) > 0:
            print(f"\nOriginal Sparse Scales - Mean: {orig_scales.mean():.4f}, Std: {orig_scales.std():.4f}, Min: {orig_scales.min():.4f}, Max: {orig_scales.max():.4f}")
        else:
            print("\nOriginal Sparse Scales - No valid data")
        
        if len(pre_scales) > 0:
            print(f"Preprocessed Sparse Scales - Mean: {pre_scales.mean():.4f}, Std: {pre_scales.std():.4f}, Min: {pre_scales.min():.4f}, Max: {pre_scales.max():.4f}")
        else:
            print("Preprocessed Sparse Scales - No valid data")
        
        # Stats for interpolated maps
        print(f"\nOriginal Interpolated Map - Mean: {orig_interp.mean():.4f}, Std: {orig_interp.std():.4f}, Min: {orig_interp.min():.4f}, Max: {orig_interp.max():.4f}")
        print(f"Preprocessed Interpolated Map - Mean: {pre_interp.mean():.4f}, Std: {pre_interp.std():.4f}, Min: {pre_interp.min():.4f}, Max: {pre_interp.max():.4f}")
        
    def batch_interpolate_scale_maps(self, pred_inv_depth, sparse_depth, valid_mask, interpolate_method='linear', normalize=True):
        """
        Batch-friendly interpolation of scale maps for sparse depth points.
        
        Args:
            pred_inv_depth: tensor of shape [B, H, W] - predicted inverse depth
            sparse_depth: tensor of shape [B, H, W] - sparse depth measurements
            valid_mask: tensor of shape [B, H, W] - binary mask where depth measurements exist
            interpolate_method: interpolation method ('linear', 'cubic', 'nearest')
            normalize: whether to normalize the output scale maps to [0, 1]
            
        Returns:
            interpolated_scale_maps: tensor of shape [B, H, W]
        """
        batch_size, height, width = pred_inv_depth.shape
        device = pred_inv_depth.device
        
        # Convert sparse depth to inverse depth
        sparse_depth_inv = torch.zeros_like(sparse_depth)
        valid_indices = valid_mask > 0
        sparse_depth_inv[valid_indices] = 1.0 / (sparse_depth[valid_indices] + 1e-6)
        
        # Calculate raw scale factors
        scale_factors = torch.zeros_like(sparse_depth_inv)
        valid_idx = valid_mask > 0
        scale_factors[valid_idx] = sparse_depth_inv[valid_idx] / (pred_inv_depth[valid_idx] + 1e-6)
        
        # Process each item in the batch
        interpolated_maps = []
        
        for b in range(batch_size):
            # Extract the data for this batch item
            current_valid = valid_mask[b].cpu().numpy().astype(bool)
            current_scales = scale_factors[b].cpu().numpy()
            
            if np.sum(current_valid) == 0:
                # No valid points in this batch item
                interpolated_maps.append(torch.ones(height, width, device=device))
                continue
                
            # Get coordinates of valid points
            y_coords, x_coords = np.nonzero(current_valid)
            knot_coords = np.stack([x_coords, y_coords], axis=1)
            knot_values = current_scales[current_valid]
            
            # Create a grid for interpolation
            grid_y, grid_x = np.mgrid[0:height, 0:width]
            grid_points = np.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
            
            # Perform interpolation
            interpolated_values = griddata(
                points=knot_coords,
                values=knot_values,
                xi=grid_points,
                method=interpolate_method,
                fill_value=1.0
            )
            
            interpolated_map = interpolated_values.reshape(height, width)
            
            # Convert back to tensor
            interpolated_map_tensor = torch.from_numpy(interpolated_map).float().to(device)
            
            # Optional normalization to [0, 1] range
            if normalize and torch.sum(current_valid) > 0:
                min_val = torch.min(scale_factors[b][valid_idx[b]])
                max_val = torch.max(scale_factors[b][valid_idx[b]])
                if max_val > min_val:
                    interpolated_map_tensor = (interpolated_map_tensor - min_val) / (max_val - min_val + 1e-6)
            
            interpolated_maps.append(interpolated_map_tensor)
        
        # Stack the results
        return torch.stack(interpolated_maps, dim=0)


    def normalize_unit_range(self, tensor):
        """Normalize tensor to [0, 1] range for each item in batch"""
        batch_size = tensor.shape[0]
        normalized = torch.zeros_like(tensor)
        
        for b in range(batch_size):
            min_val = tensor[b].min()
            max_val = tensor[b].max()
            if max_val > min_val:
                normalized[b] = (tensor[b] - min_val) / (max_val - min_val + 1e-6)
            else:
                normalized[b] = torch.zeros_like(tensor[b])
        
        return normalized

    def forward(self, tgt_img, ref_imgs, tgt_ga_depth, ref_ga_depth, 
                tgt_interp, tgt_sparse_depth, ref_interp, tgt_pose, 
                ref_pose, intrinsics, tgt_normals, tgt_depth_pred_inv, global_step, depth_only=False):
        """
        Refine metric depth scale and VIO poses and target/reference images.
        """ 
        #Step: Extract features using the base model
        ref_inputs = [
            self.preprocess_batch(ref_img, keys=["image"], device=ref_img.device)
            for ref_img in ref_imgs
        ]
        
        # Step: Compute sparse scales and valid mask
        tgt_sparse_depth_inv = utils.depth2inv(tgt_sparse_depth)  # [B, H, W]
        valid_mask = (tgt_sparse_depth_inv > 0).float()  # [B, H, W]
        sparse_scales = torch.zeros_like(tgt_sparse_depth_inv)  # [B, H, W]
        sparse_scales[valid_mask > 0] = tgt_sparse_depth_inv[valid_mask > 0] / (tgt_ga_depth[valid_mask > 0] + 1e-6)
        
        #normalize the sparse_scales
        orig_min = sparse_scales[valid_mask > 0].min()
        orig_max = sparse_scales[valid_mask > 0].max()
        sparse_scales[valid_mask > 0] = (sparse_scales[valid_mask > 0] - orig_min) / (orig_max - orig_min + 1e-6)
        
        processed_batch = self.preprocess_batch(
            tgt_img, tgt_ga_depth, tgt_ga_depth, sparse_scales, valid_mask, tgt_interp,
            keys=["image", "int_depth", "int_depth_c", "int_scales_c", "mask", "int_scales_c"],
            device=tgt_img.device
        )
        tgt_input, context_depth_input = torch.split(processed_batch, [4, 4], dim=1) #[1,4,288,384]
        init_metric_depth_inv = context_depth_input[:, 0:1, :, :]  # pred_inv_depth

        #test feat extraction only with the rgb image
        tgt_input = processed_batch[:, :3, :, :] # Shape: [1, 3, 288, 384]
        
        int_depth_pre = context_depth_input[:,0:1, :,:] #shape: [1,1,288,384]
        sparse_scales_pre = context_depth_input[:, 1:2, :, :] # Shape: [1, 1, 288, 384]
        valid_mask_pre = context_depth_input[:, 2:3, :, :] # Shape: [1, 1, 288, 384]
        # tgt_normals_pre = F.interpolate(tgt_normals.permute(0, 3, 1, 2), 
        #                                 size=(288, 384), mode='bilinear', align_corners=False) #1, 3, 288, 384
        int_interp_pre = context_depth_input[:,3:4,:,:]

        tgt_sparse_depth_inv_resized = F.interpolate(
            tgt_sparse_depth_inv.unsqueeze(1),  
            size=(tgt_input.shape[2], tgt_input.shape[3]),  
            mode='nearest',   
            align_corners=None  
        )
        #valid_mask_pre = (tgt_sparse_depth_inv_resized > 0).float()
        
        ##Visualize the sparse scales
        # self.visualize_sparse_scales(
        #     tgt_sparse_depth_inv,         # Original sparse scales [B, H, W]
        #     valid_mask,            # Original valid mask [B, H, W],
        #     tgt_interp,
        #     tgt_sparse_depth_inv_resized,     # Preprocessed sparse scales [B, 1, H, W]
        #     valid_mask_pre,         # Preprocessed valid mask [B, 1, H, W]
        #     int_interp_pre
        # )

        inv_depth_pred, outputs = self.depth_prop(tgt_input, tgt_sparse_depth_inv_resized)
        refined_inv_depth = inv_depth_pred
        
        ##get scale scaffolding from the propagated_depth
        #coarse_scales = torch.ones_like(propagated_depth_inv)
        #valid_mask_prop = (propagated_depth_inv > 0)
        #coarse_scales = torch.where(valid_mask_prop, propagated_depth_inv / (int_depth_pre + 1e-6), coarse_scales)
        #coarse_scales = (coarse_scales - coarse_scales.min()) / (coarse_scales.max() - coarse_scales.min() + 1e-6)
        ##normalize corase_scales
        
        #coarse_scales[valid_mask] = propagated_depth_inv[valid_mask] / (tgt_ga_depth[valid_mask] + 1e-6)
        #coarse_scales = (coarse_scales - coarse_scales.min()) / (coarse_scales.max() - coarse_scales.min() + 1e-6)
        #scale_scaffolding = coarse_scales
        
        # # Extract features using MidasNet #TODO: batch processing?
        # #tgt_feats = self.FeatExtractor(tgt_input) #[1, 32, 288, 384]
        # #ref_feats = [self.FeatExtractor(ref_input) for ref_input in ref_inputs] #[1, 32, 288, 384]
        # fmaps = self.fnet(torch.cat([tgt_input] + ref_inputs, dim=0))
        # fmaps = torch.split(fmaps, [tgt_input.shape[0]] * (1 + len(ref_inputs)), dim=0)
        # tgt_feats, ref_feats = fmaps[0], fmaps[1:] #[1,32,288/4,384/4]
        # tgt_larg_feat = self.fnet_midas(tgt_input) #[B, 32, 144, 192]
        # #context_feats = self.contextLearner(context_depth_input) #[1, 64, 144, 192] ga_depth + interp_scale
        # #context_feats = self.cnet_depth(context_depth_input) # (1, 128, 72, 96)
        # #context_feats = self.cnet_depth(context_depth_input)
        # # Extract sparse points and values for each sample in the batch
        # B, _, H, W = init_metric_depth_inv.shape
        # sparse_points_idx = []
        # sparse_scales_values = []
        # for b in range(B):
        #     indices = torch.where(valid_mask_pre[b,0] > 0)
        #     points = [(y.item(), x.item()) for y, x in zip(indices[0], indices[1])]
        #     scales = sparse_scales_pre[b,0][indices].detach()
        #     sparse_points_idx.append(points)
        #     sparse_scales_values.append(scales)
             
        # affinity_map = self.ap(tgt_larg_feat, sparse_points_idx, sparse_scales_values, 
        #                                         tgt_normals_pre, full_res=(H, W))
        
        # # Step 4: Prepare scale_at_points
        # N_max = max(len(points) for points in sparse_points_idx)
        # scale_at_points = torch.zeros(B, N_max, 1, device=tgt_larg_feat.device)
        # for b in range(B):
        #     num_points = len(sparse_scales_values[b])
        #     if num_points > 0:
        #         scale_at_points[b, :num_points, 0] = sparse_scales_values[b]
        
        # # Step 5: Propagate scales using affinity map 
        # H_down, W_down = tgt_larg_feat.shape[2], tgt_larg_feat.shape[3]
        # scale_scaffolding = torch.bmm(affinity_map, scale_at_points).squeeze(-1)  # [B, H_down*W_down]
        # scale_scaffolding = scale_scaffolding.view(B, 1, H_down, W_down)  # [B, 1, H_down, W_down]
        
        #
        # Step 6: Predict delta scales and confidence
        #ga_depth_feat = self.cnet_depth_affinity(int_depth_pre)
        #ga_depth_feat = self.upsample_1(ga_depth_feat)
        #context_test = torch.cat([int_depth_pre, int_interp_pre], dim=1)
        # context = torch.cat([int_depth_pre, scale_scaffolding], dim=1)
        #context = self.cnet_depth(context_test)
        #scale_map = self.scaleOutput(context)
        #delta_scales = F.relu(1.0 + scale_map)  # Ensure scale is positive
        #inv_depth_pred = init_metric_depth_inv * delta_scales
            
        if self.min_pred is not None and self.max_pred is not None:
           inv_depth_pred = torch.clamp(
               inv_depth_pred, 
               min=1.0 / self.max_pred, 
               max=1.0 / self.min_pred
           )
            
        refined_inv_depth = inv_depth_pred #[b, 1, 288, 384]
        
        depth_init_up = F.interpolate(
                refined_inv_depth,
                size=(tgt_img.shape[1], tgt_img.shape[2]),
                mode='bicubic',
                align_corners=False
        )  # shape => [B, 1, 480, 640]
        
        #fixed_tgt_pose = tgt_pose.detach().requires_grad_(False) #T_vio_cam2wld
        #refined_rel_poses = [pose_to_se3(fixed_tgt_pose.inverse() @ ref_p) for ref_p in ref_pose] #T_relative 6D
        # initial pose from VIO
        pose_list_init = ref_pose
        
        # -------- ConvGRU-Based Iterative Refinement --------            
        inv_depth_predictions = [depth_init_up] #[metric_depth_inv_tgt] #to see the history of depth predictions
        pose_predictions = [[pose_to_se3((pose).clone()) for pose in pose_list_init]] #to see the history of pose predictions
            
        # get optimization init
        if self.iter_steps > 0:
            #context_feat = self.context_conv(context_feats) #to make the output dim = hidden_dim + cost_dim #[1,160,144,192]
            hidden_d, inp_d = torch.split(context_feats, [self.hidden_dim, self.cost_dim], dim=1)
            hidden_d = torch.tanh(hidden_d) #1,128,144,192
            inp_d = torch.relu(inp_d) #1,32,144,192
                
            img_pairs = []
            for ref_img in ref_inputs:
                img_pairs.append(torch.cat([tgt_input, ref_img], dim=1))
            cnet_pose_list = self.contextPose(img_pairs)
            hidden_p_list, inp_p_list = [], []
            for cnet_pose in cnet_pose_list:
                hidden_p, inp_p = torch.split(cnet_pose, [self.hidden_dim, self.cost_dim], dim=1)
                hidden_p_list.append(torch.tanh(hidden_p))
                inp_p_list.append(torch.relu(inp_p))
            
        pose_list = pose_list_init
            
        # Step2: compute cost map and optimize depth scale iteratively
        for itr in range(self.iter_steps):
            #print("Iter: {}".format(itr))
            with torch.no_grad():
                self.log_fn(f"Iter_{itr}/hidden_state_norm", hidden_d.norm().item(), on_step=True, logger=True)
                self.log_fn(f"Iter_{itr}/input_state_norm", inp_d.norm().item(), on_step=True, logger=True)
                
            # Detach tensors to avoid backprop through refinement history
            refined_inv_depth = refined_inv_depth.detach()
            #ref_abspose_list = [fixed_tgt_pose @ se3_to_pose(pose).detach() for pose in refined_rel_poses] #(1,6) (1,6)
            pose_list = [pose.detach() for pose in pose_list]
            #refined_rel_poses = [pose.detach() for pose in refined_rel_poses] #relative pose
            #ref_abspose_list = [fixed_tgt_pose @ se3_to_pose(pose) for pose in refined_rel_poses] #4x4 absolute poses
                
            # ----------------------------
            # 1. Update scale with ConvGRU
            # ----------------------------
            depth_cost_map_func = partial(self.depth_cost_calc, 
                                        fmap=tgt_feats,
                                        fmaps_ref=ref_feats,
                                        pose_list=pose_list,
                                        tgt_pose=None,
                                        K=intrinsics,
                                        scale_factor=1.0/scale_factor)
                
            #update depth #TODO: check and understand this function
            hidden_d, up_mask_seqs, inv_depth_seqs, cost_means = self.update_block_depth(hidden_d, depth_cost_map_func,
                                                                            refined_inv_depth, inp_d,
                                                                            seq_len=self.seq_len)
                
            #we won't supervise the intermediate predictions
            #up_mask_seqs, inv_depth_seqs = [up_mask_seqs[-1]], [inv_depth_seqs[-1]]
                
            #upsample
            # for up_mask_i, inv_depth_i in zip(up_mask_seqs, inv_depth_seqs):
            #     refined_depth_inv = self.upsample_depth(
            #         inv_depth_i,  # input depth [B, 1, H/ratio, W/ratio]
            #         up_mask_i,    # upsample mask [B, 9*ratio*ratio, H/ratio, W/ratio]
            #         ratio=int(scale_factor),       # upsampling ratio
            #     )
                    
            refined_depth_inv = F.interpolate(
                inv_depth_seqs[-1],
                size=(tgt_img.shape[1], tgt_img.shape[2]),
                mode='bicubic',
                align_corners=False
            )  # shape => [B, 1, 480, 640]
                    
            inv_depth_predictions.append(refined_depth_inv)

            if self.log_fn:
                with torch.no_grad():
                    print(f"Iter {itr} cost progression: {cost_means}")
                    self.log_fn(f"Iter_{itr}/depth_mean", refined_depth_inv.mean().item(), on_step=True, logger=True)
                    self.log_fn(f"Iter_{itr}/depth_var", refined_depth_inv.var().item(), on_step=True, logger=True)
                        
            refined_inv_depth = inv_depth_seqs[-1]
                
            #### update pose using the updated depth ####
            # calc cost
            pose_cost_func_list = []
            for fmap_ref in ref_feats:
                pose_cost_func_list.append(partial(self.get_cost_each, tgt_poseC2W=None,fmap=tgt_feats, 
                                                       fmap_ref=fmap_ref,
                                                       depth=utils.inv2depth(refined_inv_depth),
                                                       K=intrinsics, scale_factor=1.0/scale_factor))
                
            pose_list_seqs = [None] * len(pose_list)
            for i, (ref_pose, hidden_p) in enumerate(zip(pose_list, hidden_p_list)):
                hidden_p, pose_seqs = self.update_block_pose(hidden_p, pose_cost_func_list[i],
                                                            ref_pose, inp_p_list[i], seq_len=self.seq_len, 
                                                            depth_only=depth_only)
                hidden_p_list[i] = hidden_p
                if not self.inter_sup:
                    pose_seqs = [pose_seqs[-1]] #take final iteration
                pose_list_seqs[i] = pose_seqs
                
            for pose_list_i in zip(*pose_list_seqs):
                pose_predictions.append([pose_to_se3(pose.clone()) for pose in pose_list_i])
                
            # Convert updated poses to relative
            pose_list = list(zip(*pose_list_seqs))[-1]
            #pose_list = [se3_to_pose(pose) for pose in pose_list]
                
            #ref_abspose_list = list(zip(*pose_list_seqs))[-1]
             #refined_rel_poses = [pose_to_se3(fixed_tgt_pose.inverse() @ ref_p) for ref_p in ref_abspose_list]
        
        if self.is_train:
            return inv_depth_predictions, \
                torch.stack([torch.stack(poses_ref, dim=1) for poses_ref in pose_predictions], dim=2), \
                    outputs #(b, n, iters, 6)
        else:
            return inv_depth_predictions[-1],\
                torch.stack(pose_predictions[-1], dim=1).view(tgt_img.shape[0], len(ref_imgs), 6) #(b, n, 6)

