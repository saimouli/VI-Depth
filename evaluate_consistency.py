import numpy as np
import torch
import torch.nn.functional as F
from data.SML_consistent_dataset import SML_consistent_dataset
from data.SML_consistent_resize import SML_consistent_resize
from utils.camera import Camera
import modules.midas.utils as utils
import cv2
import modules.midas.transforms as transforms
from modules.midas.midas_net_custom import MidasNet_small_videpth
from model.main_consistent import midasNetConsistentModule
#from visualizer.ros_visualizer import PointCloudVisualizer
#from visualize_data import project_depth_vectorize
#import rospy
from model.mhybrid_net import midasNet
from model.main import midasNetModule
#from geometry_msgs.msg import Pose, Point
import metrics
import matplotlib.pyplot as plt
from modules.interpolator import Interpolator2D, Interpolator2DWithUncertainty
from utils_eval import compute_ls_solution
from accelerated_features.modules.xfeat import XFeat
from test_densification import densify_vio_with_triangulation_and_ba

ROS_VIZ = False; EVAL = True

min_depth, max_depth = 0.2, 5.0
min_pred, max_pred = 0.1, 8.0

def get_vi_depth_depth(tgt_img, tgt_ga_depth, tgt_interp, ScaleMapLearner, ScaleMapLearner_transform, device):
    #vi-depth
    sample = {"image" : tgt_img.squeeze().cpu().numpy(), "int_depth" : tgt_ga_depth.squeeze().cpu().numpy(), 
            "int_scales" : tgt_interp.squeeze().cpu().numpy(), "int_depth_no_tf" : tgt_ga_depth.squeeze().cpu().numpy()}
    sample = ScaleMapLearner_transform(sample)
    x = torch.cat([sample["int_depth"], sample["int_scales"]], 0)
    x = x.to(device)
    d = sample["int_depth_no_tf"].to(device)
    sml_pred_inv, sml_scales = ScaleMapLearner.forward(x.unsqueeze(0), d.unsqueeze(0))
    sml_pred_inv = (torch.nn.functional.interpolate(sml_pred_inv.detach().cpu(),size=(tgt_img.shape[1], tgt_img.shape[2]),mode="bicubic",
                align_corners=False,
            )
            .cpu()
        )
    
    return sml_pred_inv

def get_cost_each(tgt_pose, pose, fmap, fmap_ref, depth, K, scale_factor):
    """
    ga_depth: (b, 1, h, w)
    map, fmap_ref: (b, c, h, w)
    """
    device = depth.device
    ref_cam = Camera(K=K.float(), Twc=pose).scaled(scale_factor).to(device)
    cam = Camera(K=K.float(), Twc=tgt_pose).scaled(scale_factor).to(device) # tcw = Identity
        
    # Reconstruct world points from target_camera
    world_points = cam.reconstruct(depth, frame='w')
    # Project world points onto reference camera
    ref_coords = ref_cam.project(world_points, frame='w', normalize=True) #(b, h, w,2)

    fmap_warped = F.grid_sample(fmap_ref, ref_coords, 
                                    mode='bilinear', padding_mode='zeros', align_corners=True) # (b, c, h, w)
        
    cost = (fmap - fmap_warped)**2
    #print("cost each: mean", cost.mean())
    return cost
    
def multi_view_reprojection_cost(ga_depth_inv, fmap, fmaps_ref, pose_list, tgt_pose, K, scale_factor):
    cost_list = []
    for pose, fmap_r in zip(pose_list, fmaps_ref):
        cost = get_cost_each(tgt_pose, pose, fmap, fmap_r, utils.inv2depth(ga_depth_inv), K, scale_factor)
        cost_list.append(cost)
    cost = torch.stack(cost_list, dim=1).mean(dim=1)
    return cost

def center_crop(tensor, target_height, target_width):
    _, h, w = tensor.shape[-3:]  # Extract the height and width of the tensor
    top = (h - target_height) // 2
    left = (w - target_width) // 2
    cropped_tensor = tensor[..., top:top + target_height, left:left + target_width]
    return cropped_tensor

def overlay_sparse_points_on_image(tgt_img, tgt_sparse_depth, color=(255, 0, 0), radius=5, thickness=-1):
    # Convert PyTorch tensors to numpy arrays
    if isinstance(tgt_img, torch.Tensor):
        tgt_img = tgt_img.cpu().numpy()
    if isinstance(tgt_sparse_depth, torch.Tensor):
        tgt_sparse_depth = tgt_sparse_depth.cpu().numpy()

    # Ensure the image is in uint8 format (scale if normalized)
    if tgt_img.max() <= 1.0:
        tgt_img = (tgt_img * 255).astype(np.uint8)
    else:
        tgt_img = tgt_img.astype(np.uint8)

    # Find non-zero sparse points
    sparse_points = np.argwhere(tgt_sparse_depth > 0)

    if len(tgt_img.shape) == 2 or tgt_img.shape[2] != 3:
        tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_GRAY2BGR)
    
    for y, x in sparse_points:
        cv2.circle(tgt_img, (x, y), radius, color, thickness)
        
    # # Overlay sparse points on the image
    # for y, x in sparse_points:
    #     # Set the pixel at (y, x) to the specified color
    #     tgt_img[y, x] = color

    return tgt_img

def plot_depth(tgt_img_cpu, tgt_gt_depth_inv_cpu, tgt_ga_depth_cpu, tgt_pred_depth_cpu):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(tgt_img_cpu)
    axes[0].set_title("RGB Image")
    axes[0].axis("off")
    
    depth_min, depth_max = tgt_gt_depth_inv_cpu.min(), tgt_gt_depth_inv_cpu.max()
    axes[1].imshow(tgt_gt_depth_inv_cpu, cmap="viridis", vmin=depth_min, vmax=depth_max)
    axes[1].set_title("Ground Truth Depth")
    axes[1].axis("off")
    
    axes[2].imshow(tgt_ga_depth_cpu, cmap="viridis", vmin=depth_min, vmax=depth_max)
    axes[2].set_title("GA Depth")
    axes[2].axis("off")
    
    axes[3].imshow(tgt_pred_depth_cpu, cmap="viridis", vmin=depth_min, vmax=depth_max)
    axes[3].set_title("SML Predicted Depth")
    axes[3].axis("off")
    
    plt.tight_layout()
    plt.show()
    
def get_ga_and_scale(depth_pred, input_sparse_depth, input_sparse_depth_valid, min_pred, max_pred):
    int_depth,_,_ = compute_ls_solution(depth_pred, input_sparse_depth, 
                                        input_sparse_depth_valid, min_pred, max_pred)
        
    # Interpolation of scale map
    assert (np.sum(input_sparse_depth_valid) >= 3), "not enough valid sparse points"
    ScaleMapInterpolator = Interpolator2D(
        pred_inv = int_depth,
        sparse_depth_inv = input_sparse_depth,
        valid = input_sparse_depth_valid,
    )
    
    ScaleMapInterpolator.generate_interpolated_scale_map(
        interpolate_method='linear', 
        fill_corners=False
    )
    int_scales = ScaleMapInterpolator.interpolated_scale_map.astype(np.float32)
    int_scales = utils.normalize_unit_range(int_scales)

    return int_depth, int_scales

def ls_without_uncertainty(d_r, d_m, valid, min_pred, max_pred):
    """
    Ordinary (un-weighted) least-squares fit of scale s and offset b:
      minimize  ∑_i (s·d_r[i] + b − d_m[i])²
    over all valid i.
    
    Returns:
      aligned : H×W float32 = s·d_r + b  (clamped)
      (s, b)  : the fitted scale and offset
    """
    # extract only the valid knots
    dr = d_r[valid].astype(np.float64)
    dm = d_m[valid].astype(np.float64)

    # build normal equations with w_i = 1
    A00 = np.sum(dr * dr)
    A01 = np.sum(dr)
    A11 = dr.shape[0]            # = ∑ 1

    b0  = np.sum(dr * dm)
    b1  = np.sum(dm)

    A = np.array([[A00, A01],
                  [A01, A11]], dtype=np.float64)
    B = np.array([b0, b1],       dtype=np.float64)

    # solve for s,b
    invA = np.linalg.inv(A)
    s, b = invA.dot(B)

    # apply and clamp
    aligned = (d_r * s + b).astype(np.float32)
    if min_pred is not None:
        aligned[aligned > 1.0/min_pred] = 1.0/min_pred
    if max_pred is not None:
        aligned[aligned < 1.0/max_pred] = 1.0/max_pred

    return aligned, (s, b)

def weighted_ls_with_uncertainity(d_r, d_m_noisy, var_m, valid, min_pred, max_pred):
    dr     = d_r[valid].astype(np.float64)
    dm     = d_m_noisy[valid].astype(np.float64)
    sigma2 = var_m[valid].astype(np.float64)
    w_raw = 1.0 /sigma2
    w     = np.clip(w_raw, a_min=1e-3, a_max=1e3)
    
    A00 = np.sum(w*dr*dr)
    A01 = np.sum(w*dr)
    A11 = np.sum(w)
    
    b0 = np.sum(w*dr*dm)
    b1 = np.sum(w*dm)
    
    A = np.array([[A00, A01], [A01, A11]])
    B = np.array([b0, b1])
    
    invA = np.linalg.inv(A)
    s,b = invA.dot(B)
    
    Var_s = invA[0,0]
    Var_b = invA[1,1]
    Cov_sb = invA[0,1]
    
    #apply and clamp
    aligned = (d_r * s + b).astype(np.float32)
    aligned[aligned>1.0/min_pred] = 1.0/min_pred
    aligned[aligned<1.0/max_pred] = 1.0/max_pred
    
    #per pixel variance
    var_map = (
      (d_r**2)*Var_s +
       Var_b +
      2*d_r*Cov_sb
    ).astype(np.float32)
    
    return aligned, (s,b), var_map

def create_vio_noise_model(X_means, randomize=True):
    Zs = X_means[:, 2]
    
    if randomize:
        noise_type = np.random.choice(['linear', 'quadratic', 'constant'])
        
        if noise_type == 'linear':
            # Linear noise model: alpha * Z
            alpha = np.random.uniform(0.01, 0.06)  # 1-6% of depth
            #min_noise = np.random.uniform(0.005, 0.02)  # 0.5-2cm minimum
            #max_noise = np.random.uniform(0.02, 0.06)  # 2-6cm maximum
            sigma_zs = np.clip(alpha * Zs, 0.005, 0.15)
        elif noise_type == 'quadratic':
            # Quadratic noise model: alpha * Z^2
            alpha      = np.random.uniform(0.005, 0.02)
            sigma_zs   = alpha * (Zs**2)
            sigma_zs   = np.clip(sigma_zs, 0.005, 0.15)
        else:
            sigma_zs   = np.random.uniform(0.02, 0.06, size=Zs.shape)

        # Randomly add outliers (5% chance)
        if np.random.random() < 0.1:  # 50% chance to add outliers
            mask       = np.random.rand(*Zs.shape) < np.random.uniform(0.01,0.05)
            sigma_zs[mask] *= np.random.uniform(2.0,4.0)
    else:
        # Default deterministic noise model for inference
        alpha = 0.06  # 2 cm of noise at 1 m, 4 cm at 2 m, etc.
        sigma_zs = np.clip(alpha * Zs, 0.01, 0.10)  # 1-10cm noise
    
    # Create covariance matrices (no noise in x,y directions)
    sigma_xs = np.full_like(sigma_zs, 0.001)
    sigma_ys = np.full_like(sigma_zs, 0.001)
    
    Sigmas = np.array([
        np.diag([sx*sx, sy*sy, sz*sz])
        for sx, sy, sz in zip(sigma_xs, sigma_ys, sigma_zs)
    ])
    
    return Sigmas
        
    
def get_ga_scale_uncertainity(depth_pred_inv, input_sparse_depth_inv, var_m, valid, min_pred, max_pred):
    
    # aligned_depth_inv, _, var_map = weighted_ls_with_uncertainity(depth_pred_inv, input_sparse_depth_inv, 
    #                                                     var_m, valid, min_pred, max_pred)
    
    aligned_depth_inv, _ = ls_without_uncertainty(depth_pred_inv, input_sparse_depth_inv, 
                                                  valid, min_pred, max_pred)
    
    assert (np.sum(valid) >= 3), "not enough valid sparse points"
    interpolator = Interpolator2DWithUncertainty(aligned_depth_inv, input_sparse_depth_inv, valid, var_m)
    scale_map, uncertainty_map = interpolator.generate_interpolated_scale_map('linear')
    
    int_scales = utils.normalize_unit_range(scale_map)
    
    
    return aligned_depth_inv, int_scales, uncertainty_map
    
def backproject_sparse(d_m, valid, fx, fy, cx, cy):
    ys, xs = np.nonzero(valid)
    Z = d_m[ys, xs]
    X = (xs - cx)/fx * Z
    Y = (ys - cy)/fy * Z
    X_means = np.stack([X, Y, Z], axis=1)
    coords = list(zip(ys, xs))
    return X_means, coords

def simulate_vio_analytic(X_means, Sigmas, coords, shape): #(N,3) proj. points, Sigmas (N,3,3)
    """
    For each sparse point i:
    - True depth = Z_i
    - Depth variance     = Σ_i[2,2]
    - Inject one zero‑mean sample n ~ N(0, Σ_i[2,2])
    - New depth = Z_i + n
    Returns
    - d_m_noisy: noisy depth
    - var_m: depth variance
    """
    H, W = shape  # Use the provided shape
    
    d_m_noisy = np.zeros((H,W), dtype=np.float32)
    var_m = np.zeros((H,W), dtype=np.float32)
    
    for i, (y,x) in enumerate(coords):
        Z = X_means[i,2]
        sigma2_z = Sigmas[i,2,2] # variance of Z
        noise = np.random.normal(0, np.sqrt(sigma2_z))
        d_m_noisy[y,x] = Z + noise
        var_m[y,x] = sigma2_z
    
    return d_m_noisy, var_m
    
def verify_uncertainty_map(noisy_sparse_depth_inv, uncertainty_map, input_sparse_depth_valid):
    import matplotlib.pyplot as plt
    
    # Create figure with subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Inverse sparse depth
    valid_y, valid_x = np.where(input_sparse_depth_valid)
    inv_depth_values = noisy_sparse_depth_inv[valid_y, valid_x]
    
    # For visualization, convert inverse depth to regular depth
    # (small inverse depth = far object, large inverse depth = close object)
    depth_values = 1.0 / inv_depth_values
    
    scatter1 = axs[0].scatter(valid_x, valid_y, c=depth_values, 
                              cmap='viridis', s=4)
    axs[0].set_title('Sparse Points (colored by depth)')
    axs[0].invert_yaxis()
    cbar1 = plt.colorbar(scatter1, ax=axs[0])
    cbar1.set_label('Depth (m)')
    
    # Plot 2: Uncertainty map
    im = axs[1].imshow(uncertainty_map, cmap='hot')
    axs[1].set_title('Log-Compressed Uncertainty Map')
    plt.colorbar(im, ax=axs[1])
    
    # Plot 3: Overlay sparse points on uncertainty map
    axs[2].imshow(uncertainty_map, cmap='hot', alpha=0.7)
    scatter3 = axs[2].scatter(valid_x, valid_y, c=depth_values, 
                              cmap='viridis', s=4, alpha=1.0)
    axs[2].set_title('Uncertainty with Sparse Points Overlay')
    axs[2].invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    
#currently evaluating the GT depth consistency TODO: include valid mask for gt depth
if __name__ == "__main__":
    dataset = SML_consistent_dataset(data_root='/media/saimouli/Data6T/datasets/void_150', mode='val', add_noise=False)
    #dataset = SML_consistent_resize(data_root='/media/saimouli/Data6T/datasets/VOID_150_small', mode='val')
    dataloader = torch.utils.data.DataLoader(dataset)
    
    sml_model_path = None #"weights/sml_model.dpredictor.dpt_hybrid.nsamples.150.ckpt"
    #sml_model_path = "weights/sml_model.dpredictor.dpt_hybrid.nsamples.150.pretrained.ckpt" #tartanair
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = midasNet(min_pred, max_pred, min_depth, max_depth, 150, sml_model_path)
    model_transforms = transforms.get_transforms("dpt_hybrid", "void", str(150))
    add_noise = False; #noise_sigma = 0.04
    # ScaleMapLearner_transform = model_transforms["sml_model"]
    # ScaleMapLearner = MidasNet_small_videpth(
    #     path=sml_model_path,
    #     min_pred=min_pred,
    #     max_pred=max_pred,
    # )
    # ScaleMapLearner.eval()
    # ScaleMapLearner.to(device)
    
    # model = midasNetModule()
    # model.load_from_checkpoint("weights/total_loss=0.095.ckpt", map_location=device)
    # model.eval()
    # model.to(device)
    
    model = midasNetConsistentModule(sml_model_path=sml_model_path, useConvGRU=True, is_train=False)
    model = model.load_from_checkpoint("weights/epoch=25-val_no_uncer/total_loss=0.080.ckpt")
    #model = model.load_from_checkpoint("weights/epoch=21-val_uncer/total_loss=0.082.ckpt")
    #model = model.load_from_checkpoint("lightning_logs/MVSML/version_0/checkpoints/epoch=8-val/total_loss=0.091.ckpt")
    #model = model.load_from_checkpoint("/home/saimouli/Documents/github/VI_Depth_sai/weights/without_costv/total_loss=0.086.ckpt")
    model.eval()
    model.to(device)
    xfeat = XFeat()
    
    first_frame = True
    threshold = 0.05  # Threshold for inlier correspondence
    rmse_list = []; rmse_list_sml = []
    inlier_ratios = []; inlier_ratios_sml = []
    overlap_ratios = []; overlap_ratios_sml = []
    poses_gt = []
    
    avg_error_w_int_depth = metrics.ErrorMetricsAverager()
    avg_error_w_pred = metrics.ErrorMetricsAverager()

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
            intrinsics, _, ref_pose_perturbed, tgt_depth_pred, _, tgt_uncer_scale = batch_data #dataset[idx] #Cam2Wld poses (R_ctoG, p_CinG)
        
        ##########################################################################
        ## add noise
        if add_noise:
            #print(f"Adding Gaussian noise with sigma={noise_sigma} to sparse points")
            tgt_sparse_depth_np = tgt_sparse_depth.squeeze().cpu().numpy()
            validity_map_bool = tgt_sparse_depth_np > 0
            input_sparse_depth_valid = (tgt_sparse_depth_np < max_depth) * (tgt_sparse_depth_np > min_depth)
            input_sparse_depth_valid = input_sparse_depth_valid.astype(bool)
            intrin_cam = intrinsics.squeeze().cpu().numpy()
            fx, fy, cx, cy = intrin_cam[0][0], intrin_cam[1][1], intrin_cam[0][2], intrin_cam[1][2]
            X_means, coords = backproject_sparse(tgt_sparse_depth_np, validity_map_bool, fx, fy, cx, cy)
            
            #assign VIO‑style covariances (here a simple diagonal for all points)
            #2) build per-point sigma_z (depth-dependent example)
            # Zs = X_means[:,2]
            # alpha = 0.02 #2 cm of noise at 1 m, 4 cm at 2 m, 6 cm at 3 m, 8 cm at 4 m, 10cm at 5m
            # sigma_zs = np.clip(alpha * Zs, 0.01, 0.10) #0.01, 0.05) #1cm - 6cm
            # # constant small xy noise
            # sigma_xs = np.zeros_like(sigma_zs)
            # sigma_ys = np.zeros_like(sigma_zs)
            # Sigmas = np.array([
            #     np.diag([sx*sx, sy*sy, sz*sz])
            #     for sx,sy,sz in zip(sigma_xs, sigma_ys, sigma_zs)
            # ])
            Sigmas = create_vio_noise_model(X_means, randomize=False)
            
            #sigma_x, sigma_y, sigma_z = 0.00, 0.00, 0.20 #0.1cm, 0.1cm , 4cm #TODO should be random
            #N = len(coords)
            #Sigmas = np.array([np.diag([sigma_x**2, sigma_y**2, sigma_z**2]) for i in range(N)])
            
            #simulate noisy sparse depths + variance
            d_m_noisy, var_m = simulate_vio_analytic(X_means, Sigmas, coords, tgt_sparse_depth_np.shape)
            #print(f"var_m stats: min={var_m[input_sparse_depth_valid].min()}, max={var_m[input_sparse_depth_valid].max()}, mean={var_m[input_sparse_depth_valid].mean()}")
            
            # #import matplotlib.pyplot as plt

            # # Create a simple figure with two subplots
            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # # Get valid coordinates
            # valid_y, valid_x = np.where(validity_map_bool)

            # # Plot original sparse depths
            # scatter1 = ax1.scatter(valid_x, valid_y, c=tgt_sparse_depth_np[valid_y, valid_x], 
            #                     cmap='viridis', s=5)
            # ax1.set_title('Original Sparse Depth')
            # ax1.invert_yaxis()  # Invert y-axis to match image coordinates
            # plt.colorbar(scatter1, ax=ax1)

            # # Plot noisy sparse depths
            # scatter2 = ax2.scatter(valid_x, valid_y, c=d_m_noisy[valid_y, valid_x], 
            #                     cmap='viridis', s=5)
            # ax2.set_title('Noisy Sparse Depth (4cm std)')
            # ax2.invert_yaxis()  # Invert y-axis to match image coordinates
            # plt.colorbar(scatter2, ax=ax2)

            # plt.tight_layout()
            # plt.show()
            
            #noise = np.random.normal(0, noise_sigma, tgt_sparse_depth_np.shape)
            #tgt_sparse_depth_np[validity_map_bool] += noise[validity_map_bool]
            
            d_m_noisy[~input_sparse_depth_valid] = np.inf
            noisy_sparse_depth_inv = 1.0 / d_m_noisy
            
            tgt_ga_depth, int_scales, uncertainity_map = get_ga_scale_uncertainity(
                tgt_depth_pred.squeeze().cpu().numpy(),
                noisy_sparse_depth_inv,
                var_m,
                input_sparse_depth_valid,
                min_pred,
                max_pred
            )
            uncertainity_map = np.log(1 + uncertainity_map)  # Compress dynamic range
            
            tgt_ga_depth = torch.from_numpy(tgt_ga_depth).unsqueeze(0).float().to(device)
            tgt_interp = torch.from_numpy(int_scales).unsqueeze(0).float().to(device)
            tgt_uncer_scale = torch.from_numpy(uncertainity_map).unsqueeze(0).unsqueeze(0).float().to(device)
            
            # tgt_sparse_depth_np[~input_sparse_depth_valid] = np.inf 
            # tgt_sparse_depth_np = 1.0 / tgt_sparse_depth_np 
            # ##recompute tgt_ga_depth
            # tgt_ga_depth, int_scales = get_ga_and_scale(tgt_depth_pred.squeeze().cpu().numpy(), tgt_sparse_depth_np, 
            #                                        input_sparse_depth_valid, min_pred, max_pred)
            # tgt_ga_depth = torch.from_numpy(tgt_ga_depth).unsqueeze(0).float().to(device)
            # tgt_interp = torch.from_numpy(int_scales).unsqueeze(0).float().to(device)
        
        #############################################################################
        ###reduce tgt sparse depth and try interpolating again and then pass
        # validity_map = (tgt_sparse_depth > 0).squeeze().cpu().numpy().astype(np.uint8)
        # print("Before Pts: ", np.count_nonzero(validity_map))
        # reduce_pts = int(np.count_nonzero(validity_map) * 0.90)
        # nonzero_indices = np.argwhere(validity_map == 1)
        # remove_indices = np.random.choice(len(nonzero_indices), size=reduce_pts, replace=False)
        # points_to_remove = nonzero_indices[remove_indices]
        # for x, y in points_to_remove:
        #     validity_map[x, y] = 0
        # print("After Pts: ", np.count_nonzero(validity_map))
        # input_sparse_depth_valid = validity_map.astype(bool)
        # tgt_sparse_depth = tgt_sparse_depth.squeeze().cpu().numpy()
        # tgt_sparse_depth[~input_sparse_depth_valid] = np.inf
        # tgt_sparse_depth = 1.0 / tgt_sparse_depth 

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #Add points using added xfeat points
        tgt_sparse_depth = tgt_sparse_depth.squeeze().cpu().numpy()
        sparse_vio_points = []
        for u in range(tgt_sparse_depth.shape[1]):
            for v in range(tgt_sparse_depth.shape[0]):
                if tgt_sparse_depth[v, u] > 0:
                    depth = tgt_sparse_depth[v, u]
                    sparse_vio_points.append((u, v, depth))
        
        tgt_gt_depth_np = utils.inv2depth(tgt_gt_depth_inv).squeeze().cpu().numpy()

        sparse_pts, new_pts = densify_vio_with_triangulation_and_ba(ref_img[0].squeeze(0).cpu(), tgt_img.squeeze(0).cpu(), ref_img[1].squeeze(0).cpu(), 
                                                                ref_gt_pose[0].squeeze(0).cpu().numpy(), tgt_pose.squeeze(0).cpu().numpy(), ref_gt_pose[1].squeeze(0).cpu().numpy(), 
                                                                intrinsics.squeeze(0).cpu().numpy(), 
                                                                sparse_vio_points, xfeat, gt_depth=tgt_gt_depth_np, 
                                                                num_new_points=2000)
        #insert new_pts to tgt_sparse_depth
        for u, v, depth in new_pts:
            tgt_sparse_depth[int(v), int(u)] = depth


        validity_map = (tgt_sparse_depth > 0).astype(np.uint8)
        input_sparse_depth_valid = validity_map.astype(bool)
        tgt_sparse_depth[~input_sparse_depth_valid] = np.inf
        tgt_sparse_depth = 1.0 / tgt_sparse_depth
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        #recompute tgt_ga_depth
        tgt_ga_depth = tgt_ga_depth.squeeze().cpu().numpy()
        tgt_ga_depth,_,_ = compute_ls_solution(tgt_depth_pred.squeeze().cpu().numpy(), tgt_sparse_depth, 
                                               input_sparse_depth_valid, min_pred, max_pred)
        tgt_ga_depth = torch.from_numpy(tgt_ga_depth).unsqueeze(0).float().to(device)
        
        ScaleMapInterpolator = Interpolator2D(
            pred_inv = tgt_ga_depth.squeeze().cpu().numpy(),
            sparse_depth_inv = tgt_sparse_depth,
            valid = input_sparse_depth_valid,
        )
        ScaleMapInterpolator.generate_interpolated_scale_map(
            interpolate_method='linear', 
            fill_corners=False
        )
        int_scales = ScaleMapInterpolator.interpolated_scale_map.astype(np.float32)
        int_scales = utils.normalize_unit_range(int_scales)
        tgt_interp = torch.from_numpy(int_scales).unsqueeze(0).float().to(device)
        tgt_sparse_depth = torch.from_numpy(tgt_sparse_depth).unsqueeze(0).float().to(device)
        ##########################################################################################
        
        ref_rel_poses = [tgt_pose.inverse() @ ref_p for ref_p in ref_pose_perturbed]
        ref_rel_gtposes = [tgt_pose.inverse() @ ref_p for ref_p in ref_gt_pose]
        
        _,H, W,_ = tgt_img.shape
        _,_, DH, DW = tgt_gt_depth_inv.shape
        
        scale_factor = DH / H
        
        tgt_gt_depth = utils.inv2depth(tgt_gt_depth_inv)
        valid_mask = (tgt_gt_depth >= min_depth) & (tgt_gt_depth <= max_depth)
        filtered_gt_depth = tgt_gt_depth * valid_mask
        filtered_gt_depth_inv = torch.zeros_like(tgt_gt_depth_inv)
        filtered_gt_depth_inv[valid_mask] = 1.0 / filtered_gt_depth[valid_mask]
        
        refined_depth_inv, refined_ref_poses = model(tgt_img, ref_img,
                                                    tgt_ga_depth, ref_ga_depth, 
                                                    tgt_interp, tgt_sparse_depth, 
                                                    ref_interp, 
                                                    tgt_pose, 
                                                    ref_rel_poses,
                                                    intrinsics,
                                                    tgt_uncer_scale,
                                                    do_scale_uncer=False)
        #compute consistent module
        # sml_depth_inv = model(tgt_img, tgt_gt_depth_inv, tgt_ga_depth, tgt_interp, ref_img, 
        #                               ref_ga_depth, ref_interp, ref_gt_depth, tgt_pose, ref_gt_pose, intrinsics)
        
        # sml_depth_inv = sml_depth_inv.detach().cpu()
        #sml_depth_inv, _ = model(tgt_sparse_depth, tgt_img, _, tgt_interp, tgt_ga_depth)
        #sml_depth_inv = sml_depth_inv.detach().cpu()
        
        #plot rgb, gt depth, ga depth, sml depth
        # plot_depth(tgt_img[0].cpu().numpy(), utils.inv2depth(tgt_gt_depth_inv[0][0]).cpu().numpy(), 
        #            utils.inv2depth(tgt_ga_depth[0]).cpu().numpy(), utils.inv2depth(sml_depth_inv[0][0]).cpu().numpy())
        
        
        #print(sml_depth_inv.shape)
        #networks take[3,288,384]size
        #resize the depth to 288, 384
        # filtered_gt_depth_inv_crop = center_crop(filtered_gt_depth_inv[0], 288, 384)
        # mask = (filtered_gt_depth_inv_crop >= min_depth) & (filtered_gt_depth_inv_crop <= max_depth)
        # tgt_img_crop = center_crop(tgt_img.permute(0,3,1,2), 288, 384).permute(0,2,3,1)
        
        # tgt_ga_depth_crop = center_crop(tgt_ga_depth, 288, 384)
        # tgt_interp_crop = center_crop(tgt_interp, 288, 384)
        # tgt_sparse_depth_crop = center_crop(tgt_sparse_depth, 288, 384)
        
        # sml_depth_inv = get_vi_depth_depth(tgt_img_crop, tgt_ga_depth_crop, tgt_interp_crop, ScaleMapLearner, ScaleMapLearner_transform, device)
        
        #sample_tgt_ = {"image" : tgt_img.squeeze().cpu().numpy(), "tgt_gt_depth_inv" : tgt_gt_depth_inv.squeeze().cpu().numpy(), 'tgt_ga_depth_inv' : tgt_ga_depth.squeeze().cpu().numpy()}
        #sample_ref1_ = {"image" : ref_imgs[0].squeeze().cpu().numpy(), "ref_gt_depth_inv" : ref_gt_depth_inv[0].squeeze().cpu().numpy(), 'ref_ga_depth_inv' : ref_ga_depth[0].squeeze().cpu().numpy()}
        #sample_ref2_ = {"image" : ref_imgs[1].squeeze().cpu().numpy(), "ref_gt_depth_inv" : ref_gt_depth_inv[1].squeeze().cpu().numpy(), 'ref_ga_depth_inv' : ref_ga_depth[1].squeeze().cpu().numpy()}
        
        #sample_tgt = ScaleMapLearner_transform(sample_tgt_)
        #sample_ref1 = ScaleMapLearner_transform(sample_ref1_)
        #sample_ref2 = ScaleMapLearner_transform(sample_ref2_)
        
        #TODO: use our cost here with GT depth
        #step1: downscale the depth and project
        cam_curr = Camera(K=intrinsics.float(), Twc=tgt_pose).scaled(scale_factor)
        pc2_gt = cam_curr.reconstruct(utils.inv2depth(filtered_gt_depth_inv), frame='w')[0].permute(1,2,0).view(-1, 3).cpu().numpy()
        #pc2_sml = cam_curr.reconstruct(utils.inv2depth(sml_depth_inv), frame='w')[0].permute(1,2,0).view(-1, 3).cpu().numpy()
        #pc_sparse = cam_curr.reconstruct(tgt_sparse_depth, frame='w')[0].permute(1,2,0).view(-1, 3).cpu().numpy()
        colors_gt = tgt_img[0].reshape(-1,3).cpu().numpy() * 255
        
        #step2: downscale the ga_depth and project
        #step3: get features and calculate cost
        #step4 optimize overfit the scale using pixelwise optimization
        #step5: visualize
        
#         prev_gt_depth_inv = filtered_gt_depth_inv
        
#         if first_frame:
#             cam_curr = Camera(K=intrinsics.float(), Twc=tgt_pose).scaled(scale_factor)
#             first_frame = False
            
#             prev_gt_depth_inv = tgt_gt_depth_inv
            
#             sml_depth_inv = get_vi_depth_depth(tgt_img, tgt_ga_depth, tgt_interp, ScaleMapLearner, ScaleMapLearner_transform, device)
            
#             prev_sml_pred = sml_depth_inv
#             continue
            
#         cam_prev = cam_curr
#         cam_curr = Camera(K=intrinsics.float(), Twc=tgt_pose).scaled(scale_factor)
        
#         #project to 3d points
#         prev_gt_depth = utils.inv2depth(prev_gt_depth_inv); tgt_gt_depth = utils.inv2depth(tgt_gt_depth_inv)
#         pc1_gt = cam_prev.reconstruct(prev_gt_depth, frame='w')[0].permute(1,2,0).view(-1,3).cpu().numpy()
#         pc2_gt = cam_curr.reconstruct(tgt_gt_depth, frame='w')[0].permute(1,2,0).view(-1, 3).cpu().numpy()
#         #filter out invalid points pc1_gt and pc2_gt based on min and max depth
#         colors_gt = tgt_img[0].view(-1,3).cpu().numpy() * 255
        
#         #vi-depth
#         sml_depth_inv = get_vi_depth_depth(tgt_img, tgt_ga_depth, tgt_interp, ScaleMapLearner, ScaleMapLearner_transform, device)
#         pc1_sml = cam_prev.reconstruct(utils.inv2depth(prev_sml_pred), frame='w')[0].permute(1,2,0).view(-1,3).cpu().numpy()
#         pc2_sml = cam_curr.reconstruct(utils.inv2depth(sml_depth_inv), frame='w')[0].permute(1,2,0).view(-1,3).cpu().numpy()
#         colors_gt = tgt_img[0].view(-1,3).cpu().numpy() * 255
        
        # compute error metrics using intermediate (globally aligned) depth
        if EVAL:
            mask = valid_mask.squeeze(0).cpu().numpy()
            error_w_int_depth = metrics.ErrorMetrics()
            error_w_int_depth.compute(
                estimate = tgt_ga_depth.cpu().detach().numpy(), 
                target = filtered_gt_depth_inv.cpu().detach().squeeze(0).numpy(), 
                valid = mask.astype(bool),
            )

            # # compute error metrics using SML output depth
            error_w_pred = metrics.ErrorMetrics()
            error_w_pred.compute(
                estimate = refined_depth_inv[-1].cpu().detach().squeeze(0).numpy(), 
                target = filtered_gt_depth_inv.cpu().detach().squeeze(0).numpy(), 
                valid = mask.astype(bool),
            )
            
            # # accumulate error metric
            avg_error_w_int_depth.accumulate(error_w_int_depth)
            avg_error_w_pred.accumulate(error_w_pred)
        
        if ROS_VIZ:
            visualizer = PointCloudVisualizer()
        
        if ROS_VIZ:
            rate = rospy.Rate(1)
            poses_gt.append(Pose(position=Point(
                x=tgt_pose[0][:,-1][0],
                y=tgt_pose[0][:,-1][1],
                z=tgt_pose[0][:,-1][2]
            )))
            visualizer.publish_path(poses_gt)
            visualizer.publish_point_cloud_gt(pc2_gt, colors_gt)
            #visualizer.publish_point_cloud_refine(pc2_sml, colors_gt)
            tgt_img_cpu = overlay_sparse_points_on_image(tgt_img[0], tgt_sparse_depth[0])
            #visualizer.pose_callback(tgt_pose[0][:,-1], tgt_pose[0][:,:3])
            visualizer.publish_tgt_img(tgt_img_cpu)
            #visualizer.publish_sparse_points(pc_sparse)
            #visualizer.publish_point_cloud_refine(pc2_sml, colors_gt)
            rate.sleep()
            
            
#         # Create Open3D point clouds
#         o3d_pc1 = o3d.geometry.PointCloud()
#         o3d_pc1.points = o3d.utility.Vector3dVector(pc1_gt)
#         o3d_pc2 = o3d.geometry.PointCloud()
#         o3d_pc2.points = o3d.utility.Vector3dVector(pc2_gt)
        
#         o3d_pc1_sml = o3d.geometry.PointCloud()
#         o3d_pc1_sml.points = o3d.utility.Vector3dVector(pc1_sml)
#         o3d_pc2_sml = o3d.geometry.PointCloud()
#         o3d_pc2_sml.points = o3d.utility.Vector3dVector(pc2_sml)

#         # Evaluate registration
#         result = o3d.pipelines.registration.evaluate_registration(
#             o3d_pc1, o3d_pc2, threshold
#         )
        
#         result_sml = o3d.pipelines.registration.evaluate_registration(
#             o3d_pc1_sml, o3d_pc2_sml, threshold
#         )
        
#         # Collect metrics
#         rmse_list.append(result.inlier_rmse)
#         rmse_list_sml.append(result_sml.inlier_rmse)
        
#         inlier_ratios.append(len(result.correspondence_set) / len(pc2_gt.reshape(-1, 3)))
#         inlier_ratios_sml.append(len(result_sml.correspondence_set) / len(pc2_sml.reshape(-1, 3)))
        
#         overlap_ratios.append(len(result.correspondence_set) / len(pc2_gt.reshape(-1, 3)))
#         overlap_ratios_sml.append(len(result_sml.correspondence_set) / len(pc2_sml.reshape(-1, 3)))

#         # Prepare for next iteration
#         prev_gt_depth_inv = tgt_gt_depth_inv
#         prev_sml_pred = sml_depth_inv
    
# # Compute averages over the entire sequence
# results = {
#     "average_rmse": np.mean(rmse_list), #Measures the geometric alignment error between the point clouds
#     "average_inlier_ratio": np.mean(inlier_ratios), #The fraction of points in pc2_gt that have valid correspondences in pc1_gt (within the threshold)
#     "average_overlap_ratio": np.mean(overlap_ratios),
# }

# results_sml = {
#     "average_rmse": np.mean(rmse_list_sml), #Measures the geometric alignment error between the point clouds
#     "average_inlier_ratio": np.mean(inlier_ratios_sml), #The fraction of points in pc2_gt that have valid correspondences in pc1_gt (within the threshold)
#     "average_overlap_ratio": np.mean(overlap_ratios_sml),
# }

# print("Depth Scale Consistency Results:", results)
# print("SML Consistency Results:", results_sml)

    if EVAL:
        # compute average error metrics
        print("Averaging metrics for globally-aligned depth over {} samples".format(
            avg_error_w_int_depth.total_count
        ))
        avg_error_w_int_depth.average()

        print("Averaging metrics for SML-aligned depth over {} samples".format(
            avg_error_w_pred.total_count
        ))
        avg_error_w_pred.average()
        
        from prettytable import PrettyTable
        summary_tb = PrettyTable()
        summary_tb.field_names = ["Metric", "GA Only", "GA+SML"]

        summary_tb.add_row(["RMSE", f"{avg_error_w_int_depth.rmse_avg:7.2f}", f"{avg_error_w_pred.rmse_avg:7.2f}"])
        summary_tb.add_row(["MAE", f"{avg_error_w_int_depth.mae_avg:7.2f}", f"{avg_error_w_pred.mae_avg:7.2f}"])
        summary_tb.add_row(["AbsRel", f"{avg_error_w_int_depth.absrel_avg:8.3f}", f"{avg_error_w_pred.absrel_avg:8.3f}"])
        summary_tb.add_row(["iRMSE", f"{avg_error_w_int_depth.inv_rmse_avg:7.2f}", f"{avg_error_w_pred.inv_rmse_avg:7.2f}"])
        summary_tb.add_row(["iMAE", f"{avg_error_w_int_depth.inv_mae_avg:7.2f}", f"{avg_error_w_pred.inv_mae_avg:7.2f}"])
        summary_tb.add_row(["iAbsRel", f"{avg_error_w_int_depth.inv_absrel_avg:8.3f}", f"{avg_error_w_pred.inv_absrel_avg:8.3f}"])
        
        print(summary_tb)