import open3d as o3d
import numpy as np
import torch
import torch.nn.functional as F
from data.SML_consistent_dataset import SML_consistent_dataset
from utils.camera import Camera
import modules.midas.utils as utils
import cv2
import modules.midas.transforms as transforms
from modules.midas.midas_net_custom import MidasNet_small_videpth
from model.main_consistent import midasNetConsistentModule
from visualizer.ros_visualizer import PointCloudVisualizer
from visualize_data import project_depth_vectorize
import rospy
from model.mhybrid_net import midasNet
from geometry_msgs.msg import Pose, Point
import metrics
import matplotlib.pyplot as plt

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
    
    
#currently evaluating the GT depth consistency TODO: include valid mask for gt depth
if __name__ == "__main__":
    dataset = SML_consistent_dataset(data_root='/media/saimouli/Data6T/datasets/VOID_150_small', mode='val')
    dataloader = torch.utils.data.DataLoader(dataset)
    
    #sml_model_path = "/home/saimouli/Documents/github/VI_Depth_sai/weights/sml_model.dpredictor.dpt_hybrid.nsamples.150.ckpt"
    sml_model_path = "weights/sml_model.dpredictor.dpt_hybrid.nsamples.150.pretrained.ckpt" #tartanair
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = midasNet(min_pred, max_pred, min_depth, max_depth, 150, sml_model_path)
    #model_transforms = transforms.get_transforms("dpt_hybrid", "void", str(150))
    #ScaleMapLearner_transform = model_transforms["sml_model"]
    # ScaleMapLearner = MidasNet_small_videpth(
    #     path=sml_model_path,
    #     min_pred=min_pred,
    #     max_pred=max_pred,
    # )
    # ScaleMapLearner.eval()
    # ScaleMapLearner.to(device)
    
    model = midasNetConsistentModule()
    model.load_from_checkpoint("lightning_logs/ckpt/checkpoints/epoch=1-val/total_loss=0.150.ckpt")
    model.eval()
    model.to(device)
    
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
            [item.to(device) if isinstance(item, torch.Tensor) else 
            [subitem.to(device) if isinstance(subitem, torch.Tensor) else subitem for subitem in item]
            if isinstance(item, list) else item
            for item in batch_data]
        )
        
        tgt_img, tgt_gt_depth_inv, tgt_ga_depth, tgt_interp, tgt_sparse_depth, ref_img, \
        ref_ga_depth, ref_interp, ref_gt_depth, ref_sparse_depth, tgt_pose, ref_pose, intrinsics = batch_data #dataset[idx] #Cam2Wld poses (R_ctoG, p_CinG)

        _,H, W,_ = tgt_img.shape
        _,_, DH, DW = tgt_gt_depth_inv.shape
        
        scale_factor = DH / H
        
        tgt_gt_depth = utils.inv2depth(tgt_gt_depth_inv)
        valid_mask = (tgt_gt_depth >= min_depth) & (tgt_gt_depth <= max_depth)
        filtered_gt_depth = tgt_gt_depth * valid_mask
        filtered_gt_depth_inv = torch.zeros_like(tgt_gt_depth_inv)
        filtered_gt_depth_inv[valid_mask] = 1.0 / filtered_gt_depth[valid_mask]
        
        sml_depth_inv = model(tgt_img, tgt_gt_depth_inv, tgt_ga_depth, tgt_interp, ref_img, 
                                      ref_ga_depth, ref_interp, ref_gt_depth, tgt_pose, ref_pose, intrinsics)
        
        sml_depth_inv = sml_depth_inv.detach().cpu()
        
        #plot rgb, gt depth, ga depth, sml depth
        plot_depth(tgt_img[0].cpu().numpy(), utils.inv2depth(tgt_gt_depth_inv[0][0]).cpu().numpy(), 
                   utils.inv2depth(tgt_ga_depth[0]).cpu().numpy(), utils.inv2depth(sml_depth_inv[0][0]).cpu().numpy())
        
        
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
        #cam_curr = Camera(K=intrinsics.float(), Twc=tgt_pose).scaled(scale_factor)
        #pc2_gt = cam_curr.reconstruct(utils.inv2depth(filtered_gt_depth_inv), frame='w')[0].permute(1,2,0).view(-1, 3).cpu().numpy()
        #pc2_sml = cam_curr.reconstruct(utils.inv2depth(sml_depth_inv), frame='w')[0].permute(1,2,0).view(-1, 3).cpu().numpy()
        #pc_sparse = cam_curr.reconstruct(tgt_sparse_depth, frame='w')[0].permute(1,2,0).view(-1, 3).cpu().numpy()
        #colors_gt = tgt_img[0].reshape(-1,3).cpu().numpy() * 255
        
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
                estimate = tgt_ga_depth.cpu().numpy(), 
                target = filtered_gt_depth_inv.cpu().squeeze(0).numpy(), 
                valid = mask.astype(bool),
            )

            # # compute error metrics using SML output depth
            error_w_pred = metrics.ErrorMetrics()
            error_w_pred.compute(
                estimate = sml_depth_inv.cpu().squeeze(0).numpy(), 
                target = filtered_gt_depth_inv.cpu().squeeze(0).numpy(), 
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