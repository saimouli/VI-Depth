import os
import argparse

import torch
import imageio
import numpy as np

from tqdm import tqdm
from PIL import Image

import modules.midas.utils as utils

import pipeline
import metrics

from geometry_msgs.msg import Pose, Point
import matplotlib.pyplot as plt
import rospy
from visualizer.ros_visualizer import PointCloudVisualizer

ROS_VIZ = True

def project_depth_vectorize(depth_img, img, p_CinG, R_CtoG, cam_K, normals=None, scale=None, shift=None):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().permute(1,2,0).numpy()
    p_CinG = p_CinG.reshape((3,1))

    valid_mask = (depth_img >= 0.1) & (depth_img <= 8)
    y_coords, x_coords = np.where(valid_mask)

    valid_depth_values = depth_img[y_coords, x_coords]

    pixel_coordinates = np.vstack((x_coords, y_coords, np.ones_like(x_coords)))
    normalized_camera_coordinates = np.linalg.solve(cam_K, pixel_coordinates)
    normalized_camera_coordinates *= valid_depth_values

    pFinC = np.vstack((normalized_camera_coordinates, np.ones_like(x_coords)))
    p_CinG_broadcasted = np.tile(p_CinG.reshape(3, 1), (1, pFinC.shape[1]))
    pFinG = np.dot(R_CtoG, pFinC[:3, :]) + p_CinG_broadcasted

    bgr_values = img[y_coords, x_coords]*255.0

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

def evaluate(dataset_path, depth_predictor, nsamples, sml_model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)
    
    # ranges for VOID
    min_depth, max_depth = 0.2, 5.0
    min_pred, max_pred = 0.1, 8.0
    
    # instantiate method
    method = pipeline.VIDepth(
        depth_predictor, nsamples, sml_model_path, 
        min_pred, max_pred, min_depth, max_depth, device
    )
    
    # get inputs
    with open(f"{dataset_path}/test_image.txt") as f: 
        test_image_list = [line.rstrip() for line in f]
    
    if ROS_VIZ:
        visualizer = PointCloudVisualizer()
    
    # initialize error aggregators
    avg_error_w_int_depth = metrics.ErrorMetricsAverager()
    avg_error_w_pred = metrics.ErrorMetricsAverager()

    if ROS_VIZ:
        rate = rospy.Rate(1)
    
    poses = []
        
    for i in tqdm(range(len(test_image_list))):
        
        #image
        input_image_fp = os.path.join(dataset_path, test_image_list[i])
        input_image = utils.read_image(input_image_fp)
        
        #poses list
        pose_fp = input_image_fp.replace("image", "absolute_pose").replace(".png", ".txt")
        #Cam2Wld
        pose_CtoG = np.loadtxt(pose_fp)
        R_CtoG = pose_CtoG[:3, :3]
        p_CinG = pose_CtoG[:3, 3]
        R_GtoC = R_CtoG.transpose()

        poses.append(Pose(position=Point(
            x=p_CinG[0],
            y=p_CinG[1],
            z=p_CinG[2]
        )))
        
        cam_K = np.loadtxt(dataset_path + "/K.txt")
        # sparse depth
        input_sparse_depth_fp = input_image_fp.replace("image", "sparse_depth")
        input_sparse_depth = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32) / 256.0
        input_sparse_depth[input_sparse_depth <= 0] = 0.0
        
        validity_map = None

        target_depth_fp = input_image_fp.replace("image", "ground_truth")
        target_depth = np.array(Image.open(target_depth_fp), dtype=np.float32) / 256.0
        target_depth[target_depth <= 0] = 0.0
        
        # target depth valid/mask
        mask = (target_depth < max_depth)
        if min_depth is not None:
            mask *= (target_depth > min_depth)
        target_depth[~mask] = np.inf  # set invalid depth
        target_depth = 1.0 / target_depth
        
        output = method.run(input_image, input_sparse_depth, validity_map, device)
        

        # compute error metrics using intermediate (globally aligned) depth
        error_w_int_depth = metrics.ErrorMetrics()
        error_w_int_depth.compute(
            estimate = output["ga_depth"], 
            target = target_depth, 
            valid = mask.astype(bool),
        )

        # compute error metrics using SML output depth
        error_w_pred = metrics.ErrorMetrics()
        error_w_pred.compute(
            estimate = output["sml_depth"], 
            target = target_depth, 
            valid = mask.astype(bool),
        )
        
        # accumulate error metric
        avg_error_w_int_depth.accumulate(error_w_int_depth)
        avg_error_w_pred.accumulate(error_w_pred)
        
        if ROS_VIZ:
            points_refine, colors_refine, _ = project_depth_vectorize(1.0/output["sml_depth"], input_image, p_CinG, R_CtoG, cam_K)
            points_gt, colors_gt, _ = project_depth_vectorize(1.0/target_depth, input_image, p_CinG, R_CtoG, cam_K)
            
            visualizer.publish_path(poses)
            visualizer.pose_callback(p_CinG, R_CtoG)
            visualizer.publish_point_cloud_refine(points_refine, colors_refine)
            visualizer.publish_point_cloud_gt(points_gt, colors_gt)
            rate.sleep()
    
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
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset_path", type=str, default="/home/rpng/Documents/sai_ws/splat_vins_repos_test/VI-Depth/input")
    
    parser.add_argument("--depth_predictor", type=str, default='dpt_hybrid')
    
    parser.add_argument("--nsamples", type=int, default=150, help="Number of samples for SML")
    
    parser.add_argument("--sml_model_path", type=str, default="/home/rpng/Documents/sai_ws/splat_vins_repos_test/VI-Depth/weights/sml_model.dpredictor.dpt_hybrid.nsamples.150.ckpt")
    
    args = parser.parse_args()
    print(args)
    
    evaluate(
        args.dataset_path, 
        args.depth_predictor, 
        args.nsamples, 
        args.sml_model_path)