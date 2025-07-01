#!/usr/bin/env python3

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
import glob
from visualizer.ros_visualizer import PointCloudVisualizer
from model.mhybrid_net import midasNet
from data.SML_dataset import SML_dataset
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader
import cv2
import modules.midas.transforms as transforms
from modules.midas.midas_net_custom import MidasNet_small_videpth
from evaluate_consistency import overlay_sparse_points_on_image
import pandas as pd

ROS_VIZ = True

def project_depth_vectorize(depth_img, img, p_CinG, R_CtoG, cam_K, normals=None, scale=None, shift=None):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy() #.permute(1,2,0).numpy()
    p_CinG = p_CinG.reshape((3,1))

    valid_mask = (depth_img >= 0.2) & (depth_img <= 8)
    y_coords, x_coords = np.where(valid_mask)

    valid_depth_values = depth_img[y_coords, x_coords]

    pixel_coordinates = np.vstack((x_coords, y_coords, np.ones_like(x_coords)))
    normalized_camera_coordinates = np.linalg.solve(cam_K, pixel_coordinates)
    normalized_camera_coordinates *= valid_depth_values

    pFinC = np.vstack((normalized_camera_coordinates, np.ones_like(x_coords)))
    p_CinG_broadcasted = np.tile(p_CinG.reshape(3, 1), (1, pFinC.shape[1]))
    pFinG = np.dot(R_CtoG, pFinC[:3, :]) + p_CinG_broadcasted

    bgr_values = img[y_coords, x_coords] * 255.0

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
    min_pred, max_pred = 0.1, 5.0
    
    # instantiate method
    method = pipeline.VIDepth(
        depth_predictor, nsamples, sml_model_path, 
        min_pred, max_pred, min_depth, max_depth, device
    )

    #model = midasNet(min_pred, max_pred, min_depth, max_depth, nsamples, sml_model_path)
    
    # get inputs
    # with open(f"{dataset_path}/test_image.txt") as f: 
    #     test_image_list = [line.rstrip() for line in f]
    # test_image_list = sorted(test_image_list)
    
    #read all the list of images in folder
    dataset_path_fld = os.path.join(dataset_path, "image")
    test_image_list = sorted([os.path.basename(f) for f in glob.glob(os.path.join(dataset_path_fld, "*.png"))])
    #append dataset_path_fld to the image list
    test_image_list = [os.path.join(dataset_path_fld, f) for f in test_image_list]
    
    if ROS_VIZ:
        visualizer = PointCloudVisualizer()
    
    # initialize error aggregators
    avg_error_w_int_depth = metrics.ErrorMetricsAverager()
    avg_error_w_pred = metrics.ErrorMetricsAverager()

    if ROS_VIZ:
        rate = rospy.Rate(1)
    
    poses = []
    cam_K = np.loadtxt(dataset_path + "/K.txt")
        
    for i in tqdm(range(len(test_image_list))):
        
        #image
        input_image_fp = os.path.join(dataset_path, test_image_list[i])
        input_image = utils.read_image(input_image_fp)
        
        #poses list
        pose_fp = input_image_fp.replace("image", "absolute_pose").replace(".png", ".txt")
        ##Cam2Wld
        pose_CtoG = np.loadtxt(pose_fp)
        R_CtoG = pose_CtoG[:3, :3]
        R_CtoG = R_CtoG#.transpose() #I did transpose for tartan data OV
        p_CinG = pose_CtoG[:3, 3]
        R_GtoC = R_CtoG.transpose()

        poses.append(Pose(position=Point(
            x=p_CinG[0],
            y=p_CinG[1],
            z=p_CinG[2]
        )))
        
        ## sparse depth
        input_sparse_depth_fp = input_image_fp.replace("image", "sparse_depth")
        input_sparse_depth = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32) / 256.0
        input_sparse_depth[input_sparse_depth <= 0] = 0.0
        
        #print how many sparse points
        print("# Sparse Pts: ", np.count_nonzero(input_sparse_depth))
        
        # validity_map_fp = input_image_fp.replace("image", "validity_map")
        # validity_map = np.array(Image.open(validity_map_fp), dtype=np.float32)
        # assert(np.all(np.unique(validity_map) == [0, 256]))
        # validity_map[validity_map > 0] = 1
        validity_map=None

        # print("Before Pts: ", np.count_nonzero(validity_map))
        # reduce_pts = int(np.count_nonzero(validity_map) * 0.65)
        # nonzero_indices = np.argwhere(validity_map == 1)
        # remove_indices = np.random.choice(len(nonzero_indices), size=reduce_pts, replace=False)
        # points_to_remove = nonzero_indices[remove_indices]
        # for x, y in points_to_remove:
        #     validity_map[x, y] = 0
        # print("After Pts: ", np.count_nonzero(validity_map))
        
        target_depth_fp = input_image_fp.replace("image", "ground_truth")
        target_depth = np.array(Image.open(target_depth_fp), dtype=np.float32) / 256.0
        target_depth[target_depth <= 0] = 0.0

        #Load the consistent scale depth
        # refine_depth_fp = input_image_fp.replace("image", "output/model_v3/depth")
        # refine_depth_fp = refine_depth_fp.replace(".png", ".npy")
        # refine_depth_test = np.array(np.load(refine_depth_fp))
        # #resize to  480, 640
        # refine_depth_test = cv2.resize(refine_depth_test, (640, 480), interpolation=cv2.INTER_NEAREST)
        # refine_depth_test[refine_depth_test <= 0] = 0.0
        # valid_mask = (target_depth > 0) & (refine_depth_test > 0)
        # scaling_factor = 0.0526 #np.median(target_depth[valid_mask] / refine_depth_test[valid_mask])
        # print(scaling_factor)
        # scaled_refine_depth_test = refine_depth_test * scaling_factor
        # scaled_refine_depth_test[~valid_mask] = 0.0

        
        # target depth valid/mask
        mask = (target_depth < max_depth)
        if min_depth is not None:
            mask *= (target_depth > min_depth)
        target_depth[~mask] = np.inf  # set invalid depth
        target_depth = 1.0 / target_depth
        
        output = method.run(input_image, input_sparse_depth, validity_map, device)
        ga_depth = output["ga_depth"]
        sml_depth = output["sml_depth"]
        sml_depth_viz = 1.0/ sml_depth
        sml_depth_viz[~mask] = np.inf 
        
        #ga_depth, sml_depth = model.forward(input_sparse_depth_inv, input_image, depth_pred_inv, interp_scale, GA_depth_inv, None)

        # compute error metrics using intermediate (globally aligned) depth
        error_w_int_depth = metrics.ErrorMetrics()
        error_w_int_depth.compute(
            estimate = ga_depth, 
            target = target_depth, 
            valid = mask.astype(bool),
        )

        # compute error metrics using SML output depth
        error_w_pred = metrics.ErrorMetrics()
        error_w_pred.compute(
            estimate = sml_depth, 
            target = target_depth, 
            valid = mask.astype(bool),
        )
        
        # accumulate error metric
        avg_error_w_int_depth.accumulate(error_w_int_depth)
        avg_error_w_pred.accumulate(error_w_pred)
        
        if ROS_VIZ and i % 5 ==0:
            points_refine, colors_refine, _ = project_depth_vectorize(sml_depth_viz, input_image, p_CinG, R_CtoG, cam_K)
            #points_ga, colors_ga, _ = project_depth_vectorize(1.0/ga_depth, input_image, p_CinG, R_CtoG, cam_K)
            points_gt, colors_gt, _ = project_depth_vectorize(1.0/target_depth, input_image, p_CinG, R_CtoG, cam_K)
            #point_consis, colors_consis, _ = project_depth_vectorize(scaled_refine_depth_test, input_image, p_CinG, R_CtoG, cam_K)
            pc_sparse, _, _ = project_depth_vectorize(input_sparse_depth, input_image, p_CinG, R_CtoG, cam_K)
            
            visualizer.publish_path(poses)
            tgt_img = input_image*255.0
            tgt_img = overlay_sparse_points_on_image(tgt_img, 1.0/input_sparse_depth)
            visualizer.pose_callback(p_CinG, R_CtoG)
            visualizer.publish_point_cloud_refine(points_refine, colors_refine)
            visualizer.publish_point_cloud_gt(points_gt, colors_gt)
            visualizer.publish_tgt_img(tgt_img)
            visualizer.publish_sparse_points(pc_sparse)
            #visualizer.publish_point_cloud_refine(point_consis, colors_consis)
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

def resize_with_aspect_ratio(image, target_width, ensure_multiple_of, interpolation=cv2.INTER_CUBIC):
    original_height, original_width = image.shape[:2]
    aspect_ratio = original_height / original_width

    # Calculate new dimensions
    new_width = target_width
    new_height = int(round(new_width * aspect_ratio))
        
    # Ensure height is a multiple of the given value
    new_height = (new_height // ensure_multiple_of) * ensure_multiple_of
        
    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    return resized_image

@torch.no_grad()
def evaluate_light(dataset_path, depth_predictor, nsamples, sml_model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)
    
    # ranges for VOID
    min_depth, max_depth = 0.2, 5.0
    min_pred, max_pred = 0.1, 8.0

    model = midasNet(min_pred, max_pred, min_depth, max_depth, nsamples, sml_model_path)
    test_dataset = SML_dataset(dataset_path, mode="val", depth_scale=256.0)
    data_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    #read all the list of images in folder
    dataset_path_fld = os.path.join(dataset_path, "image")
    test_image_list = sorted([os.path.basename(f) for f in glob.glob(os.path.join(dataset_path_fld, "*.png"))])
    
    if ROS_VIZ:
        visualizer = PointCloudVisualizer()
        rate = rospy.Rate(1)
    
    avg_error_w_int_depth = metrics.ErrorMetricsAverager()
    avg_error_w_pred = metrics.ErrorMetricsAverager()

    poses = []
    cam_K = np.loadtxt(dataset_path + "/K.txt")
    model_transforms = transforms.get_transforms("dpt_hybrid", "void", str(150))
    ScaleMapLearner_transform = model_transforms["sml_model"]
    ScaleMapLearner = MidasNet_small_videpth(
        path=sml_model_path,
        min_pred=min_pred,
        max_pred=max_pred,
    )
    ScaleMapLearner.eval()
    ScaleMapLearner.to(device)
        
    #for i in tqdm(range(len(test_image_list))):
    for batch_idx, (image, gt_depth_inv, sparse_depth_inv, depth_pred_inv, ga_depth_inv, int_scales, mask, pose_CtoG) in enumerate(data_loader):
        #image
        #input_image_fp = os.path.join(dataset_path_fld, test_image_list[batch_idx])
        #input_image = utils.read_image(input_image_fp)
        
        # #poses list
        # pose_fp = input_image_fp.replace("image", "absolute_pose").replace(".png", ".txt")
        # #Cam2Wld
        # pose_CtoG = np.loadtxt(pose_fp)
        pose_CtoG = pose_CtoG.numpy()[0]
        R_CtoG = pose_CtoG[:3, :3]
        p_CinG = pose_CtoG[:3, 3]
        R_GtoC = R_CtoG.transpose()

        poses.append(Pose(position=Point(
            x=p_CinG[0],
            y=p_CinG[1],
            z=p_CinG[2]
        )))

        sample = {"image" : image.squeeze().cpu().numpy(), "int_depth" : ga_depth_inv.squeeze().cpu().numpy(), 
                  "int_scales" : int_scales.squeeze().cpu().numpy(), "int_depth_no_tf" : ga_depth_inv.squeeze().cpu().numpy()}
        sample = ScaleMapLearner_transform(sample)
        x = torch.cat([sample["int_depth"], sample["int_scales"]], 0)
        x = x.to(device)
        d = sample["int_depth_no_tf"].to(device)
        sml_pred, sml_scales = ScaleMapLearner.forward(x.unsqueeze(0), d.unsqueeze(0))
        sml_pred = (torch.nn.functional.interpolate(sml_pred,size=(image.shape[1], image.shape[2]),mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
        #cam_K = np.loadtxt(dataset_path + "/K.txt")
        # sparse depth
        # input_sparse_depth_fp = input_image_fp.replace("image", "sparse_depth")
        # input_sparse_depth = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32) / 256.0
        # input_sparse_depth[input_sparse_depth <= 0] = 0.0
        
        # validity_map = None

        # target_depth_fp = input_image_fp.replace("image", "ground_truth")
        # target_depth = np.array(Image.open(target_depth_fp), dtype=np.float32) / 256.0
        # target_depth[target_depth <= 0] = 0.0
        
        ## target depth valid/mask
        # mask = (target_depth < max_depth)
        # if min_depth is not None:
        #     mask *= (target_depth > min_depth)
        # target_depth[~mask] = np.inf  # set invalid depth
        # target_depth = 1.0 / target_depth
        
        
        #output = method.run(input_image, input_sparse_depth, validity_map, device)
        #ga_depth = output["ga_depth"]
        #sml_depth = output["sml_depth"]
        #ga_depth, sml_depth = model.forward(input_sparse_depth_inv, input_image, depth_pred_inv, interp_scale, GA_depth_inv, None)

        # compute error metrics using intermediate (globally aligned) depth
        mask = mask[0].squeeze(0).numpy()
        error_w_int_depth = metrics.ErrorMetrics()
        error_w_int_depth.compute(
            estimate = ga_depth_inv[0].numpy(), 
            target = gt_depth_inv[0][0].numpy(), 
            valid = mask.astype(bool),
        )

        # # compute error metrics using SML output depth
        error_w_pred = metrics.ErrorMetrics()
        error_w_pred.compute(
            estimate = sml_pred, 
            target = gt_depth_inv[0][0].numpy(), 
            valid = mask.astype(bool),
        )
        
        # # accumulate error metric
        avg_error_w_int_depth.accumulate(error_w_int_depth)
        avg_error_w_pred.accumulate(error_w_pred)
        
        if ROS_VIZ and batch_idx % 4 ==0:
            #resize target depth to 480, 640
            #target_depth = torch.nn.functional.interpolate(gt_depth_inv, size=(480, 640), mode='bicubic')[0][0].numpy()
            #image = torch.nn.functional.interpolate(image, size=(480, 640), mode='bicubic')[0]
            depth_gt = 1.0/gt_depth_inv.numpy()
            depth_gt[depth_gt == float("inf")] = 0

            ga_depth = 1.0/ga_depth_inv.numpy()
            ga_depth[ga_depth == float("inf")] = 0

            sml_pred = 1.0/sml_pred
            sml_pred[sml_pred == float("inf")] = 0

            points_refine, colors_refine, _ = project_depth_vectorize(sml_pred, image[0], p_CinG, R_CtoG, cam_K)
            #points_ga, colors_ga, _ = project_depth_vectorize(ga_depth, image[0], p_CinG, R_CtoG, cam_K)
            points_gt, colors_gt, _ = project_depth_vectorize(depth_gt[0][0], image[0], p_CinG, R_CtoG, cam_K)
            #gt_depth_inv[0][0].numpy()
            
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


def load_ov_poses(pose_file_path):
    """Loads poses from ov_poses.txt file."""
    poses_dict = {}
    try:
        with open(pose_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 8:
                    timestamp = float(parts[0])
                    pose_data = np.array([float(p) for p in parts[1:]]) # tx ty tz qx qy qz qw
                    poses_dict[timestamp] = pose_data
    except FileNotFoundError:
        print(f"Error: Pose file not found at {pose_file_path}")
    return poses_dict
    
@torch.no_grad()
def evaluate_table(dataset_path, depth_predictor, nsamples, sml_model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)
    
    # ranges for VOID
    min_depth, max_depth = 0.2, 10.0
    min_pred, max_pred = 0.1, 10.0
    
    # instantiate method
    method = pipeline.VIDepth(
        depth_predictor, nsamples, sml_model_path, 
        min_pred, max_pred, min_depth, max_depth, device
    )
    
    #read all the list of images in folder
    dataset_path_fld = os.path.join(dataset_path, "image")
    test_image_list = sorted([os.path.basename(f) for f in glob.glob(os.path.join(dataset_path_fld, "*.png"))])
    #append dataset_path_fld to the image list
    test_image_list = [os.path.join(dataset_path_fld, f) for f in test_image_list]
    test_image_list = test_image_list[10:250:3]

    # --- Load poses from ov_poses.txt ---
    pose_file_path = os.path.join(dataset_path, "ov_pose.txt")
    all_poses_data = load_ov_poses(pose_file_path)
    # ------------------------------------
    
    if ROS_VIZ:
        visualizer = PointCloudVisualizer()
    
    # initialize error aggregators
    avg_error_w_int_depth = metrics.ErrorMetricsAverager()
    avg_error_w_pred = metrics.ErrorMetricsAverager()

    if ROS_VIZ:
        rate = rospy.Rate(1)
    
    poses = []
    cam_K = np.loadtxt(dataset_path + "/K.txt")
        
    for i in tqdm(range(len(test_image_list))):
        
        #image
        input_image_fp = os.path.join(dataset_path, test_image_list[i])
        input_image = utils.read_image(input_image_fp)
        
        #poses list
        img_name = os.path.basename(input_image_fp).replace(".png", "")
        pose_data = all_poses_data[float(img_name)]
        p_CinG = pose_data[0:3]
        quat_CtoG = pose_data[3:7] # qx, qy, qz, qw
        R_CtoG = R.from_quat(quat_CtoG).as_matrix()
        
        poses.append(Pose(position=Point(
            x=p_CinG[0],
            y=p_CinG[1],
            z=p_CinG[2]
        )))
        
        ## sparse depth
        input_sparse_depth_fp = input_image_fp.replace("image", "sparse_depth")
        input_sparse_depth = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32) / 1000.0
        input_sparse_depth[input_sparse_depth <= 0] = 0.0
        
        #validity_map_fp = #input_image_fp.replace("image", "validity_map")
        #count non zero pixels in sparse depth and init validity map
        validity_map = None #np.zeros(input_sparse_depth.shape, dtype=np.float32)
        #validity_map[input_sparse_depth > 0] = 1.0
        
        #assert(np.all(np.unique(validity_map) == [0, 256]))
        #validity_map[validity_map > 0] = 1

        # reduce_pts = int(np.count_nonzero(validity_map) * 0.65)
        nonzero_indices = np.argwhere(input_sparse_depth > 0)
        
        # remove_indices = np.random.choice(len(nonzero_indices), size=reduce_pts, replace=False)
        # points_to_remove = nonzero_indices[remove_indices]
        # for x, y in points_to_remove:
        #     validity_map[x, y] = 0
        # print("After Pts: ", np.count_nonzero(validity_map))
        
        target_depth_fp = input_image_fp.replace("image", "gt_depth")
        target_depth = np.array(Image.open(target_depth_fp), dtype=np.float32) / 1000.0
        target_depth[target_depth <= 0] = 0.0
        
        # replace the sparse depth with gt depth values and add more points
        for x, y in nonzero_indices:
            #print(input_sparse_depth[x, y], target_depth[x, y])
            input_sparse_depth[x, y] = target_depth[x, y]

        candidates = np.argwhere((target_depth > 0) & (input_sparse_depth == 0))
        num_to_add = min(100, len(candidates))  # avoid error if fewer than 100 candidates
        add_indices = np.random.choice(len(candidates), size=num_to_add, replace=False)
        points_to_add = candidates[add_indices]
        for x, y in points_to_add:
            input_sparse_depth[x, y] = target_depth[x, y]

        # target depth valid/mask
        mask = (target_depth < max_depth)
        if min_depth is not None:
            mask *= (target_depth > min_depth)
        target_depth[~mask] = np.inf  # set invalid depth
        target_depth_inv = 1.0 / target_depth
        
        print("Before Pts: ", np.count_nonzero(input_sparse_depth))

        #resize img, sparse depth and target depth to 480, 640
        # input_image = resize_with_aspect_ratio(input_image, 640, 32, interpolation=cv2.INTER_LINEAR)
        # input_sparse_depth = resize_with_aspect_ratio(input_sparse_depth, 640, 32, interpolation=cv2.INTER_NEAREST)
        # target_depth = resize_with_aspect_ratio(target_depth, 640, 32, interpolation=cv2.INTER_NEAREST) 
        # target_depth_inv = resize_with_aspect_ratio(target_depth_inv, 640, 32, interpolation=cv2.INTER_NEAREST)
        # mask = resize_with_aspect_ratio(mask.astype(np.uint8), 640, 32, interpolation=cv2.INTER_NEAREST)

        #center crop img, depth and mask
        # def center_crop(img, target_height, target_width):
        #     h, w = img.shape[:2]
        #     if target_height > h or target_width > w:
        #         raise ValueError(f"Target size ({target_height}, {target_width}) "
        #                         f"is larger than image size ({h}, {w}).")
        #     top = max((h - target_height) // 2, 0)
        #     left = max((w - target_width) // 2, 0)
        #     return img[top:top + target_height, left:left + target_width]
        
        # input_image = center_crop(input_image, 288, 384)
        # input_sparse_depth = center_crop(input_sparse_depth, 288, 384)
        # target_depth = center_crop(target_depth, 288, 384)
        # target_depth_inv = center_crop(target_depth_inv, 288, 384)
        # mask = center_crop(mask, 288, 384)
        
        output = method.run(input_image, input_sparse_depth, validity_map, device)
        ga_depth = output["ga_depth"]
        sml_depth = output["sml_depth"]
        #ga_depth, sml_depth = model.forward(input_sparse_depth_inv, input_image, depth_pred_inv, interp_scale, GA_depth_inv, None)

        # compute error metrics using intermediate (globally aligned) depth
        error_w_int_depth = metrics.ErrorMetrics()
        error_w_int_depth.compute(
            estimate = ga_depth, 
            target = target_depth_inv, 
            valid = mask.astype(bool),
        )

        # compute error metrics using SML output depth
        error_w_pred = metrics.ErrorMetrics()
        error_w_pred.compute(
            estimate = sml_depth, 
            target = target_depth_inv, 
            valid = mask.astype(bool),
        )
        
        # accumulate error metric
        avg_error_w_int_depth.accumulate(error_w_int_depth)
        avg_error_w_pred.accumulate(error_w_pred)
        
        if ROS_VIZ and i % 10 ==0:
            #remove boarder sml_depth values

            points_refine, colors_refine, _ = project_depth_vectorize(1.0/sml_depth, input_image, p_CinG, R_CtoG, cam_K)
            #points_ga, colors_ga, _ = project_depth_vectorize(1.0/ga_depth, input_image, p_CinG, R_CtoG, cam_K)
            points_gt, colors_gt, _ = project_depth_vectorize(1.0/target_depth_inv, input_image, p_CinG, R_CtoG, cam_K)
            
            pc_sparse, _, _ = project_depth_vectorize(input_sparse_depth, input_image, p_CinG, R_CtoG, cam_K)
            #point_consis, colors_consis, _ = project_depth_vectorize(scaled_refine_depth_test, input_image, p_CinG, R_CtoG, cam_K)
            
            tgt_img = input_image * 255.0
            tgt_img = overlay_sparse_points_on_image(tgt_img, 1.0/input_sparse_depth)
            visualizer.publish_path(poses)
            visualizer.pose_callback(p_CinG, R_CtoG)
            visualizer.publish_point_cloud_refine(points_refine, colors_refine)
            visualizer.publish_point_cloud_gt(points_gt, colors_gt)
            visualizer.publish_tgt_img(tgt_img)
            visualizer.publish_sparse_points(pc_sparse)
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

def plot_gt_traj(data_path):
    def load_trajectory_data(traj_file):
        """Load trajectory data from TUM format file"""
        data = np.loadtxt(traj_file)
        positions = data[:, 0:3] # tx, ty, tz
        quaternions = data[:, 3:8]  # qx, qy, qz, qw
        return positions, quaternions
        #data = np.loadtxt(traj_file, skiprows=1)  # Skip header
        #timestamps = data[:, 0]
        #positions = data[:, 1:4]
        #quaternions = data[:, 4:8]  # qx, qy, qz, qw
        #return timestamps, positions, quaternions

    def load_camera_timestamps(timestamp_file):
        """Load camera timestamps"""
        return np.loadtxt(timestamp_file)

    def find_closest_trajectory_pose(target_time, traj_timestamps, positions, quaternions):
        """Find closest trajectory pose to target timestamp"""
        idx = np.argmin(np.abs(traj_timestamps - target_time))
        return positions[idx], quaternions[idx], idx
    
    def read_decode_depth(depthpath):
        depth_rgba = cv2.imread(depthpath, cv2.IMREAD_UNCHANGED)
        depth = depth_rgba.view("<f4")
        return np.squeeze(depth, axis=-1)

    #poses list
    visualizer = PointCloudVisualizer()
    rate = rospy.Rate(10)
    poses = []

    #traj_file = os.path.join(data_path, "imu", "easy_traj.txt")
    #raw_data = np.loadtxt(traj_file, delimiter=' ')
    rgb_dir = os.path.join(data_path, "image_lcam_front")
    depth_dir = os.path.join(data_path, "depth_lcam_front")
    traj_file = os.path.join(data_path, "pose_lcam_front.txt")
    #camera_timestamps_file = os.path.join(data_path, "imu", "cam_time.txt")
    
    #if raw_data.ndim == 1: # Handle case with only one line in the file
    #    raw_data = raw_data.reshape(1, -1)
    
    #if raw_data.shape[1] != 7:
    #    return 
    
    cam_K = np.array([[320.0, 0.0, 320.0],
                      [0.0, 320.0, 320.0],
                      [0.0, 0.0, 1.0]])
    
    positions, quaternions = load_trajectory_data(traj_file)
    #camera_timestamps = load_camera_timestamps(camera_timestamps_file)
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    
    for i, (rgb_file, depth_file) in enumerate(zip(rgb_files, depth_files)):
        # if i >= len(camera_timestamps):
        #     break
            
        # cam_time = camera_timestamps[i]
        
        # Find closest trajectory pose
        # p_CinG, quat_CtoG, traj_idx = find_closest_trajectory_pose(
        #     cam_time, traj_timestamps, positions, quaternions)
        p_CinG = positions[i]
        quat_CtoG = quaternions[i]
        
        # Convert quaternion to rotation matrix
        # Note: quaternion format is [qx, qy, qz, qw]
        quat_scipy = [quat_CtoG[0], quat_CtoG[1], quat_CtoG[2], quat_CtoG[3]]  # [qx, qy, qz, qw]
        R_CtoG = R.from_quat(quat_scipy).as_matrix()
        
        # Load images
        try:
            rgb_img = cv2.imread(rgb_file)
            depth_img = read_decode_depth(depth_file)
            
            if rgb_img is None or depth_img is None:
                print(f"Failed to load images at index {i}")
                continue
                
            # Convert BGR to RGB
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            
            # Convert depth to meters if needed (depends on your depth format)
            # Assuming depth is in millimeters, convert to meters
            if depth_img.dtype == np.uint16:
                depth_img = depth_img.astype(np.float32) / 1000.0
                
        except Exception as e:
            print(f"Error loading images at index {i}: {e}")
            continue
        
        # Project depth to 3D points
        points_gt, colors_gt, _ = project_depth_vectorize(
            depth_img, rgb_img, p_CinG, R_CtoG, cam_K)
        
        # Create pose message
        pose = Pose()
        pose.position = Point(x=p_CinG[0], y=p_CinG[1], z=p_CinG[2])
        pose.orientation.x = quat_CtoG[0]
        pose.orientation.y = quat_CtoG[1] 
        pose.orientation.z = quat_CtoG[2]
        pose.orientation.w = quat_CtoG[3]
        
        poses.append(pose)
        
        # Publish visualization
        visualizer.publish_path(poses)
        visualizer.publish_tgt_img(rgb_img)
        if len(points_gt) > 0:
            visualizer.publish_point_cloud_gt(points_gt, colors_gt)
        
        # print(f"Processing frame {i+1}/{len(rgb_files)}, "
        #       f"timestamp: {cam_time:.3f}, "
        #       f"trajectory idx: {traj_idx}, "
        #       f"points: {len(points_gt)}")
        
        if rospy.is_shutdown():
            break
            
        rate.sleep()

def evaluate_tartan_covar(dataset_path, depth_predictor, nsamples, sml_model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ranges for VOID
    min_depth, max_depth = 0.2, 5.0
    min_pred, max_pred = 0.1, 5.0
    
    # instantiate method
    method = pipeline.VIDepth(
        depth_predictor, nsamples, sml_model_path, 
        min_pred, max_pred, min_depth, max_depth, device
    )

    # get inputs - read all images in folder
    dataset_path_fld = os.path.join(dataset_path, "image")
    test_image_list = sorted([os.path.basename(f) for f in glob.glob(os.path.join(dataset_path_fld, "*.png"))])
    test_image_list = [os.path.join(dataset_path_fld, f) for f in test_image_list]
    
    if ROS_VIZ:
        visualizer = PointCloudVisualizer()
    
    # initialize error aggregators
    avg_error_w_int_depth = metrics.ErrorMetricsAverager()
    avg_error_w_pred = metrics.ErrorMetricsAverager()

    if ROS_VIZ:
        rate = rospy.Rate(5)
    
    poses = []
    cam_K = np.array([[320.0, 0.0, 320.0],
                      [0.0, 320.0, 240.0],
                      [0.0, 0.0, 1.0]])
    all_slam_errors = []
    all_slam_variances = []
    all_slam_abs_errors = []
    all_active_errors = []

    for i in tqdm(range(0,2000,2)):
        
        # image
        input_image_fp = os.path.join(dataset_path, test_image_list[i])
        input_image = utils.read_image(input_image_fp)
        
        # poses list
        pose_fp = input_image_fp.replace("image", "ov_pose").replace(".png", ".txt")
        pose_CtoG = np.loadtxt(pose_fp)
        R_CtoG = pose_CtoG[:3, :3]
        R_CtoG = R_CtoG #.transpose()  # transpose for tartan data OV JPL to global
        p_CinG = pose_CtoG[:3, 3]

        poses.append(Pose(position=Point(
            x=p_CinG[0],
            y=p_CinG[1],
            z=p_CinG[2]
        )))
        
        ## sparse depth
        input_sparse_depth_fp = input_image_fp.replace("image", "sparse_depth")
        input_sparse_depth = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32) / 256.0
        input_sparse_depth[input_sparse_depth <= 0] = 0.0

        input_active_depth = np.zeros_like(input_sparse_depth)
        input_slam_depth = np.zeros_like(input_sparse_depth)
        
        ## covariance for SLAM points
        input_covariance_fp = input_image_fp.replace("image", "covariance")
        input_covariance = None
        if os.path.exists(input_covariance_fp):
            # Load covariance image (scaled by 1000x and saved as 16-bit)
            input_covariance = np.array(Image.open(input_covariance_fp), dtype=np.float32) / 1000.0
            input_covariance[input_covariance <= 0] = 0.0
        
        print("# Sparse Pts: ", np.count_nonzero(input_sparse_depth))
        if np.count_nonzero(input_sparse_depth) < 3:
            continue
        
        # ground truth depth
        target_depth_fp = input_image_fp.replace("image", "ground_truth")
        target_depth = np.array(Image.open(target_depth_fp), dtype=np.float32) / 256.0
        target_depth[target_depth <= 0] = 0.0

        # Create validity map and separate SLAM vs active tracking points
        # Note: sparse depth contains BOTH SLAM features AND active tracking points
        # SLAM features have covariance data, active tracking points don't
        validity_map = np.zeros(input_sparse_depth.shape, dtype=np.float32)
        slam_point_map = np.zeros(input_sparse_depth.shape, dtype=np.float32)
        active_point_map = np.zeros(input_sparse_depth.shape, dtype=np.float32)
        
        for x in range(input_sparse_depth.shape[0]):
            for y in range(input_sparse_depth.shape[1]):
                if input_sparse_depth[x, y] > min_depth and input_sparse_depth[x,y] < max_depth:
                    validity_map[x, y] = 1
                    
                    # Check if this point has covariance data (indicates SLAM point)
                    if input_covariance is not None and input_covariance[x, y] > 0:
                        slam_point_map[x, y] = 1
                        input_slam_depth[x,y] = input_sparse_depth[x,y]
                    else:
                        active_point_map[x, y] = 1
                        input_active_depth[x,y] = input_sparse_depth[x,y]
        
        # Print statistics about SLAM vs active points
        num_slam_points = np.count_nonzero(slam_point_map)
        num_active_points = np.count_nonzero(active_point_map)
        print(f"SLAM points: {num_slam_points}, Active points: {num_active_points}")

        if ((num_active_points + num_slam_points) < 50):
            continue
        
        # Visualize how good the sparse SLAM points are vs active points
        slam_depth_errors = []
        active_track_errors = []
        slam_errors_for_analysis = []
        slam_variances_for_analysis = []
        slam_abs_errors_for_analysis = []     

        for x in range(validity_map.shape[0]):
            for y in range(validity_map.shape[1]):
                if validity_map[x, y] > 0:
                    if target_depth[x, y] <= min_depth or target_depth[x, y] > max_depth:
                        continue
                    depth_error = input_sparse_depth[x, y] - target_depth[x, y]
                    abs_error = np.abs(depth_error)
                    
                    if slam_point_map[x, y] == 1:
                        slam_depth_errors.append(depth_error)
                        # Use covariance for outlier detection threshold for SLAM points
                        variance_threshold = input_covariance[x, y] if input_covariance is not None else 10
                        outlier_threshold = 2.0 * np.sqrt(variance_threshold)  # 3-sigma rule
                        if input_covariance is not None and input_covariance[x, y] > 0:
                                variance = input_covariance[x, y]
                                slam_errors_for_analysis.append(depth_error)
                                slam_variances_for_analysis.append(variance)
                                slam_abs_errors_for_analysis.append(abs_error)
                                
                                # Also collect for global analysis across all frames
                                all_slam_errors.append(depth_error)
                                all_slam_variances.append(variance)
                                all_slam_abs_errors.append(abs_error) 

                        print(f"SLAM point: depth={input_sparse_depth[x, y]:.3f}, "
                              f"(x={x}, y={y}),"
                              f"gt={target_depth[x, y]:.3f}, error={depth_error:.3f}, "
                              f"variance={input_covariance[x, y]:.6f}, threshold={outlier_threshold:.3f}")
                              
                        # if np.abs(depth_error) > outlier_threshold:
                        #     input_sparse_depth[x, y] = target_depth[x, y]
                        #     print(f"  -> Outlier detected and corrected")
                    else:
                        active_track_errors.append(depth_error)
                        all_active_errors.append(depth_error)

                        print(f"Active point: depth={input_sparse_depth[x, y]:.3f}, "
                              f"gt={target_depth[x, y]:.3f}, error={depth_error:.3f}")
                        
                        # Fixed threshold for active points (no covariance available)
                        # fixed_threshold = 0.10
                        
                        # if np.abs(depth_error) > fixed_threshold:
                        #     input_sparse_depth[x, y] = target_depth[x, y]
        
        # Print error statistics
        if slam_depth_errors:
            slam_rmse = np.sqrt(np.mean(np.array(slam_depth_errors)**2))
            print(f"SLAM points RMSE: {slam_rmse:.4f}")
        if active_track_errors:
            active_rmse = np.sqrt(np.mean(np.array(active_track_errors)**2))
            print(f"Active points RMSE: {active_rmse:.4f}")
        
        # target depth valid/mask
        mask = (target_depth < max_depth)
        if min_depth is not None:
            mask *= (target_depth > min_depth)
        target_depth[~mask] = np.inf
        target_depth = 1.0 / target_depth
        
        output = method.run(input_image, input_sparse_depth, validity_map, device)
        ga_depth = output["ga_depth"]
        sml_depth = output["sml_depth"]
        sml_depth_viz = 1.0 / sml_depth
        sml_depth_viz[~mask] = np.inf 
        
        # compute error metrics using intermediate (globally aligned) depth
        error_w_int_depth = metrics.ErrorMetrics()
        error_w_int_depth.compute(
            estimate=ga_depth, 
            target=target_depth, 
            valid=mask.astype(bool),
        )

        # compute error metrics using SML output depth
        error_w_pred = metrics.ErrorMetrics()
        error_w_pred.compute(
            estimate=sml_depth, 
            target=target_depth, 
            valid=mask.astype(bool),
        )
        
        # accumulate error metric
        avg_error_w_int_depth.accumulate(error_w_int_depth)
        avg_error_w_pred.accumulate(error_w_pred)
        
        if ROS_VIZ and i % 5 == 0:
            points_refine, colors_refine, _ = project_depth_vectorize(sml_depth_viz, input_image, p_CinG, R_CtoG, cam_K)
            points_gt, colors_gt, _ = project_depth_vectorize(1.0/target_depth, input_image, p_CinG, R_CtoG, cam_K)
            #pc_sparse, _, _ = project_depth_vectorize(input_sparse_depth, input_image, p_CinG, R_CtoG, cam_K)
            pc_active, _, _ = project_depth_vectorize(input_active_depth, input_image, p_CinG, R_CtoG, cam_K)
            pc_slam, _, _ = project_depth_vectorize(input_slam_depth, input_image, p_CinG, R_CtoG, cam_K)
    
            
            visualizer.publish_path(poses)
            tgt_img = input_image * 255.0
            tgt_img = overlay_sparse_points_on_image(tgt_img, 1.0/input_sparse_depth)
            visualizer.pose_callback(p_CinG, R_CtoG)
            visualizer.publish_point_cloud_refine(points_refine, colors_refine)
            visualizer.publish_point_cloud_gt(points_gt, colors_gt)
            visualizer.publish_tgt_img(tgt_img)
            visualizer.publish_active_points(pc_active)
            visualizer.publish_slam_points(pc_slam)
            #visualizer.publish_sparse_points(pc_sparse)
                
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
    
    ### ========== VARIANCE vs ERROR ANALYSIS AND VISUALIZATION ==========
    ### Create comprehensive plots for variance vs error analysis
    # if len(all_slam_variances) > 10:  # Need enough data points for meaningful plots
    #     import matplotlib.pyplot as plt
        
    #     all_slam_errors_np = np.array(all_slam_errors)
    #     all_slam_variances_np = np.array(all_slam_variances)
    #     all_slam_abs_errors_np = np.array(all_slam_abs_errors)
    #     all_slam_std_devs = np.sqrt(all_slam_variances_np)
        
    #     # Calculate overall correlations
    #     corr_error_variance = np.corrcoef(all_slam_abs_errors_np, all_slam_variances_np)[0, 1]
    #     corr_error_std = np.corrcoef(all_slam_abs_errors_np, all_slam_std_devs)[0, 1]
        
    #     # Calculate sigma bounds
    #     within_1sigma = np.sum(all_slam_abs_errors_np <= all_slam_std_devs) / len(all_slam_abs_errors_np) * 100
    #     within_2sigma = np.sum(all_slam_abs_errors_np <= 2*all_slam_std_devs) / len(all_slam_abs_errors_np) * 100
    #     within_3sigma = np.sum(all_slam_abs_errors_np <= 3*all_slam_std_devs) / len(all_slam_abs_errors_np) * 100
        
    #     print(f"\n=== OVERALL SLAM Error vs Variance Analysis ===")
    #     print(f"Total SLAM points analyzed: {len(all_slam_variances)}")
    #     print(f"Correlation |error| vs variance: {corr_error_variance:.4f}")
    #     print(f"Correlation |error| vs std_dev: {corr_error_std:.4f}")
    #     print(f"Errors within 1σ: {within_1sigma:.1f}% (expected ~68%)")
    #     print(f"Errors within 2σ: {within_2sigma:.1f}% (expected ~95%)")
    #     print(f"Errors within 3σ: {within_3sigma:.1f}% (expected ~99.7%)")
        
    #     # Create subplots
    #     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
    #     # Plot 1: Absolute Error vs Standard Deviation (scatter)
    #     ax1.scatter(all_slam_std_devs, all_slam_abs_errors_np, alpha=0.6, s=20)
    #     ax1.plot([0, max(all_slam_std_devs)], [0, max(all_slam_std_devs)], 'r--', 
    #             label=f'Perfect calibration (y=x)')
    #     ax1.set_xlabel('Predicted Standard Deviation (m)')
    #     ax1.set_ylabel('Absolute Error (m)')
    #     ax1.set_title(f'Error vs Uncertainty\nCorrelation: {corr_error_std:.3f}')
    #     ax1.legend()
    #     ax1.grid(True, alpha=0.3)
        
    #     # Plot 2: Histogram comparison of errors and std devs
    #     ax2.hist(all_slam_abs_errors_np, bins=30, alpha=0.7, label='Absolute Errors', density=True)
    #     ax2.hist(all_slam_std_devs, bins=30, alpha=0.7, label='Standard Deviations', density=True)
    #     ax2.set_xlabel('Value (m)')
    #     ax2.set_ylabel('Density')
    #     ax2.set_title('Distribution Comparison')
    #     ax2.legend()
    #     ax2.grid(True, alpha=0.3)
        
    #     # Plot 3: Error vs Standard Deviation (2D histogram/heatmap)
    #     h, xedges, yedges = np.histogram2d(all_slam_std_devs, all_slam_abs_errors_np, bins=30)
    #     extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #     im = ax3.imshow(h.T, extent=extent, origin='lower', aspect='auto', cmap='Blues')
    #     ax3.plot([0, max(all_slam_std_devs)], [0, max(all_slam_std_devs)], 'r--', linewidth=2)
    #     ax3.set_xlabel('Predicted Standard Deviation (m)')
    #     ax3.set_ylabel('Absolute Error (m)')
    #     ax3.set_title('Error vs Uncertainty Density')
    #     plt.colorbar(im, ax=ax3, label='Count')
        
    #     # Plot 4: Calibration analysis - ratio of error to std dev
    #     error_to_std_ratio = all_slam_abs_errors_np / (all_slam_std_devs + 1e-8)  # Add small epsilon to avoid division by zero
    #     ax4.hist(error_to_std_ratio, bins=50, alpha=0.7, edgecolor='black')
    #     ax4.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Perfect calibration (ratio=1)')
    #     ax4.axvline(x=np.mean(error_to_std_ratio), color='green', linestyle='-', linewidth=2, 
    #                label=f'Mean ratio: {np.mean(error_to_std_ratio):.2f}')
    #     ax4.set_xlabel('|Error| / Standard Deviation')
    #     ax4.set_ylabel('Count')
    #     ax4.set_title('Calibration Quality\n(Ideal mean ratio ≈ 0.8)')
    #     ax4.legend()
    #     ax4.grid(True, alpha=0.3)
        
    #     plt.tight_layout()
    #     plt.savefig(f'{dataset_path}/variance_error_analysis.png', dpi=300, bbox_inches='tight')
    #     plt.show()
        
    #     # Additional analysis
    #     print(f"Mean |error|/std_dev ratio: {np.mean(error_to_std_ratio):.3f} (ideal ≈ 0.8)")
    #     print(f"Std of |error|/std_dev ratio: {np.std(error_to_std_ratio):.3f}")
        
    #     # Calibration quality assessment
    #     if 0.7 <= np.mean(error_to_std_ratio) <= 0.9:
    #         print("✓ Well-calibrated: Error/std ratio near ideal")
    #     elif np.mean(error_to_std_ratio) < 0.7:
    #         print("⚠ Over-confident: Predicted uncertainties too high")
    #     else:
    #         print("⚠ Under-confident: Predicted uncertainties too low")
            
    #     if corr_error_std > 0.4:
    #         print("✓ Strong correlation: Uncertainty is predictive of error")
    #     elif corr_error_std > 0.2:
    #         print("~ Moderate correlation: Some predictive power")
    #     else:
    #         print("⚠ Weak correlation: Uncertainty not predictive of error")
            
    #     # Create additional statistical summary
    #     print(f"\n=== Statistical Summary ===")
    #     print(f"Mean variance: {np.mean(all_slam_variances_np):.6f}")
    #     print(f"Mean std deviation: {np.mean(all_slam_std_devs):.4f}")
    #     print(f"Mean absolute error: {np.mean(all_slam_abs_errors_np):.4f}")
    #     print(f"SLAM RMSE: {np.sqrt(np.mean(all_slam_errors_np**2)):.4f}")
        
    #     if len(all_active_errors) > 0:
    #         active_rmse = np.sqrt(np.mean(np.array(all_active_errors)**2))
    #         print(f"Active points RMSE: {active_rmse:.4f}")
    #         print(f"SLAM vs Active RMSE improvement: {((active_rmse - np.sqrt(np.mean(all_slam_errors_np**2))) / active_rmse * 100):.1f}%")
    
    # else:
    #     print("Not enough SLAM points for variance analysis plots")

def evaluate_tartan(dataset_path, depth_predictor, nsamples, sml_model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)
    
    # ranges for VOID
    min_depth, max_depth = 0.2, 5.0
    min_pred, max_pred = 0.1, 5.0
    
    # instantiate method
    method = pipeline.VIDepth(
        depth_predictor, nsamples, sml_model_path, 
        min_pred, max_pred, min_depth, max_depth, device
    )

    #model = midasNet(min_pred, max_pred, min_depth, max_depth, nsamples, sml_model_path)
    
    # get inputs
    # with open(f"{dataset_path}/test_image.txt") as f: 
    #     test_image_list = [line.rstrip() for line in f]
    # test_image_list = sorted(test_image_list)
    
    #read all the list of images in folder
    dataset_path_fld = os.path.join(dataset_path, "image")
    test_image_list = sorted([os.path.basename(f) for f in glob.glob(os.path.join(dataset_path_fld, "*.png"))])
    #append dataset_path_fld to the image list
    test_image_list = [os.path.join(dataset_path_fld, f) for f in test_image_list]
    
    if ROS_VIZ:
        visualizer = PointCloudVisualizer()
    
    # initialize error aggregators
    avg_error_w_int_depth = metrics.ErrorMetricsAverager()
    avg_error_w_pred = metrics.ErrorMetricsAverager()

    if ROS_VIZ:
        rate = rospy.Rate(5)
    
    poses = []
    cam_K = np.array([[320.0, 0.0, 320.0],
                      [0.0, 320.0, 240.0],
                      [0.0, 0.0, 1.0]])
        
    for i in tqdm(range(0,200,2)): #len(test_image_list))
        
        #image
        input_image_fp = os.path.join(dataset_path, test_image_list[i])
        input_image = utils.read_image(input_image_fp)
        
        #poses list
        pose_fp = input_image_fp.replace("image", "absolute_pose").replace(".png", ".txt")
        ##Cam2Wld
        pose_CtoG = np.loadtxt(pose_fp)
        R_CtoG = pose_CtoG[:3, :3]
        R_CtoG = R_CtoG.transpose() #I did transpose for tartan data OV
        p_CinG = pose_CtoG[:3, 3]
        #R_GtoC = R_CtoG.transpose()

        poses.append(Pose(position=Point(
            x=p_CinG[0],
            y=p_CinG[1],
            z=p_CinG[2]
        )))
        
        ## sparse depth
        input_sparse_depth_fp = input_image_fp.replace("image", "sparse_depth")
        input_sparse_depth = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32) / 256.0
        input_sparse_depth[input_sparse_depth <= 0] = 0.0

        #print how many sparse points
        print("# Sparse Pts: ", np.count_nonzero(input_sparse_depth))
        if np.count_nonzero(input_sparse_depth) < 3:
            continue
        
        
        target_depth_fp = input_image_fp.replace("image", "ground_truth")
        target_depth = np.array(Image.open(target_depth_fp), dtype=np.float32) / 256.0
        target_depth[target_depth <= 0] = 0.0

        validity_map=np.zeros(input_sparse_depth.shape, dtype=np.float32)
        for x in range(input_sparse_depth.shape[0]):
            for y in range(input_sparse_depth.shape[1]):
                if input_sparse_depth[x, y] > 0:
                    validity_map[x, y] = 1
            
                    
        #visualize how good is the ov sparse slam points
        for x in range(validity_map.shape[0]):
            for y in range(validity_map.shape[1]):
                if validity_map[x, y] > 0:
                    print(input_sparse_depth[x, y], target_depth[x, y], input_sparse_depth[x, y] - target_depth[x, y])
                    if np.abs(input_sparse_depth[x, y] - target_depth[x, y]) > 0.10:
                        input_sparse_depth[x, y] = target_depth[x, y]

        
        # target depth valid/mask
        mask = (target_depth < max_depth)
        if min_depth is not None:
            mask *= (target_depth > min_depth)
        target_depth[~mask] = np.inf  # set invalid depth
        target_depth = 1.0 / target_depth
        
        output = method.run(input_image, input_sparse_depth, validity_map, device)
        ga_depth = output["ga_depth"]
        sml_depth = output["sml_depth"]
        sml_depth_viz = 1.0/ sml_depth
        sml_depth_viz[~mask] = np.inf 
        
        #ga_depth, sml_depth = model.forward(input_sparse_depth_inv, input_image, depth_pred_inv, interp_scale, GA_depth_inv, None)

        # compute error metrics using intermediate (globally aligned) depth
        error_w_int_depth = metrics.ErrorMetrics()
        error_w_int_depth.compute(
            estimate = ga_depth, 
            target = target_depth, 
            valid = mask.astype(bool),
        )

        # compute error metrics using SML output depth
        error_w_pred = metrics.ErrorMetrics()
        error_w_pred.compute(
            estimate = sml_depth, 
            target = target_depth, 
            valid = mask.astype(bool),
        )
        
        # accumulate error metric
        avg_error_w_int_depth.accumulate(error_w_int_depth)
        avg_error_w_pred.accumulate(error_w_pred)
        
        if ROS_VIZ and i % 5 ==0:
            points_refine, colors_refine, _ = project_depth_vectorize(sml_depth_viz, input_image, p_CinG, R_CtoG, cam_K)
            #points_ga, colors_ga, _ = project_depth_vectorize(1.0/ga_depth, input_image, p_CinG, R_CtoG, cam_K)
            points_gt, colors_gt, _ = project_depth_vectorize(1.0/target_depth, input_image, p_CinG, R_CtoG, cam_K)
            #point_consis, colors_consis, _ = project_depth_vectorize(scaled_refine_depth_test, input_image, p_CinG, R_CtoG, cam_K)
            pc_sparse, _, _ = project_depth_vectorize(input_sparse_depth, input_image, p_CinG, R_CtoG, cam_K)
            
            visualizer.publish_path(poses)
            tgt_img = input_image*255.0
            tgt_img = overlay_sparse_points_on_image(tgt_img, 1.0/input_sparse_depth)
            visualizer.pose_callback(p_CinG, R_CtoG)
            visualizer.publish_point_cloud_refine(points_refine, colors_refine)
            visualizer.publish_point_cloud_gt(points_gt, colors_gt)
            visualizer.publish_tgt_img(tgt_img)
            visualizer.publish_sparse_points(pc_sparse)
            #visualizer.publish_point_cloud_refine(point_consis, colors_consis)
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
    
    parser.add_argument("--dataset_path", type=str, default="/media/saimouli/RPNG_FLASH_4/datasets/tartan/oldtown/Easy/P000/VI_depth")
    
    parser.add_argument("--depth_predictor", type=str, default='dpt_hybrid')
    
    parser.add_argument("--nsamples", type=int, default=150, help="Number of samples for SML")
    
    parser.add_argument("--sml_model_path", type=str, default="/home/saimouli/Documents/github/VI_Depth_sai/weights/sml_model.dpredictor.dpt_hybrid.nsamples.150.pretrained.ckpt")
    #/home/saimouli/Documents/github/VI_Depth_sai/weights/sml_model.dpredictor.dpt_hybrid.nsamples.150.pretrained.ckpt
    
    args = parser.parse_args()
    print(args)

    #plot_gt_traj("/media/saimouli/RPNG_FLASH_4/datasets/tartanv2/CarWelding/Data_easy/P001")
    evaluate_tartan_covar(
        args.dataset_path, 
        args.depth_predictor, 
        args.nsamples, 
        args.sml_model_path)
    
    # evaluate(
    #     args.dataset_path, 
    #     args.depth_predictor, 
    #     args.nsamples, 
    #     args.sml_model_path)
    
    # evaluate_table(
    #     args.dataset_path, 
    #     args.depth_predictor, 
    #     args.nsamples, 
    #     args.sml_model_path)
    
    # evaluate_light(
    #     args.dataset_path, 
    #     args.depth_predictor, 
    #     args.nsamples, 
    #     args.sml_model_path)