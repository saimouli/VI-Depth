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
from torch.utils.data import DataLoader
import cv2
import modules.midas.transforms as transforms
from modules.midas.midas_net_custom import MidasNet_small_videpth

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
    min_pred, max_pred = 0.1, 8.0
    
    # instantiate method
    method = pipeline.VIDepth(
        depth_predictor, nsamples, sml_model_path, 
        min_pred, max_pred, min_depth, max_depth, device
    )

    #model = midasNet(min_pred, max_pred, min_depth, max_depth, nsamples, sml_model_path)
    
    # get inputs
    with open(f"{dataset_path}/test_image.txt") as f: 
        test_image_list = [line.rstrip() for line in f]
    
    #read all the list of images in folder
    #dataset_path_fld = os.path.join(dataset_path, "image")
    #test_image_list = sorted([os.path.basename(f) for f in glob.glob(os.path.join(dataset_path_fld, "*.png"))])
    
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
        
        # sparse depth
        # input_sparse_depth_fp = input_image_fp.replace("image", "sparse_depth")
        # input_sparse_depth = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32) / 256.0
        # input_sparse_depth[input_sparse_depth <= 0] = 0.0
        
        validity_map = None

        target_depth_fp = input_image_fp.replace("image", "ground_truth")
        target_depth = np.array(Image.open(target_depth_fp), dtype=np.float32) / 256.0
        target_depth[target_depth <= 0] = 0.0

        #Load the consistent scale depth
        refine_depth_fp = input_image_fp.replace("image", "output/model_v3/depth")
        refine_depth_fp = refine_depth_fp.replace(".png", ".npy")
        refine_depth_test = np.array(np.load(refine_depth_fp))
        #resize to  480, 640
        refine_depth_test = cv2.resize(refine_depth_test, (640, 480), interpolation=cv2.INTER_NEAREST)
        refine_depth_test[refine_depth_test <= 0] = 0.0
        valid_mask = (target_depth > 0) & (refine_depth_test > 0)
        scaling_factor = 0.0526 #np.median(target_depth[valid_mask] / refine_depth_test[valid_mask])
        print(scaling_factor)
        scaled_refine_depth_test = refine_depth_test * scaling_factor
        scaled_refine_depth_test[~valid_mask] = 0.0

        
        # # target depth valid/mask
        # mask = (target_depth < max_depth)
        # if min_depth is not None:
        #     mask *= (target_depth > min_depth)
        # target_depth[~mask] = np.inf  # set invalid depth
        # target_depth = 1.0 / target_depth
        
        # output = method.run(input_image, input_sparse_depth, validity_map, device)
        # ga_depth = output["ga_depth"]
        # sml_depth = output["sml_depth"]
        # #ga_depth, sml_depth = model.forward(input_sparse_depth_inv, input_image, depth_pred_inv, interp_scale, GA_depth_inv, None)

        # # compute error metrics using intermediate (globally aligned) depth
        # error_w_int_depth = metrics.ErrorMetrics()
        # error_w_int_depth.compute(
        #     estimate = ga_depth, 
        #     target = target_depth, 
        #     valid = mask.astype(bool),
        # )

        # # compute error metrics using SML output depth
        # error_w_pred = metrics.ErrorMetrics()
        # error_w_pred.compute(
        #     estimate = sml_depth, 
        #     target = target_depth, 
        #     valid = mask.astype(bool),
        # )
        
        # # accumulate error metric
        # avg_error_w_int_depth.accumulate(error_w_int_depth)
        # avg_error_w_pred.accumulate(error_w_pred)
        
        if ROS_VIZ and i % 2 ==0:
            #points_refine, colors_refine, _ = project_depth_vectorize(1.0/sml_depth, input_image, p_CinG, R_CtoG, cam_K)
            #points_ga, colors_ga, _ = project_depth_vectorize(1.0/ga_depth, input_image, p_CinG, R_CtoG, cam_K)
            points_gt, colors_gt, _ = project_depth_vectorize(target_depth, input_image, p_CinG, R_CtoG, cam_K)
            point_consis, colors_consis, _ = project_depth_vectorize(scaled_refine_depth_test, input_image, p_CinG, R_CtoG, cam_K)
            
            visualizer.publish_path(poses)
            visualizer.pose_callback(p_CinG, R_CtoG)
            #visualizer.publish_point_cloud_refine(points_refine, colors_refine)
            visualizer.publish_point_cloud_gt(points_gt, colors_gt)
            visualizer.publish_point_cloud_refine(point_consis, colors_consis)
            rate.sleep()
    
    # # compute average error metrics
    # print("Averaging metrics for globally-aligned depth over {} samples".format(
    #     avg_error_w_int_depth.total_count
    # ))
    # avg_error_w_int_depth.average()

    # print("Averaging metrics for SML-aligned depth over {} samples".format(
    #     avg_error_w_pred.total_count
    # ))
    # avg_error_w_pred.average()
    
    # from prettytable import PrettyTable
    # summary_tb = PrettyTable()
    # summary_tb.field_names = ["Metric", "GA Only", "GA+SML"]

    # summary_tb.add_row(["RMSE", f"{avg_error_w_int_depth.rmse_avg:7.2f}", f"{avg_error_w_pred.rmse_avg:7.2f}"])
    # summary_tb.add_row(["MAE", f"{avg_error_w_int_depth.mae_avg:7.2f}", f"{avg_error_w_pred.mae_avg:7.2f}"])
    # summary_tb.add_row(["AbsRel", f"{avg_error_w_int_depth.absrel_avg:8.3f}", f"{avg_error_w_pred.absrel_avg:8.3f}"])
    # summary_tb.add_row(["iRMSE", f"{avg_error_w_int_depth.inv_rmse_avg:7.2f}", f"{avg_error_w_pred.inv_rmse_avg:7.2f}"])
    # summary_tb.add_row(["iMAE", f"{avg_error_w_int_depth.inv_mae_avg:7.2f}", f"{avg_error_w_pred.inv_mae_avg:7.2f}"])
    # summary_tb.add_row(["iAbsRel", f"{avg_error_w_int_depth.inv_absrel_avg:8.3f}", f"{avg_error_w_pred.inv_absrel_avg:8.3f}"])
    
    # print(summary_tb)

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
    min_depth, max_depth = 0.2, 8.0
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
        mask = mask[0][0].numpy()
        error_w_int_depth = metrics.ErrorMetrics()
        error_w_int_depth.compute(
            estimate = ga_depth_inv[0].numpy(), 
            target = gt_depth_inv[0].numpy(), 
            valid = mask.astype(bool),
        )

        # # compute error metrics using SML output depth
        error_w_pred = metrics.ErrorMetrics()
        error_w_pred.compute(
            estimate = sml_pred, 
            target = gt_depth_inv[0].numpy(), 
            valid = mask.astype(bool),
        )
        
        # # accumulate error metric
        avg_error_w_int_depth.accumulate(error_w_int_depth)
        avg_error_w_pred.accumulate(error_w_pred)
        
        if ROS_VIZ and batch_idx % 4 ==0:
            #resize target depth to 480, 640
            #target_depth = torch.nn.functional.interpolate(gt_depth_inv, size=(480, 640), mode='bicubic')[0][0].numpy()
            #image = torch.nn.functional.interpolate(image, size=(480, 640), mode='bicubic')[0]
            depth_gt = 1.0/gt_depth_inv[0].numpy()
            depth_gt[depth_gt == float("inf")] = 0

            ga_depth = 1.0/ga_depth_inv[0].numpy()
            ga_depth[ga_depth == float("inf")] = 0

            sml_pred = 1.0/sml_pred
            sml_pred[sml_pred == float("inf")] = 0

            points_refine, colors_refine, _ = project_depth_vectorize(sml_pred, image[0], p_CinG, R_CtoG, cam_K)
            #points_ga, colors_ga, _ = project_depth_vectorize(ga_depth, image[0], p_CinG, R_CtoG, cam_K)
            #points_gt, colors_gt, _ = project_depth_vectorize(depth_gt, image[0], p_CinG, R_CtoG, cam_K)
            #gt_depth_inv[0][0].numpy()
            
            visualizer.publish_path(poses)
            visualizer.pose_callback(p_CinG, R_CtoG)
            visualizer.publish_point_cloud_refine(points_refine, colors_refine)
            #visualizer.publish_point_cloud_gt(points_gt, colors_gt)
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
    
    parser.add_argument("--dataset_path", type=str, default="/media/saimouli/Data6T/sc_depth/void/office3")
    
    parser.add_argument("--depth_predictor", type=str, default='dpt_hybrid')
    
    parser.add_argument("--nsamples", type=int, default=150, help="Number of samples for SML")
    
    parser.add_argument("--sml_model_path", type=str, default="/home/saimouli/Documents/github/VI_Depth_sai/weights/sml_model.dpredictor.dpt_hybrid.nsamples.150.ckpt")
    
    args = parser.parse_args()
    print(args)
    
    evaluate(
        args.dataset_path, 
        args.depth_predictor, 
        args.nsamples, 
        args.sml_model_path)