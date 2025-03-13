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
import cv2
import skimage.io
from utils_warp import inverse_warp, forward_warp, forward_warp_test, inverse_warp_test, photometric_loss, inverse_warp_sfm
import rospy
from visualizer.ros_visualizer import PointCloudVisualizer
import re
from natsort import natsorted
from scipy.spatial.transform import Rotation as R

from dataset import load_dataset
from config_utils import load_config
from munch import munchify
import kornia

def project_depth_vectorize(depth_img, img, p_CinG, R_CtoG, cam_K):
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

    bgr_values = img[y_coords, x_coords]#*255.0

    points = pFinG.T.tolist()
    colors = bgr_values.tolist()

    points = np.asarray(points).reshape(-1,3)
    colors = np.asarray(colors).reshape(-1,3)

    return points, colors

def test_warp(R_CtoG1, p_CinG1, R_CtoG2, p_CinG2, depth1, K, rgb1, rgb2):
    # Compute relative pose transformation from image1 to image2
    R_1to2 = R_CtoG2 @ R_CtoG1.transpose()
    t_1to2 = p_CinG2 - R_1to2 @ p_CinG1

    # Warp depth from image1 to image2
    h, w = depth1.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    uv1 = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3)
    d1 = depth1.flatten()

    # 3D points in camera1 frame
    points3D_C1 = (uv1 * d1[:, np.newaxis]) @ np.linalg.inv(K).T

    # Transform to camera2 frame
    points3D_C2 = (R_1to2 @ points3D_C1.T + t_1to2[:, np.newaxis]).T

    # Project to image2
    uv2_proj = (K @ points3D_C2.T).T
    uv2_proj /= uv2_proj[:, 2][:, np.newaxis]
    uv2_proj = uv2_proj[:, :2].reshape(h, w, 2)

    # Interpolate RGB image from image1 to image2
    rgb1_warped = cv2.remap(rgb1, uv2_proj[..., 0].astype(np.float32), uv2_proj[..., 1].astype(np.float32), 
                                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # Overlay the warped image on image2
    overlay = cv2.addWeighted(rgb2, 0.5, rgb1_warped, 0.5, 0)

    # Save or display the result
    #cv2.imwrite('Warped_rgb1.png', rgb1_warped)
    #cv2.imwrite('Overlay.png', overlay)
    plt.imshow(overlay)
    plt.show()
    #cv2.imshow('Warped RGB1', rgb1_warped)
    #cv2.imshow('Overlay', overlay)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def infer_depth(input_image_fp, max_depth, min_depth, device, method):
    input_image = utils.read_image(input_image_fp)

    #poses list
    pose_fp = input_image_fp.replace("image", "absolute_pose").replace(".png", ".txt")
    #Cam2Wld
    pose_CtoG = np.loadtxt(pose_fp)
    R_CtoG = pose_CtoG[:3, :3]
    p_CinG = pose_CtoG[:3, 3]
    R_GtoC = R_CtoG.transpose()

    # cam_K = np.loadtxt(dataset_path + "/K.txt")
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

    ga_depth = output["ga_depth"]
    est_depth = output["sml_depth"]

    return ga_depth, est_depth, target_depth, R_CtoG, p_CinG

# test warping from one depth image to another
# scale the depth from one depth frame to another one on void150 dataset
def run_sml(dataset_path, depth_predictor, nsamples, sml_model_path):
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
    
    # poses = []

    input_image_fp_1 = os.path.join(dataset_path, test_image_list[0])
    input_image_fp_2 = os.path.join(dataset_path, test_image_list[1])

    rgb1 = utils.read_image(input_image_fp_1)
    rgb2 = utils.read_image(input_image_fp_2)

    # poses list
    pose_fp_1 = input_image_fp_1.replace("image", "absolute_pose").replace(".png", ".txt")
    pose_fp_2 = input_image_fp_2.replace("image", "absolute_pose").replace(".png", ".txt")

    pose_C1toG = np.loadtxt(pose_fp_1)
    pose_C2toG = np.loadtxt(pose_fp_2)

    # Extract rotation and translation components
    R_C1toG = pose_C1toG[:3, :3]
    p_C1inG = pose_C1toG[:3, 3]
    p_GinC1 = -R_C1toG.transpose() @ p_C1inG
    R_GtoC1 = R_C1toG.transpose()

    R_C2toG = pose_C2toG[:3, :3]
    p_C2inG = pose_C2toG[:3, 3]
    R_GtoC2 = R_C2toG.transpose()

    # Compute relative pose transformation from image2 to image1
    R_C2toC1 = R_GtoC1 @ R_C2toG
    p_C2inC1 = p_GinC1 + R_GtoC1 @ p_C2inG

    cam_K = np.loadtxt(dataset_path + "/K.txt")

    ga_depth_1, est_depth_1, target_depth_1, _, _ = infer_depth(input_image_fp_1, max_depth, min_depth, device, method)
    ga_depth_2, est_depth_2, target_depth_2, _, _ = infer_depth(input_image_fp_2, max_depth, min_depth, device, method)
 
    #plt.figure(1);plt.imshow(1.0/target_depth_1)
    #plt.figure(2);plt.imshow(1.0/target_depth_2);plt.show()

    #Warp from image1 to image2
    # Prepare inputs for warping
    img1 = torch.tensor(rgb1).permute(2, 0, 1).unsqueeze(0).float().to(device)  # Convert to tensor and permute dimensions
    depth2 = torch.tensor(est_depth_2).unsqueeze(0).unsqueeze(0).float().to(device)  # Convert to tensor
    depth1 = torch.tensor(est_depth_1).unsqueeze(0).unsqueeze(0).float().to(device)  # Convert to tensor
    pose_2to1 = torch.tensor(np.hstack([R_C2toC1, p_C2inC1.reshape(3, 1)])).unsqueeze(0).float().to(device)  # Convert to tensor
    intrinsics = torch.tensor(cam_K).unsqueeze(0).float().to(device)  # Convert to tensor

    ## Warp image1 to image2's viewpoint
    #warped_img1, im_val = inverse_warp(img1, depth2, pose_2to1, intrinsics)
    
    #warps an image from image 1 (src) to image2 (target)
    warped_img2, w_depth_2, w_val_2 = forward_warp(img1, 1.0/depth1, pose_2to1, intrinsics, upscale=1)

    print("warped img1 shape: ", warped_img2.shape)
    # Convert tensors to numpy for visualization
    warped_img2_np = warped_img2.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    rgb2_np = rgb2.astype(np.uint8)
    rgb1_np = rgb1.astype(np.uint8)

    # Visualize overlay of image2 and warped image1
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image 2')
    plt.imshow(rgb1_np)
    plt.subplot(1, 2, 2)
    plt.title('Warped Image 1 to Image 2')
    plt.imshow(warped_img2_np)
    plt.show()

    #test_warp(R_CtoG1, p_CinG1, R_CtoG2, p_CinG2, 1.0/target_depth_1, cam_K, rgb1, rgb2)

    # for i in tqdm(range(len(test_image_list))):
    #     #image
    #     input_image_fp = os.path.join(dataset_path, test_image_list[i])
    #     input_image = utils.read_image(input_image_fp)

    #     #poses list
    #     pose_fp = input_image_fp.replace("image", "absolute_pose").replace(".png", ".txt")
    #     #Cam2Wld
    #     pose_CtoG = np.loadtxt(pose_fp)
    #     R_CtoG = pose_CtoG[:3, :3]
    #     p_CinG = pose_CtoG[:3, 3]
    #     R_GtoC = R_CtoG.transpose()

    #     poses.append(Pose(position=Point(
    #         x=p_CinG[0],
    #         y=p_CinG[1],
    #         z=p_CinG[2]
    #     )))

    #     cam_K = np.loadtxt(dataset_path + "/K.txt")
    #     # sparse depth
    #     input_sparse_depth_fp = input_image_fp.replace("image", "sparse_depth")
    #     input_sparse_depth = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32) / 256.0
    #     input_sparse_depth[input_sparse_depth <= 0] = 0.0
        
    #     validity_map = None

    #     target_depth_fp = input_image_fp.replace("image", "ground_truth")
    #     target_depth = np.array(Image.open(target_depth_fp), dtype=np.float32) / 256.0
    #     target_depth[target_depth <= 0] = 0.0
        
    #     # target depth valid/mask
    #     mask = (target_depth < max_depth)
    #     if min_depth is not None:
    #         mask *= (target_depth > min_depth)
    #     target_depth[~mask] = np.inf  # set invalid depth
    #     target_depth = 1.0 / target_depth

    #     output = method.run(input_image, input_sparse_depth, validity_map, device)

    #     ga_depth = output["ga_depth"]
    #     est_depth = output["sml_depth"]

    #     # plot subplot the ga_depth, sml_depth, target_depth two figures
    #     plt.figure()
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(ga_depth)
    #     plt.title("global ailgned Depth")
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(est_depth)
    #     plt.title("Estimated Depth")
    #     plt.subplot(1, 3, 3)
    #     plt.imshow(target_depth)
    #     plt.title("Ground Truth Depth")
    #     plt.show()

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1

# def compute_pairwise_loss():

def visualize(images, titles, figsize=(15, 15)):
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    if len(images) == 1:
        axes = [axes]
    for img, title, ax in zip(images, titles, axes):
        if len(img.shape) == 3 and img.shape[0] == 3:
            img = img.permute(1, 2, 0).cpu().numpy()
        elif len(img.shape) == 2:
            img = img.cpu().numpy()
        ax.imshow(img, cmap='viridis')
        ax.set_title(title)
        ax.axis('off')
    plt.show()

def test_sim():
    visualizer= PointCloudVisualizer()
    rate = rospy.Rate(1)

    poses = []

    fx = 600.0; fy = 600.0
    cx = 599.5; cy = 339.5
    camera_intrinsics = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    #cam2wld
    root_dir = '/media/saimouli/RPNG_FLASH_4/datasets/vmap/room_0/imap/01/'
    traj_file = os.path.join(root_dir, "traj_w_c.txt")
    TCW_ = np.loadtxt(traj_file, delimiter=" ").reshape([-1, 4, 4])
    #TCw = np.linalg.inv(Twc)

    #get image list in the folder
    rgb_folder = "/media/saimouli/RPNG_FLASH_4/datasets/vmap/room_0/imap/01/rgb/"
    depth_folder = "/media/saimouli/RPNG_FLASH_4/datasets/vmap/room_0/imap/01/depth/"

    rgb_files = [f for f in os.listdir(rgb_folder) if f.endswith('.png')]
    rgb_files = natsorted(rgb_files, key=extract_number)

    depth_files = [f for f in os.listdir(depth_folder) if f.endswith('.png')]
    depth_files = natsorted(depth_files, key=extract_number)

    idx = 0; idx2 = 6
    transformation1 = TCW_[idx]
    R_C1toG = transformation1[:3, :3]
    p_C1inG = transformation1[:3, 3]
    p_GinC1 = -R_C1toG.transpose() @ p_C1inG
    R_GtoC1 = R_C1toG.transpose()

    transformation2 = TCW_[idx2]
    R_C2toG = transformation2[:3, :3]
    p_C2inG = transformation2[:3, 3]
    p_GinC2 = -R_C2toG.transpose() @ p_C2inG
    R_GtoC2 = R_C2toG.transpose()

    R_C2toC1 = R_GtoC1 @ R_C2toG
    R_C1toC2 = R_C2toC1.transpose()

    p_C2inC1 = p_GinC1 + R_GtoC1 @ p_C2inG
    p_C1inC2 = -R_C1toC2 @ p_C2inC1

    rgb1_name = rgb_folder + rgb_files[idx]
    print("rgb1_name: ", rgb1_name)
    rgb1 = cv2.imread(rgb1_name)
    rgb1 = cv2.cvtColor(rgb1, cv2.COLOR_BGR2RGB)
    depth1_name = depth_folder + depth_files[idx]
    depth1_data = cv2.imread(depth1_name, cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth1 = np.nan_to_num(depth1_data, nan=0.)/1000.0

    rgb2_name = rgb_folder + rgb_files[idx2]
    print("rgb2_name: ", rgb2_name)
    rgb2 = cv2.imread(rgb2_name)
    rgb2 = cv2.cvtColor(rgb2, cv2.COLOR_BGR2RGB)
    depth2_name = depth_folder + depth_files[idx2]
    depth2_data = cv2.imread(depth2_name, cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth2 = np.nan_to_num(depth2_data, nan=0.)/1000.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img1 = torch.tensor(rgb1).permute(2, 0, 1).unsqueeze(0).float().to(device)  # Convert to tensor and permute dimensions
    img2 = torch.tensor(rgb2).permute(2, 0, 1).unsqueeze(0).float().to(device)  # Convert to tensor and permute dimensions
    depth2 = torch.tensor(depth2).unsqueeze(0).unsqueeze(0).float().to(device)  # Convert to tensor
    depth1 = torch.tensor(depth1).unsqueeze(0).unsqueeze(0).float().to(device)  # Convert to tensor
    pose_2to1 = torch.tensor(np.hstack([R_C2toC1, p_C2inC1.reshape(3, 1)])).unsqueeze(0).float().to(device)  # Convert to tensor
    pose_1to2 = torch.tensor(np.hstack([R_C1toC2, p_C1inC2.reshape(3, 1)])).unsqueeze(0).float().to(device)
    intrinsics = torch.tensor(camera_intrinsics).unsqueeze(0).float().to(device)  # Convert to tensor

    ##ref img: img1; target img: img2
    ref_img_warped, projected_depth, computed_depth = inverse_warp_sfm(img1, depth2, depth1, pose_2to1, intrinsics, padding_mode='zeros')
    ref_img_warped_2, projected_depth_2, computed_depth_2 = inverse_warp_sfm(img2, depth1, depth2, pose_1to2, intrinsics, padding_mode='zeros')

    diff_depth = (computed_depth - projected_depth).abs() / (computed_depth + projected_depth) 
    ## masking zero values
    valid_mask_ref = (ref_img_warped.abs().mean(
         dim=1, keepdim=True) > 1e-3).float()
    valid_mask_tgt = (img2.abs().mean(dim=1, keepdim=True) > 1e-3).float()
    valid_mask = valid_mask_tgt * valid_mask_ref

    diff_color = (img2-ref_img_warped).abs().mean(dim=1, keepdim=True)
    diff_img = (img2-ref_img_warped).abs().clamp(0, 1)

    # Visualize results
    visualize([rgb1, rgb2, ref_img_warped.squeeze(0).permute(1,2,0).cpu().numpy().astype(np.uint8), ref_img_warped_2.squeeze(0).permute(1,2,0).cpu().numpy().astype(np.uint8)],
            ["Reference Image (Prev)", "Target Image (Curr)", "Warped img1to2", "Warped img2to1"])
    
    
    #warped_img2, w_depth_2, w_val_2 = forward_warp(img1, depth1, pose_2to1, intrinsics, upscale=1)
    #warped_img1, im_val = inverse_warp(img1, depth1, pose_2to1, intrinsics) 
    #warped_img1, _ = inverse_warp_test(img2, depth2, pose_2to1, intrinsics)
    #warped_img2, _ = forward_warp_test(img1, depth1, pose_1to2, intrinsics, upscale=1)
    #warped_img1_np = warped_img1.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    #warped_img2_np = warped_img2.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    #rgb2_np = rgb2.astype(np.uint8)
    #rgb1_np = rgb1.astype(np.uint8)
    # Overlay the warped image on the target image
    #alpha = 0.5
    #overlay_img = (alpha * rgb1_np + (1 - alpha) * warped_img1_np).astype(np.uint8)
    #overlay_img2 = (alpha * rgb2_np + (1 - alpha) * warped_img2_np).astype(np.uint8)

    #print(photometric_loss())

    # # Plotting
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 4, 1)
    # plt.title('Source Image (t1)')
    # plt.imshow(rgb1_np)
    # plt.axis('off')

    # plt.subplot(1, 4, 2)
    # plt.title('Target Image (t2)')
    # plt.imshow(rgb2_np)
    # plt.axis('off')

    # plt.subplot(1, 4, 3)
    # plt.title('Warped Img2 to Img1')
    # plt.imshow(warped_img1_np)
    # plt.axis('off')
    # plt.show()

    # plt.subplot(1, 4, 4)
    # plt.title('Warped Img1 to Img2')
    # plt.imshow(warped_img2_np)
    # plt.axis('off')
    # plt.show()

    # for idx in range(0, 1000, 50):
    #     Twc_idx = Twc[idx]
    #     #Tcw = np.linalg.inv(Twc_idx)
    #     transformation1 = Twc_idx
    #     R_C1toG = transformation1[:3, :3]
    #     p_C1inG = transformation1[:3, 3]
    #     p_GinC1 = -R_C1toG.transpose() @ p_C1inG
    #     R_GtoC1 = R_C1toG.transpose()
        
    #     #rgb1 = utils.read_image(input_image_fp_1)
    #     rgb1_name = rgb_folder + rgb_files[idx]
    #     print("rgb1_name: ", rgb1_name)
    #     rgb1 = cv2.imread(rgb1_name)
    #     rgb1 = cv2.cvtColor(rgb1, cv2.COLOR_BGR2RGB)
    #     #depth1 = cv2.imread(depth_fp_1, -1).astype(np.float32).transpose(1,0)
    #     depth_name = depth_folder + depth_files[idx]
    #     depth_data = cv2.imread(depth_name, cv2.IMREAD_UNCHANGED).astype(np.float32)
    #     depth1 = np.nan_to_num(depth_data, nan=0.)/1000.0
    #     print("depth_name: ", depth_name)
    #     #depth1 = cv2.imread(depth_fp_1, cv2.IMREAD_UNCHANGED)[..., None]
    #     #print("depth min: ", np.min(depth1), "depth max: ", np.max(depth1))
    #     #rgb2 = cv2.imread(rgb_folder + rgb_files[idx+1]) #utils.read_image(rgb_files[idx+1])
    #     #rgb2 = cv2.cvtColor(rgb2, cv2.COLOR_BGR2RGB)
    #     #depth2 = cv2.imread(depth_fp_2, cv2.IMREAD_UNCHANGED)[..., None]
    #     #depth2_data = cv2.imread(depth_folder + depth_files[idx+1], -1).astype(np.float32).transpose(1,0)
    #     #depth2 = np.nan_to_num(depth2_data, nan=0.)/1000.0

    #     points_gt, colors_gt = project_depth_vectorize(depth1, rgb1, p_C1inG, R_C1toG, camera_intrinsics)
        
    #     poses.append(Pose(position=Point(
    #         x=p_C1inG[0],
    #         y=p_C1inG[1],
    #         z=p_C1inG[2]
    #     )))
        
    #     visualizer.publish_path(poses)
    #     visualizer.publish_point_cloud_gt(points_gt, colors_gt)
    #     rate.sleep()

    # plt.figure(1);plt.imshow(rgb1)
    # plt.figure(2);plt.imshow(rgb2)
    # plt.figure(1);plt.imshow(depth1)
    # plt.figure(2);plt.imshow(depth2)
    # plt.show()

    ##Put text to rgb1
    #font = cv2.FONT_HERSHEY_SIMPLEX
    #org = (150, 150)
    #fontScale = 9
    #color = (255, 0, 0)
    #thickness = 2
    #rgb1 = cv2.putText(rgb1, 'Image 1', org, font, fontScale, color, thickness, cv2.LINE_AA)
    #rgb2 = cv2.putText(rgb2, 'Image 2', org, font, fontScale, color, thickness, cv2.LINE_AA)

    #test_warp(transformation1, transformation2, depth, camera_intrinsics, img1, img2)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # img1 = torch.tensor(rgb1).permute(2, 0, 1).unsqueeze(0).float().to(device)
    # depth1 = torch.tensor(depth1).unsqueeze(0).unsqueeze(0).float().to(device)
    # img2 = torch.tensor(rgb2).permute(2, 0, 1).unsqueeze(0).float().to(device)
    # depth2 = torch.tensor(depth2).unsqueeze(0).unsqueeze(0).float().to(device)
    # pose_2to1 = torch.tensor(np.hstack([R_C2toC1, p_C2inC1.reshape(3, 1)])).unsqueeze(0).float().to(device)  # Convert to tensor
    # intrinsics = torch.tensor(camera_intrinsics).unsqueeze(0).float().to(device)
    
    #fwd warp the images from img1 to img2


    #warped_img2, w_depth_2, w_val_2 = forward_warp(img1, depth1, pose_2to1, intrinsics, upscale=1)
    #warped_img2, im_val = inverse_warp(img1, depth1, pose_2to1, intrinsics) 
    #warped_img2_np = warped_img2.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    #rgb2_np = rgb2.astype(np.uint8)
    #rgb1_np = rgb1.astype(np.uint8)

    # Visualize overlay of image2 and warped image1
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.title('Original Image 2')
    # plt.imshow(rgb2_np)
    # plt.subplot(1, 2, 2)
    # plt.title('Warped Image 1 to Image 2')
    # plt.imshow(warped_img2_np)
    # plt.show()


def tum_data():
    visualizer= PointCloudVisualizer()
    rate = rospy.Rate(1)
    config = load_config("/home/saimouli/Documents/github/VI_Depth_sai/configs/mono/tum/office.yaml")
    model_params = munchify(config["model_params"])
    dataset = load_dataset(model_params, model_params.source_path, config=config)
    poses = []

    for idx in range(0, 600, 5):
        gt_img, gt_depth, gt_pose = dataset[idx]

        gt_pose = gt_pose.detach().cpu().numpy()
        #gt_img = gt_img.detach().cpu().permute(1,2,0).numpy()
        p_C1inG = gt_pose[:3, 3]
        R_C1toG = gt_pose[:3, :3]
        points_gt, colors_gt = project_depth_vectorize(gt_depth, gt_img, p_C1inG, R_C1toG, dataset.K)

        poses.append(Pose(position=Point(
            x=p_C1inG[0],
            y=p_C1inG[1],
            z=p_C1inG[2]
        )))
        
        visualizer.publish_path(poses)
        visualizer.publish_point_cloud_gt(points_gt, colors_gt)
        rate.sleep()




if __name__ == "__main__":
    pass
    #run_sml(
    #    dataset_path="/media/saimouli/RPNG_FLASH_4/datasets/VOID_150/classroom6",
    #    depth_predictor="dpt_hybrid",
    #    nsamples=150,
    #    sml_model_path="/home/saimouli/Documents/github/VI_Depth_sai/weights/sml_model.dpredictor.dpt_hybrid.nsamples.150.ckpt")

    #test_sim()
    #tum_data()
    