import numpy as np
#import matplotlib.pyplot as plt

import pipeline
from utils_eval import compute_ls_solution
import modules.midas.utils as utils
from modules.interpolator import Interpolator2D, PolynomialInterpolator2D
from tqdm import tqdm
from PIL import Image
import torch
from collections import OrderedDict
import matplotlib.pyplot as plt
import cv2
#from model.mhybrid_consistent_net import InitDepth
from model.main_consistent import midasNetConsistentModule
from data.SML_consistent_resize import SML_consistent_resize
from torch.nn import functional as F
import modules.midas.transforms as transforms
#from test_scaffolding import plot_surface_normals

def load_sparse_depth(input_sparse_depth_fp):
    input_sparse_depth = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32) / 256.0
    input_sparse_depth[input_sparse_depth <= 0] = 0.0
    return input_sparse_depth

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

def get_ga_and_scale(depth_pred, input_sparse_depth, input_sparse_depth_valid, min_pred, max_pred):
    int_depth,_,_ = compute_ls_solution(depth_pred, input_sparse_depth, input_sparse_depth_valid, min_pred, max_pred)
        
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

def get_ga_scale_using_init_depth(depth_pred_rel, init_depth_inv, input_sparse_depth, input_sparse_depth_valid, min_pred, max_pred):
    ga_depth_inv,_,_ = compute_ls_solution(depth_pred_rel, input_sparse_depth, input_sparse_depth_valid, min_pred, max_pred)


    #init scales from the init_depth_inv/int_depth
    valid_mask = (init_depth_inv > (1/0.2)) * (init_depth_inv < (1/5.0))
    valid_mask = valid_mask.astype(bool)
    #global_scales = np.ones_like(ga_depth_inv)
    #global_scales[valid_mask] = init_depth_inv[valid_mask] / ga_depth_inv[valid_mask]
    #int_scales = utils.normalize_unit_range(global_scales)

    ScaleMapInterpolator = Interpolator2D(
        pred_inv = ga_depth_inv,
        sparse_depth_inv = init_depth_inv, #init_depth_inv,
        valid = input_sparse_depth_valid,
    )
    
    ScaleMapInterpolator.generate_interpolated_scale_map(
        interpolate_method='linear',
        fill_corners=False
    )
    int_scales = ScaleMapInterpolator.interpolated_scale_map.astype(np.float32)
    int_scales = utils.normalize_unit_range(int_scales)

    return ga_depth_inv, int_scales

def save_depth_image_as_npy(depth_array, save_path):
    # Ensure depth_array is in 32-bit float format
    depth_array = depth_array.astype(np.float32)
    
    # Save the depth image as a .npy file
    np.save(save_path, depth_array)

def load_depth_image_from_npy(load_path):
    # Load the depth image from the .npy file
    depth_array = np.load(load_path)# + '.npy')
    
    return depth_array

def strategic_scaffold_filling(input_sparse_depth, 
                               plane_masks,
                               plane_params, 
                               intrinsic,
                               min_points_per_plane = 3, 
                               fill_ratio=0.0005,
                               inlier_threshold=0.05,
                               edge_buffer=2):
    
    height, width = input_sparse_depth.shape
    valid_mask = input_sparse_depth > 0.2
    densified_depth = input_sparse_depth.copy()

    num_planes = plane_masks.shape[0]
    print("Number of planes: ", num_planes)
    
    fx = intrinsic[0, 0]; fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]; cy = intrinsic[1, 2]
    edge_kernel = np.ones((3, 3), np.uint8)
    
    for plane_idx in range(num_planes):
        # Create binary mask for this plane
        plane_mask = plane_masks[plane_idx] > 0.7
        
        # Skip empty planes
        if np.sum(plane_mask) < 100:
            continue
        
        if edge_buffer > 0:
            eroded_mask = cv2.erode(plane_mask.astype(np.uint8), edge_kernel, iterations=edge_buffer)
            # plt.figure(1)
            # plt.imshow(eroded_mask.astype(bool))
            edge_mask = plane_mask & ~eroded_mask.astype(bool)
            # plt.figure(2)
            # plt.imshow(edge_mask)
            # plt.show()
        else:
            edge_mask = np.zeros_like(plane_mask, dtype=bool)
            
        # Find sparse points in this plane
        good_points_mask = plane_mask & valid_mask & ~edge_mask
        n_good_points = np.sum(good_points_mask)
        print("Number of good points: ", n_good_points)
        points_in_plane = good_points_mask
        n_points = n_good_points
        
        if n_good_points >= min_points_per_plane:
            a, b, c = plane_params[plane_idx]
            normal_length = np.sqrt(a*a + b*b + c*c)
            
            if normal_length < 1e-6:
                continue
            
            d = normal_length
            normal = np.array([a, b, c]) / d
            
            y_valid, x_valid = np.where(points_in_plane)
            depths_valid = input_sparse_depth[points_in_plane]
            
            # Convert 2D points to 3D using camera intrinsics
            Z = depths_valid
            X = (x_valid - cx) * Z / fx
            Y = (y_valid - cy) * Z / fy
            
            # Calculate plane size and points to add
            plane_size = np.sum(plane_mask)
            n_points_to_add = min(int(plane_size * fill_ratio), plane_size - n_points)
            
            if n_points_to_add > 0:
                # Extract sparse point coordinates and depths
                empty_pixels = plane_mask & (~valid_mask)
                y_empty, x_empty = np.where(empty_pixels)
                
                if len(y_empty) > n_points_to_add: #we randomly sample points to reduce the affects of bad planes
                    indices = np.random.choice(len(y_empty), n_points_to_add, replace=False)
                    y_sampled = y_empty[indices]
                    x_sampled = x_empty[indices]
                else:
                    y_sampled = y_empty
                    x_sampled = x_empty
                
                # For each sampled point, assign depth from nearest sparse point
                for i in range(len(y_sampled)):
                    y, x = y_sampled[i], x_sampled[i]
                    # Calculate depth using plane equation: d = a*X + b*Y + c*Z
                    # Solving for Z: Z = d / (a*(u-cx)/fx + b*(v-cy)/fy + c)
                    denominator = (normal[0] * (x - cx) / fx + 
                                   normal[1] * (y - cy) / fy + 
                                   normal[2])
                    
                    if abs(denominator) > 1e-6:
                        depth_value = d / denominator
                    
                        if depth_value > 0:
                            densified_depth[y, x] = depth_value
            
    return densified_depth

def visualize_depth_completion_process(input_image, input_sparse_depth, plane_masks, plane_norms,
                                       densified_depth, probability_threshold=0.7):
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
        
# This script is used to save ga depth, depth, and interpolated needed for training
def save_priors(data_dir):
    device = "cuda"; nsamples = 150; sml_model_path = ""
    depth_predictor = "dpt_hybrid"
    #data_dir = "/media/saimouli/RPNG_FLASH_4/datasets/VOID_150"
    #read all the folders in the data_dir except .txt files

    import os
    folders = [f for f in os.listdir(data_dir) if not f.endswith('.txt')]
    print("Folder length: ", len(folders))

    # for each folder, read the images in image folder
    for folder in folders:
        print("Folder: ", folder)

        image_folder = os.path.join(data_dir, folder, "image")
        # get list of images in the image folder
        images = [f for f in os.listdir(image_folder) if f.endswith('.png')]
        images_path = [os.path.join(image_folder, f) for f in images]

        sparse_folder = os.path.join(data_dir, folder, "sparse_depth")
        sprase_depth_path = [os.path.join(sparse_folder, f) for f in images]

        min_depth, max_depth = 0.1, 5.0
        min_pred, max_pred = 0.1, 5.0

        # Instantiate method
        method = pipeline.VIDepth(
            depth_predictor, nsamples, sml_model_path, 
            min_pred, max_pred, min_depth, max_depth, device
        )

        print("Images: ", len(images))
        print("Sparse: ", len(sprase_depth_path))

        save_folder = os.path.join(data_dir, folder)
        save_folder = os.path.join(save_folder, "depth_infer_dpt")
        print("Save folder: ", save_folder)
        os.makedirs(save_folder, exist_ok=True)
        save_folder = os.path.join(data_dir, folder)
        save_folder = os.path.join(save_folder, "ga_depth_inv")
        os.makedirs(save_folder, exist_ok=True)
        save_folder = os.path.join(data_dir, folder)
        save_folder = os.path.join(save_folder, "interp_scale")
        os.makedirs(save_folder, exist_ok=True)

        for i in tqdm(range(len(images))):
            input_image_fp = images_path[i]
            input_sparse_depth_fp = sprase_depth_path[i]
            input_image = utils.read_image(input_image_fp)
            input_sparse_depth = load_sparse_depth(input_sparse_depth_fp)

            depth_infer_inv = method.infer_depth(input_image)
            input_sparse_depth_valid = (input_sparse_depth < max_depth) * (input_sparse_depth > min_depth)

            input_sparse_depth_valid = input_sparse_depth_valid.astype(bool)
            input_sparse_depth[~input_sparse_depth_valid] = np.inf # set invalid depth
            input_sparse_depth_inv = 1.0 / input_sparse_depth
            
            #if sparse depth points are less than 3 continue
            print("# of sparse points: ", np.sum(input_sparse_depth_valid))
            if np.sum(input_sparse_depth_valid) < 50:
                continue

            ga_depth_inv, interp_scale = get_ga_and_scale(depth_infer_inv, input_sparse_depth_inv, 
                                                          input_sparse_depth_valid, min_pred, max_pred )

            # plt.figure(1);plt.imshow(ga_depth_inv)
            # plt.figure(2);plt.imshow(interp_scale)
            # plt.show()

            ## save the images in the respective folders
            save_image_path = os.path.join(data_dir, folder, "depth_infer_dpt", images[i])
            save_image_path = save_image_path.replace('.png', '.npy')

            save_depth_image_as_npy(depth_infer_inv, save_image_path)
            save_image_path = os.path.join(data_dir, folder, "ga_depth_inv", images[i])
            save_image_path = save_image_path.replace('.png', '.npy')
            #Image.fromarray(ga_depth_inv).save(save_image_path)
            save_depth_image_as_npy(ga_depth_inv, save_image_path)
            save_image_path = os.path.join(data_dir, folder, "interp_scale", images[i])
            save_image_path = save_image_path.replace('.png', '.npy')
            #Image.fromarray(interp_scale).save(save_image_path)
            save_depth_image_as_npy(interp_scale, save_image_path)

            #depth_infer_load = load_depth_image_from_npy(save_image_path) #os.path.join(data_dir, folder, "depth_infer_dpt", images[i]))
            # if not np.array_equal(depth_infer_load, depth_infer_inv):
            #     # throw error
            #     print("Depth infer not equal")
            #     break
            #test = 0
        # for list in image_folder read the image and sparse depth from respective folders

def rotation_angle(R1, R2):
    """Compute the angle (in degrees) between two rotation matrices using trace."""
    R = R1 @ R2.T  # Relative rotation
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)  # Numerical stability
    return np.degrees(np.arccos(cos_theta))  # Convert to degrees
 
def create_frame_index(data_dir):
    # Creates frame that has motion > 0.1 cm
    import os
    folders = [f for f in os.listdir(data_dir) if not f.endswith('.txt')]
    print("Folder length: ", len(folders))

    for folder in folders:
        print("Folder: ", folder)
        image_folder = os.path.join(data_dir, folder, "image")
        # get list of images in the image folder
        images = [f for f in os.listdir(image_folder) if f.endswith('.png')]
        images.sort()
        images_path = [os.path.join(image_folder, f) for f in images]

        pose_folder = os.path.join(data_dir, folder, "absolute_pose")
        poses = [f for f in os.listdir(pose_folder) if f.endswith('.txt')]
        poses.sort()
        pose_path = [os.path.join(pose_folder, f) for f in poses]

        index = [0]; frame_names = [images[0]]
        for idx in range(1, len(images)):

            #frame1 = cv2.imread(images_path[index[-1]])
            #frame2 = cv2.imread(images_path[idx])

            pose1 = np.loadtxt(pose_path[index[-1]])
            pose2 = np.loadtxt(pose_path[idx])

            #if pose movement is > 0.1m?
            pose_diff = np.linalg.norm(pose1[:3, 3] - pose2[:3, 3])
            R1, R2 = pose1[:3, :3], pose2[:3, :3]
            rot_angle = rotation_angle(R1, R2)
            
            if pose_diff < 0.05: # and rot_angle < 6:
                continue
            index.append(idx)
            frame_names.append(images[idx])

        print(len(images), len(frame_names))
        np.savetxt(os.path.join(data_dir, folder, "frame_index.txt"), index, fmt='%d', delimiter='\n')

def save_normals(data_dir):
    import os
    folders = [f for f in os.listdir(data_dir) if not f.endswith('.txt')]
    print("Folder length: ", len(folders))
    
    for folder in folders:
        print("Folder: ", folder)

        depth_folder = os.path.join(data_dir, folder, "depth_infer_dpt")
        depth_path = [os.path.join(depth_folder, f) for f in os.listdir(depth_folder) if f.endswith('.npy')]
        
        save_folder = os.path.join(data_dir, folder)
        save_folder = os.path.join(save_folder, "dpt_normals")
        os.makedirs(save_folder, exist_ok=True)
        
        for i in tqdm(range(len(depth_path))):
            dpt_depth = load_depth_image_from_npy(depth_path[i])
            
            normals_cam = utils.compute_normals(dpt_depth)
            save_image_path = os.path.join(data_dir, folder, "dpt_normals", os.path.basename(depth_path[i]))
            save_depth_image_as_npy(normals_cam, save_image_path)
            
def save_init_depth(data_dir):
    # prepare the dataset loader
    import os
    model = midasNetConsistentModule(sml_model_path=None, useConvGRU=True, is_train=False)
    model = model.load_from_checkpoint("/home/saimouli/Documents/github/VI_Depth_sai/weights/init_depth/epoch=20-val/total_loss=0.137.ckpt")
    model.eval()
    device = "cuda"
    model = model.to(device)

    # folders = [f for f in os.listdir(data_dir) if not f.endswith('.txt')]
    # print("Folder length: ", len(folders))

    # for folder in folders:
    #     print("Folder: ", folder)
    #     image_folder = os.path.join(data_dir, folder, "image")
    #     images = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    #     images_path = [os.path.join(image_folder, f) for f in images]

    #     sparse_folder = os.path.join(data_dir, folder, "sparse_depth")
    #     sprase_depth_path = [os.path.join(sparse_folder, f) for f in images]

    #     min_depth, max_depth = 0.1, 5.0
    #     min_pred, max_pred = 0.1, 8.0

    #     save_folder = os.path.join(data_dir, folder)

    #     for i in tqdm(range(len(images))):
    #         input_image_fp = images_path[i]
    #         input_sparse_depth_fp = sprase_depth_path[i]
    #         tgt_img = torch.from_numpy(utils.read_image(input_image_fp)).unsqueeze(0).to(device)
    #         tgt_sparse_depth = torch.from_numpy(load_sparse_depth(input_sparse_depth_fp)).unsqueeze(0).to(device)
    #         #dummy ref pose tgtpose and intrinsics, ref_img, tgt_ga_depth, tgt_interp
    #         tgt_pose = torch.eye(4).unsqueeze(0).to(device)
    #         ref_pose = [torch.eye(4).unsqueeze(0).to(device)]
    #         intrinsics = torch.eye(4).unsqueeze(0).to(device)
    #         ref_img = [tgt_img]
    #         tgt_ga_depth = torch.zeros_like(tgt_sparse_depth).to(device)
    #         tgt_interp = torch.zeros_like(tgt_sparse_depth).to(device)

    #         init_depth_inv, _ = model.forward(tgt_img, ref_img,tgt_ga_depth, 
    #             None, tgt_interp, tgt_sparse_depth, 
    #             None, tgt_pose, 
    #             ref_pose, intrinsics)
            
    #         plt.imshow( utils.inv2depth(init_depth_inv[-1][0][0].detach().cpu()).numpy() )
    #         plt.show()
            
            # input_sparse_depth_valid = (input_sparse_depth > 0.2) * (input_sparse_depth < 5.0)
            # input_sparse_depth_inv = utils.depth2inv(input_sparse_depth)
            # ga_depth_inv, init_scales_inv = get_ga_scale_using_init_depth(tgt_depth_pred_rel[0].detach().cpu().numpy(), 
            #                                                           init_depth_inv[-1][0][0].detach().cpu().numpy(), 
            #                                                         input_sparse_depth_inv[0].detach().cpu().numpy(), 
            #                                                         input_sparse_depth_valid[0].detach().cpu().numpy(),
            #                                                         0.2, 5.0)   
    dataset = torch.utils.data.DataLoader(SML_consistent_resize(data_root=data_dir, mode='val', sequence_length=1))
    for batch_data in dataset:

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

        tgt_img, tgt_gt_depth_inv, tgt_ga_depth, tgt_interp, tgt_sparse_depth, ref_img, \
        ref_ga_depth, ref_interp, _, _, tgt_pose, ref_pose, intrinsics, \
            tgt_pose_per,ref_pose_per, tgt_depth_pred_rel, tgt_path = batch_data
        
        folder_path = os.path.dirname(os.path.dirname(tgt_path[0])) 
        filename = os.path.basename(tgt_path[0])
        filename_no_ext = os.path.splitext(filename)[0]

        ga_depth_dir = os.path.join(folder_path, "ga_depth_inv")
        interp_scale_dir = os.path.join(folder_path, "interp_scale_init_depth")
        os.makedirs(ga_depth_dir, exist_ok=True)
        os.makedirs(interp_scale_dir, exist_ok=True)
        ref_img = [tgt_img]; ref_ga_depth = [tgt_ga_depth]; ref_interp = [tgt_interp]; ref_pose = [tgt_pose]
        init_depth_inv, _ = model.forward(tgt_img, ref_img,tgt_ga_depth, 
                ref_ga_depth, tgt_interp, tgt_sparse_depth, 
                ref_interp, tgt_pose, 
                ref_pose, intrinsics)
        
        input_sparse_depth_valid = (tgt_sparse_depth > 0.2) * (tgt_sparse_depth < 5.0)
        tgt_sparse_depth_inv = utils.depth2inv(tgt_sparse_depth)
        ga_depth_inv, init_scales_inv = get_ga_scale_using_init_depth(tgt_depth_pred_rel[0].detach().cpu().numpy(), 
                                                                      init_depth_inv[-1][0][0].detach().cpu().numpy(), 
                                                                    tgt_sparse_depth_inv[0].detach().cpu().numpy(), 
                                                                    input_sparse_depth_valid[0].detach().cpu().numpy(),
                                                                    0.2, 5.0)

        init_depth = utils.inv2depth(init_depth_inv)

        #save the ga_depth and interp_scales
        ga_depth_save_path = os.path.join(ga_depth_dir, f"{filename_no_ext}.npy")
        interp_scale_save_path = os.path.join(interp_scale_dir, f"{filename_no_ext}.npy")
        save_depth_image_as_npy(ga_depth_inv, ga_depth_save_path)
        save_depth_image_as_npy(init_scales_inv, interp_scale_save_path)

        print(f"Saved ga_depth_inv to: {ga_depth_save_path}")
        print(f"Saved interp_scale_inv to: {interp_scale_save_path}")

        # plt.subplot(1, 4, 1)
        # plt.imshow(tgt_img[0].cpu().detach().numpy())
        # plt.subplot(1, 4, 2)
        # plt.imshow(init_depth[-1][0, 0].cpu().detach().numpy())
        # plt.subplot(1, 4, 3)
        # plt.imshow(init_scales_inv)
        # #plot histogram of scales
        # plt.subplot(1, 4, 4)
        # plt.hist(init_scales_inv.flatten(), bins=100)
        # plt.show()

def clean_dir(data_dir, ref_subfolder="depth_infer_dpt", subfolders_to_clean=["image", "sparse_depth", "absolute_pose", "ground_truth", "ov_pose", "interp_scale"]):
    #Cleans up the data_dir by removing files from subfolders that do not have a corresponding reference .npy file in ref_subfolder.
    import os
    folders = [f for f in os.listdir(data_dir) if not f.endswith('.txt')]
    for folder in folders:
        print("Folder: ", folder)
        ref_folder = os.path.join(data_dir, folder, ref_subfolder)
        valid_files = set()
        if not os.path.exists(ref_folder):
            print(f"Reference folder {ref_folder} does not exist, skipping.")
            continue
        for fname in os.listdir(ref_folder):
            if fname.endswith('.npy'):
                base = os.path.splitext(fname)[0]
                valid_files.add(base)
        print(f"Found {len(valid_files)} valid frames in {ref_subfolder}")
        
        for sub in subfolders_to_clean:
            sub_folder = os.path.join(data_dir, folder, sub)
            if not os.path.exists(sub_folder):
                print(f"Subfolder {sub_folder} does not exist, skipping.")
                continue

            for fname in os.listdir(sub_folder):
                base, ext = os.path.splitext(fname)
                # You may want to adjust extensions as needed (e.g., .png, .txt)
                if base not in valid_files:
                    fp = os.path.join(sub_folder, fname)
                    print(f"Removing: {fp}")
                    os.remove(fp)

    print("Cleanup complete.")

    
    
if __name__ == "__main__":
    data_dir = "/home/sai/Documents/data/tartan_ov/training" #"/media/saimouli/RPNG_FLASH_4/datasets/VOID_150/training"

    #save_priors(data_dir)
    #save_normals(data_dir)
    #clean_dir(data_dir)
    create_frame_index(data_dir)
    
    
    # import matplotlib.pyplot as plt
    # normals = np.load("/media/saimouli/Data6T/datasets/VOID_150_small/testing/copyroom4/dpt_normals/1552625608.9718.npy")
    # img = utils.read_image("/media/saimouli/Data6T/datasets/VOID_150_small/testing/copyroom4/image/1552625608.9718.png")
    # plt.imshow(img)
    # ax = plt.gca()
    # for y in range(0, normals.shape[0], 10):
    #     for x in range(0, normals.shape[1], 10):
    #         if np.isnan(normals[y, x]).any():
    #             continue
    #         ax.arrow(x, y, normals[y, x, 0] * 5, normals[y, x, 1] * 5,
    #                  head_width=2, head_length=2, fc='r', ec='r')
    # plt.show()

