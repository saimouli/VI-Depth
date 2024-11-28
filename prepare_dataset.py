import numpy as np
#import matplotlib.pyplot as plt

import pipeline
from utils_eval import compute_ls_solution
import modules.midas.utils as utils
from modules.interpolator import Interpolator2D
from tqdm import tqdm
from PIL import Image
import cv2

def load_sparse_depth(input_sparse_depth_fp):
    input_sparse_depth = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32) / 256.0
    input_sparse_depth[input_sparse_depth <= 0] = 0.0
    return input_sparse_depth

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

def save_depth_image_as_npy(depth_array, save_path):
    # Ensure depth_array is in 32-bit float format
    depth_array = depth_array.astype(np.float32)
    
    # Save the depth image as a .npy file
    np.save(save_path, depth_array)

def load_depth_image_from_npy(load_path):
    # Load the depth image from the .npy file
    depth_array = np.load(load_path)# + '.npy')
    
    return depth_array

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

        min_depth, max_depth = 0.1, 8.0
        min_pred, max_pred = 0.1, 8.0

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

            ga_depth_inv, interp_scale = get_ga_and_scale(depth_infer_inv, input_sparse_depth_inv, input_sparse_depth_valid, min_pred, max_pred )

            # plt.figure(1);plt.imshow(ga_depth_inv)
            # plt.figure(2);plt.imshow(interp_scale)
            # plt.show()

            ## save the images in the respective folders
            save_image_path = os.path.join(data_dir, folder, "depth_infer_dpt", images[i])
            save_image_path = save_image_path.replace('.png', '.npy')

            #save_depth_image_as_npy(depth_infer_inv, save_image_path)
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

            frame1 = cv2.imread(images_path[index[-1]])
            frame2 = cv2.imread(images_path[idx])

            pose1 = np.loadtxt(pose_path[index[-1]])
            pose2 = np.loadtxt(pose_path[idx])

            #if pose movement is > 0.1cm
            pose_diff = np.linalg.norm(pose1[:3, 3] - pose2[:3, 3])
            if pose_diff < 0.1:
                continue
            index.append(idx)
            frame_names.append(images[idx])

        print(len(images), len(frame_names))
        np.savetxt(os.path.join(data_dir, folder, "frame_index.txt"), index, fmt='%d', delimiter='\n')




if __name__ == "__main__":
    data_dir = "/media/saimouli/Data6T/datasets/VOID_150/training" #"/media/saimouli/RPNG_FLASH_4/datasets/VOID_150/training"
    save_priors(data_dir)

    #create_frame_index(data_dir)

