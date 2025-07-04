import sys
import numpy as np
import torch.utils.data
import os
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../modules"))
if module_path not in sys.path:
    sys.path.append(module_path)
import midas.utils as utils
from PIL import Image
#from path import Path
from pathlib import Path
import matplotlib.pyplot as plt
cam_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if cam_path not in sys.path:
    sys.path.append(cam_path)
from utils.camera import Camera
import torch.nn.functional as F
from torch.utils.data import Dataset
from pytorch3d.transforms import se3_exp_map, se3_log_map
import cv2
from utils.camera import Camera, pose_to_se3, se3_to_pose, se3_update

def load_input_image(input_image_fp):
    return utils.read_image(input_image_fp)

def load_sparse_depth(input_sparse_depth_fp, depth_scale):
    input_sparse_depth = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32) / depth_scale
    input_sparse_depth[input_sparse_depth <= 0] = 0.0
    return input_sparse_depth

def load_depth_image_from_npy(load_path):
    # Load the depth image from the .npy file
    depth_array = np.load(load_path)
    
    return depth_array

def generate_sample_index(num_frames, skip_frames, sequence_length):
    sample_index_list = []
    k = skip_frames
    demi_length = (sequence_length-1)//2
    shifts = list(range(-demi_length * k,
                        demi_length * k + 1, k))
    shifts.pop(demi_length)

    if num_frames > sequence_length:
        for i in range(demi_length * k, num_frames-demi_length * k):
            sample_index = {'tgt_idx': i, 'ref_idx': []}
            for j in shifts:
                sample_index['ref_idx'].append(i+j)
            sample_index_list.append(sample_index)

    return sample_index_list


class SML_tartan_consistent_dataset(Dataset):
    def __init__(self,
                 data_root,
                 mode="train",
                 sequence_length=3,
                 depth_scale=256.0,
                ):
        #self.root = Path(root)/'training'
        self.root = Path(data_root)
        print("Root: ", self.root)
        self.depth_scale = depth_scale
        if mode == "train":
            self.scenes = [subfolder for subfolder in (self.root / 'training').iterdir() if subfolder.is_dir()]
            # scene_list_path = self.root/'train_image.txt'
            # self.scenes = [self.root/folder[:-1]
            #             for folder in open(scene_list_path)]
        if mode == "val":
            self.scenes = [subfolder for subfolder in (self.root / 'testing').iterdir() if subfolder.is_dir()]
            # scene_list_path = self.root/'test_image.txt'
            # self.scenes = [self.root/folder[:-1]
            #             for folder in open(scene_list_path)]
        
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        # k skip frames
        sequence_set = []

        for scene in self.scenes:
            intrinsics = np.array([
                [320,   0, 320],
                [  0, 320, 240],
                [  0,   0,   1]
            ], dtype=np.float32)

            # Load image paths
            images_path = scene / 'image'
            imgs = sorted(images_path.glob('*.png'))

            # Load frame index
            frame_index = [int(index) for index in open(scene / 'frame_index.txt')]
            imgs = [imgs[d] for d in frame_index]

            #load tgt target depth
            # depth_pred_path = scene / 'depth_infer_dpt'
            # depth_pred = sorted(depth_pred_path.glob('*.npy'))
            # depth_pred = [depth_pred[d] for d in frame_index]
            
            # Load ga depth inverse
            ga_depth_inv_path = scene / 'ga_depth_inv'
            ga_depth_inv = sorted(ga_depth_inv_path.glob('*.npy'))
            ga_depth_inv = [ga_depth_inv[d] for d in frame_index]

            #Load sparse depth points 
            sparse_depth_path = scene / 'sparse_depth'
            sparse_depth = sorted(sparse_depth_path.glob('*.png'))
            sparse_depth = [sparse_depth[d] for d in frame_index]
            
            # Load interpolated scaffolding
            interp_depth_path = scene / 'interp_scale'
            interp_depth = sorted(interp_depth_path.glob('*.npy'))
            interp_depth = [interp_depth[d] for d in frame_index]

            # Load poses
            poses_path = scene / 'absolute_pose'
            poses = sorted(poses_path.glob('*.txt'))
            poses = [poses[d] for d in frame_index]
            
            #get gt depths
            gt_depth_path = scene / 'ground_truth'
            gt_depth = sorted(gt_depth_path.glob('*.png'))
            gt_depth = [gt_depth[d] for d in frame_index]

            if len(imgs) < sequence_length:
                continue

            sample_index_list = generate_sample_index(
                len(imgs), 1, sequence_length)
            
            for sample_index in sample_index_list:
                sample = {'intrinsics': intrinsics,
                          'tgt_img': imgs[sample_index['tgt_idx']]}
                #sample['tgt_depth_pred'] = depth_pred[sample_index['tgt_idx']]
                sample['tgt_ga_depth'] = ga_depth_inv[sample_index['tgt_idx']]
                sample['tgt_gt_depth'] = gt_depth[sample_index['tgt_idx']]
                sample['tgt_pose'] = poses[sample_index['tgt_idx']]
                sample['tgt_interp'] = interp_depth[sample_index['tgt_idx']]
                sample['tgt_pose'] = poses[sample_index['tgt_idx']]
                sample['tgt_sparse_depth'] = sparse_depth[sample_index['tgt_idx']]


                sample['ref_imgs'] = []; sample['ref_ga_depth'] = []
                sample['ref_gt_depth'] = []; sample['ref_pose'] = []
                sample['ref_interp'] = []; sample['ref_pose'] = []
                sample['ref_sparse_depth'] = []
                for j in sample_index['ref_idx']:
                    sample['ref_imgs'].append(imgs[j])
                    sample['ref_ga_depth'].append(ga_depth_inv[j])
                    sample['ref_interp'].append(interp_depth[j])
                    sample['ref_pose'].append(poses[j])
                    sample['ref_gt_depth'].append(gt_depth[j])
                    sample['ref_sparse_depth'].append(sparse_depth[j])
                sequence_set.append(sample)

        self.samples = sequence_set
    
    def convert_to_4x4(self, pose):
        if pose.shape == (3, 4):
            pose_4x4 = np.eye(4)
            pose_4x4[:3, :4] = pose
            return pose_4x4
        elif pose.shape == (4, 4):
            return pose
        else:
            raise ValueError("Pose should be of shape (3, 4) or (4, 4)")
        
    def data_augment(self, img):
        random_gamma = np.random.uniform(0.9, 1.1)
        random_brightness = np.random.uniform(0.8, 1.2)
        random_colors = np.random.uniform(0.8, 1.2, [3])

        img = 255.0 * ((img / 255.0) ** random_gamma)
        img *= random_brightness
        img *= np.reshape(random_colors, [1, 3, 1, 1])
        img = np.clip(img, 0.0, 255.0).astype(np.uint8)
        return img

    # Add SE(3) perturbations (mimic VIO drift)
    def add_perturbation(self, pose, max_trans=0.10, max_rot_deg=4.0):
        """Add random SE(3) perturbation to pose (3x4 numpy array)"""
        if isinstance(pose, np.ndarray):
            pose = torch.from_numpy(pose).float()

        # Convert to 4x4 matrix [R|t; 0|1]
        pose_mat = pose
        
        # Generate random SE(3) perturbation parameters
        # Translation: random direction with magnitude between 0-3cm
        trans_mag = torch.rand(1) * max_trans  # [0, 0.03]
        trans_dir = torch.randn(3)
        trans_dir /= torch.norm(trans_dir)  # random unit vector
        translation = trans_dir * trans_mag
        
        # Rotation: random axis with angle between 0-2 degrees
        # In reality for VIO only the yaw angle drifts significantly
        rot_deg = torch.rand(1) * max_rot_deg  # [0, 2]
        rot_rad = torch.deg2rad(rot_deg)
        #axis = torch.tensor([0.0, 0.0, 1.0]) #z-axis for yaw
        axis = torch.randn(3)
        axis /= torch.norm(axis)  # random unit axis
        rotation = axis * rot_rad
        
        # Combine into 6D se3 vector [translation, rotation]
        se3_params = torch.cat([translation, rotation])  # [6]
        
        # Convert to perturbation matrix
        perturb_mat = se3_exp_map(se3_params.unsqueeze(0))[0]  # [4,4]
        #convert perturb map to [R 0|T 1] to [R T; 0 1]
        perturb_mat_standard = perturb_mat.clone()
        perturb_mat_standard[:3, 3] = perturb_mat[3, :3] 
        perturb_mat_standard[3, :3] = 0 
        
        
        # Apply perturbation in LOCAL frame: T_new = T_original @ perturb_mat
        perturbed_mat = pose_mat @ perturb_mat_standard
        
        return perturbed_mat
    
    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_input_image(str(sample['tgt_img']))
        tgt_gt_depth = load_sparse_depth(str(sample['tgt_gt_depth']), depth_scale=self.depth_scale)
        tgt_sparse_depth = load_sparse_depth(str(sample['tgt_sparse_depth']), depth_scale=self.depth_scale)
        tgt_ga_depth = load_depth_image_from_npy(str(sample['tgt_ga_depth']))
        tgt_interp = load_depth_image_from_npy(str(sample['tgt_interp']))
        tgt_pose = self.convert_to_4x4(np.loadtxt(str(sample['tgt_pose'])))
        #tgt_depth_pred = load_depth_image_from_npy(str(sample['tgt_depth_pred']))

        ref_img = [load_input_image(str(ref_img)) for ref_img in sample['ref_imgs']]
        ref_ga_depth = [load_depth_image_from_npy(str(ref_ga_depth)) for ref_ga_depth in sample['ref_ga_depth']]
        ref_sparse_depth = [load_sparse_depth(str(ref_sparse_depth), depth_scale=self.depth_scale) for ref_sparse_depth in sample['ref_sparse_depth']]
        ref_interp = [load_depth_image_from_npy(str(ref_interp)) for ref_interp in sample['ref_interp']]
        ref_gt_depth = [load_sparse_depth(str(ref_gt_depth), depth_scale=self.depth_scale) for ref_gt_depth in sample['ref_gt_depth']]
        ref_pose = [self.convert_to_4x4(np.loadtxt(pose)) for pose in sample['ref_pose']]

        intrinsics = np.copy(sample['intrinsics'])
        
        # Get original image dimensions
        h, w = tgt_img.shape[:2]
        
        mask = (tgt_gt_depth < 5.0)
        mask *= (tgt_gt_depth > 0.2)
        tgt_gt_depth[~mask] = np.inf
        tgt_gt_depth_inv = 1.0 / tgt_gt_depth
        tgt_gt_depth_inv[tgt_gt_depth_inv == float("inf")] = 0
        tgt_gt_depth_inv = torch.from_numpy(tgt_gt_depth_inv).unsqueeze(0)
    
        
        tgt_img, tgt_gt_depth_inv, tgt_ga_depth, tgt_sparse_depth, tgt_interp, tgt_pose = [
            T.astype(np.float32) if isinstance(T, np.ndarray) else T for T in [
                tgt_img, tgt_gt_depth_inv, tgt_ga_depth, tgt_sparse_depth, tgt_interp, tgt_pose
            ]
        ]
        
        #Convert ref_img, ref_pose, ref_interp, intrinsics tensor to float32
        ref_img, ref_ga_depth, ref_interp, ref_gt_depth, ref_sparse_depth, ref_pose, intrinsics = [
            [torch.from_numpy(item).float() if isinstance(item, np.ndarray) else item for item in T]
            if isinstance(T, list) else torch.from_numpy(T).float() if isinstance(T, np.ndarray) else T
            for T in [ref_img, ref_ga_depth, ref_interp, ref_gt_depth, ref_sparse_depth, ref_pose, intrinsics]
        ]
        
        tgt_pose_perturbed = tgt_pose #self.add_perturbation(tgt_pose)
        ref_pose_perturbed = [self.add_perturbation(p) for p in ref_pose]
        
        #img, gt_depth, ga_depth, interp_scale
        return tgt_img, tgt_gt_depth_inv, tgt_ga_depth, tgt_interp, tgt_sparse_depth, ref_img, \
            ref_ga_depth, ref_interp, ref_gt_depth, ref_sparse_depth, tgt_pose, ref_pose, intrinsics, \
            tgt_pose_perturbed, ref_pose_perturbed
    
    def __len__(self):
        return len(self.samples)


