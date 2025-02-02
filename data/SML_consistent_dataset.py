import numpy as np
import torch.utils.data
import os
import sys
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if module_path not in sys.path:
    sys.path.append(module_path)
import modules.midas.utils as utils
from PIL import Image
#from path import Path
from pathlib import Path
import matplotlib.pyplot as plt
from utils.camera import Camera
import torch.nn.functional as F
from torch.utils.data import Dataset

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


class SML_consistent_dataset(Dataset):
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
            intrinsics = np.genfromtxt(
                scene/'K.txt').astype(np.float32).reshape((3, 3))

            # Load image paths
            images_path = scene / 'image'
            imgs = sorted(images_path.glob('*.png'))

            # Load frame index
            frame_index = [int(index) for index in open(scene / 'frame_index.txt')]
            imgs = [imgs[d] for d in frame_index]

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

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_input_image(str(sample['tgt_img']))
        tgt_gt_depth = load_sparse_depth(str(sample['tgt_gt_depth']), depth_scale=self.depth_scale)
        tgt_sparse_depth = load_sparse_depth(str(sample['tgt_sparse_depth']), depth_scale=self.depth_scale)
        tgt_ga_depth = load_depth_image_from_npy(str(sample['tgt_ga_depth']))
        tgt_interp = load_depth_image_from_npy(str(sample['tgt_interp']))
        tgt_pose = np.loadtxt(str(sample['tgt_pose']))

        ref_img = [load_input_image(str(ref_img)) for ref_img in sample['ref_imgs']]
        ref_ga_depth = [load_depth_image_from_npy(str(ref_ga_depth)) for ref_ga_depth in sample['ref_ga_depth']]
        ref_sparse_depth = [load_sparse_depth(str(ref_sparse_depth), depth_scale=self.depth_scale) for ref_sparse_depth in sample['ref_sparse_depth']]
        ref_interp = [load_depth_image_from_npy(str(ref_interp)) for ref_interp in sample['ref_interp']]
        ref_gt_depth = [load_sparse_depth(str(ref_gt_depth), depth_scale=self.depth_scale) for ref_gt_depth in sample['ref_gt_depth']]
        ref_pose = [np.loadtxt(pose) for pose in sample['ref_pose']]
        intrinsics = np.copy(sample['intrinsics'])

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
        
        #convert ref_img, ref_pose, ref_interp, intrinsics tensor to float 32
        ref_img, ref_ga_depth, ref_interp, ref_gt_depth, ref_sparse_depth, ref_pose, intrinsics = [
            [torch.from_numpy(item).float() if isinstance(item, np.ndarray) else item for item in T]
            if isinstance(T, list) else torch.from_numpy(T).float() if isinstance(T, np.ndarray) else T
            for T in [ref_img, ref_ga_depth, ref_interp, ref_gt_depth, ref_sparse_depth, ref_pose, intrinsics]
        ]
        
        #img, gt_depth, ga_depth, interp_scale
        return tgt_img, tgt_gt_depth_inv, tgt_ga_depth, tgt_interp, tgt_sparse_depth, ref_img, \
            ref_ga_depth, ref_interp, ref_gt_depth, ref_sparse_depth, tgt_pose, ref_pose, intrinsics
    
    def __len__(self):
        return len(self.samples)


# if __name__ == "__main__":
#     #dataset = SML_consistent_dataset(data_root='/media/saimouli/Data6T/datasets/VOID_150_test', mode='val', sequence_length=3)
#     dataset = torch.utils.data.DataLoader(SML_consistent_dataset(data_root='/media/saimouli/Data6T/datasets/VOID_150_test', mode='val'))
#     # rr.init("3d_points_visualization", spawn=True)

#     #for idx in range(len(dataset)):
#     for batch_data in dataset:
#         tgt_img, tgt_gt_depth_inv, tgt_ga_depth, tgt_interp, ref_imgs, \
#         ref_ga_depth, ref_interp, ref_gt_depth, tgt_pose, ref_pose, intrinsics = batch_data #dataset[idx] #Cam2Wld poses (R_ctoG, p_CinG)

#         B, H, W,_ = tgt_img.shape
#         _, _, DH, DW = tgt_gt_depth_inv.shape
#         scale_factor = DW / float(W)
        
#         perturbed_pose = ref_pose[0].clone().detach().requires_grad_(True)
#         optimizer = torch.optim.Adam([perturbed_pose], lr=1e-3)
#         cam = Camera(K=intrinsics.float(), Twc=tgt_pose).scaled(scale_factor)
#         ref_cam1 = Camera(K=intrinsics.float(), Twc=ref_pose[0]).scaled(scale_factor)
#         ref_cam2 = Camera(K=intrinsics.float(), Twc=ref_pose[1]).scaled(scale_factor)

#         gt_depth = utils.inv2depth(tgt_gt_depth_inv)
#         world_points = cam.reconstruct(gt_depth, frame='w')
        
#         # for i in range(100):
#         #     optimizer.zero_grad()

#         #     ref_cam1_perturbed = Camera(K=intrinsics.float(), Twc=perturbed_pose).scaled(scale_factor)
#         #     ref_coords1 = ref_cam1_perturbed.project(world_points, frame='w', normalize=True)  # (b, h, w, 2)
#         #     warped_ref1 = F.grid_sample(
#         #         ref_imgs[0].permute(0, 3, 1, 2), ref_coords1, mode='bilinear', padding_mode='zeros', align_corners=True
#         #     )

#         #     loss = F.l1_loss(warped_ref1, tgt_img.permute(0, 3, 1, 2))
#         #     loss.backward()
#         #     optimizer.step()

#         #     print(f"Iteration {i}/{10}, Loss: {loss.item()}")

#         #     with torch.no_grad():
#         #         ref_coords1_optimized = ref_cam1_perturbed.project(world_points, frame='w', normalize=True)
#         #         warped_ref1_optimized = F.grid_sample(
#         #             ref_imgs[0].permute(0, 3, 1, 2), ref_coords1_optimized, mode='bilinear', padding_mode='zeros', align_corners=True
#         #         )

#         #         # Convert tensors to numpy for visualization
#         #         tgt_img_np = tgt_img[0].cpu().numpy()  # (C, H, W) -> (H, W, C)
#         #         warped_ref1_np = warped_ref1_optimized.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)

#         #         tgt_img_cv2 = (tgt_img_np * 255).astype(np.uint8)  # Convert target image to uint8
#         #         warped_ref1_cv2 = (warped_ref1_np * 255).astype(np.uint8)  # Convert warped image to uint8

#         #         # Convert RGB to BGR for OpenCV
#         #         tgt_img_cv2 = cv2.cvtColor(tgt_img_cv2, cv2.COLOR_RGB2BGR)
#         #         warped_ref1_cv2 = cv2.cvtColor(warped_ref1_cv2, cv2.COLOR_RGB2BGR)

#         #         # Create a side-by-side visualization
#         #         combined = np.hstack((tgt_img_cv2, warped_ref1_cv2))

#         #         # Add text labels to each image
#         #         font = cv2.FONT_HERSHEY_SIMPLEX
#         #         cv2.putText(combined, "Target Image", (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
#         #         cv2.putText(combined, "Optimized Warped Ref Image", (tgt_img_cv2.shape[1] + 50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

#         #         # Show the images using OpenCV
#         #         cv2.imshow("Comparison", combined)
#         #         cv2.waitKey(10)  # Wait for a key press to close the window
#         #         #cv2.destroyAllWindows()
#         # #rr.log("3d_points", rr.Points3D(world_points[0].permute(1,2,0).view(-1,3).cpu().numpy()))

#         # Project world points into reference cameras
#         ref_coords1 = ref_cam1.project(world_points, frame='w', normalize=True)  # (b, h, w, 2)
#         ref_coords2 = ref_cam2.project(world_points, frame='w', normalize=True)  # (b, h, w, 2)

#         # Warp reference images into the target view
#         warped_ref1 = F.grid_sample(ref_imgs[0].permute(0, 3, 1, 2), ref_coords1, mode='bilinear', padding_mode='zeros', align_corners=True)
#         warped_ref2 = F.grid_sample(ref_imgs[1].permute(0, 3, 1, 2), ref_coords2, mode='bilinear', padding_mode='zeros', align_corners=True)

#         # Convert tensors to numpy for visualization
#         tgt_img_np = (tgt_img[0].cpu().numpy() * 255).astype(np.uint8)  # Target image
#         warped_ref1_np = (warped_ref1.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # Warped Ref 1
#         warped_ref2_np = (warped_ref2.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # Warped Ref 2

#         # Convert images to BGR for OpenCV visualization
#         tgt_img_bgr = cv2.cvtColor(tgt_img_np, cv2.COLOR_RGB2BGR)
#         warped_ref1_bgr = cv2.cvtColor(warped_ref1_np, cv2.COLOR_RGB2BGR)
#         warped_ref2_bgr = cv2.cvtColor(warped_ref2_np, cv2.COLOR_RGB2BGR)

#         # Create overlays of the target image and the warped reference images
#         overlay_ref1 = cv2.addWeighted(tgt_img_bgr, 0.5, warped_ref1_bgr, 0.5, 0)
#         overlay_ref2 = cv2.addWeighted(tgt_img_bgr, 0.5, warped_ref2_bgr, 0.5, 0)

#         # Plot the aligned images
#         fig, axes = plt.subplots(1, 3, figsize=(15, 5))

#         # Target image
#         axes[0].imshow(cv2.cvtColor(tgt_img_bgr, cv2.COLOR_BGR2RGB))
#         axes[0].set_title("Target Image")
#         axes[0].axis("off")

#         # Target + Warped Ref Image 1
#         axes[1].imshow(cv2.cvtColor(overlay_ref1, cv2.COLOR_BGR2RGB))
#         axes[1].set_title("Overlay: Target + Warped Ref Image 1")
#         axes[1].axis("off")

#         # Target + Warped Ref Image 2
#         axes[2].imshow(cv2.cvtColor(overlay_ref2, cv2.COLOR_BGR2RGB))
#         axes[2].set_title("Overlay: Target + Warped Ref Image 2")
#         axes[2].axis("off")

#         plt.tight_layout()
#         plt.show()
                


        # #tgt_img_np = tgt_img.transpose(1, 2, 0)
        # #tgt_img_np = tgt_img_np.astype(np.uint8) 

        # #ref_imgs_np = [ref_img.transpose(1, 2, 0).astype(np.uint8) for ref_img in ref_imgs]
        # fig, axes = plt.subplots(1, len(ref_imgs) + 1, figsize=(15, 5))
        # mid_idx = len(ref_imgs) // 2
        # for i, ref_img_np in enumerate(ref_imgs[:mid_idx]):
        #     axes[i].imshow(ref_img_np)
        #     axes[i].set_title(f"Ref Image {i + 1}")
        #     axes[i].axis("off")

        # # Plot the target image in the center
        # axes[mid_idx].imshow(tgt_img)
        # axes[mid_idx].set_title("Target Image")
        # axes[mid_idx].axis("off")

        # for i, ref_img_np in enumerate(ref_imgs[mid_idx:], start=mid_idx + 1):
        #     axes[i].imshow(ref_img_np)
        #     axes[i].set_title(f"Ref Image {i + 1}")
        #     axes[i].axis("off")
        
        # plt.tight_layout()
        # plt.show()

