from model.mhybrid_consistent_net import midasConsNet
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
from typing import Any
import matplotlib.pyplot as plt
import torchvision
import cv2
import modules.midas.utils as utils
from metrics import rmse, mae, absrel, inv_rmse, inv_mae, inv_absrel
import metrics
from sklearn.decomposition import PCA
from utils.camera import Camera, se3_to_pose
from pytorch3d.transforms import se3_exp_map, se3_log_map
from utils.pose import Pose

class midasNetConsistentModule(pl.LightningModule):
    def __init__(self, lr: float = 0.1, wd: float = 0.1, min_pred: float = 0.1, 
                 max_pred: float = 8.0, min_depth: float = 0.2, 
                 max_depth: float = 5.0, nsamples: int = 150, img_h=480, 
                 img_w=640, sml_model_path: str = None, useConvGRU: bool = True, is_train: bool = True,
                 *args: Any, **kwargs: Any) -> None:
        super(midasNetConsistentModule, self).__init__(*args, **kwargs)
        self.model = midasConsNet(min_pred, max_pred, min_depth, max_depth, nsamples, 
                                  sml_model_path, is_train=is_train, log_fn=self.log, isConvGRU=useConvGRU)
        #print model params
        print("useConvGRU: ", useConvGRU)
        print("is_train: ", is_train)
        print("Model Parameters: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        self.lr = lr
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.wd = wd
        self.orig_h = img_h
        self.orig_w = img_w
        self.abs_loss = nn.L1Loss()
        self.useConvGRU = useConvGRU

    def on_fit_start(self):
        """Ensure that metric averaging uses the correct device after model initialization."""
        self.avg_error_w_int_depth = metrics.ErrorMetricsAverager_DDP(self.device)
        self.avg_error_w_pred = metrics.ErrorMetricsAverager_DDP(self.device)
        self.avg_error_pose_pred = metrics.ErrorMetricsAverager_DDP(self.device)

    def on_validation_start(self):
        self.avg_error_w_int_depth = metrics.ErrorMetricsAverager_DDP(self.device)
        self.avg_error_w_pred = metrics.ErrorMetricsAverager_DDP(self.device)
        self.avg_error_pose_pred = metrics.ErrorMetricsAverager_DDP(self.device)
        
    def load_from_pth(self, file_path):
        state_dict = torch.load(file_path, map_location=self.device)
        self.load_state_dict(state_dict)
    
    def forward(self, tgt_img, ref_img,tgt_ga_depth, 
                ref_ga_depth, tgt_interp, 
                ref_interp, tgt_pose, 
                ref_pose, intrinsics):
        refined_depth_inv, refined_ref_poses = self.model(tgt_img, ref_img,
                                                            tgt_ga_depth, ref_ga_depth, 
                                                            tgt_interp, ref_interp, 
                                                            tgt_pose, 
                                                            ref_pose, 
                                                            intrinsics)

        return refined_depth_inv,refined_ref_poses 
    
    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr'] 
        self.log("learning_rate", current_lr, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_epoch_end(self, outputs):
        # Get synchronized metrics
        ga_metrics = self.avg_error_w_int_depth.get_metrics()
        pred_metrics = self.avg_error_w_pred.get_metrics()
        pose_metrics = self.avg_error_pose_pred.get_pose_metrics()
        
        # Only log from main process
        if self.trainer.is_global_zero:
            table = (
                "Metric                | GA Depth (mm) | Predicted Depth (mm)\n"
                "----------------------|---------------|----------------------\n"
                f"RMSE                 | {ga_metrics['rmse']:.4f} | {pred_metrics['rmse']:.4f}\n"
                f"MAE                  | {ga_metrics['mae']:.4f} | {pred_metrics['mae']:.4f}\n"
                f"AbsRel               | {ga_metrics['absrel']:.4f} | {pred_metrics['absrel']:.4f}\n"
                f"Inv RMSE (1/km)      | {ga_metrics['inv_rmse']:.4f} | {pred_metrics['inv_rmse']:.4f}\n"
                f"Inv MAE (1/km)       | {ga_metrics['inv_mae']:.4f} | {pred_metrics['inv_mae']:.4f}\n"
                f"Inv AbsRel (1/km)    | {ga_metrics['inv_absrel']:.4f} | {pred_metrics['inv_absrel']:.4f}\n"  # Fixed newline
                "----------------------|---------------|----------------------\n"
                f"Trans RMSE (m)       | -             | {pose_metrics['trans_rmse']:.4f}\n"
                f"Rot RMSE (deg)       | -             | {pose_metrics['rot_rmse']:.4f}"
            )

            self.logger.experiment.add_text(
                "Validation Metrics", 
                table, 
                global_step=self.global_step
            )
            print(f"\nValidation Metrics (Total Samples: {ga_metrics['total_count']}):\n{table}")

        # Reset accumulators for next epoch
        self.avg_error_w_int_depth.reset()
        self.avg_error_w_pred.reset()
        self.avg_error_pose_pred.reset()
        
    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, stage="val")
        pred_depth = loss["pred_depth"]  # (B, H, W)
        gt_depth = loss["gt_depth"]      # (B, H, W)
        ga_depth = loss["ga_depth"]      # (B, H, W)
        ref_gt_poses = loss["ref_gt_poses"]
        ref_pred_poses = loss["ref_pred_poses"]
        
        batch_size = pred_depth.shape[0]
        max_depth, min_depth = self.max_depth, self.min_depth
        
        for i in range(batch_size):
            tgt_gt = gt_depth[i]
            tgt_ga = ga_depth[i]
            pred = pred_depth[i]
            
            valid_mask = (tgt_gt >= min_depth) & (tgt_gt <= max_depth)
            ga_metrics = metrics.ErrorMetrics_DDP()
            ga_metrics.compute(tgt_ga, tgt_gt, valid_mask)
            self.avg_error_w_int_depth.accumulate(ga_metrics)
            
            pred_metrics = metrics.ErrorMetrics_DDP()
            pred_metrics.compute(pred, tgt_gt, valid_mask)
            self.avg_error_w_pred.accumulate(pred_metrics)
            
            for ref_idx in range(len(ref_pred_poses)):
                pred_pose = ref_pred_poses[ref_idx][i]
                gt_pose = ref_gt_poses[ref_idx][i]
                
                pose_metrics = metrics.ErrorMetrics_DDP()
                pose_metrics.compute_pose(pred_pose, gt_pose)
                self.avg_error_pose_pred.accumulate_pose(pose_metrics)
                        
        return loss
        
    def compute_exp_weighted_l1loss(self, metric_depth_pred, tgt_gt_depth, gamma=0.85):
        total_loss = 0.0
        total_weight = 0.0
        num_levels = len(metric_depth_pred)
        
        valid_mask = (tgt_gt_depth > 0).float().detach()
        
        for i, pred_depth in enumerate(metric_depth_pred):
            weight = gamma ** (num_levels - i - 1)
            total_weight += weight
            
            l1_loss = torch.mean(torch.abs(tgt_gt_depth - pred_depth) * valid_mask)
            loss_i = l1_loss
            total_loss += weight * loss_i
        
        return total_loss / total_weight

    @torch.no_grad()
    def visualize_reproj_error(
        self,
        ref_img,          # Target image [B, H, W, 3] (numpy array, RGB, 0-255)
        proj_pred,        # Predicted reprojected points [B, H, W, 2]
        proj_gt,          # Ground truth reprojected points [B, H, W, 2]
        valid_mask,       # Valid mask [B, H, W]
        sample_points=100,  # Number of points to sample for visualization
        mode="train"
    ):
        """
        Visualize reprojection error by overlaying predicted and GT points on the target image.
        """
        # Convert tensors to numpy arrays
        if isinstance(ref_img, torch.Tensor):
            ref_img = ref_img.cpu().numpy()
        if isinstance(proj_pred, torch.Tensor):
            proj_pred = proj_pred.cpu().numpy()
        if isinstance(proj_gt, torch.Tensor):
            proj_gt = proj_gt.cpu().numpy()
        if isinstance(valid_mask, torch.Tensor):
            valid_mask = valid_mask.cpu().numpy()

        # Ensure images are in [0, 255] range
        if ref_img.max() <= 1.0:
            ref_img = (ref_img * 255).astype(np.uint8)

        # Determine number of batches to plot
        num_batches = min(3, ref_img.shape[0])
        
        combined_image = None
        # # Create subplot grid
        # fig, axes = plt.subplots(1, num_batches, figsize=(5*num_batches, 5))
        # if num_batches == 1:
        #     axes = [axes]  # Ensure axes is always iterable

        for batch_idx in range(num_batches):
            batch_img = ref_img[batch_idx].copy()  # Copy to avoid modifying original image
            batch_proj_pred = proj_pred[batch_idx]
            batch_proj_gt = proj_gt[batch_idx]
            batch_valid_mask = valid_mask[batch_idx]

            H, W = batch_img.shape[:2]
            y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            y_samples = y_coords.flatten()
            x_samples = x_coords.flatten()
            sample_indices = np.random.choice(
                H * W, 
                size=min(sample_points, H * W), 
                replace=False
            )
            
            valid_indices = np.where(batch_valid_mask.flatten() == 1)[0]
            valid_sample_indices = np.intersect1d(valid_indices, sample_indices)
            if len(valid_sample_indices) == 0:
                continue  # Skip if no valid points
            
            y_valid = y_samples[valid_sample_indices]
            x_valid = x_samples[valid_sample_indices]
            
            pred_px = ((batch_proj_pred[y_valid, x_valid] + 1) * np.array([W/2, H/2])).astype(int)
            gt_px = ((batch_proj_gt[y_valid, x_valid] + 1) * np.array([W/2, H/2])).astype(int)
            
            # Draw GT points (Green) and Predicted points (Red)
            for g, p in zip(gt_px, pred_px):
                cv2.circle(batch_img, (g[0], g[1]), 4, (0, 255, 0), -1)  # GT (Green)
                cv2.circle(batch_img, (p[0], p[1]), 4, (0, 0, 255), -1)  # Predicted (Red)
                cv2.line(batch_img, (g[0], g[1]), (p[0], p[1]), (255, 255, 0), 2)  # Cyan line

            # Add batch label at the top of the image
            cv2.putText(batch_img, f"Batch {batch_idx}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Stack images horizontally
            if combined_image is None:
                combined_image = batch_img
            else:
                combined_image = np.hstack((combined_image, batch_img))

            if combined_image is None:
                return 
            
            combined_image = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
            img_tensor = torch.from_numpy(combined_image).permute(2, 0, 1)
            
            self.logger.experiment.add_image(f"{mode}_Reprojection Error", 
                                            img_tensor, 
                                            global_step=self.global_step,
                                            dataformats="CHW")

        #axes[-1].legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        #plt.tight_layout()
        #plt.show()
            
    def compute_reproj_loss(self, depth_pred, depth_gt, tgt_pose_pred, tgt_pose_gt,
                            ref_pose_pred, ref_pose_gt, K,ref_img, log_tb=False, mode="train"):
        #Loss = ||π(T_pred * X_pred) - π(T_gt * X_gt)||
        #where X_pred = π^-1(x, D_pred), X_gt = π^-1(x, D_gt)
        device = depth_pred.device
        B, _, H, W = depth_pred.shape
        scale_factor = H / depth_pred.shape[2] 
        N_ref = len(ref_pose_pred)
            
        depth_pred = torch.clamp(depth_pred, min=0.2, max=5.0)
        depth_gt = torch.clamp(depth_gt, min=0.2, max=5.0)
        valid_depth_mask = ((depth_gt > 0.2) & (depth_gt <= 5.0)).bool().detach()
            
        # Reconstruct 3D points in world coordinates
        def reconstruct_points(pose, depth):
            cam = Camera(K=K, Twc=pose).scaled(scale_factor).to(device)
            return cam.reconstruct(depth, frame='w')  # [B, 3, H, W]

        points_pred = reconstruct_points(tgt_pose_pred, depth_pred)
        points_gt = reconstruct_points(tgt_pose_gt, depth_gt)

        # Projection function
        def project(pose, points):
            cam = Camera(K=K, Twc=pose).scaled(scale_factor).to(device)
            return cam.project(points, frame='w', normalize=True)  # [B, H, W, 2]

        total_loss = 0
        valid_count = 1e-6  # Avoid division by zero

        for ref_idx in range(N_ref):
            # Get reference poses for current view
            ref_pose_p = ref_pose_pred[ref_idx]  # [1, 4, 4]
            ref_pose_g = ref_pose_gt[ref_idx]    # [1, 4, 4]

            # Project points
            proj_pred = project(ref_pose_p, points_pred)  # [1, H, W, 2]
            proj_gt = project(ref_pose_g, points_gt)     # [1, H, W, 2]

            # Calculate valid projections
            valid_proj = proj_pred.abs().max(dim=-1)[0] <= 1.0
            valid_proj &= (proj_gt.abs().max(dim=-1)[0] <= 1.0)    
            valid = valid_depth_mask.squeeze(1) & valid_proj  # [1, H, W]

            # Calculate Huber loss
            error = F.huber_loss(proj_pred, proj_gt, reduction='none').mean(-1)
            total_loss += (error * valid).sum()
            valid_count += valid.sum()

            # Visualize first sample in batch and first reference view
            with torch.no_grad():
                if ref_idx == 0 and log_tb:
                    self.visualize_reproj_error(
                        ref_img[ref_idx],
                        proj_pred.detach().cpu().numpy(),
                        proj_gt.detach().cpu().numpy(),
                        valid.bool().cpu().numpy(),
                        mode=mode
                    )

        return total_loss / valid_count
        
    def compute_loss(self, pred_depth, gt_depth, log_variance=None, mask=None):
        """
        Computes the loss for depth prediction based on L1 depth loss and multiscale gradient matching.

        Args:
        - pred_depth (torch.Tensor): Predicted depth map (B, C, H, W)
        - gt_depth (torch.Tensor): Ground truth depth map (B, C, H, W)
        - log_variance (torch.Tensor, optional): Log variance (uncertainty) map (B, 1, H, W)
        
        Returns:
        - loss (torch.Tensor): Total loss (depth loss + 0.5 * gradient loss)
        - loss_info (dict): Detailed information about individual loss components
        """
        # Ensure valid mask for pixels with ground truth
        valid_mask = ((gt_depth > self.min_depth) & (gt_depth <= self.max_depth)).float().detach()
        M = valid_mask.sum()

        if log_variance is not None:
            depth_diff = (pred_depth * valid_mask) - (gt_depth * valid_mask)
            weighted_l1_loss = 0.5 * torch.exp(-log_variance) * (depth_diff ** 2) + 0.5 * log_variance
            l1_depth_loss = (weighted_l1_loss * valid_mask).sum() / M
        else:
            # L1 Depth loss
            l1_depth_loss = F.l1_loss(pred_depth * valid_mask, gt_depth * valid_mask, reduction='sum') / M

        if mask is not None:
            mask_entropy = F.binary_cross_entropy_with_logits(mask, valid_mask, reduction='mean')
            
        # Multiscale gradient matching loss
        def compute_gradient_loss(pred, gt):
            diff = gt - pred
            grad_x_pred = torch.abs(diff[:, :, :, :-1] - diff[:, :, :, 1:])
            grad_y_pred = torch.abs(diff[:, :, :-1, :] - diff[:, :, 1:, :])
            return (grad_x_pred.mean() + grad_y_pred.mean()) / M

        grad_loss = 0.0
        for scale in range(3):  # K = 3 levels
            scaled_pred = F.interpolate(pred_depth, scale_factor=1 / (2 ** scale), mode='bilinear', align_corners=False)
            scaled_gt = F.interpolate(gt_depth, scale_factor=1 / (2 ** scale), mode='bilinear', align_corners=False)
            grad_loss += compute_gradient_loss(scaled_pred, scaled_gt)
        
        grad_loss /= 3  # Average over K = 3 scales

        # Total loss
        total_loss = l1_depth_loss + 0.5 * grad_loss

        # Loss info dictionary for logging purposes
        loss_info = {
            'l1_depth_loss': l1_depth_loss.item(),
            'gradient_loss': grad_loss.item(),
            'total_loss': total_loss.item()
        }

        if log_variance is not None:
            loss_info['uncertainty_loss'] = 0.5 * log_variance.mean().item()

        return total_loss, loss_info
    
    def calculate_grudepth_loss(self, inv_depths, gt_inv_depths):
        """
        Calculate the supervised loss.

        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            List of predicted inverse depth maps
        gt_inv_depths : list of torch.Tensor [B,1,H,W]
            List of ground-truth inverse depth maps

        Returns
        -------
        loss : torch.Tensor [1]
            Average supervised loss for all scales
        """
        num_scales = len(inv_depths)
        total_loss = 0
        total_w = 0
        gamma = 0.85
        min_disp = self.min_depth
        max_disp = self.max_depth
        for i in range(num_scales):
            w = gamma**(num_scales - i - 1)
            total_w += w
            
            valid = ((gt_inv_depths > min_disp) & (gt_inv_depths < max_disp)).detach()
            valid = valid.squeeze(1)

            loss_depth = torch.mean(valid * torch.abs(gt_inv_depths - inv_depths[i]).squeeze(1))
            loss_i = loss_depth
            total_loss += w * loss_i
        
        return total_loss / total_w
    
    
    def get_ref_coords(self, pose, K, depth, scale_factor, device):
        if not isinstance(pose, Pose):
            pose = Pose(pose)

        cam = Camera(K=K.float()).scaled(scale_factor).to(device) # tcw = Identity
        ref_cam = Camera(K=K.float(), Twc=pose).scaled(scale_factor).to(device)
    
        # Reconstruct world points from target_camera
        world_points = cam.reconstruct(depth, frame='w')
        # Project world points onto reference camera
        ref_coords = ref_cam.project(world_points, frame='w', normalize=True) #(b, h, w,2)
        valid_mask = (ref_coords >= -1) & (ref_coords <= 1)
        return ref_coords, valid_mask
    
    def calc_pose_loss(self, pred_poses, gt_pose_context, gt_depth, K):
        device = gt_pose_context[0].device
        scale_factor = 1
        MAX_ERROR = 1
        total_loss = 0
        w_totoal = 0
        gamma = 0.85
        iter = len(pred_poses[0])
        
        min_depth = self.min_depth
        max_depth = self.max_depth / 4.0
        
        gt_depth_mask = (gt_depth > min_depth) & (gt_depth < max_depth)
        gt_depth_mask = gt_depth_mask.permute(0, 2, 3 ,1)
        for i in range(iter):
            loss_it = 0
            w = gamma**(iter - i - 1)
            w_totoal += w
                
            for view_i, gt_pose_view in enumerate(gt_pose_context):
                pred_pose_view = pred_poses[view_i][i]
                
                coords_gt, mask_gt = self.get_ref_coords(gt_pose_view, K, gt_depth, scale_factor, device)
                coords_pred, mask_pred = self.get_ref_coords(pred_pose_view, K, gt_depth, scale_factor, device)
                valid_mask = mask_gt * mask_pred * gt_depth_mask
                reproj_diff = valid_mask * torch.abs(coords_pred - coords_gt).clamp(-MAX_ERROR, MAX_ERROR)
                reporj_loss = torch.mean(reproj_diff)
                loss_it += reporj_loss
                
            loss_it = loss_it / len(gt_pose_context) 
            total_loss += loss_it * w
        
        return total_loss / w_totoal
    
    def _common_step(self, batch, batch_idx, stage="train"):
        #input_sparse_depth, input_image, rel_depth_pred, depth_gt, validity_map = batch
        tgt_img, tgt_gt_depth_inv, tgt_ga_depth, tgt_interp,_, ref_imgs, \
        ref_ga_depth, ref_interp, _, _, tgt_pose, ref_gt_pose, intrinsics, \
            tgt_pose_perturbed, ref_pose_perturbed,_ = batch
        
        gt_depth = utils.inv2depth(tgt_gt_depth_inv)
        
        # if self.useConvGRU:
        #     metric_depth_inv_pred = self.model(tgt_img, ref_imgs, tgt_ga_depth, ref_ga_depth, 
        #                                     tgt_interp, ref_interp,
        #                                     tgt_pose, ref_pose, intrinsics)
            
        #     # Compute depth loss with depth uncertainty
        #     # loss, loss_info = self.compute_loss(utils.inv2depth(metric_depth_inv_pred), 
        #     #                                    utils.inv2depth(tgt_gt_depth_inv),
        #     #                                    log_variance=None)
            
        #     metric_depth_pred = utils.inv2depth(metric_depth_inv_pred)
        #     total_loss = self.compute_exp_weighted_l1loss(metric_depth_pred, 
        #                                             gt_depth)
        # else:
        
        #convert to relative poses making the target pose identity
        with torch.no_grad():
            ref_rel_poses = [tgt_pose.inverse() @ ref_p for ref_p in ref_pose_perturbed]
            ref_rel_gtposes = [tgt_pose.inverse() @ ref_p for ref_p in ref_gt_pose]
            
        # refined_depth_inv, refined_target_pose, refined_ref_poses, warping_vis = self.model(tgt_img, ref_imgs,
        #                                                                                     tgt_ga_depth, ref_ga_depth, 
        #                                                                                     tgt_interp, ref_interp, 
        #                                                                                     tgt_pose, 
        #                                                                                     ref_rel_poses, 
        #                                                                                     intrinsics)
        
        #(b, n, iters, 6) 
        if self.current_epoch < 10:
            self.model.iter_steps=0   
            refined_depth_inv, refined_ref_poses = self.model(tgt_img, ref_imgs,
                                                            tgt_ga_depth, ref_ga_depth, 
                                                            tgt_interp, ref_interp, 
                                                            tgt_pose, 
                                                            ref_rel_gtposes, 
                                                            intrinsics)
            
        elif self.current_epoch < 30:
            self.model.iter_steps=3
            refined_depth_inv, refined_ref_poses = self.model(tgt_img, ref_imgs,
                                                            tgt_ga_depth, ref_ga_depth, 
                                                            tgt_interp, ref_interp, 
                                                            tgt_pose, 
                                                            ref_rel_gtposes, 
                                                            intrinsics)
        
        # else:
        #     self.model.iter_steps=3
        #     refined_depth_inv, refined_ref_poses = self.model(tgt_img, ref_imgs,
        #                                                     tgt_ga_depth, ref_ga_depth, 
        #                                                     tgt_interp, ref_interp, 
        #                                                     tgt_pose, 
        #                                                     ref_rel_poses, 
        #                                                     intrinsics)
        # 1. Depth L1 Loss
        # depth_loss,_ = self.compute_loss(utils.inv2depth(refined_depth_inv),
        #                         gt_depth,
        #                         log_variance=None,
        #                         mask=None)
        
        depth_loss = self.calculate_grudepth_loss(utils.inv2depth(refined_depth_inv), 
                                            utils.inv2depth(tgt_gt_depth_inv))
        
        #visualize flag
        if batch_idx %10 == 0:
            log_tb = True
        else:
            log_tb = False
        
        # 2. Reprojection loss
        # reproj_loss = self.compute_reproj_loss(
        #     gt_depth, gt_depth,
        #     tgt_pose,             # Predicted target pose
        #     tgt_pose,             # GT target pose
        #     refined_ref_poses,    # Predicted reference poses
        #     ref_gt_pose,             # GT reference poses
        #     intrinsics,
        #     ref_imgs,
        #     log_tb=log_tb,
        #     mode=stage
        # )
        
        # refined_ref_poses shape: (b, num_ref_views, num_iters, 6)
        pred_poses = []
        for ref_idx in range(refined_ref_poses.shape[1]):  # Iterate through reference views
            view_poses = []
            for iter_idx in range(refined_ref_poses.shape[2]):  # Iterate through refinement steps
                # Extract pose vector [b, 6]
                pose_vec = refined_ref_poses[:, ref_idx, iter_idx, :]
                # Convert to Pose object
                view_poses.append(se3_to_pose(pose_vec))
            pred_poses.append(view_poses)  # shape: (num_ref_views, num_iters)

        # Format ground truth poses ---------------------------------------------------
        # Assuming ref_gt_pose is list of absolute poses: [pose_ref1, pose_ref2,...]
        # Convert to relative poses if needed (based on your pose parametrization)
        #refined_rel_poses = [pose_to_se3(fixed_tgt_pose.inverse() @ ref_p) for ref_p in ref_pose]
        # gt_rel_poses = []
        # tgt_pose_inv = tgt_pose.inverse()  # Assuming tgt_pose is fixed
        # for ref_pose in ref_gt_pose:
        #     # Convert absolute pose to relative pose
        #     rel_pose = tgt_pose_inv * ref_pose
        #     gt_rel_poses.append(rel_pose)
            
        reproj_loss = self.calc_pose_loss(pred_poses, ref_rel_gtposes, 
                                          gt_depth, intrinsics)
        
        #pose regularization
        # pose_reg = torch.norm(se3_log_map(refined_target_pose @ tgt_pose.inverse()))
        # for refined_pose, gt_pose in zip(refined_ref_poses, ref_pose):
        #     pose_reg += torch.norm(se3_log_map(refined_pose @ gt_pose.inverse()))
    
        #total_loss = loss
        #if self.current_epoch < 0:
        #   total_loss = depth_loss
        #else:
        total_loss = depth_loss + 1.0 * reproj_loss #+ 0.01 * pose_reg
        
        #self.logger.experiment.add_scalar(f"{stage}_loss", loss, self.global_step)
        #self.log(f"{stage}/pose_reg_loss", pose_reg, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/reproj_loss", reproj_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/l1_depth_loss", depth_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/total_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        
        with torch.no_grad():
            self.log_refinement_progress(tgt_img, ref_imgs, gt_depth,
                                        utils.inv2depth(refined_depth_inv), pred_poses,
                                        ref_rel_gtposes, intrinsics, mode=stage)
            
            #if len(warping_vis) > 0:
            #    self.log_warping(warping_vis, mode=stage)
            # self.log_img_tensorboard(tgt_img, gt_depth, utils.inv2depth(tgt_ga_depth).unsqueeze(0).permute(1,0,2,3), 
            #                          utils.inv2depth(refined_depth_inv), batch_idx, self.current_epoch, mode=stage)
        return {
            "loss": total_loss,
            "mode": stage,
            "pred_depth": refined_depth_inv[-1].detach(),
            "gt_depth": tgt_gt_depth_inv.detach(),
            "ga_depth": tgt_ga_depth.unsqueeze(0).permute(1,0,2,3).detach(),
            "ref_gt_poses": ref_rel_gtposes,
            "ref_pred_poses": [view_poses[-1] for view_poses in pred_poses],
        }
        #TODO: compute loss for the pose as well
        

    @torch.no_grad()
    def log_refinement_progress(self, tgt_img, ref_imgs, gt_depth, 
                                inv_depth_predictions, refined_ref_poses,
                                ref_gt_pose, intrinsics,
                                mode="train"):
        """
        Visualize refinement progress with:
        - Depth predictions
        - Depth errors
        - Reprojection alignment using current poses vs GT poses
        """
        device = tgt_img.device
        num_iters = len(inv_depth_predictions)
        plot_iters = [0, num_iters//2, -1]  # First, middle, final
        B, C, H, W = tgt_img.shape
        
        # Select first sample in batch
        idx = 0
        tgt_img = tgt_img[idx].unsqueeze(0).detach().cpu()
        gt_depth = gt_depth[idx].detach().cpu()#.unsqueeze(0)
        ref_img = ref_imgs[0][idx].unsqueeze(0).detach().cpu()  # First reference image
        
        # 1. Precompute GT projections --------------------------------------------
        # Using GT depth and GT poses
        gt_depth_metric = utils.inv2depth(gt_depth)
        gt_proj, gt_valid = self.compute_reprojection(
            depth=gt_depth_metric,
            ref_pose=ref_gt_pose[0][0].unsqueeze(0),  # First reference view
            K=intrinsics[0].unsqueeze(0)
        )
        
        def apply_colormap(tensor, colormap="viridis", vmin=None, vmax=None):

            tensor_np = tensor.squeeze(0).cpu().numpy()
            # Use percentile-based normalization for better visualization
            if vmin is None:
                vmin = np.percentile(tensor_np, 2)  # Avoid extreme minimums
            if vmax is None:
                vmax = np.percentile(tensor_np, 98)  # Avoid extreme maximums
            norm = (tensor_np - vmin) / (vmax - vmin + 1e-6)
            norm = np.clip(norm, 0, 1)
            cmap = plt.get_cmap(colormap)
            colored = cmap(norm)[:, :, :3]  # Drop alpha channel

            return torch.from_numpy(colored).permute(2, 0, 1).float()
        
        # 2. Depth Visualization --------------------------------------------------
        def depth_to_rgb(depth, cmap='viridis'):
            depth = depth.float()
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
            return apply_colormap(depth_norm, cmap)

        def overlay_text_on_image(image, text, position=(10, 30), font_scale=0.5, color=(255, 0, 0)):
            image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            cv2.putText(image_np, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness=1, lineType=cv2.LINE_AA)
            return torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0 
    
        depth_grid = []
        gt_depth_grid = []
        error_grid = []
        reproj_grid = []

        for iter_idx in plot_iters:
            # Get current refinement results
            pred_depth = utils.inv2depth(inv_depth_predictions[iter_idx][idx])
            pred_pose = refined_ref_poses[0][iter_idx][0].unsqueeze(0)  # First ref view
            
            # Depth error
            error = torch.abs(pred_depth.cpu() - gt_depth.squeeze())
            depth_error_rmse = torch.sqrt((error ** 2).mean()).item()

            # 
            gt_pose = ref_gt_pose[0][0].unsqueeze(0)
            R_pred, T_pred = pred_pose[:, :3, :3], pred_pose[:, :3, 3]
            R_gt, T_gt = gt_pose[:, :3, :3], gt_pose[:, :3, 3]
            
            R_diff = R_gt.transpose(1, 2) @ R_pred 
            trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(-1)
            trace_clamped = torch.clamp((trace - 1) / 2, -1, 1)
            
            R_error = torch.acos(trace_clamped) * (180 / np.pi)
            T_error = torch.norm(T_gt - T_pred, dim=-1).mean().item()
            
            # 3. Compute current predictions --------------------------------------
            pred_proj, pred_valid = self.compute_reprojection(
                depth=gt_depth_metric,
                ref_pose=pred_pose,
                K=intrinsics[0].unsqueeze(0)
            )
            
            # 4. Create reprojection visualization --------------------------------
            reproj_img = self.visualize_reproj_pair(
                ref_img=ref_img,
                pred_proj=pred_proj,
                gt_proj=gt_proj,
                valid_mask=pred_valid.cpu() & gt_valid if 'gt_valid' in locals() else pred_valid.cpu(),
                iter_idx=iter_idx
            )
            
            #convert predicted depth to rgb
            pred_depth_rgb = depth_to_rgb(pred_depth).detach().cpu()
            gt_depth_rgb = depth_to_rgb(gt_depth.squeeze()).detach().cpu()
            
            # Overlay error text
            pred_depth_rgb = overlay_text_on_image(pred_depth_rgb, f"Depth RMSE: {depth_error_rmse:.3f}", position=(10, 20))
            reproj_img = overlay_text_on_image(reproj_img/255.0, f"R: {R_error.item():.3f}°, T: {T_error:.3f}", position=(10, 20))
            
            depth_grid.append(pred_depth_rgb.unsqueeze(0))  # (1, 3, H, W)
            gt_depth_grid.append(gt_depth_rgb.unsqueeze(0))
            
            
            error = torch.abs(pred_depth.cpu() - gt_depth.squeeze())
            error_rgb = apply_colormap(error.squeeze(0), 'Reds').detach().cpu()
            error_grid.append(error_rgb.unsqueeze(0))  # (1, 3, H, W)
            
            reproj_grid.append(reproj_img.unsqueeze(0).detach().cpu()) 
            
            

        # depth_grid.append(depth_to_rgb(pred_depth).unsqueeze(0).permute(0, 3, 1, 2))
        # error = torch.abs(pred_depth.cpu() - gt_depth.squeeze())
        # error_grid.append(apply_colormap(error.squeeze(0), 'Reds').unsqueeze(0).permute(0, 3, 1, 2))
        # reproj_grid.append(reproj_img.unsqueeze(0).permute(0, 3, 1, 2))
    
        # 5. Combine all components -----------------------------------------------
        depth_row = torch.cat(depth_grid, dim=-1) if depth_grid else None
        gt_depth_row = torch.cat(gt_depth_grid, dim=-1) if gt_depth_grid else None
        error_row = torch.cat(error_grid, dim=-1) if error_grid else None
        reproj_row = torch.cat(reproj_grid, dim=-1) if reproj_grid else None
        
        final_grid = torch.cat(
            [row for row in [depth_row, gt_depth_row, error_row, reproj_row] if row is not None], 
            dim=2
        ).squeeze(0)
        final_grid = final_grid.to(dtype=torch.float32)
        
        # Log to TensorBoard
        self.logger.experiment.add_image(
            f"{mode}_refinement",
            final_grid,
            global_step=self.global_step,
            dataformats="CHW"
        )

    def compute_reprojection(self, depth, ref_pose, K):
        """Compute projected coordinates in reference view"""
        # Reconstruct 3D points in world coordinates
        tgt_cam = Camera(K=K.float()).scaled(1.0).to(depth.device)
        ref_cam = Camera(K=K.float(), Twc=ref_pose).scaled(1.0).to(depth.device)
        
        wld_points = tgt_cam.reconstruct(depth, frame='w')
        # Project to reference view
        proj = ref_cam.project(wld_points, frame='w', normalize=True)
        
        # Validate projections
        valid = (proj.abs() <= 1.0).all(dim=-1, keepdim=True)
        return proj, valid

    def visualize_reproj_pair(self, ref_img, pred_proj, gt_proj, 
                              valid_mask, iter_idx, num_points=200):
        """Visualize predicted vs GT reprojections on reference image"""
        if isinstance(ref_img, torch.Tensor):
            ref_img = ref_img.squeeze(0).cpu().numpy()
        if isinstance(pred_proj, torch.Tensor):
            pred_proj = pred_proj.cpu().numpy()
        if isinstance(gt_proj, torch.Tensor):
            gt_proj = gt_proj.cpu().numpy()
        if isinstance(valid_mask, torch.Tensor):
            valid_mask = valid_mask.squeeze(3).cpu().numpy()
        
        # print("tgt_img", ref_img.shape)
        # print("pred_proj", pred_proj.shape)
        # print("gt_proj", gt_proj.shape)
        # print("valid_mask", valid_mask.shape)
        
        # Ensure target image is in [0, 255] range
        if ref_img.max() <= 1.0:
            ref_img = (ref_img * 255).astype(np.uint8)
        vis_img = ref_img.copy()
        #cv2.imshow("ref_img", ref_img)
        # Sample random points for visualization
        B, H, W = valid_mask.shape
        y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        y_coords = y_coords.flatten()
        x_coords = x_coords.flatten()

        # Randomly sample points
        sample_indices = np.random.choice(H * W, size=num_points, replace=False)
        y_samples = y_coords[sample_indices]
        x_samples = x_coords[sample_indices]

        valid_samples = valid_mask[0, y_samples, x_samples]
        pred_samples = pred_proj[0, y_samples, x_samples][valid_samples == 1]
        gt_samples = gt_proj[0, y_samples, x_samples][valid_samples == 1]

        # Convert normalized coordinates to pixel coordinates
        pred_px = (pred_samples + 1) * np.array([W, H]) / 2  # [-1, 1] → [0, W] and [0, H]
        gt_px = (gt_samples + 1) * np.array([W, H]) / 2
        
        # Draw correspondences
        for g, p in zip(gt_px, pred_px):
            g_coord = (int(round(g[0])), int(round(g[1])))
            p_coord = (int(round(p[0])), int(round(p[1])))
            cv2.circle(vis_img, g_coord, 4, (0, 255, 0), -1)
            cv2.circle(vis_img, p_coord, 4, (0, 0, 255), -1)
            cv2.line(vis_img, g_coord, p_coord, (255, 255, 0), 2)

        
        cv2.putText(vis_img, f"itr {iter_idx}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # cv2.imshow("reproj_img", vis_img)
        # cv2.waitKey(0)
        # Convert back to tensor
        vis_img = torch.from_numpy(vis_img).permute(2,0,1).float() #/ 255.0
        return vis_img

    @torch.no_grad()
    def log_img_tensorboard(self, images, gt_depth, ga_depth, pred_depth, batch_idx, epoch, mode="train"):
        """
        Top row: RGB | Predicted Depth | GA Depth
        Bottom row: GT Depth | |Pred - GT| (error) | |GA - GT| (error, red indicates high error)
        _________________________________________________________________________________________
        images (tensor): [B,3,H,W] RGB images (normalized to [0,1]).
        gt_depth (tensor): [B,1,H,W] ground-truth depth.
        ga_depth (tensor): [B,1,H,W] globally aligned depth.
        pred_depth (tensor): [B,1,H,W] predicted depth.
        batch_idx (int): Current batch index (for logging).
        epoch (int): Current epoch (for logging).
        mode (str): "train", "val", or "test" to label the log.
        """
        
        # Sample the first 4 images and depth maps from the batch
        num_samples = min(2, images.shape[0])
        images = images[:num_samples]
        gt_depth = gt_depth[:num_samples]
        ga_depth = ga_depth[:num_samples]
        pred_depth = pred_depth[:num_samples]
        
        valid_mask = (gt_depth > 0).float()
        error_pred = torch.abs(pred_depth - gt_depth)* valid_mask
        error_ga = torch.abs(ga_depth - gt_depth)* valid_mask
        
        error_max = torch.quantile(error_pred, 0.98).item()
        error_max_ga = torch.quantile(error_ga, 0.98).item()
        
        def apply_colormap(tensor, colormap="viridis", vmin=None, vmax=None):

            tensor_np = tensor.cpu().numpy()
            # Use percentile-based normalization for better visualization
            if vmin is None:
                vmin = np.percentile(tensor_np, 2)  # Avoid extreme minimums
            if vmax is None:
                vmax = np.percentile(tensor_np, 98)  # Avoid extreme maximums
            norm = (tensor_np - vmin) / (vmax - vmin + 1e-6)
            norm = np.clip(norm, 0, 1)
            cmap = plt.get_cmap(colormap)
            colored = cmap(norm)[:, :, :3]  # Drop alpha channel

            return torch.from_numpy(colored).permute(2, 0, 1).float()

        composite_images = []
        for i in range(num_samples):
            rgb = images[i].detach().cpu()
            gt_d = gt_depth[i].squeeze(0).detach().cpu()
            ga_d = ga_depth[i].squeeze(0).detach().cpu()
            pred_d = pred_depth[i].squeeze(0).detach().cpu()
            err_pred = error_pred[i].squeeze(0).detach().cpu()
            err_ga = error_ga[i].squeeze(0).detach().cpu()

            # Use the improved colormap function with dynamic vmin, vmax
            pred_d_color = apply_colormap(pred_d, colormap="viridis")
            ga_d_color = apply_colormap(ga_d, colormap="viridis")
            gt_d_color = apply_colormap(gt_d, colormap="viridis")
            
            err_pred_color = apply_colormap(err_pred, colormap="Reds_r", vmax=error_max)
            err_ga_color = apply_colormap(err_ga, colormap="Reds_r", vmax=error_max_ga)

            # Stack images: [RGB, Predicted Depth, GA Depth] on top of [GT Depth, Pred Error, GA Error]
            row1 = torch.cat([rgb.permute(2,0,1), pred_d_color, ga_d_color], dim=2)
            row2 = torch.cat([gt_d_color, err_pred_color, err_ga_color], dim=2)
            composite = torch.cat([row1, row2], dim=1)  # Stack rows vertically
            composite_images.append(composite)

        grid = torchvision.utils.make_grid(composite_images, nrow=1, padding=10)
    
        # colorbar = np.zeros((50, 256, 3), dtype=np.uint8)
        # for i in range(256):
        #     colorbar[:, i] = np.array(plt.get_cmap("viridis")(i / 255.0)[:3]) * 255
        # colorbar = torch.from_numpy(colorbar).permute(2, 0, 1).float() / 255.0
        # # Concatenate the colorbar to the right of the grid
        # desired_height = grid.shape[1]
        # colorbar_resized = F.interpolate(colorbar.unsqueeze(0), 
        #                                  size=(desired_height, colorbar.shape[2]), 
        #                                  mode='bilinear', align_corners=False).squeeze(0)
        # final_grid = torch.cat([grid, colorbar_resized], dim=2)
        
        # Log to TensorBoard
        self.logger.experiment.add_image(
            f"{mode}_depth_vis",
            grid,
            global_step=self.global_step,
            dataformats="CHW",
        )
    
    @torch.no_grad()
    def log_warping(self, warping_vis, mode="train"):
        """
        Logs warping and cost volume visualizations to TensorBoard.

        Args:
            warping_vis (list[dict]): A list of dictionaries (one per reference view) with keys:
                - 'src_feat': Source feature map, tensor of shape [B, C, H, W]
                - 'warped_feat': Warped reference feature map, tensor of shape [B, C, H, W]
                - 'valid_mask': Valid mask from warping, tensor of shape [B, H, W]
                - 'cost': Cost volume (error map), tensor of shape [B, 1, H, W]
            global_step (int): Current training step.
        """
        def feature_to_rgb(feature):
            pca = PCA(n_components=3)
            flat_feat = feature.permute(1,2,0).reshape(-1, feature.shape[0]).cpu().numpy()
            pca_feat = pca.fit_transform(flat_feat)
            pca_feat = (pca_feat - pca_feat.min()) / (pca_feat.max() - pca_feat.min())
            return torch.tensor(pca_feat.reshape(*feature.shape[1:], 3)).permute(2,0,1)
        
        def feature_to_rgb(feature):
            pca = PCA(n_components=3)
            flat_feat = feature.permute(1,2,0).reshape(-1, feature.shape[0]).cpu().numpy()
            pca_feat = pca.fit_transform(flat_feat)
            pca_feat = (pca_feat - pca_feat.min()) / (pca_feat.max() - pca_feat.min())
            return torch.tensor(pca_feat.reshape(*feature.shape[1:], 3)).permute(2,0,1)
        
        # Loop over each reference view that was warped
        for idx, vis in enumerate(warping_vis):
            # Select the first sample in the batch for logging.
            src_feat = vis['src_feat']         # [C, H, W]
            warped_feat = vis['warped_feat']     # [C, H, W]
            valid_mask = vis['valid_mask']       # [H, W]
            cost = vis['cost']                   # [1, H, W]

            # Visualize features by averaging over channels
            src_feat_vis = feature_to_rgb(src_feat) #src_feat.mean(dim=0, keepdim=True)       # [1, H, W]
            warped_feat_vis = feature_to_rgb(warped_feat) #warped_feat.mean(dim=0, keepdim=True) # [1, H, W]

            # Normalize cost volume to [0, 1] for visualization purposes
            cost_min = cost.min()
            cost_max = cost.max()
            cost_vis = (cost - cost_min) / (cost_max - cost_min + 1e-6)  # [1, H, W]
            cost_colored_np = plt.cm.jet(cost_vis[0].cpu().numpy())[...,:3]  # Convert to RGB
            cost_vis = torch.from_numpy(cost_colored_np).permute(2, 0, 1).float()
            
            # The valid mask is already a binary map; add a channel dimension for logging.
            valid_mask_vis = valid_mask.repeat(3,1,1).cpu() #valid_mask.unsqueeze(0)  # [3, H, W]
            overlay = 0.7*src_feat_vis + 0.3*valid_mask_vis

            # Log images using TensorBoard (dataformats: "CHW" means Channel, Height, Width)
            self.logger.experiment.add_image(
                f"Warping/View_{mode}/Overlay", 
                overlay, 
                self.global_step, 
                dataformats="CHW"
            )
            self.logger.experiment.add_image(
                f"Warping/View_{mode}/Src_Feature", 
                src_feat_vis, 
                self.global_step, 
                dataformats="CHW"
            )
            self.logger.experiment.add_image(
                f"Warping/View_{mode}/Warped_Feature", 
                warped_feat_vis, 
                self.global_step, 
                dataformats="CHW"
            )
            self.logger.experiment.add_image(
                f"Warping/View_{mode}/Cost", 
                cost_vis, 
                self.global_step, 
                dataformats="CHW"
            )
            self.logger.experiment.add_image(
                f"Warping/View_{mode}/Valid_Mask", 
                valid_mask_vis, 
                self.global_step, 
                dataformats="CHW"
            )
            
            # Optionally, log the average cost as a scalar so you can see if it decreases over time.
            avg_cost = cost_vis.mean().item()
            self.log(
                f"Warping/View_{mode}/Avg_Cost", 
                avg_cost, 
                prog_bar=True, 
                on_epoch=True, 
                on_step=True, 
                sync_dist=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.lr,
                                      weight_decay=self.wd,
                                      betas=(0.9, 0.999)
                                      )

        return optimizer