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
from utils.camera import Camera
from pytorch3d.transforms import se3_exp_map, se3_log_map

class midasNetConsistentModule(pl.LightningModule):
    def __init__(self, lr: float = 0.1, wd: float = 0.1, min_pred: float = 0.1, 
                 max_pred: float = 8.0, min_depth: float = 0.2, 
                 max_depth: float = 5.0, nsamples: int = 150, img_h=480, 
                 img_w=640, sml_model_path: str = None, useConvGRU: bool = False,
                 *args: Any, **kwargs: Any) -> None:
        super(midasNetConsistentModule, self).__init__(*args, **kwargs)
        self.model = midasConsNet(min_pred, max_pred, min_depth, max_depth, nsamples, 
                                  sml_model_path, log_fn=self.log, isConvGRU=useConvGRU)
        #print model params
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

    def on_validation_start(self):
        self.avg_error_w_int_depth = metrics.ErrorMetricsAverager_DDP(self.device)
        self.avg_error_w_pred = metrics.ErrorMetricsAverager_DDP(self.device)
          
    def load_from_pth(self, file_path):
        state_dict = torch.load(file_path, map_location=self.device)
        self.load_state_dict(state_dict)
    
    def forward(self, tgt_img, tgt_gt_depth_inv, tgt_ga_depth, tgt_interp, ref_imgs,
                ref_ga_depth, ref_interp, ref_gt_depth, tgt_pose, ref_pose, intrinsics):
        _, pred_inv_depth,_ =  self.model(tgt_img, ref_imgs, tgt_ga_depth, ref_ga_depth, tgt_interp,
                          ref_interp, tgt_pose, ref_pose, intrinsics)

        return pred_inv_depth
    
    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr'] 
        self.log("learning_rate", current_lr, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_epoch_end(self, outputs):
        # Get synchronized metrics
        ga_metrics = self.avg_error_w_int_depth.get_metrics()
        pred_metrics = self.avg_error_w_pred.get_metrics()
        
        # Only log from main process
        if self.trainer.is_global_zero:
            table = (
                "Metric                | GA Depth (mm) | Predicted Depth (mm)\n"
                "----------------------|---------------|--------------------\n"
                f"RMSE                 | {ga_metrics['rmse']:.4f} | {pred_metrics['rmse']:.4f}\n"
                f"MAE                  | {ga_metrics['mae']:.4f} | {pred_metrics['mae']:.4f}\n"
                f"AbsRel               | {ga_metrics['absrel']:.4f} | {pred_metrics['absrel']:.4f}\n"
                f"Inv RMSE (1/km)      | {ga_metrics['inv_rmse']:.4f} | {pred_metrics['inv_rmse']:.4f}\n"
                f"Inv MAE (1/km)       | {ga_metrics['inv_mae']:.4f} | {pred_metrics['inv_mae']:.4f}\n"
                f"Inv AbsRel (1/km)    | {ga_metrics['inv_absrel']:.4f} | {pred_metrics['inv_absrel']:.4f}"
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
        
    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, stage="val")
        pred_depth = loss["pred_depth"]  # (B, H, W)
        gt_depth = loss["gt_depth"]      # (B, H, W)
        ga_depth = loss["ga_depth"]      # (B, H, W)
        
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
            
        return loss

    def validation_epoch_end(self, outputs):
        # Get synchronized metrics
        ga_metrics = self.avg_error_w_int_depth.get_metrics()
        pred_metrics = self.avg_error_w_pred.get_metrics()
        
        # Only log from main process
        if self.trainer.is_global_zero:
            table = (
                "Metric                | GA Depth (mm) | Predicted Depth (mm)\n"
                "----------------------|---------------|--------------------\n"
                f"RMSE                 | {ga_metrics['rmse']:.4f} | {pred_metrics['rmse']:.4f}\n"
                f"MAE                  | {ga_metrics['mae']:.4f} | {pred_metrics['mae']:.4f}\n"
                f"AbsRel               | {ga_metrics['absrel']:.4f} | {pred_metrics['absrel']:.4f}\n"
                f"Inv RMSE (1/km)      | {ga_metrics['inv_rmse']:.4f} | {pred_metrics['inv_rmse']:.4f}\n"
                f"Inv MAE (1/km)       | {ga_metrics['inv_mae']:.4f} | {pred_metrics['inv_mae']:.4f}\n"
                f"Inv AbsRel (1/km)    | {ga_metrics['inv_absrel']:.4f} | {pred_metrics['inv_absrel']:.4f}"
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

    def compute_reproj_loss(self, depth_pred, depth_gt, tgt_pose_pred, tgt_pose_gt,
                            ref_pose_pred, ref_pose_gt, K):
        #Loss = ||π(T_pred * X_pred) - π(T_gt * X_gt)||
        #where X_pred = π^-1(x, D_pred), X_gt = π^-1(x, D_gt)
        valid_depth_mask = ((depth_gt > self.min_depth) & (depth_gt <= self.max_depth)).float().detach()
        device = depth_pred.device
        B, _, H, W = depth_pred.shape
        scale_factor = 1.0
        
        # Reconstruct 3D points using PREDICTED target pose and depth(global frame)
        tgt_cam_pred = Camera(K=K, Twc=tgt_pose_pred).scaled(scale_factor).to(device)
        points_pred_world = tgt_cam_pred.reconstruct(depth_pred*valid_depth_mask, frame='w')
        
        # Reconstruct GT points using GT target pose and depth (global frame)
        tgt_cam_gt = Camera(K=K, Twc=tgt_pose_gt).scaled(scale_factor).to(device)
        points_gt_world = tgt_cam_gt.reconstruct(depth_gt*valid_depth_mask, frame='w')
        
        # Project using PREDICTED reference poses (global frame)
        proj_pred = []
        for ref_pose in ref_pose_pred:
            ref_cam_pred = Camera(K=K, Twc=ref_pose)
            proj = ref_cam_pred.project(points_pred_world, frame='w', normalize=True)
            proj_pred.append(proj)
        
        # Project using GT reference poses and depth (global frame)
        proj_gt = []
        for ref_pose in ref_pose_gt:
            ref_cam_gt = Camera(K=K, Twc=ref_pose)
            proj = ref_cam_gt.project(points_gt_world, frame='w', normalize=True)
            proj_gt.append(proj)
        
        # Compute masked reprojection error
        loss = 0
        for p_pred, p_gt in zip(proj_pred, proj_gt):
            valid_mask = (p_pred.abs().max(dim=-1)[0] <= 1.0) & (p_gt.abs().max(dim=-1)[0] <= 1.0)
            valid_mask = valid_mask.float().detach() * valid_depth_mask.squeeze(1)
            
            error = torch.norm(p_pred - p_gt, dim=-1) * valid_mask
            loss += error.sum() / (valid_mask.sum() + 1e-6)
            
        return loss / len(proj_pred)
        
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
    
    def _common_step(self, batch, batch_idx, stage="train"):
        #input_sparse_depth, input_image, rel_depth_pred, depth_gt, validity_map = batch
        tgt_img, tgt_gt_depth_inv, tgt_ga_depth, tgt_interp,_, ref_imgs, \
        ref_ga_depth, ref_interp, ref_gt_depth, _, tgt_pose, ref_pose, intrinsics = batch

        #convert to 4x4 pose matrix from 3x4
        
        
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
        refined_depth_inv, refined_target_pose, refined_ref_poses, warping_vis = self.model(tgt_img, ref_imgs,
                                                                                            tgt_ga_depth, ref_ga_depth, 
                                                                                            tgt_interp, ref_interp, tgt_pose, 
                                                                                            ref_pose, intrinsics)
            
        # 1. Depth L1 Loss
        depth_loss,_ = self.compute_loss(utils.inv2depth(refined_depth_inv),
                                gt_depth,
                                log_variance=None,
                                mask=None)
        
        # 2. Reprojection loss (global frame)
        reproj_loss = self.compute_reproj_loss(
            utils.inv2depth(refined_depth_inv), gt_depth,
            refined_target_pose,  # Predicted target pose
            tgt_pose,             # GT target pose
            refined_ref_poses,    # Predicted reference poses
            ref_pose,             # GT reference poses
            intrinsics
        )
        
        #pose regularization
        # pose_reg = torch.norm(se3_log_map(refined_target_pose @ tgt_pose.inverse()))
        # for refined_pose, gt_pose in zip(refined_ref_poses, ref_pose):
        #     pose_reg += torch.norm(se3_log_map(refined_pose @ gt_pose.inverse()))
    
        #total_loss = loss
        if self.current_epoch < 5:
           total_loss = depth_loss
        else:
           total_loss = depth_loss + 0.8 * reproj_loss #+ 0.01 * pose_reg
        
        #self.logger.experiment.add_scalar(f"{stage}_loss", loss, self.global_step)
        #self.log(f"{stage}/pose_reg_loss", pose_reg, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/reproj_loss", reproj_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/l1_depth_loss", depth_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/total_loss", total_loss, on_step=True, on_epoch=True, sync_dist=True)
        
        with torch.no_grad():
            if batch_idx % 10 == 0:
                if len(warping_vis) > 0:
                    self.log_warping(warping_vis, mode=stage)
                self.log_img_tensorboard(tgt_img, gt_depth, utils.inv2depth(tgt_ga_depth).unsqueeze(0).permute(1,0,2,3), 
                                         utils.inv2depth(refined_depth_inv), batch_idx, self.current_epoch, mode=stage)
        return {
            "loss": total_loss,
            "mode": stage,
            "pred_depth": refined_depth_inv.detach(),
            "gt_depth": tgt_gt_depth_inv.detach(),
            "ga_depth": tgt_ga_depth.unsqueeze(0).permute(1,0,2,3).detach(),
        }
        #TODO: compute loss for the pose as well
        

    # @torch.no_grad()
    # def log_img_tensorboard(self, images, depth_maps, batch_idx, epoch, mode="train"):
    #     depth_maps = torch.clip(depth_maps, 0, 1)
    #     img_vis = images[0].detach().cpu()
    #     img_vis = torch.from_numpy((img_vis.numpy() * 255).astype('uint8')).permute(2, 0, 1) / 255.0  # Convert to CHW
    #     processed_maps = []
    #     for depth_map in depth_maps:
    #         depth_map_np = depth_map.squeeze(0).detach().cpu().numpy()
    #         depth_map_np = (depth_map_np - np.min(depth_map_np)) / (
    #             np.max(depth_map_np) - np.min(depth_map_np)
    #         )
    #         colored_map = plt.get_cmap("jet")(depth_map_np)[ #blue low error, green to yellow medium error, red high error
    #             :, :, :3
    #         ]  # Apply colormap and remove alpha channel
    #         colored_map_tensor = (
    #             torch.from_numpy(colored_map).float().permute(
    #                 2, 0, 1).unsqueeze(0)
    #         )
    #         processed_maps.append(colored_map_tensor)
    #     all_maps = torch.cat(processed_maps, dim=0)
    #     all_maps = torch.cat([img_vis.unsqueeze(0), all_maps], dim=0)

    #     self.logger.experiment.add_image(
    #         mode,
    #         torchvision.utils.make_grid(all_maps, nrow=4, padding=4),
    #         global_step=self.global_step,
    #         dataformats="CHW",
    #     )

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