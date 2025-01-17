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

class midasNetConsistentModule(pl.LightningModule):
    def __init__(self, lr: float = 0.1, wd: float = 0.1, min_pred: float = 0.1, 
                 max_pred: float = 8.0, min_depth: float = 0.2, 
                 max_depth: float = 5.0, nsamples: int = 150, img_h=480, img_w=640, sml_model_path: str = None,
                 *args: Any, **kwargs: Any) -> None:
        super(midasNetConsistentModule, self).__init__(*args, **kwargs)
        self.model = midasConsNet(min_pred, max_pred, min_depth, max_depth, nsamples, sml_model_path)
        self.lr = lr
        self.wd = wd
        self.orig_h = img_h
        self.orig_w = img_w
        self.abs_loss = nn.L1Loss()

    def load_from_pth(self, file_path):
        state_dict = torch.load(file_path, map_location=self.device)
        self.load_state_dict(state_dict)
    
    def forward(self, tgt_img, tgt_gt_depth_inv, tgt_ga_depth, tgt_interp, ref_imgs,
                ref_ga_depth, ref_interp, ref_gt_depth, tgt_pose, ref_pose, intrinsics):
        return self.model(tgt_img, tgt_gt_depth_inv, tgt_ga_depth, tgt_interp, ref_imgs,
                          ref_ga_depth, ref_interp, ref_gt_depth, tgt_pose, ref_pose, intrinsics)
    
    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr'] 
        self.log("learning_rate", current_lr, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, stage="val")
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

    def compute_loss(self, pred_depth, gt_depth, log_variance=None):
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
        valid_mask = (gt_depth > 0).float()
        M = valid_mask.sum()

        if log_variance is not None:
            depth_diff = (pred_depth * valid_mask) - (gt_depth * valid_mask)
            weighted_l1_loss = 0.5 * torch.exp(-log_variance) * (depth_diff ** 2) + 0.5 * log_variance
            l1_depth_loss = (weighted_l1_loss * valid_mask).sum() / M
        else:
            # L1 Depth loss
            l1_depth_loss = F.l1_loss(pred_depth * valid_mask, gt_depth * valid_mask, reduction='sum') / M

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
        tgt_img, tgt_gt_depth_inv, tgt_ga_depth, tgt_interp, ref_imgs, \
        ref_ga_depth, ref_interp, ref_gt_depth, tgt_pose, ref_pose, intrinsics = batch


        metric_depth_inv_pred = self.model(tgt_img, ref_imgs, tgt_ga_depth, ref_ga_depth, 
                                           tgt_interp, ref_interp,
                                           tgt_pose, ref_pose, intrinsics)
        
        # Compute depth loss with depth uncertainty
        # loss, loss_info = self.compute_loss(utils.inv2depth(metric_depth_inv_pred), 
        #                                    utils.inv2depth(tgt_gt_depth_inv),
        #                                    log_variance=None)
        
        gt_depth = utils.inv2depth(tgt_gt_depth_inv)
        metric_depth_pred = utils.inv2depth(metric_depth_inv_pred)
        total_loss = self.compute_exp_weighted_l1loss(metric_depth_pred, 
                                                gt_depth)
        
        #self.logger.experiment.add_scalar(f"{stage}_loss", loss, self.global_step)
        self.log(f"{stage}/total_loss", total_loss, on_step=True, on_epoch=True)
        
        with torch.no_grad():
            if batch_idx % 10 == 0:
                depth_gt_vis = gt_depth[0]

                metric_pred_vis = metric_depth_pred[-1][0]

                GA_pred = 1.0 / tgt_ga_depth[0]
                GA_pred[GA_pred == float("inf")] = 0

                t_gt = depth_gt_vis - depth_gt_vis.min()
                t_gt = t_gt / t_gt.max()

                t_pred = metric_pred_vis - metric_pred_vis.min()
                t_pred = t_pred / t_pred.max()

                t_ga = GA_pred - GA_pred.min()
                t_ga = t_ga / t_ga.max()

                t = torch.concat([t_gt, t_pred, t_ga.unsqueeze(0)], dim=0)
                self.log_img_tensorboard(tgt_img, t, batch_idx, self.current_epoch, mode=stage)
            
        return total_loss
        #TODO: compute loss for the pose as well
        

    @torch.no_grad()
    def log_img_tensorboard(self, images, depth_maps, batch_idx, epoch, mode="train"):
        depth_maps = torch.clip(depth_maps, 0, 1)
        img_vis = images[0].detach().cpu()
        img_vis = torch.from_numpy((img_vis.numpy() * 255).astype('uint8')).permute(2, 0, 1) / 255.0  # Convert to CHW
        processed_maps = []
        for depth_map in depth_maps:
            depth_map_np = depth_map.squeeze(0).detach().cpu().numpy()
            depth_map_np = (depth_map_np - np.min(depth_map_np)) / (
                np.max(depth_map_np) - np.min(depth_map_np)
            )
            colored_map = plt.get_cmap("jet")(depth_map_np)[
                :, :, :3
            ]  # Apply colormap and remove alpha channel
            colored_map_tensor = (
                torch.from_numpy(colored_map).float().permute(
                    2, 0, 1).unsqueeze(0)
            )
            processed_maps.append(colored_map_tensor)
        all_maps = torch.cat(processed_maps, dim=0)
        all_maps = torch.cat([img_vis.unsqueeze(0), all_maps], dim=0)

        self.logger.experiment.add_image(
            mode,
            torchvision.utils.make_grid(all_maps, nrow=4, padding=4),
            global_step=self.global_step,
            dataformats="CHW",
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.lr,
                                      weight_decay=self.wd,
                                      betas=(0.9, 0.999)
                                      )

        return optimizer