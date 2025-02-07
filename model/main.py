from model.mhybrid_net import midasNet
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
from typing import Any
#from utils.loss import MSGradientLoss, SILogLoss
import matplotlib.pyplot as plt
import torchvision
import time
import cv2
import metrics

class midasNetModule(pl.LightningModule):
    def __init__(self, lr: float = 0.1, wd: float = 0.1, min_pred: float = 0.1, 
                 max_pred: float = 8.0, min_depth: float = 0.2, 
                 max_depth: float = 5.0, nsamples: int = 150, img_h=480, img_w=640, sml_model_path: str = None,
                 *args: Any, **kwargs: Any) -> None:
        super(midasNetModule, self).__init__(*args, **kwargs)
        self.model = midasNet(min_pred, max_pred, min_depth, max_depth, nsamples, sml_model_path)
        self.lr = lr
        self.wd = wd
        self.orig_h = img_h
        self.orig_w = img_w
        self.max_depth = max_depth
        self.min_depth = min_depth
        #self.grad_loss = MSGradientLoss(num_scales=4)
        #self.abs_loss = nn.L1Loss()
        #self.silog = SILogLoss()
        self.total_val_img = 0
        # self.avg_error_w_int_depth = metrics.ErrorMetricsAverager_DDP(self.device)
        # self.avg_error_w_pred = metrics.ErrorMetricsAverager_DDP(self.device)
    
    def on_fit_start(self):
        """Ensure that metric averaging uses the correct device after model initialization."""
        self.avg_error_w_int_depth = metrics.ErrorMetricsAverager_DDP(self.device)
        self.avg_error_w_pred = metrics.ErrorMetricsAverager_DDP(self.device)
    
    def compute_loss(self, pred_depth, gt_depth):
        """
        Computes the loss for depth prediction based on L1 depth loss and multiscale gradient matching.

        Args:
        - pred_depth (torch.Tensor): Predicted depth map (B, C, H, W)
        - gt_depth (torch.Tensor): Ground truth depth map (B, C, H, W)
        
        Returns:
        - loss (torch.Tensor): Total loss (depth loss + 0.5 * gradient loss)
        - loss_info (dict): Detailed information about individual loss components
        """
        # Ensure valid mask for pixels with ground truth
        valid_mask = ((gt_depth > 0) & (gt_depth <= 5)).float()
        M = valid_mask.sum()

        ## L1 Depth loss
        l1_depth_loss = F.l1_loss(pred_depth * valid_mask, gt_depth * valid_mask, reduction='sum') / M
        # alpha = 1e-7
        # beta = 0.15
        # g = torch.log(pred_depth + alpha) - torch.log(gt_depth + alpha)
        # var_g = torch.var(g[valid_mask > 0])
        # mean_g = torch.mean(g[valid_mask > 0])
        # scale_invariant_loss = 10 * torch.sqrt(var_g + beta * mean_g**2)
        

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

        # Compute ground truth normals
        gt_grad_x = gt_depth[:, :, :, :-1] - gt_depth[:, :, :, 1:]
        gt_grad_y = gt_depth[:, :, :-1, :] - gt_depth[:, :, 1:, :]
        
        min_height = min(gt_grad_x.size(2), gt_grad_y.size(2))
        min_width = min(gt_grad_x.size(3), gt_grad_y.size(3))
        gt_grad_x = gt_grad_x[:, :, :min_height, :min_width]
        gt_grad_y = gt_grad_y[:, :, :min_height, :min_width]
        gt_normal_z = torch.ones_like(gt_grad_x)
        
    
        gt_normal = torch.cat([gt_grad_x, gt_grad_y, gt_normal_z], dim=1)
        gt_normal = F.normalize(gt_normal, dim=1)  # Normalize ground truth normals

        # Compute predicted normals
        pred_grad_x = pred_depth[:, :, :, :-1] - pred_depth[:, :, :, 1:]
        pred_grad_y = pred_depth[:, :, :-1, :] - pred_depth[:, :, 1:, :]
        
        pred_grad_x = pred_grad_x[:, :, :min_height, :min_width]
        pred_grad_y = pred_grad_y[:, :, :min_height, :min_width]
        pred_normal_z = torch.ones_like(pred_grad_x)
        
        pred_normal = torch.cat([pred_grad_x, pred_grad_y, pred_normal_z], dim=1)
        pred_normal = F.normalize(pred_normal, dim=1)  # Normalize predicted normals
        valid_mask_cropped = valid_mask[:, :, :min_height, :min_width]
        normal_loss = F.l1_loss(pred_normal * valid_mask_cropped, gt_normal * valid_mask_cropped, reduction='sum') / M
    
        # Total loss
        total_loss = l1_depth_loss + 0.5 * grad_loss + 0.7 * normal_loss

        # Loss info dictionary for logging purposes
        loss_info = {
            'l1_depth_loss': l1_depth_loss.item(),
            'gradient_loss': grad_loss.item(),
            'total_loss': total_loss.item(),
            'normal_loss': normal_loss.item()
        }

        return total_loss, loss_info
    
    def load_from_pth(self, file_path):
        state_dict = torch.load(file_path, map_location=self.device)
        self.load_state_dict(state_dict)
    
    def forward(self, input_sparse_depth, input_image, rel_depth_pred, interp_scale, ga_depth_inv):
        metric_depth_inv_pred, GA_depth_inv =  self.model(input_sparse_depth, input_image, rel_depth_pred, interp_scale, ga_depth_inv, None)
        return metric_depth_inv_pred, GA_depth_inv
    
    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr'] 
        self.log("learning_rate", current_lr, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, stage="val")
        pred_depth = loss["pred_depth"]  # (B, H, W)
        gt_depth = loss["gt_depth"]      # (B, H, W)
        ga_depth = loss["ga_depth"]      # (B, H, W)

        batch_size = pred_depth.shape[0]
        max_depth, min_depth = self.max_depth, self.min_depth

        for i in range(batch_size):
            # Get tensors directly (assuming they're already on correct device)
            tgt_gt = gt_depth[i]
            tgt_ga = ga_depth[i]
            pred = pred_depth[i]

            # Compute valid mask
            valid_mask = (tgt_gt >= min_depth) & (tgt_gt <= max_depth)

            # Compute metrics for GA depth
            ga_metrics = metrics.ErrorMetrics_DDP()
            ga_metrics.compute(tgt_ga, tgt_gt, valid_mask)
            self.avg_error_w_int_depth.accumulate(ga_metrics)

            # Compute metrics for predicted depth
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
        
    def _common_step(self, batch, batch_idx, stage="train"):
        #input_sparse_depth, input_image, rel_depth_pred, depth_gt, validity_map = batch
        #t1 = time.time()
        input_image, depth_gt_inv, input_sparse_depth, rel_depth_pred, ga_depth_inv, interp_scale, mask,_ = batch
        #print("Time taken to load batch: ", time.time() - t1)
        #metric_depth_pred, GA_depth, mask = self.model(input_sparse_depth, input_image, rel_depth_pred, None)
        #t1 = time.time()
        metric_depth_inv_pred, GA_depth_inv = self.model(input_sparse_depth, input_image, rel_depth_pred, 
                                                         interp_scale, ga_depth_inv, None)
        #print("Time taken to forward pass: ", time.time() - t1)

        # metric_depth_pred = torch.nn.functional.interpolate(
        #     metric_depth_pred,
        #     size=(self.orig_h, self.orig_w),  # Assuming self.orig_h, self.orig_w match depth_gt dimensions
        #     mode="bicubic",
        #     align_corners=False,
        # )
        # # Resize mask to match depth_gt dimensions
        # mask_resized = torch.nn.functional.interpolate(
        #     mask,
        #     size=(self.orig_h, self.orig_w),
        #     mode="nearest"  # Use nearest for masks to preserve binary values
        # )
        # depth_gt_inv_resized = torch.nn.functional.interpolate(
        #     depth_gt_inv.unsqueeze(1), size=metric_depth_inv_pred.shape[2:], mode="bicubic", align_corners=False
        # )
        #t1 = time.time()
        depth_gt = 1.0 / depth_gt_inv
        depth_gt[depth_gt == float("inf")] = 0
        metric_depth_pred = 1.0 / metric_depth_inv_pred
        metric_depth_pred[metric_depth_pred == float("inf")] = 0

        loss,loss_info = self.compute_loss(metric_depth_pred, depth_gt)
        #print("Time taken to compute loss: ", time.time() - t1)
        
        #resize metric_depth_pred to match depth_gt dimensions
        #l_grad = self.grad_loss(metric_depth_pred, depth_gt)
        #siloss = self.silog(metric_depth_pred, depth_gt, mask)
        
        #self.logger.experiment.add_scalar(f"{stage}_grad_loss", l_grad, self.global_step)
        #self.logger.experiment.add_scalar(f"{stage}_siloss", siloss, self.global_step)

        # self.log(stage+"_gradloss", l_grad, on_epoch=True, prog_bar=True, sync_dist=True)
        # self.log(stage+"_silossloss", siloss, on_epoch=True, prog_bar=True, sync_dist=True)

        #loss = siloss + l_grad * 0.5
        #self.log(stage+"_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/total_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/gradloss", loss_info['gradient_loss'], prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/l1loss", loss_info['l1_depth_loss'], prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/normal_loss", loss_info['normal_loss'], prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        if batch_idx % 40 == 0:
            #t1 = time.time()
            depth_gt_vis = depth_gt[0]

            metric_pred_vis = metric_depth_pred[0]

            GA_pred = 1.0 / GA_depth_inv[0]
            GA_pred[GA_pred == float("inf")] = 0

            t_gt = depth_gt_vis - depth_gt_vis.min()
            t_gt = t_gt / t_gt.max()

            t_pred = metric_pred_vis - metric_pred_vis.min()
            t_pred = t_pred / t_pred.max()

            t_ga = GA_pred - GA_pred.min()
            t_ga = t_ga / t_ga.max()

            t = torch.concat([t_gt, t_pred, t_ga.unsqueeze(0)], dim=0)
            self.log_img_tensorboard(input_image, t, batch_idx, self.current_epoch, mode=stage)
            #print("Time taken to log to tensorboard: ", time.time() - t1)

        return {
            #loss, input_image, metric_depth_inv_pred, GA_depth_inv, depth_gt_inv
            "loss": loss,
            "mode": stage,
            "pred_depth": metric_depth_inv_pred.detach(),
            "gt_depth": depth_gt_inv.detach(),
            "ga_depth": ga_depth_inv.unsqueeze(0).permute(1,0,2,3).detach(),
        }

    
    def visualize_depth_diff(self, target, pred):
        abs_diff = torch.abs(pred - target)
        abs_diff_vis = (abs_diff - abs_diff.min()) / (abs_diff.max() - abs_diff.min() + 1e-6)
        abs_diff_vis = abs_diff_vis[0].squeeze(0).detach().cpu().numpy() 

        abs_diff_colormap = cv2.applyColorMap((abs_diff_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
        return abs_diff_colormap

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

        # depth_gt[depth_gt == float("inf")] = 0
        # metric_depth_pred[metric_depth_pred == float("inf")] = 0
        # GA_depth[GA_depth == float("inf")] = 0

        # # Apply valid masks
        # valid_mask_gt = (depth_gt[0] >= 0.1) & (depth_gt[0] <= 8)
        # valid_mask_depth = (metric_depth_pred[0] >= 0.1) & (metric_depth_pred[0] <= 8)
        # valid_mask_GA = (GA_depth[0] >= 0.1) & (GA_depth[0] <= 8)

        # # Mask invalid regions (set them to 0)
        # depth_gt_vis = depth_gt[0] * valid_mask_gt
        # metric_depth_pred_vis = metric_depth_pred[0] * valid_mask_depth
        # GA_depth_vis = GA_depth[0] * valid_mask_GA

        # # Normalize and colormap the absolute differences
        # abs_diff_metric_colormap = self.visualize_depth_diff(depth_gt_vis, metric_depth_pred_vis)
        # abs_diff_GA_colormap = self.visualize_depth_diff(depth_gt_vis, GA_depth_vis)

        # # Prepare input image for visualization
        # img_vis = images[0].permute(1, 2, 0).detach().cpu()  # Convert to HWC
        # img_vis = torch.from_numpy((img_vis.numpy() * 255).astype('uint8')).permute(2, 0, 1) / 255.0  # Convert to CHW

        # # Normalize depth maps for consistent visualization
        # depth_gt_colormap = self.normalize_and_colormap(depth_gt_vis)  # Add batch dim for processing
        # metric_depth_pred_colormap = self.normalize_and_colormap(metric_depth_pred_vis)
        # GA_depth_colormap = self.normalize_and_colormap(GA_depth_vis)

        # # Convert absolute difference colormaps to tensors
        # abs_diff_metric_tensor = torch.from_numpy(abs_diff_metric_colormap).permute(2, 0, 1) / 255.0
        # abs_diff_GA_tensor = torch.from_numpy(abs_diff_GA_colormap).permute(2, 0, 1) / 255.0

        # # Concatenate horizontally for TensorBoard visualization
        # horz_grid = torch.cat(
        #     (
        #         img_vis.cpu(),  # Input image
        #         depth_gt_colormap.cpu(),  # Ground truth depth
        #         metric_depth_pred_colormap.cpu(),  # Predicted depth
        #         GA_depth_colormap.cpu(),  # GA depth
        #     ),
        #     dim=2,  # Concatenate along width
        # )

        # horz_abs_grid = torch.cat(
        #     (
        #         abs_diff_metric_tensor.cpu(),  # Absolute difference with metric depth
        #         abs_diff_GA_tensor.cpu(),  # Absolute difference with GA depth
        #     ),
        #     dim=2,  # Concatenate along width
        # )

        # # Log the horizontal grid to TensorBoard
        # self.logger.experiment.add_image(f"{mode}_visualization_grid", horz_grid, self.global_step, dataformats="CHW")
        # self.logger.experiment.add_image(f"{mode}_abs_diff_grid", horz_abs_grid, self.global_step, dataformats="CHW")
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.lr,
                                      weight_decay=self.wd,
                                      betas=(0.9, 0.999)
                                      )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                            mode='min', 
                                                            factor=0.5, 
                                                            patience=1, 
                                                            min_lr=1e-6,
                                                            verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/total_loss",  # Replace with your actual validation loss metric key
            },
        }