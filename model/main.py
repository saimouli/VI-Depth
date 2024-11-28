from model.mhybrid_net import midasNet
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
from typing import Any
from utils.loss import MSGradientLoss, SILogLoss
import matplotlib.pyplot as plt
import torchvision
import cv2

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
        self.grad_loss = MSGradientLoss(num_scales=4)
        self.abs_loss = nn.L1Loss()
        self.silog = SILogLoss()
    
    def load_from_pth(self, file_path):
        state_dict = torch.load(file_path, map_location=self.device)
        self.load_state_dict(state_dict)
    
    def forward(self, input_sparse_depth, input_image, depth_pred, validity_map):
        return self.model(input_sparse_depth, input_image, depth_pred, validity_map)
    
    def training_step(self, batch, batch_idx):
        loss, _, _, _, _= self._common_step(batch, batch_idx)
        
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr'] 
        self.log("learning_rate", current_lr, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, input_img, metric_depth_pred, GA_depth, depth_gt = self._common_step(batch, batch_idx, stage="val")
        return loss
    
    def test_step():
        pass

    def _common_step(self, batch, batch_idx, stage="train"):
        #input_sparse_depth, input_image, rel_depth_pred, depth_gt, validity_map = batch
        input_image, depth_gt_inv, input_sparse_depth, rel_depth_pred, ga_depth_inv, interp_scale, mask = batch
        #metric_depth_pred, GA_depth, mask = self.model(input_sparse_depth, input_image, rel_depth_pred, None)
        metric_depth_inv_pred, GA_depth_inv = self.model(input_sparse_depth, input_image, rel_depth_pred, 
                                                         interp_scale, ga_depth_inv, None)

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

        l_grad = self.grad_loss(metric_depth_inv_pred, depth_gt_inv)
        siloss = self.silog(metric_depth_inv_pred, depth_gt_inv, mask)
        
        self.logger.experiment.add_scalar(f"{stage}_grad_loss", l_grad, self.global_step)
        self.logger.experiment.add_scalar(f"{stage}_siloss", siloss, self.global_step)

        # self.log(stage+"_gradloss", l_grad, on_epoch=True, prog_bar=True, sync_dist=True)
        # self.log(stage+"_silossloss", siloss, on_epoch=True, prog_bar=True, sync_dist=True)

        loss = siloss + l_grad * 0.5
        #self.log(stage+"_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.logger.experiment.add_scalar(f"{stage}_loss", loss, self.global_step)

        # if batch_idx % 10 == 0:
        #     self.log_img_tensorboard(input_image, 1.0/depth_gt_inv, 1.0/metric_depth_inv_pred, 1.0/GA_depth_inv, 
        #                              batch_idx, self.current_epoch, mode=stage)

        return loss, input_image, metric_depth_inv_pred, GA_depth_inv, depth_gt_inv

    def normalize_and_colormap(self, depth_map, cmap="jet"):
        """
        Normalize and apply a colormap to a single depth map.
        """
        depth_map = depth_map.detach().squeeze(0).cpu().numpy()
        depth_min = np.min(depth_map[depth_map > 0])
        depth_max = np.max(depth_map)
        depth = (depth_map - depth_min) / (depth_max - depth_min + 1e-6) 
        depth =  (depth * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        #depth_min, depth_max = depth_map.min(), depth_map.max()
        #depth_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-6)
        #depth_uint8 = (depth_norm * 255).byte().numpy()
        #depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        depth_colormap_tensor = torch.from_numpy(depth_vis).permute(2, 0, 1) / 255.0
        return depth_colormap_tensor
    
    def visualize_depth_diff(self, target, pred):
        abs_diff = torch.abs(pred - target)
        abs_diff_vis = (abs_diff - abs_diff.min()) / (abs_diff.max() - abs_diff.min() + 1e-6)
        abs_diff_vis = abs_diff_vis[0].squeeze(0).detach().cpu().numpy() 

        abs_diff_colormap = cv2.applyColorMap((abs_diff_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
        return abs_diff_colormap

    @torch.no_grad()
    def log_img_tensorboard(self, images, depth_gt, metric_depth_pred, GA_depth, batch_idx, epoch, mode="train"):
        # if mode == "train":
        #     global_step = epoch * len(self.trainer.train_dataloader) + batch_idx
        # else:
        #     global_step = epoch * len(self.trainer.val_dataloaders) + batch_idx
        
        #images_grid = torchvision.utils.make_grid(images, nrow=4, normalize=True, scale_each=True)

        # depth_gt_vis = [self.normalize_and_colormap(d.unsqueeze(0)) for d in depth_gt[:3]]  # Adding channel dimension
        # metric_depth_pred_vis = [self.normalize_and_colormap(d[0]) for d in metric_depth_pred[:3]]
        # GA_depth_vis = [self.normalize_and_colormap(d[0]) for d in GA_depth[:3]]

        # depth_gt_grid = torchvision.utils.make_grid(torch.stack(depth_gt_vis), nrow=4)
        # metric_depth_pred_grid = torchvision.utils.make_grid(torch.stack(metric_depth_pred_vis), nrow=4)
        # GA_depth_grid = torchvision.utils.make_grid(torch.stack(GA_depth_vis), nrow=4)

        # plt.imshow(depth_gt[0].squeeze(0).cpu().numpy())
        # plt.show()
        # plt.imshow(metric_depth_pred[0].squeeze(0).cpu().numpy())
        # plt.show()

        depth_gt_vis = depth_gt[0].cpu() #self.normalize_and_colormap(depth_gt[0])
        metric_depth_pred_vis = metric_depth_pred[0].cpu() #self.normalize_and_colormap(metric_depth_pred[0])
        GA_depth_vis = GA_depth[0].cpu() #self.normalize_and_colormap(GA_depth[0])
        img_vis = images[0].permute(1, 2, 0).detach().cpu()

        img_vis = torch.from_numpy((img_vis.numpy() * 255).astype('uint8')).permute(2, 0, 1) / 255.0
        horz_grid = torch.cat((img_vis, depth_gt_vis, metric_depth_pred_vis, GA_depth_vis), dim=2)
        # Stack rows vertically
        self.logger.experiment.add_image(f"{mode}_visualization_grid", horz_grid, self.global_step, dataformats="CHW")

        # self.logger.experiment.add_image(f"{mode}_input_images", img_vis, self.global_step, dataformats="HWC")
        # self.logger.experiment.add_image(f"{mode}_depth_gt", depth_gt_vis, self.global_step, dataformats="CHW")
        # self.logger.experiment.add_image(f"{mode}_metric_depth_pred", metric_depth_pred_vis, self.global_step, dataformats="CHW")
        # self.logger.experiment.add_image(f"{mode}_GA_depth", GA_depth_vis, self.global_step, dataformats="CHW")

        # abs_diff = torch.abs(metric_depth_pred - depth_gt)
        # abs_diff_vis = (abs_diff - abs_diff.min()) / (abs_diff.max() - abs_diff.min() + 1e-6)
        # abs_diff_vis = abs_diff_vis[0].squeeze(0).cpu().numpy() 

        # abs_diff_colormap = cv2.applyColorMap((abs_diff_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
        abs_diff_metric_colormap = self.visualize_depth_diff(depth_gt[0], metric_depth_pred[0])
        abs_diff_GA_colormap = self.visualize_depth_diff(depth_gt[0], GA_depth[0])


        self.logger.experiment.add_image(f"{mode}_absolute_difference", abs_diff_metric_colormap, self.global_step, dataformats="HWC")
        self.logger.experiment.add_image(f"{mode}_absolute_difference", abs_diff_GA_colormap, self.global_step, dataformats="HWC")

        # self.log(f"{mode}_RMSE", torch.sqrt(torch.mean((metric_depth_pred - depth_gt) ** 2)), 
        #          on_epoch=True, prog_bar=True, sync_dist=True)

        # self.logger.experiment.add_images(f'{mode}_images', images.detach().cpu().numpy(), global_step)
        # self.logger.experiment.add_images(f'{mode}_depth_gt', depth_gt.detach().cpu().numpy(), global_step)
        # self.logger.experiment.add_images(f'{mode}_metric_depth_pred', metric_depth_pred.detach().cpu().numpy(), global_step)
        # self.logger.experiment.add_images(f'{mode}_GA_depth', GA_depth.detach().cpu().numpy(), global_step)
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.lr,
                                      weight_decay=self.wd,
                                      betas=(0.9, 0.999)
                                      )

        return optimizer