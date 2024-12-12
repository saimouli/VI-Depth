from model.mhybrid_net import midasNet
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
from typing import Any
import matplotlib.pyplot as plt
import torchvision
import cv2

class midasNetConsistentModule(pl.LightningModule):
    def __init__(self, lr: float = 0.1, wd: float = 0.1, min_pred: float = 0.1, 
                 max_pred: float = 8.0, min_depth: float = 0.2, 
                 max_depth: float = 5.0, nsamples: int = 150, img_h=480, img_w=640, sml_model_path: str = None,
                 *args: Any, **kwargs: Any) -> None:
        super(midasNetConsistentModule, self).__init__(*args, **kwargs)
        self.model = midasNet(min_pred, max_pred, min_depth, max_depth, nsamples, sml_model_path)
        self.lr = lr
        self.wd = wd
        self.orig_h = img_h
        self.orig_w = img_w
        self.abs_loss = nn.L1Loss()

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
    
    def _common_step(self, batch, batch_idx, stage="train"):
        #input_sparse_depth, input_image, rel_depth_pred, depth_gt, validity_map = batch
        tgt_img, tgt_gt_depth, tgt_ga_depth, tgt_interp, \
                ref_img, ref_ga_depth, ref_interp, ref_gt_depth,\
                tgt_pose, ref_pose, intrinsics = batch
        test = 0
        

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