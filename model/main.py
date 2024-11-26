from model.mhybrid_net import midasNet
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
from typing import Any
from utils.loss import MSGradientLoss, SILogLoss

class midasNetModule(pl.LightningModule):
    def __init__(self, lr: float = 0.1, wd: float = 0.1, min_pred: float = 0.1, 
                 max_pred: float = 8.0, min_depth: float = 0.2, 
                 max_depth: float = 5.0, nsamples: int = 150, sml_model_path: str = None, 
                 *args: Any, **kwargs: Any) -> None:
        super(midasNetModule, self).__init__(*args, **kwargs)
        self.model = midasNet(min_pred, max_pred, min_depth, max_depth, nsamples, sml_model_path)
        self.lr = lr
        self.wd = wd
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
        input_image, depth_gt, input_sparse_depth, rel_depth_pred = batch
        metric_depth_pred, GA_depth, mask = self.model(input_sparse_depth, input_image, rel_depth_pred, None)

        l_grad = self.grad_loss(metric_depth_pred, depth_gt)
        siloss = self.silog(metric_depth_pred, depth_gt, mask)
        
        self.log(stage+"_gradloss", l_grad, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(stage+"_silossloss", siloss, on_epoch=True, prog_bar=True, sync_dist=True)

        loss = siloss + l_grad * 0.5
        self.log(stage+"_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        if batch_idx % 10 == 0:
            self.log_img_tensorboard(input_image, depth_gt, metric_depth_pred, GA_depth, batch_idx, self.current_epoch, mode=stage)

        return loss, input_image, metric_depth_pred, GA_depth, depth_gt
    
    def log_img_tensorboard(self, images, depth_gt, metric_depth_pred, GA_depth, batch_idx, epoch, mode="train"):
        if mode == "train":
            global_step = epoch * len(self.trainer.train_dataloader) + batch_idx
        else:
            global_step = epoch * len(self.trainer.val_dataloader) + batch_idx
        
        self.logger.experiment.add_images(f'{mode}_images', images.detach().cpu().numpy(), global_step)
        self.logger.experiment.add_images(f'{mode}_depth_gt', depth_gt.detach().cpu().numpy(), global_step)
        self.logger.experiment.add_images(f'{mode}_metric_depth_pred', metric_depth_pred.detach().cpu().numpy(), global_step)
        self.logger.experiment.add_images(f'{mode}_GA_depth', GA_depth.detach().cpu().numpy(), global_step)
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.lr,
                                      weight_decay=self.wd,
                                      betas=(0.9, 0.999)
                                      )

        return optimizer