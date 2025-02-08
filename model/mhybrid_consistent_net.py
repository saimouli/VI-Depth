import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if module_path not in sys.path:
    sys.path.append(module_path)
from modules.midas.midas_net_cons_custom import MidasNet_small_cons_videpth
import numpy as np
import modules.midas.utils as utils
import modules.midas.transforms as transforms
from utils.camera import Camera
from functools import partial
#from modules.midas.blocks import OutputConv

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class ProjectionInputDepth(nn.Module):
    def __init__(self, cost_dim, hidden_dim, out_chs):
        super().__init__()
        self.out_chs = out_chs
        self.convc1 = nn.Conv2d(cost_dim, hidden_dim, 1, padding=0)
        self.convc2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        
        self.convd1 = nn.Conv2d(1, hidden_dim, 7, padding=3)
        self.convd2 = nn.Conv2d(hidden_dim, 64, 3, padding=1)
        
        self.convd = nn.Conv2d(64+hidden_dim, out_chs - 1, 3, padding=1)
        
    def forward(self, depth, cost):
        #cost -> shape [B, cost_dim, H, W]
        cor = F.relu(self.convc1(cost))
        cor = F.relu(self.convc2(cor))

        # depth -> shape [B, 1, H, W]
        dept = F.relu(self.convd1(depth))
        dept = F.relu(self.convd2(dept))
        cor_dfm = torch.cat([cor, dept], dim=1)
        
        out_d = F.relu(self.convd(cor_dfm))
        # final shape -> [B, out_chs, H, W]
        return torch.cat([out_d, depth], dim=1)

#Limitations: 
# limited receptive field: (1,5) and (5,1) convolutions separable
# lack of global context
class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

class OutputScaleConv(nn.Module):
    """Output conv block.
    """

    def __init__(self, features, groups, activation, non_negative):

        super(OutputScaleConv, self).__init__()

        self.output_conv = nn.Sequential(
            nn.Conv2d(features, features//2, kernel_size=3, stride=1, padding=1, groups=groups),
            #nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(features//2, 32, kernel_size=3, stride=1, padding=1),
            activation,
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

    def forward(self, x):        
        return self.output_conv(x)
    
class BasicUpdateBlockDepth(nn.Module):
    def __init__(self, hidden_dim=128, cost_dim=64, ratio=3, context_dim=64, min_pred=None, max_pred=None, log_fn=None):
        super(BasicUpdateBlockDepth, self).__init__()
                
        self.encoder = ProjectionInputDepth(cost_dim=cost_dim, hidden_dim=hidden_dim, out_chs=hidden_dim)
        self.depth_gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=self.encoder.out_chs+context_dim)
        self.depth_head = OutputScaleConv(features=128, groups=1, activation=nn.ReLU(False), non_negative=False)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim*2, ratio*ratio*9, 1, padding=0))
        self.min_pred = min_pred
        self.max_pred = max_pred
        self.log_fn = log_fn

    def forward(self, hidden, cost_func, inv_depth, context, seq_len=4):
        inv_depth_list = [] 
        mask_list = []
        
        for i in range(seq_len):
            cost = cost_func(inv_depth)
            #print("cost {}: mean{}".format(i, cost.mean()))
            if self.log_fn:
                self.log_fn(f"UpdateBlock/Iteration_{i}/cost_mean", cost.mean().item(), on_step=True, logger=True)
                
            input_features = self.encoder(inv_depth, cost)
            inp_i = torch.cat([context, input_features], dim=1)

            if self.log_fn:
                self.log_fn(f"UpdateBlock/Iteration_{i}/hidden_state_norm_before", hidden.norm().item(), on_step=True, logger=True)
                
            hidden = self.depth_gru(hidden, inp_i)
            
            if self.log_fn:
                self.log_fn(f"UpdateBlock/Iteration_{i}/hidden_state_norm_after", hidden.norm().item(), on_step=True, logger=True)
                
            delta_scales = self.depth_head(hidden)
            delta_scales = F.relu(1.0 + delta_scales) #ensure positive scale
            print(f"inv_depth mean step {i}: {inv_depth.mean().item()}")
            print("delta_scales mean step", i, delta_scales.mean().item())
            
            if self.log_fn:
                self.log_fn(f"UpdateBlock/Iteration_{i}/delta_scales_mean", delta_scales.mean().item(), on_step=True, logger=True)
                
            inv_depth_pred = inv_depth * delta_scales

            # clamp pred to min and max
            if self.min_pred is not None and self.max_pred is not None:
                inv_depth_pred = torch.clamp(
                    inv_depth_pred, 
                    min=1.0 / self.max_pred, 
                    max=1.0 / self.min_pred
                )   
            # if self.min_pred is not None:
            #     min_pred_inv = 1.0/self.min_pred
            #     inv_depth_pred[inv_depth_pred > min_pred_inv] = min_pred_inv
            #     #pred[pred < self.min_pred] = self.min_pred
            # if self.max_pred is not None:
            #     max_pred_inv = 1.0/self.max_pred
            #     inv_depth_pred[inv_depth_pred < max_pred_inv] = max_pred_inv
            
             # *** Update for next iteration ***
            inv_depth = inv_depth_pred
            # scale mask to balence gradients
            mask = 0.25 * self.mask(hidden) #helps numerical stability how?
            
            inv_depth_list.append(inv_depth_pred)
            mask_list.append(mask)
            
        return hidden, mask_list, inv_depth_list

# class DifferentiableGaussNewton(nn.Module):
#     def __init__(self, num_steps=3):
#         super().__init__()
#         self.num_steps = num_steps
    
#     def forward(self, features, depth, poses, sparse_points, K):
#         """
#         features: (B, C, H, W) - Target/reference features
#         depth: (B, 1, H, W) - Current depth estimate
#         poses: (B, N, 4, 4) - Camera poses (N = num keyframes)
#         K: (B, 3, 3) - Intrinsics
#         """
#         b, _, h, w = depth.shape
#         device = depth.device
        
#         for _ in range(self.num_steps):
#             # Compute Jacobian and residuals
#             J_pose, residuals = self.compute_jacobian_residuals(
#                 features, depth, poses, sparse_points, K
#             )
            
#             #Solve Gauss-Newton step: ΔT = -(J^T J)^-1 J^T r
#             J_T = J_pose.transpose(-1, -2)
#             H = J_T @ J_pose
#             delta = -torch.linalg.inv(H + 1e-3 * torch.eye(6, device=device)) @ (J_T @ residuals)
#             # Update poses (Lie algebra se3)
#             poses = self.update_pose(poses, delta)
#         return poses
    
#     def compute_jacobian_residuals(self, features, depth, poses, sparse_points, K):
#         J_pose = 0; residuals = 0
#         return J_pose, residuals
    
#     def update_pose(self, poses, delta):
#         delat_mat = pp.se3.exp(delta)
#         return pose @ delta_mat

  
class midasConsNet(nn.Module):
    def __init__(self, min_pred, max_pred, min_depth, max_depth, nsamples, sml_model_path, 
                 is_train=True, log_fn=None, isConvGRU=False):
        super(midasConsNet, self).__init__()
        self.is_train = is_train
        self.log_fn = log_fn  # Logger function from Lightning module
        self.UseConvGRU = isConvGRU
        
        self.min_pred, self.max_pred = min_pred, max_pred
        self.min_depth, self.max_depth = min_depth, max_depth

        model_transforms = transforms.get_transforms("dpt_hybrid", "void", str(nsamples))
        self.ScaleMapLearner_transform = model_transforms["sml_model"]

        self.FeatExtractor = MidasNet_small_cons_videpth(
            path=sml_model_path,
            min_pred=self.min_pred,
            max_pred=self.max_pred,
            output_downsample=True,
            backbone="efficientnet_lite3",
        )
        self.FeatExtractor.train()
        
        self.contextLearner = MidasNet_small_cons_videpth(
            in_channels=2,
            path=sml_model_path,
            min_pred=self.min_pred,
            max_pred=self.max_pred,
            output_downsample=True,
            backbone="efficientnet_lite3",
        )
        self.contextLearner.train()
    
        self.hidden_dim = 128
        self.cost_dim = 64
        self.iter_steps = 3
        
        self.context_conv = nn.Conv2d(
            in_channels=64,
            out_channels=self.hidden_dim + self.cost_dim, 
            kernel_size=3, stride=1, padding=1
        )
        self.maskNet = nn.Conv2d(64, 1, kernel_size=3, padding=1) 
        
        if self.UseConvGRU:
            self.update_block_depth = BasicUpdateBlockDepth(hidden_dim=self.hidden_dim, 
                                                            cost_dim=self.cost_dim,
                                                            ratio=3, 
                                                            context_dim=self.cost_dim,
                                                            min_pred=self.min_pred,
                                                            max_pred=self.max_pred)
            
        else:
            self.scaleOutput = OutputScaleConv(features=64, groups=1, activation=nn.ReLU(False), non_negative=False)
        
    def upsample_depth(self, depth, mask, ratio):
        """ Upsample depth field [H/ratio, W/ratio, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = depth.shape

        mask = mask.view(N, 1, 9, ratio, ratio, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(depth, [3,3], padding=1)
        up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 1, ratio*H, ratio*W)
    
    def get_cost_each(self, tgt_pose, pose, fmap, fmap_ref, depth, K, scale_factor):
        """
            ga_depth: (b, 1, h, w)
            fmap, fmap_ref: (b, c, h, w)
        """
        device = depth.device
        ref_cam = Camera(K=K.float(), Twc=pose).scaled(scale_factor).to(device)
        cam = Camera(K=K.float(), Twc=tgt_pose).scaled(scale_factor).to(device) # tcw = Identity
        
        # Reconstruct world points from target_camera
        world_points = cam.reconstruct(depth, frame='w')
        # Project world points onto reference camera
        ref_coords = ref_cam.project(world_points, frame='w', normalize=True) #(b, h, w,2)
        #with torch.no_grad():
        #    valid_mask = (ref_coords.abs().max(dim=-1)[0] <= 1).float()  # [B, H, W]
        fmap_warped = F.grid_sample(fmap_ref, ref_coords, 
                                    mode='bilinear', padding_mode='zeros', align_corners=True) # (b, c, h, w)
        
        cost = (fmap - fmap_warped)**2
        #cost_l1 = torch.abs(fmap - fmap_warped).mean()
        #cost_ssim = (1 - self.ssim_loss(fmap, fmap_warped)).mean()
        #cost = 0.85 * cost_l1 + 0.15 * cost_ssim 
        #if self.global_step % 50 == 0:
            #self.log_feature_pca(fmap, fmap_warped, self.global_step)
        #print("cost each: mean", cost.mean())
        
        return cost
    
    def depth_cost_calc(self, inv_depth, fmap, fmaps_ref, pose_list, tgt_pose, K, scale_factor):
        cost_list = []
        mask_list = []
        for pose, fmap_r in zip(pose_list, fmaps_ref):
            cost = self.get_cost_each(tgt_pose, pose, fmap, fmap_r, utils.inv2depth(inv_depth), K, scale_factor)
            
            #mask_weight = torch.sigmoid(self.maskNet(fmap))
            #mask_list.append(mask_weight)
            #weighted_cost = cost * mask_weight #apply learned weighting
            cost_list.append(cost)  # (b, c,h, w) (1,64,144,192)
            
        # cost = torch.stack(cost_list, dim=1).min(dim=1)[0]
        #print(f"Cost each view : {[cost.mean().item() for cost in cost_list]}")
        #cost_stack = torch.stack(cost_list, dim=1)
        #mask_stack = torch.stack(mask_list, dim=1)
        #cost = (cost_stack.sum(dim=1) / (mask_stack.sum(dim=1) + 1e-6))  # Avoid division by zero
        cost = torch.stack(cost_list, dim=1).mean(dim=1)
        #print("cost mean", cost.mean())
        return cost

    def preprocess_batch(self, *args, keys, device=None):
        assert len(args) == len(keys)
        
        batch_size = args[0].shape[0]
        batch_inputs = []
        device = device or args[0].device

        for i in range(batch_size):
            sample = {key: arg[i].squeeze().cpu().numpy() for key, arg in zip(keys, args)}

            sample = self.ScaleMapLearner_transform(sample)
            
            x = torch.cat([torch.tensor(sample[key]) for key in keys], 0)
            x = x.to(device)
            batch_inputs.append(x)
        
        return torch.stack(batch_inputs, dim=0)

    def forward(self, tgt_img, ref_imgs, tgt_ga_depth, ref_ga_depth, 
                tgt_interp, ref_interp, tgt_pose, ref_pose, intrinsics):
        """
        Refine metric depth given ground truth poses and target/reference images.
        """

        #Step 1: Extract features using the base model
        #tgt_input = self.preprocess_batch(tgt_img, tgt_ga_depth, keys=["image", "int_depth"], device=tgt_img.device)
        ref_inputs = [
            self.preprocess_batch(ref_img, ref_depth, keys=["image", "ga_depth"], device=ref_img.device)
            for ref_img, ref_depth in zip(ref_imgs, ref_ga_depth)
        ]
        
        # context_input = self.preprocess_batch(tgt_ga_depth, tgt_interp, 
        #                                       keys=["int_depth", "int_scales"], device=tgt_ga_depth.device)
        
        processed_batch = self.preprocess_batch(
            tgt_img, tgt_ga_depth, tgt_ga_depth, tgt_interp, 
            keys=["image", "int_depth", "int_depth_c", "int_scales_c"], 
            device=tgt_img.device
        )
        tgt_input, context_input = torch.split(processed_batch, [4, 2], dim=1) #[1,4,288,384]
        #init_metric_depth_inv = context_input[:, 0:1, :, :]
        
        # Extract features using MidasNet #TODO: batch processing?
        tgt_feats = self.FeatExtractor(tgt_input) #[1, 64, 120, 160]
        ref_feats = [self.FeatExtractor(ref_input) for ref_input in ref_inputs] #[1, 64, 120, 160]
        context_feats = self.contextLearner(context_input) #[1, 64, 144, 192]
        
        #step2: get the init scaled depth from the model
        #metric_depth_inv_tgt, _ = self.ScaleMapLearner(x, d)
        scale_factor =  tgt_img.permute(0,3,1,2).shape[2] / tgt_feats.shape[2] #->convert to 160,192?
        
        #poses
        #pose_list_init = []
        
        #Intialize depth and poses
        #metric_depth_inv_tgt = tgt_ga_depth #[1, 480, 640] to [1,1,120,160]
        metric_depth_inv_tgt = F.interpolate(
            tgt_ga_depth.unsqueeze(1), size=(tgt_feats.shape[2], tgt_feats.shape[3]), mode='bicubic'
        )
        
        if not self.UseConvGRU:
            scale_map = self.scaleOutput(context_feats)
            delta_scales = F.relu(1.0 + scale_map)  # Ensure scale is positive
            inv_depth_pred = metric_depth_inv_tgt * delta_scales
            
            if self.min_pred is not None and self.max_pred is not None:
                inv_depth_pred = torch.clamp(
                    inv_depth_pred, 
                    min=1.0 / self.max_pred, 
                    max=1.0 / self.min_pred
                )  
                
            #apply scale to depth before computing cost
            tgt_pose = tgt_pose.detach()
            ref_pose = [pose.detach() for pose in ref_pose]
            depth_cost_map = self.depth_cost_calc(inv_depth_pred, tgt_feats, ref_feats, 
                                                  pose_list=ref_pose, tgt_pose=tgt_pose,
                                                  K=intrinsics, scale_factor=1.0/scale_factor)
            
            # refined_depth_inv = self.upsample_depth(
            #     inv_depth_pred,  # input depth [B, 1, H/ratio, W/ratio]
            #     up_mask_i,    # upsample mask [B, 9*ratio*ratio, H/ratio, W/ratio]
            #     ratio=int(scale_factor),       # upsampling ratio
            # )
            
            #TODO: propagate depth maybe SPN
                    
            refined_depth_inv = F.interpolate(
                inv_depth_pred,
                size=(tgt_img.shape[1], tgt_img.shape[2]),
                mode='bicubic',
                align_corners=False
            )  # shape => [B, 1, 480, 640]
                
            return depth_cost_map, refined_depth_inv
        
        #pose_list = pose_list_init
        inv_depth_predictions = [] #[metric_depth_inv_tgt] #to see the history of depth predictions
        
        # get optimization init
        if self.iter_steps > 0:
            context_feat = self.context_conv(context_feats) #to make the output dim = hidden_dim + cost_dim #[1,160,144,192]
            hidden_d, inp_d = torch.split(context_feat, [self.hidden_dim, self.cost_dim], dim=1)
            hidden_d = torch.tanh(hidden_d) #1,128,144,192
            inp_d = torch.relu(inp_d) #1,32,144,192
        
        # step2 compute cost map and optimize depth scale iteratively
        for itr in range(self.iter_steps):
            print("Iter: {}".format(itr))
            self.log_fn(f"Iter_{itr}/hidden_state_norm", hidden_d.norm().item(), on_step=True, logger=True)
            self.log_fn(f"Iter_{itr}/input_state_norm", inp_d.norm().item(), on_step=True, logger=True)
            
            metric_depth_inv_tgt = metric_depth_inv_tgt.detach()
            tgt_pose = tgt_pose.detach()
            ref_pose = [pose.detach() for pose in ref_pose]
            # --------------------------
            # 1. Update scale with ConvGRU
            # --------------------------
            depth_cost_map_func = partial(self.depth_cost_calc, 
                                        fmap=tgt_feats,
                                        fmaps_ref=ref_feats,
                                        pose_list=ref_pose,
                                        tgt_pose=tgt_pose,
                                        K=intrinsics,
                                        scale_factor=1.0/scale_factor)
            
            #update depth
            hidden_d, up_mask_seqs, inv_depth_seqs = self.update_block_depth(hidden_d, depth_cost_map_func,
                                                                             metric_depth_inv_tgt, inp_d,
                                                                             seq_len=5)
            
            #we won't supervise the intermediate predictions
            up_mask_seqs, inv_depth_seqs = [up_mask_seqs[-1]], [inv_depth_seqs[-1]]
            for up_mask_i, inv_depth_i in zip(up_mask_seqs, inv_depth_seqs):
                refined_depth_inv = self.upsample_depth(
                    inv_depth_i,  # input depth [B, 1, H/ratio, W/ratio]
                    up_mask_i,    # upsample mask [B, 9*ratio*ratio, H/ratio, W/ratio]
                    ratio=int(scale_factor),       # upsampling ratio
                )
                
                refined_depth_inv = F.interpolate(
                    refined_depth_inv,
                    size=(tgt_img.shape[1], tgt_img.shape[2]),
                    mode='bicubic',
                    align_corners=False
                )  # shape => [B, 1, 480, 640]
                
                inv_depth_predictions.append(refined_depth_inv)

                if self.log_fn:
                    self.log_fn(f"Iter_{itr}/depth_mean", refined_depth_inv.mean().item(), on_step=True, logger=True)
                    self.log_fn(f"Iter_{itr}/depth_var", refined_depth_inv.var().item(), on_step=True, logger=True)
                    
            metric_depth_inv_tgt = inv_depth_seqs[-1]
            
            # --------------------------
            # 2. Differentiable Pose Refinement
            # --------------------------
            # poses = self.gn_layer(
            #     features=tgt_feats,
            #     depth=utils.inv2depth(metric_depth_inv_tgt),
            #     poses=poses,
            #     K=intrinsics
            # )

        if self.is_train:
            return inv_depth_predictions
        else:
            return inv_depth_predictions[-1]

