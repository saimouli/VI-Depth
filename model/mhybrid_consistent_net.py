import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if module_path not in sys.path:
    sys.path.append(module_path)
from modules.midas.midas_net_cons_custom import MidasNet_small_cons_videpth, ResNetEncoder
import numpy as np
import modules.midas.utils as utils
import modules.midas.transforms as transforms
from utils.camera import Camera, pose_to_se3, se3_to_pose, se3_update
#import pypose as pp 
from utils.pose import Pose
from functools import partial
#from pytorch3d.transforms import se3_exp_map, se3_log_map
import torchvision
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
    def __init__(self, cost_dim, hidden_dim, out_chs, downsample_ratio=2):
        super().__init__()
        self.out_chs = out_chs
        self.convc1 = nn.Conv2d(cost_dim, hidden_dim, 1, padding=0)
        self.convc2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        
        self.convd1 = nn.Conv2d(1, hidden_dim, 7, padding=3)
        self.convd2 = nn.Conv2d(hidden_dim, 64, 3, padding=1)
        
        self.convd = nn.Conv2d(64+hidden_dim, out_chs - 1, 3, padding=1)
        
        # Downsample layer
        if downsample_ratio > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(out_chs, out_chs, kernel_size=3, stride=downsample_ratio, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.downsample = nn.Identity()
        
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
        out = torch.cat([out_d, depth], dim=1)
        return self.downsample(out)

class ProjectionInputPose(nn.Module):
    def __init__(self, cost_dim, hidden_dim, out_chs, downsample_ratio=8):
        super().__init__()
        self.out_chs = out_chs
        self.convc1 = nn.Conv2d(cost_dim, hidden_dim, 1, padding=0)
        self.convc2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
               
        self.convp1 = nn.Conv2d(6, hidden_dim, 7, padding=3)
        self.convp2 = nn.Conv2d(hidden_dim, 64, 3, padding=1)
        
        self.convp = nn.Conv2d(64+hidden_dim, out_chs - 6, 3, padding=1)
        # Downsample layer
        if downsample_ratio > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(out_chs, out_chs, kernel_size=3, stride=downsample_ratio, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.downsample = nn.Identity()
            
    def forward(self, pose, cost):
        bs, _, h, w = cost.shape
        cor = F.relu(self.convc1(cost))
        cor = F.relu(self.convc2(cor))
        
        pose = pose.view(bs, 6, 1, 1).repeat(1, 1, h, w)
        pfm = F.relu(self.convp1(pose))
        pfm = F.relu(self.convp2(pfm))
        cor_pfm = torch.cat([cor, pfm], dim=1)
                
        out_p = F.relu(self.convp(cor_pfm))
        out= torch.cat([out_p, pose], dim=1)
        return self.downsample(out)
    
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
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(features//2, 32, kernel_size=3, stride=1, padding=1),
            activation,
            
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            activation,
            
            # Final 1x1 conv to get single-channel scale map
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            # Ensure positive scale factors
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

    def forward(self, x):        
        return self.output_conv(x)
    
class BasicUpdateBlockDepth(nn.Module):
    def __init__(self, hidden_dim=128, cost_dim=64, ratio=3, context_dim=64, min_pred=None, max_pred=None, log_fn=None):
        super(BasicUpdateBlockDepth, self).__init__()
                
        self.encoder = ProjectionInputDepth(cost_dim=1, hidden_dim=hidden_dim, out_chs=hidden_dim, downsample_ratio=1)
        self.depth_gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=self.encoder.out_chs+context_dim)
        self.depth_head = OutputScaleConv(features=hidden_dim, groups=1, activation=nn.ReLU(False), non_negative=False)
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
        cost_means = []
        for i in range(seq_len):
            cost, _ = cost_func(inv_depth)
            #print("cost {}: mean{}".format(i, cost.mean()))
            
            # If necessary, downsample inv_depth and context to match the cost resolution.
            if cost.shape[-2:] != inv_depth.shape[-2:]:
                inv_depth_low = F.interpolate(inv_depth, size=cost.shape[-2:], mode='bilinear', align_corners=True)
            else:
                inv_depth_low = inv_depth
                
            if self.log_fn:
                with torch.no_grad():
                    self.log_fn(f"UpdateBlock/Iteration_{i}/cost_mean", cost.mean().item(), on_step=True, logger=True)
                    cost_means.append(cost.mean().item())
                
            input_features = self.encoder(inv_depth_low, cost) #(b,1,72,96),(b,32,72,96) 
            inp_i = torch.cat([context, input_features], dim=1)#(2,32,72,96),(2,128,288,384) 

            #if self.log_fn:
            #    self.log_fn(f"UpdateBlock/Iteration_{i}/hidden_state_norm_before", hidden.norm().item(), on_step=True, logger=True)
                
            hidden = self.depth_gru(hidden, inp_i)
            
            #if self.log_fn:
            #    self.log_fn(f"UpdateBlock/Iteration_{i}/hidden_state_norm_after", hidden.norm().item(), on_step=True, logger=True)
                
            delta_scales = self.depth_head(hidden)
            delta_scales = 1.0 + 0.5 * torch.tanh(delta_scales)  # [0.5, 1.5] range #ensure positive scale
            #print(f"inv_depth mean step {i}: {inv_depth.mean().item()}")
            #print("delta_scales mean step", i, delta_scales.mean().item())
            
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
            
             # *** Update for next iteration ***
            inv_depth = inv_depth_pred
            # scale mask to balence gradients
            mask = 0.25 * self.mask(hidden) #helps numerical stability
            
            inv_depth_list.append(inv_depth_pred)
            mask_list.append(mask)
            
        return hidden, mask_list, inv_depth_list, cost_means
    
class PoseHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128):
        super(PoseHead, self).__init__()
        
        self.conv1_pose = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2_pose = nn.Conv2d(hidden_dim, 6, 3, padding=1)

        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x_p):
        out = self.conv2_pose(self.relu(self.conv1_pose(x_p))).mean(3).mean(2)
        return torch.cat([out[:, :3], 0.1 * out[:, 3:]], dim=1)
    
class BasicUpdateBlockPose(nn.Module):
    def __init__(self, hidden_dim=128, cost_dim=64, ratio=3, context_dim=64, log_fn=None):
        super(BasicUpdateBlockPose, self).__init__()
        self.encoder = ProjectionInputPose(cost_dim=1, hidden_dim=hidden_dim, out_chs=hidden_dim, downsample_ratio=2)
        self.pose_gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=self.encoder.out_chs+context_dim)
        self.pose_head = PoseHead(hidden_dim, hidden_dim=hidden_dim)
        
    def forward(self, hidden_p, cost_func, ref_pose, inp, seq_len=4, depth_only=False):
        pose_list = []
        #convert to ref_pose from 4x4 to 6x1
        ref_pose_init = pose_to_se3(ref_pose)
        for i in range(seq_len):
            res = cost_func(poseC2W=ref_pose)
            cost = res['cost']
            input_features = self.encoder(ref_pose_init, cost)
            inp_i = torch.cat([inp, input_features], dim=1) #(b, 32, 36, 48)
                
            hidden_p = self.pose_gru(hidden_p, inp_i)
            delta_pose = self.pose_head(hidden_p)
            
            if depth_only:
                pose = ref_pose
            else:
                pose = ref_pose @ se3_to_pose(delta_pose)
            pose_list.append(pose)
        return hidden_p, pose_list
class UpMaskNet(nn.Module):
    def __init__(self, hidden_dim=128, ratio=8):
        super(UpMaskNet, self).__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim*2, ratio*ratio*9, 1, padding=0))

    def forward(self, feat):
        # scale mask to balence gradients
        mask = .25 * self.mask(feat)
        return mask
    
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

        self.hidden_dim = 128 #96
        self.cost_dim = 64 #32
        self.iter_steps = 3
        self.seq_len = 3
        
        # Feature extractor (frozen)
        # self.FeatExtractor = MidasNet_small_cons_videpth(
        #     path=sml_model_path,
        #     in_channels=3,
        #     features=32,
        #     min_pred=self.min_pred,
        #     max_pred=self.max_pred,
        #     output_downsample=True,
        #     backbone="efficientnet_lite3",
        # )
        # for param in self.FeatExtractor.parameters():
        #     param.requires_grad = False
        # self.FeatExtractor.eval()
        
        self.fnet = ResNetEncoder(out_chs=self.cost_dim, stride=4)
        self.cnet_depth = ResNetEncoder(out_chs=self.hidden_dim + self.cost_dim, stride=4, context_num=2, pretrained=False)
    
        # self.contextLearner = MidasNet_small_cons_videpth(
        #     in_channels=2,
        #     features=64,
        #     path=sml_model_path,
        #     min_pred=self.min_pred,
        #     max_pred=self.max_pred,
        #     output_downsample=False,
        #     backbone="efficientnet_lite3",
        # )
        # self.contextLearner.train()
        
        # self.context_conv = nn.Conv2d(
        #     in_channels=64,
        #     out_channels=self.hidden_dim + self.cost_dim, 
        #     kernel_size=3, stride=1, padding=1
        # )
        #self.refine_net = DepthPoseRefineNet(hidden_dim=self.hidden_dim)
        
        if self.UseConvGRU:
            self.contextPose = ResNetEncoder(out_chs=self.hidden_dim+self.cost_dim, 
                                           stride=8, num_input_images=2)#pose
            
            self.update_block_depth = BasicUpdateBlockDepth(hidden_dim=self.hidden_dim, 
                                                            cost_dim=self.cost_dim,
                                                            ratio=3, 
                                                            context_dim=self.cost_dim,
                                                            min_pred=self.min_pred,
                                                            max_pred=self.max_pred,
                                                            log_fn=self.log_fn)
            
            self.update_block_pose = BasicUpdateBlockPose(hidden_dim=self.hidden_dim,
                                                          cost_dim=self.cost_dim,
                                                          ratio=3,
                                                          context_dim=self.cost_dim,
                                                          )
            self.inter_sup = False
            
        else:
            pass
            #self.refine_net = DepthPoseRefineNet(hidden_dim=self.hidden_dim)
            
        self.scaleOutput = OutputScaleConv(features=self.hidden_dim + self.cost_dim, groups=1, 
                                            activation=nn.ReLU(False), non_negative=False)

    def freeze_pose_branch(self, freeze=True):
        """
        Freeze or unfreeze the pose refinement branch of the model
        """
        if freeze:
            # Freeze pose update block and related components
            for param in self.update_block_pose.parameters():
                param.requires_grad = False
            
            # Also freeze contextPose if using ConvGRU
            for param in self.contextPose.parameters():
                param.requires_grad = False
        else:
            # Unfreeze pose update block and related components
            for param in self.update_block_pose.parameters():
                param.requires_grad = True
            
            # Also unfreeze contextPose if using ConvGRU
            for param in self.contextPose.parameters():
                param.requires_grad = True
                
                
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

    def clam_depth(self, inv_depth_pred):
        if self.min_pred is not None and self.max_pred is not None:
            inv_depth_pred = torch.clamp(
                inv_depth_pred, 
                min=1.0 / self.max_pred, 
                max=1.0 / self.min_pred
            )
        return inv_depth_pred

    def get_cost_each(self, tgt_poseC2W, poseC2W, fmap, fmap_ref, depth, K, scale_factor):
        """
            ga_depth: (b, 1, h, w)
            fmap, fmap_ref: (b, c, h, w)
        """
        device = depth.device
        ref_cam = Camera(K=K.float(), Twc=poseC2W).scaled(scale_factor).to(device)
        cam = Camera(K=K.float(), Twc=tgt_poseC2W).scaled(scale_factor).to(device) # tcw = Identity
        #resize depth to same size as the feature map
        depth_small = F.interpolate(depth, size=(fmap.shape[2], fmap.shape[3]), mode='bilinear', align_corners=False)
        # Reconstruct world points from target_camera
        world_points = cam.reconstruct(depth_small, frame='w')
        # Project world points onto reference camera
        ref_coords = ref_cam.project(world_points, frame='w', normalize=True) #(b, h, w,2)
        with torch.no_grad():
           valid_mask = (ref_coords.abs().max(dim=-1)[0] <= 1).float()  # [B, H, W]
           
        fmap_warped = F.grid_sample(fmap_ref, ref_coords, 
                                    mode='bilinear', padding_mode='zeros', align_corners=True) # (b, c, h, w)
        
        #cost = (fmap - fmap_warped)**2 * valid_mask.unsqueeze(1) #cost = (fmap * fmap_warped).sum(dim=1, keepdim=True) #try correlation
        #cost = cost.mean(dim=1, keepdim=True)
        fmap_norm = F.normalize(fmap, p=2, dim=1)
        fmap_warped_norm = F.normalize(fmap_warped, p=2, dim=1)
        correlation = (fmap_norm * fmap_warped_norm).sum(dim=1, keepdim=True)
        cost = (1.0 - correlation) * valid_mask.unsqueeze(1)

        return {
            'cost': cost,
            'fmap': fmap,
            'fmap_warped': fmap_warped,
            'valid_mask': valid_mask
        }
    
    def depth_cost_calc(self, inv_depth, fmap, fmaps_ref, pose_list, tgt_pose, K, scale_factor):
        cost_list = []
        warping_vis = []
        for idx, (pose, fmap_r) in enumerate(zip(pose_list, fmaps_ref)):
            result = self.get_cost_each(tgt_pose, pose, fmap, fmap_r, 
                                      utils.inv2depth(inv_depth), K, scale_factor)
            
            if idx == 0 and self.is_train:
                warping_vis.append({
                    'src_feat': result['fmap'][0].detach(),
                    'warped_feat': result['fmap_warped'][0].detach(),
                    'valid_mask': result['valid_mask'][0].detach(),
                    'cost': result['cost'][0].detach()
                })

            cost_list.append(result['cost'])  # (b, c,h, w) (1,64,144,192)
        
        cost = torch.stack(cost_list, dim=1).mean(dim=1)
        return cost, warping_vis

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
                tgt_interp, ref_interp, tgt_pose, ref_pose, intrinsics, depth_only=False):
        """
        Refine metric depth scale and VIO poses and target/reference images.
        """
        #Step 1: Extract features using the base model
        ref_inputs = [
            self.preprocess_batch(ref_img, keys=["image"], device=ref_img.device)
            for ref_img in ref_imgs
        ]
        
        processed_batch = self.preprocess_batch(
            tgt_img, tgt_ga_depth, tgt_ga_depth, tgt_interp, 
            keys=["image", "int_depth", "int_depth_c", "int_scales_c"], 
            device=tgt_img.device
        )
        tgt_input, context_depth_input = torch.split(processed_batch, [4, 2], dim=1) #[1,4,288,384]
        init_metric_depth_inv = context_depth_input[:, 0:1, :, :] #Ga_depth
        
        #test feat extraction only with the rgb image
        tgt_input = processed_batch[:, :3, :, :] # Shape: [1, 3, H, W]
        
        # Extract features using MidasNet #TODO: batch processing?
        #tgt_feats = self.FeatExtractor(tgt_input) #[1, 32, 288, 384]
        #ref_feats = [self.FeatExtractor(ref_input) for ref_input in ref_inputs] #[1, 32, 288, 384]
        fmaps = self.fnet(torch.cat([tgt_input] + ref_inputs, dim=0))
        fmaps = torch.split(fmaps, [tgt_input.shape[0]] * (1 + len(ref_inputs)), dim=0)
        tgt_feats, ref_feats = fmaps[0], fmaps[1:] #[1,32,384/4,384/4]
        
        #context_feats = self.contextLearner(context_depth_input) #[1, 64, 144, 192] ga_depth + interp_scale
        context_feats = self.cnet_depth(context_depth_input) # (1, 128, 72, 96)
        #step2: get the init scaled depth from the model
        #metric_depth_inv_tgt, _ = self.ScaleMapLearner(x, d)
        scale_factor =  tgt_img.permute(0,3,1,2).shape[2] / tgt_feats.shape[2]
        
        # Step 3: get initial depth and pose
        scale_map = self.scaleOutput(context_feats)
        delta_scales = F.relu(1.0 + scale_map)  # Ensure scale is positive
        inv_depth_pred = init_metric_depth_inv * delta_scales
            
        if self.min_pred is not None and self.max_pred is not None:
            inv_depth_pred = torch.clamp(
                inv_depth_pred, 
                min=1.0 / self.max_pred, 
                max=1.0 / self.min_pred
            )
            
        refined_inv_depth = inv_depth_pred #[b, 1, 288, 384]
        
        depth_init_up = F.interpolate(
                refined_inv_depth,
                size=(tgt_img.shape[1], tgt_img.shape[2]),
                mode='bicubic',
                align_corners=False
        )  # shape => [B, 1, 480, 640]
        
        #fixed_tgt_pose = tgt_pose.detach().requires_grad_(False) #T_vio_cam2wld
        #refined_rel_poses = [pose_to_se3(fixed_tgt_pose.inverse() @ ref_p) for ref_p in ref_pose] #T_relative 6D
        # initial pose from VIO
        pose_list_init = ref_pose
        
        # -------- ConvGRU-Based Iterative Refinement --------            
        inv_depth_predictions = [depth_init_up] #[metric_depth_inv_tgt] #to see the history of depth predictions
        pose_predictions = [[pose_to_se3((pose).clone()) for pose in pose_list_init]] #to see the history of pose predictions
            
        # get optimization init
        if self.iter_steps > 0:
            #context_feat = self.context_conv(context_feats) #to make the output dim = hidden_dim + cost_dim #[1,160,144,192]
            hidden_d, inp_d = torch.split(context_feats, [self.hidden_dim, self.cost_dim], dim=1)
            hidden_d = torch.tanh(hidden_d) #1,128,144,192
            inp_d = torch.relu(inp_d) #1,32,144,192
                
            img_pairs = []
            for ref_img in ref_inputs:
                img_pairs.append(torch.cat([tgt_input, ref_img], dim=1))
            cnet_pose_list = self.contextPose(img_pairs)
            hidden_p_list, inp_p_list = [], []
            for cnet_pose in cnet_pose_list:
                hidden_p, inp_p = torch.split(cnet_pose, [self.hidden_dim, self.cost_dim], dim=1)
                hidden_p_list.append(torch.tanh(hidden_p))
                inp_p_list.append(torch.relu(inp_p))
            
        pose_list = pose_list_init
            
        # Step2: compute cost map and optimize depth scale iteratively
        for itr in range(self.iter_steps):
            #print("Iter: {}".format(itr))
            with torch.no_grad():
                self.log_fn(f"Iter_{itr}/hidden_state_norm", hidden_d.norm().item(), on_step=True, logger=True)
                self.log_fn(f"Iter_{itr}/input_state_norm", inp_d.norm().item(), on_step=True, logger=True)
                
            # Detach tensors to avoid backprop through refinement history
            refined_inv_depth = refined_inv_depth.detach()
            #ref_abspose_list = [fixed_tgt_pose @ se3_to_pose(pose).detach() for pose in refined_rel_poses] #(1,6) (1,6)
            pose_list = [pose.detach() for pose in pose_list]
            #refined_rel_poses = [pose.detach() for pose in refined_rel_poses] #relative pose
            #ref_abspose_list = [fixed_tgt_pose @ se3_to_pose(pose) for pose in refined_rel_poses] #4x4 absolute poses
                
            # ----------------------------
            # 1. Update scale with ConvGRU
            # ----------------------------
            depth_cost_map_func = partial(self.depth_cost_calc, 
                                        fmap=tgt_feats,
                                        fmaps_ref=ref_feats,
                                        pose_list=pose_list,
                                        tgt_pose=None,
                                        K=intrinsics,
                                        scale_factor=1.0/scale_factor)
                
            #update depth #TODO: check and understand this function
            hidden_d, up_mask_seqs, inv_depth_seqs, cost_means = self.update_block_depth(hidden_d, depth_cost_map_func,
                                                                            refined_inv_depth, inp_d,
                                                                            seq_len=self.seq_len)
                
            #we won't supervise the intermediate predictions
            #up_mask_seqs, inv_depth_seqs = [up_mask_seqs[-1]], [inv_depth_seqs[-1]]
                
            #upsample
            # for up_mask_i, inv_depth_i in zip(up_mask_seqs, inv_depth_seqs):
            #     refined_depth_inv = self.upsample_depth(
            #         inv_depth_i,  # input depth [B, 1, H/ratio, W/ratio]
            #         up_mask_i,    # upsample mask [B, 9*ratio*ratio, H/ratio, W/ratio]
            #         ratio=int(scale_factor),       # upsampling ratio
            #     )
                    
            refined_depth_inv = F.interpolate(
                inv_depth_seqs[-1],
                size=(tgt_img.shape[1], tgt_img.shape[2]),
                mode='bicubic',
                align_corners=False
            )  # shape => [B, 1, 480, 640]
                    
            inv_depth_predictions.append(refined_depth_inv)

            if self.log_fn:
                with torch.no_grad():
                    print(f"Iter {itr} cost progression: {cost_means}")
                    self.log_fn(f"Iter_{itr}/depth_mean", refined_depth_inv.mean().item(), on_step=True, logger=True)
                    self.log_fn(f"Iter_{itr}/depth_var", refined_depth_inv.var().item(), on_step=True, logger=True)
                        
            refined_inv_depth = inv_depth_seqs[-1]
                
            #### update pose using the updated depth ####
            # calc cost
            pose_cost_func_list = []
            for fmap_ref in ref_feats:
                pose_cost_func_list.append(partial(self.get_cost_each, tgt_poseC2W=None,fmap=tgt_feats, 
                                                       fmap_ref=fmap_ref,
                                                       depth=utils.inv2depth(refined_inv_depth),
                                                       K=intrinsics, scale_factor=1.0/scale_factor))
                
            pose_list_seqs = [None] * len(pose_list)
            for i, (ref_pose, hidden_p) in enumerate(zip(pose_list, hidden_p_list)):
                hidden_p, pose_seqs = self.update_block_pose(hidden_p, pose_cost_func_list[i],
                                                            ref_pose, inp_p_list[i], seq_len=self.seq_len, 
                                                            depth_only=depth_only)
                hidden_p_list[i] = hidden_p
                if not self.inter_sup:
                    pose_seqs = [pose_seqs[-1]] #take final iteration
                pose_list_seqs[i] = pose_seqs
                
            for pose_list_i in zip(*pose_list_seqs):
                pose_predictions.append([pose_to_se3(pose.clone()) for pose in pose_list_i])
                
            # Convert updated poses to relative
            pose_list = list(zip(*pose_list_seqs))[-1]
            #pose_list = [se3_to_pose(pose) for pose in pose_list]
                
            #ref_abspose_list = list(zip(*pose_list_seqs))[-1]
             #refined_rel_poses = [pose_to_se3(fixed_tgt_pose.inverse() @ ref_p) for ref_p in ref_abspose_list]
                
        if self.is_train:
            return inv_depth_predictions, \
                torch.stack([torch.stack(poses_ref, dim=1) for poses_ref in pose_predictions], dim=2) #(b, n, iters, 6)
        else:
            return inv_depth_predictions[-1],\
                torch.stack(pose_predictions[-1], dim=1).view(tgt_img.shape[0], len(ref_imgs), 6) #(b, n, 6)

