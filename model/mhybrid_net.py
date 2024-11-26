import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.midas.midas_net_custom import MidasNet_small_videpth
import numpy as np
import modules.midas.utils as utils
import modules.midas.transforms as transforms
import utils.log_utils as log_utils
from utils.common_op import resize_and_pad
from modules.estimator import LeastSquaresEstimator
from modules.interpolator import Interpolator2D

class midasNet(nn.Module):
    def __init__(self, min_pred, max_pred, min_depth, max_depth, nsamples, sml_model_path):
        super(midasNet, self).__init__()
        #self.invert_depth = invert_depth
        self.midas_model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        #self.midas_model.load("model-f6b98070.pt")
        self.midas_model.requires_grad_(False)
        self.midas_model.eval()

        self.min_pred, self.max_pred = min_pred, max_pred
        self.min_depth, self.max_depth = min_depth, max_depth

        model_transforms = transforms.get_transforms("dpt_hybrid", "void", str(nsamples))
        self.depth_model_transform = model_transforms["depth_model"]
        self.ScaleMapLearner_transform = model_transforms["sml_model"]

        self.ScaleMapLearner = MidasNet_small_videpth(
            path=sml_model_path,
            min_pred=self.min_pred,
            max_pred=self.max_pred,
            backbone="efficientnet_lite3",
        )
        self.ScaleMapLearner.train()
    
    def forward(self, input_sparse_depth, input_image, depth_pred, validity_map):
        #input_height, input_width = np.shape(input_image)[0], np.shape(input_image)[1]
        input_sparse_depth_valid = (input_sparse_depth < self.max_depth) * (input_sparse_depth > self.min_depth)
        if validity_map is not None:
            input_sparse_depth_valid *= validity_map.astype(np.bool)
        
        input_sparse_depth_valid = input_sparse_depth_valid.to(torch.bool)
        input_sparse_depth[~input_sparse_depth_valid] = np.inf # set invalid depth
        input_sparse_depth = 1.0 / input_sparse_depth

        # global scale and shift alignment
        GlobalAlignment = LeastSquaresEstimator(
            estimate=depth_pred,
            target=input_sparse_depth,
            valid=input_sparse_depth_valid.requires_grad_(False),
        )
        GlobalAlignment.compute_scale_and_shift()
        GlobalAlignment.apply_scale_and_shift()
        GlobalAlignment.clamp_min_max(clamp_min=self.min_pred, clamp_max=self.max_pred)
        int_depth = GlobalAlignment.output.astype(np.float32)

        # interpolation of scale map
        assert (np.sum(input_sparse_depth_valid) >= 3), "not enough valid sparse points"
        ScaleMapInterpolator = Interpolator2D(
            pred_inv = int_depth,
            sparse_depth_inv = input_sparse_depth,
            valid = input_sparse_depth_valid,
        )
        ScaleMapInterpolator.generate_interpolated_scale_map(
            interpolate_method='linear', 
            fill_corners=False
        )
        int_scales = ScaleMapInterpolator.interpolated_scale_map.astype(np.float32)
        int_scales = utils.normalize_unit_range(int_scales)

        sample = {"image" : input_image, 
                  "int_depth" : int_depth, #LS aligned depth
                  "int_scales" : int_scales, #interpolated scale
                  "int_depth_no_tf" : int_depth}
        sample = self.ScaleMapLearner_transform(sample)
        x = torch.cat([sample["int_depth"], sample["int_scales"]], 0) #TODO 1 or 0?
        d = sample["int_depth_no_tf"]

        ## run SML model
        metric_depth, sml_scales = self.ScaleMapLearner.forward(x.unsqueeze(0), d.unsqueeze(0))

        mask = torch.logical_and(metric_depth > 0,metric_depth < 8)
        return metric_depth, int_depth, mask