import torch
import torch.nn as nn
from torch.nn import functional as F

from .base_model import BaseModel
from .blocks import FeatureFusionBlock_custom, _make_encoder, OutputConv
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

def weights_init(m):
    import math
    # initialize from normal (Gaussian) distribution
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

# class ResNetEncoder(models.ResNet):
#     """Constructs a resnet model with varying number of input images.
#     Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
#     """
#     def __init__(self, num_layers=18, num_input_images=1, pretrained=True, out_chs=32, stride=8):
#         layers = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
#         block = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]    
#         self.upsample_mode = "bilinear"
#         super(ResNetEncoder, self).__init__(block, layers)
        
#         self.inplanes = 64
#         self.conv1 = nn.Conv2d(
#             num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.stride = stride
        
#         # Modify layers based on stride setting
#         if stride == 8:
#             self.layer1 = self._make_layer(block, 64, layers[0], stride=2)   # (H/8, W/8)
#             self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # (H/16, W/16)
#             self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # (H/32, W/32)
        
#         elif stride == 4:
#             self.layer1 = self._make_layer(block, 64, layers[0], stride=1)   # (H/4, W/4)
#             self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # (H/8, W/8)
#             self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # (H/16, W/16)
        
#         elif stride == 2:
#             self.layer1 = self._make_layer(block, 64, layers[0], stride=1)   # (H/4, W/4)
#             self.layer2 = self._make_layer(block, 128, layers[1], stride=1)  # (H/4, W/4)
#             self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # (H/8, W/8)

#         else:
#             raise NotImplementedError("Stride should be 8, 4, or 2.")
        
#         # Upsampling layers
#         self.upconv1 = nn.Sequential(nn.Conv2d(256, 128, 3, 1, padding=1), nn.ReLU(inplace=True))
#         self.upconv1_fusion = nn.Sequential(nn.Conv2d(256, 128, 3, 1, padding=1), nn.ReLU(inplace=True))

#         self.upconv2 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, padding=1), nn.ReLU(inplace=True))
#         self.upconv2_fusion = nn.Sequential(nn.Conv2d(128, 64, 3, 1, padding=1), nn.ReLU(inplace=True))

#         self.upconv3 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, padding=1), nn.ReLU(inplace=True))
#         self.out_conv = nn.Conv2d(32, out_chs, 3, 1, padding=1)  # Final output (b, 32, H/2, W/2) if stride=2

#         del self.layer4
#         del self.fc
#         del self.avgpool

#         # Load pretrained weights if necessary
#         if pretrained:
#             loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
#             loaded['conv1.weight'] = torch.cat([loaded['conv1.weight']] * num_input_images, 1) / num_input_images
#             loaded_flilter = {k:v for k, v in loaded.items() if "layer4" not in k and "fc" not in k}
#             try:
#                 self.load_state_dict(loaded_flilter)
#             except Exception as e:
#                 self.load_state_dict(loaded_flilter, strict=False)
        
#     def forward(self, x):
#         feats = {}
#         # if input is list, combine batch dimension
#         is_list = isinstance(x, tuple) or isinstance(x, list)
#         if is_list:
#             num = len(x)
#             batch_dim = x[0].shape[0]
#             x = torch.cat(x, dim=0)

#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)  # (H/4, W/4)

#         x = self.layer1(x)
#         feats["s4"] = x  # (H/4, W/4)
#         x = self.layer2(x)
#         feats["s8"] = x  # (H/8, W/8)
#         x = self.layer3(x)
#         feats["s16"] = x  # (H/16, W/16)

#         # Upsampling based on stride setting
#         if self.stride == 8:
#             x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)  # (H/16 → H/8)
#             x = self.upconv1(x)
#             x = self.upconv1_fusion(torch.cat([x, feats["s8"]], dim=1))
#             x = self.out_conv(x)  # Final: (b, 32, H/8, W/8)

#         elif self.stride == 4:
#             x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)  # (H/16 → H/8)
#             x = self.upconv1(x)
#             x = self.upconv1_fusion(torch.cat([x, feats["s8"]], dim=1))

#             x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)  # (H/8 → H/4)
#             x = self.upconv2(x)
#             x = self.upconv2_fusion(torch.cat([x, feats["s4"]], dim=1))
            
#             x = self.out_conv(x)  # Final: (b, 32, H/4, W/4)

#         elif self.stride == 2:
#             x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)  # (H/16 → H/8)
#             x = self.upconv1(x)
#             x = self.upconv1_fusion(torch.cat([x, feats["s8"]], dim=1))

#             x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)  # (H/8 → H/4)
#             feats_s4_resized = F.interpolate(feats["s4"], size=x.shape[2:], mode=self.upsample_mode)  # Ensure same size
#             x = self.upconv2(x)
#             x = self.upconv2_fusion(torch.cat([x, feats_s4_resized], dim=1))

#             #x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)  # (H/4 → H/2)
#             x = self.upconv3(x)
#             #x = self.out_conv(x)  # Final: (b, 32, H/2, W/2)
            
#         if is_list:
#             x = torch.split(x, [batch_dim] * num, dim=0)

#         return x

# class ResNetEncoder(models.ResNet):
#     """Constructs a resnet model with varying number of input images.
#     Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
#     """
#     def __init__(self, num_layers=18, num_input_images=1, pretrained=True, out_chs=32, stride=8, context_num=None):
#         layers = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
#         block = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]    
#         self.upsample_mode = "bilinear"
#         super(ResNetEncoder, self).__init__(block, layers)
        
#         self.inplanes = 64
#         if context_num is not None:
#             self.conv1 = nn.Conv2d(
#                 context_num, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         else:
#             self.conv1 = nn.Conv2d(
#                 num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        
#         self.stride = stride
#         if stride == 8:
#             self.upconv1 = nn.Sequential(nn.Conv2d(256, 128, 3, 1, padding=1), nn.ReLU(inplace=True))
#             self.upconv1_fusion = nn.Sequential(nn.Conv2d(256, 128, 3, 1, padding=1), nn.ReLU(inplace=True))
#             self.out_conv = nn.Conv2d(128, out_chs, 3, 1, padding=1)
                
#         elif stride == 4:
#             self.upconv1 = nn.Sequential(nn.Conv2d(256, 128, 3, 1, padding=1), nn.ReLU(inplace=True))
#             self.upconv1_fusion = nn.Sequential(nn.Conv2d(256, 128, 3, 1, padding=1), nn.ReLU(inplace=True))
#             self.upconv2 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, padding=1), nn.ReLU(inplace=True))
#             self.upconv2_fusion = nn.Sequential(nn.Conv2d(128, 64, 3, 1, padding=1), nn.ReLU(inplace=True))
#             self.out_conv = nn.Conv2d(64, out_chs, 3, 1, padding=1)
            
#         else:
#             raise NotImplementedError 

#         # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         # del self.layer3
#         del self.layer4
#         del self.fc
#         del self.avgpool

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
        
#         if pretrained:
#             loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
#             loaded['conv1.weight'] = torch.cat([loaded['conv1.weight']] * num_input_images, 1) / num_input_images
#             loaded_flilter = {k:v for k, v in loaded.items() if "layer4" not in k and "fc" not in k}
#             try:
#                 print("load pretrained model from:", models.resnet.model_urls['resnet{}'.format(num_layers)])
#                 self.load_state_dict(loaded_flilter)
#             except Exception as e:
#                 # print(e)
#                 self.load_state_dict(loaded_flilter, strict=False)
        
#     def forward(self, x):
#         feats = {}
#         # if input is list, combine batch dimension
#         is_list = isinstance(x, tuple) or isinstance(x, list)
#         if is_list:
#             num = len(x)
#             batch_dim = x[0].shape[0]
#             x = torch.cat(x, dim=0)

#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         self.layer1_features = x # Store layer1 features
#         feats["s4"] = x
        
#         x = self.layer2(x)
#         self.layer2_features = x  # Store layer2 features
#         feats["s8"] = x
        
#         x = self.layer3(x)
#         self.layer3_features = x  # Store layer3 features
        
#         if self.stride == 8:
#             x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)
#             x = self.upconv1(x)
#             x = self.upconv1_fusion(torch.cat([x, feats["s8"]], dim=1))
#             x = self.out_conv(x)
            
#         elif self.stride == 4:
#             x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)
#             x = self.upconv1(x)
#             x = self.upconv1_fusion(torch.cat([x, feats["s8"]], dim=1)) 
            
#             x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)
#             x = self.upconv2(x)
#             x = self.upconv2_fusion(torch.cat([x, feats["s4"]], dim=1))
            
#             x = self.out_conv(x)
        
#         if is_list:
#             x = torch.split(x, [batch_dim] * num, dim=0)

#         return x

class ResNetEncoder(models.ResNet):
    """Constructs a resnet model with multiscale outputs (1/2, 1/4, 1/8).
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, num_layers=18, num_input_images=1, pretrained=True, out_chs=32, context_num=None):
        layers = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
        block = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]    
        self.upsample_mode = "bilinear"
        super(ResNetEncoder, self).__init__(block, layers)
        
        self.inplanes = 64
        if context_num is not None:
            self.conv1 = nn.Conv2d(
                context_num, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(
                num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        
        # Output convolutions for each scale
        # 1/8 scale (after layer3)
        self.out_conv_1_8 = nn.Conv2d(256, out_chs, 3, 1, padding=1)
        
        # 1/4 scale (after layer2)
        self.out_conv_1_4 = nn.Conv2d(128, out_chs, 3, 1, padding=1)
        
        # 1/2 scale (after layer1)
        self.out_conv_1_2 = nn.Conv2d(64, out_chs, 3, 1, padding=1)
        
        # Optional: If you want to process layer features before output
        self.refine_1_4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.refine_1_2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Delete unused layers
        del self.layer4
        del self.fc
        del self.avgpool

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Load pretrained weights if specified
        if pretrained:
            loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
            loaded['conv1.weight'] = torch.cat([loaded['conv1.weight']] * num_input_images, 1) / num_input_images
            loaded_filter = {k:v for k, v in loaded.items() if "layer4" not in k and "fc" not in k}
            try:
                print("load pretrained model from:", models.resnet.model_urls['resnet{}'.format(num_layers)])
                self.load_state_dict(loaded_filter)
            except Exception as e:
                # print(e)
                self.load_state_dict(loaded_filter, strict=False)
        
    def forward(self, x):
        # Create dictionary to store multiscale outputs
        multiscale_features = {}
        
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            num = len(x)
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        # Initial conv and pooling (1/4 scale)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Store features at 1/2 scale (before maxpool)
        features_1_2_pre = x
        
        x = self.maxpool(x)

        # Layer 1 (1/4 scale)
        x = self.layer1(x)
        features_1_4 = x
        
        # Layer 2 (1/8 scale)
        x = self.layer2(x)
        features_1_8_pre = x
        
        # Layer 3 (1/16 scale)
        x = self.layer3(x)
        
        # Process and store 1/8 scale features
        features_1_8 = self.out_conv_1_8(x)
        
        # Process and store 1/4 scale features
        features_1_4 = self.refine_1_4(features_1_8_pre)
        features_1_4 = self.out_conv_1_4(features_1_4)
        
        # Process and store 1/2 scale features
        features_1_2 = self.refine_1_2(features_1_2_pre)
        features_1_2 = self.out_conv_1_2(features_1_2)
        
        # Handle list input case
        if is_list:
            features_1_8 = torch.split(features_1_8, [batch_dim] * num, dim=0)
            features_1_4 = torch.split(features_1_4, [batch_dim] * num, dim=0)
            features_1_2 = torch.split(features_1_2, [batch_dim] * num, dim=0)
            return features_1_2, features_1_4, features_1_8
        
        # Return multiscale features
        return features_1_2, features_1_4, features_1_8
    
#Extract features from the backbone network
class MidasNet_small_cons_videpth(BaseModel):
    """Network for monocular depth estimation.
    """
    def __init__(self, device = 'cuda', path=None, features=64, backbone="efficientnet_lite3", 
                non_negative=False, exportable=True, channels_last=False, align_corners=True,
                blocks={'expand': True}, in_channels=4, regress='r', min_pred=None, 
                max_pred=None, output_downsample=None):
        super(MidasNet_small_cons_videpth, self).__init__()

        use_pretrained = False if path else True
                
        self.channels_last = channels_last
        self.blocks = blocks
        self.backbone = backbone
        self.output_downsample = output_downsample

        self.groups = 1

        # for model output
        self.regress = regress
        self.min_pred = min_pred
        self.max_pred = max_pred

        features1=features
        features2=features
        features3=features
        features4=features
        self.expand = False
        if "expand" in self.blocks and self.blocks['expand'] == True:
            self.expand = True
            features1=features
            features2=features*2
            features3=features*4
            features4=features*8

        self.first = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        self.first.apply(weights_init)

        self.pretrained, self.scratch = _make_encoder(self.backbone, features, 
                                                      use_pretrained, groups=self.groups, 
                                                      expand=self.expand, exportable=exportable)

        self.scratch.activation = nn.ReLU(False)    

        self.scratch.refinenet4 = FeatureFusionBlock_custom(features4, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet3 = FeatureFusionBlock_custom(features3, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet2 = FeatureFusionBlock_custom(features2, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet1 = FeatureFusionBlock_custom(features1, self.scratch.activation, deconv=False, bn=False, align_corners=align_corners)

        self.scratch_upsample = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, groups=self.groups),
            nn.Upsample(scale_factor=2, mode="bilinear"),
        )
        #self.scratch.output_conv = OutputConv(features, self.groups, self.scratch.activation, non_negative)

        #self.scale_map_learner = ScaleMapLearner(input_channels=features1 + in_channels, hidden_channels=features, output_channels=1)

        if path:
            self.load(path)
    
    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (rgb img, ga depth)
            f (tensor): features

        Returns:
            tensor: depth
        """
        if self.channels_last==True:
            #print("self.channels_last = ", self.channels_last)
            x.contiguous(memory_format=torch.channels_last)
        
        layer_0 = self.first(x)

        layer_1 = self.pretrained.layer1(layer_0)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)
        
        # Apply refinement layers:
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        if self.output_downsample:
            path_1 = self.scratch_upsample(path_1)
        # if self.output_downsample:
        #     target_h = 120
        #     target_w = 160
        #     path_1 = F.interpolate(path_1, size=(target_h, target_w),
        #                             mode='bicubic', align_corners=self.scratch.refinenet1.align_corners)
        return path_1 #[1, 64, 120, 160]
        