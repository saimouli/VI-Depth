import numpy as np
import torch, torchvision

def compute_loss(image,
                 output_depth, 
                 ground_truth,
                 loss_func,
                 w_smoothness,
                 loss_smoothness_kernel_size):
    
    loss = 0.0
    loss_supervised = 0.0
    loss_smoothness = 0.0
    
    validity_map_ground_truth = ground_truth > 0
    
    if not isinstance(output_depth, list):
        output_depth = [output_depth]
    
    for scale, output in enumerate(output_depth):
        output_height, output_width = output.shape[-2:]
        target_height, target_width = ground_truth.shape[-2:]
        
        if output_height > target_height and output_width > target_width:
            output = torch.nn.functional.interpolate(
                output,
                size=(target_height, target_width),
                mode='bicubic',
                align_corners=False)
        
        w_scale = 1.0 / (2 ** (len(output_depth) - scale - 1))
        
        if loss_func == 'l1':
            loss_supervised = loss_supervised + w_scale * l1_loss(
                output[validity_map_ground_truth],
                ground_truth[validity_map_ground_truth])
    
        elif loss_func == 'l2':
            loss_supervised = loss_supervised + w_scale * l2_loss(
                output[validity_map_ground_truth],
                ground_truth[validity_map_ground_truth])
            
        elif loss_func == 'smoothl1':
            loss_supervised = loss_supervised + w_scale * smooth_l1_loss(
                output[validity_map_ground_truth],
                ground_truth[validity_map_ground_truth])
            
        else:
            raise ValueError(f'Unknown loss function: {loss_func}')
        
        if w_smoothness > 0.0:
            if loss_smoothness_kernel_size <= 1:
                loss_smoothness = loss_smoothness + w_scale * smoothness_loss_func(
                    image=image,
                    predict=output)
                
        #     else:
        #         loss_smoothness_kernel_size = \
        #             [1, 1, loss_smoothness_kernel_size, loss_smoothness_kernel_size]

        #         loss_smoothness = loss_smoothness + w_scale * sobel_smoothness_loss_func(
        #             image=image,
        #             predict=output,
        #             filter_size=loss_smoothness_kernel_size)
        
        loss = loss_supervised + w_smoothness * loss_smoothness
        
        loss_info = {
            'loss': loss,
            'loss_supervised': loss_supervised,
            'loss_smoothness': loss_smoothness
        }
        
        return loss, loss_info
        
def smooth_l1_loss(src, tgt):
    '''
    Computes smooth_l1 loss

    Arg(s):
        src : torch.Tensor[float32]
            N x 3 x H x W source image
        tgt : torch.Tensor[float32]
            N x 3 x H x W target image
    Returns:
        float : mean smooth l1 loss across batch
    '''

    return torch.nn.functional.smooth_l1_loss(src, tgt, reduction='mean')

def l1_loss(src, tgt):
    '''
    Computes l1 loss

    Arg(s):
        src : torch.Tensor[float32]
            N x 3 x H x W source image
        tgt : torch.Tensor[float32]
            N x 3 x H x W target image
    Returns:
        float : mean l1 loss across batch
    '''

    return torch.nn.functional.l1_loss(src, tgt, reduction='mean')

def l2_loss(src, tgt):
    '''
    Computes l2 loss

    Arg(s):
        src : torch.Tensor[float32]
            N x 3 x H x W source image
        tgt : torch.Tensor[float32]
            N x 3 x H x W target image
    Returns:
        float : mean l2 loss across batch
    '''

    return torch.nn.functional.mse_loss(src, tgt, reduction='mean')

def smoothness_loss_func(predict, image):
    '''
    Computes the local smoothness loss

    Arg(s):
        predict : tensor
            N x 1 x H x W predictions
        image : tensor
            N x 3 x H x W RGB image
        w : tensor
            N x 1 x H x W weights
    Returns:
        tensor : smoothness loss
    '''

    predict_dy, predict_dx = gradient_yx(predict)
    image_dy, image_dx = gradient_yx(image)

    # Create edge awareness weights
    weights_x = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

    smoothness_x = torch.mean(weights_x * torch.abs(predict_dx))
    smoothness_y = torch.mean(weights_y * torch.abs(predict_dy))

    return smoothness_x + smoothness_y

def sobel_smoothness_loss_func(predict, image, weights, filter_size=[1, 1, 7, 7]):
    '''
    Computes the local smoothness loss using sobel filter

    Arg(s):
        predict : tensor
            N x 1 x H x W predictions
        image : tensor
            N x 3 x H x W RGB image
        w : tensor
            N x 1 x H x W weights
    Returns:
        tensor : smoothness loss
    '''

    device = predict.device

    predict = torch.nn.functional.pad(
        predict,
        (filter_size[-1]//2, filter_size[-1]//2, filter_size[-2]//2, filter_size[-2]//2),
        mode='replicate')

    gx, gy = sobel_filter(filter_size)
    gx = gx.to(device)
    gy = gy.to(device)

    predict_dy = torch.nn.functional.conv2d(predict, gy)
    predict_dx = torch.nn.functional.conv2d(predict, gx)
    if image.shape[1] == 3:
        image = image[:, 0, :, :] * 0.30 + image[:, 1, :, :] * 0.59 + image[:, 2, :, :] * 0.11
        image = torch.unsqueeze(image, 1)
    image = torch.nn.functional.pad(image, (1, 1, 1, 1), mode='replicate')

    gx_i, gy_i = sobel_filter([1, 1, 3, 3])
    gx_i = gx_i.to(device)
    gy_i = gy_i.to(device)

    image_dy = torch.nn.functional.conv2d(image, gy_i)
    image_dx = torch.nn.functional.conv2d(image, gx_i)

    # Create edge awareness weights
    weights_x = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

    smoothness_x = torch.mean(weights * weights_x * torch.abs(predict_dx))
    smoothness_y = torch.mean(weights * weights_y * torch.abs(predict_dy))

    return (smoothness_x + smoothness_y) / float(filter_size[-1] * filter_size[-2])


'''
Helper functions for constructing loss functions
'''
def gradient_yx(T):
    '''
    Computes gradients in the y and x directions

    Arg(s):
        T : tensor
            N x C x H x W tensor
    Returns:
        tensor : gradients in y direction
        tensor : gradients in x direction
    '''

    dx = T[:, :, :, :-1] - T[:, :, :, 1:]
    dy = T[:, :, :-1, :] - T[:, :, 1:, :]
    return dy, dx

def sobel_filter(filter_size=[1, 1, 3, 3]):
    Gx = torch.ones(filter_size)
    Gy = torch.ones(filter_size)

    Gx[:, :, :, filter_size[-1] // 2] = 0
    Gx[:, :, (filter_size[-2] // 2), filter_size[-1] // 2 - 1] = 2
    Gx[:, :, (filter_size[-2] // 2), filter_size[-1] // 2 + 1] = 2
    Gx[:, :, :, filter_size[-1] // 2:] = -1*Gx[:, :, :, filter_size[-1] // 2:]

    Gy[:, :, filter_size[-2] // 2, :] = 0
    Gy[:, :, filter_size[-2] // 2 - 1, filter_size[-1] // 2] = 2
    Gy[:, :, filter_size[-2] // 2 + 1, filter_size[-1] // 2] = 2
    Gy[:, :, filter_size[-2] // 2+1:, :] = -1*Gy[:, :, filter_size[-2] // 2+1:, :]

    return Gx, Gy