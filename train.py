import os, datetime
import cv2
import numpy as np
import torch, torchvision
from torch.utils.tensorboard import SummaryWriter

from modules.midas.midas_net_custom import MidasNet_small_videpth
import modules.midas.transforms as transforms
import modules.midas.utils as utils
from modules.interpolator import Interpolator2D

import utils.log_utils as log_utils
from utils.loss import compute_loss
from utils_eval import compute_ls_solution
from data.SML_dataset import SML_dataset

import time

def infer_depth(DepthModel, device, input_image, depth_model_transform):
    DepthModel.eval()
    DepthModel.to(device)
    
    input_height, input_width = input_image.shape[:2]
    
    sample = {"image" : input_image}
    sample = depth_model_transform(sample)
    im = sample["image"].to(device)
    
    with torch.no_grad():
        depth_pred = DepthModel.forward(im.unsqueeze(0))
        depth_pred = (
            torch.nn.functional.interpolate(
                depth_pred.unsqueeze(1),
                size=(input_height, input_width),
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
    return depth_pred
    

def train(
        # data input
        train_dataset_path,
        
        # training
        learning_rates,
        learning_schedule,
        batch_size,
        n_step_summary,
        n_step_per_checkpoint,
        
        # loss
        loss_func,
        w_smoothness,
        loss_smoothness_kernel_size,
        
        # model
        chkpt_path,
        min_pred_depth,
        max_pred_depth,
        checkpoint_dir,
        n_threads,
        DepthModel,
    ):
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    depth_model_checkpoint_path = os.path.join(checkpoint_dir, 'model-{}.pth')
    log_path = os.path.join(checkpoint_dir, 'results.txt')
    event_path = os.path.join(checkpoint_dir, 'events')
    
    log_utils.log_params(log_path, locals())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(f"{train_dataset_path}/train_image.txt") as f: 
        train_image_list = [train_dataset_path + "/" + line.rstrip() for line in f]
    
    train_gt_depth_list = [image_path.replace('image', 'ground_truth') for image_path in train_image_list]
    sparse_depth_list = [image_path.replace('image', 'sparse_depth') for image_path in train_image_list]
    
    n_train_sample = len(train_image_list)
    n_train_step = learning_schedule[-1] * np.ceil(n_train_sample / batch_size).astype(np.int32)
    
        
    train_dataloader = torch.utils.data.DataLoader(
        SML_dataset(
            image_paths = train_image_list,
            gt_depth_paths = train_gt_depth_list,
            sparse_paths = sparse_depth_list,
            depth_scale = 1000.0,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_threads)
    
    # transform
    model_transforms = transforms.get_transforms('dpt_hybrid', 'void', '150')
    depth_model_transform = model_transforms["depth_model"]
    ScaleMapLearner_transform = model_transforms["sml_model"]
    
    # build SML model
    ScaleMapLearner = MidasNet_small_videpth(
        device = device,
        min_pred = min_pred_depth,
        max_pred = max_pred_depth,
    )
    
    '''
    Train model
    '''
    # init optim with learning rate
    learning_schedule_pos = 0
    learning_rate = learning_rates[0]
    
    # Initialize optimizer with starting learning rate
    parameters_model = list(ScaleMapLearner.parameters())
    optimizer = torch.optim.Adam([
        {
            'params': parameters_model
        }],
        lr=learning_rate)
    
    # Set up tensorboard summary writers
    train_summary_writer = SummaryWriter(event_path + '-train')
    
    # Start training
    train_step = 0
    
    if chkpt_path is not None and chkpt_path != '':
        ScaleMapLearner.load(chkpt_path)
    
    for g in optimizer.param_groups:
        g['lr'] = learning_rate
    
    time_start = time.time()
    
    print('Start training', log_path)
    for epoch in range(1, learning_schedule[-1] + 1):
        print('Epoch', epoch)
        
        #set learning rate
        if epoch > learning_schedule[learning_schedule_pos]:
            learning_schedule_pos = learning_schedule_pos + 1
            learning_rate = learning_rates[learning_schedule_pos]
            
            #update learning rate of all optimizers
            for g in optimizer.param_groups:
                g['lr'] = learning_rate
        
        # train the mode
        for batch_data in train_dataloader:
            train_step += 1
            batch_data = [
                in_.to(device) for in_ in batch_data
            ]
            
            image, gt_depth, sparse_depth = batch_data
            
            # sparse depth
            sparse_depth_valid = (sparse_depth < max_pred_depth) * (sparse_depth > min_pred_depth)
            sparse_depth_valid = sparse_depth_valid.bool()
            sparse_depth[~sparse_depth_valid] = np.inf
            sparse_depth = 1.0 / sparse_depth
            
            batch_size = sparse_depth.shape[0]
            
            # each time empty batch
            batch_x = []; batch_d = []; batch_image = []; batch_gt = []; batch_sparse = []
            
            for i in range(batch_size):
                sparse_depth_i = sparse_depth[i].squeeze().cpu().numpy()
                sparse_depth_valid_i = sparse_depth_valid[i].squeeze().cpu().numpy()
                depth_pred_i = infer_depth(DepthModel, device, image[i].squeeze().cpu().numpy(), depth_model_transform)
                
                int_depth_i,_,_ = compute_ls_solution(depth_pred_i, sparse_depth_i, sparse_depth_valid_i, min_pred_depth, max_pred_depth)
                ScaleMapInterpolator = Interpolator2D(
                    pred_inv = int_depth_i,
                    sparse_depth_inv = sparse_depth_i,
                    valid = sparse_depth_valid_i,
                )
                ScaleMapInterpolator.generate_interpolated_scale_map(
                    interpolate_method='linear', 
                    fill_corners=False
                )
                int_scales_i = ScaleMapInterpolator.interpolated_scale_map.astype(np.float32)
                int_scales_i = utils.normalize_unit_range(int_scales_i)
        
                sample = {
                    'image': image[i].squeeze().cpu().numpy(),
                    'gt_depth': gt_depth[i].squeeze().cpu().numpy(),
                    'sparse_depth': sparse_depth[i].squeeze().cpu().numpy(),
                    'int_depth': int_depth_i,
                    'int_scales': int_scales_i,
                    'int_depth_no_tf': int_depth_i}
                
                sample = ScaleMapLearner_transform(sample)
                
                x = torch.cat((sample['int_depth'], sample['int_scales']), dim=0)
                x = x.to(device)
                d = sample['int_depth_no_tf'].to(device)
                batch_x.append(x)
                batch_d.append(d)
                batch_image.append(sample['image'].to(device))
                batch_gt.append(sample['gt_depth'].to(device))
                batch_sparse.append(sample['sparse_depth'].to(device))
            
            x = torch.stack(batch_x, 0)
            d = torch.stack(batch_d, 0)
            batch_image = torch.stack(batch_image, 0)
            batch_gt = torch.stack(batch_gt, 0)
            batch_sparse = torch.stack(batch_sparse, 0)
            
            # perform forward pass
            sml_pred, sml_scales = ScaleMapLearner.forward(x, d)
            # inverse depth to depth
            d = 1.0 / d
            sml_pred = 1.0 / sml_pred
            
            # Compute loss function
            validity_map_loss_smoothness = torch.where(
                batch_gt > 0,
                torch.zeros_like(batch_gt),
                torch.ones_like(batch_gt))
            
            loss, loss_info = compute_loss(
                image=d,
                output_depth=sml_pred,
                ground_truth=batch_gt,
                loss_func=loss_func,
                w_smoothness=w_smoothness,
                loss_smoothness_kernel_size=loss_smoothness_kernel_size
            )
            
            print('{}/{} epoch:{}: {}'.format(train_step % n_train_step, n_train_step, epoch, loss.item()))
            
            # Compute gradient and backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (train_step % n_step_summary) == 0:
                with torch.no_grad():
                    log_summary(
                        summary_writer=train_summary_writer,
                        tag='train',
                        step=train_step,
                        max_predict_depth=max_pred_depth,
                        image=batch_image,
                        input_depth=d,
                        output_depth=sml_pred,
                        ground_truth=batch_gt,
                        scalars=loss_info,
                        n_display=min(4, batch_size))
            
            if (train_step % n_step_per_checkpoint) == 0:
                time_elapse = (time.time() - time_start) / 3600
                time_remain = (n_train_step - train_step) * time_elapse / train_step
                
                print('Step={:6}/{} Loss={:.5f} Time Elapsed={:.2f}h Time Remaining={:.2f}h'.format(
                    train_step, n_train_step, loss.item(), time_elapse, time_remain), log_path)
                # Save chkpt
                ScaleMapLearner.save(depth_model_checkpoint_path.format(train_step))
    
    # save checkpoints
    ScaleMapLearner.save(depth_model_checkpoint_path.format(train_step))
            
def log_summary(summary_writer,
                tag,
                step,
                max_predict_depth,
                image=None,
                input_depth=None,
                input_response=None,
                output_depth=None,
                ground_truth=None,
                scalars={},
                n_display=4):

    with torch.no_grad():

        display_summary_image = []
        display_summary_depth = []

        display_summary_image_text = tag
        display_summary_depth_text = tag

        if image is not None:
            image_summary = image[0:n_display, ...]

            display_summary_image_text += '_image'
            display_summary_depth_text += '_image'

            # Add to list of images to log
            display_summary_image.append(
                torch.cat([
                    image_summary.cpu(),
                    torch.zeros_like(image_summary, device=torch.device('cpu'))],
                    dim=-1))

            display_summary_depth.append(display_summary_image[-1])

        if output_depth is not None:
            output_depth_summary = output_depth[0:n_display, ...]

            display_summary_depth_text += '_output_depth'

            # Add to list of images to log
            n_batch, _, n_height, n_width = output_depth_summary.shape

            display_summary_depth.append(
                torch.cat([
                    log_utils.colorize(
                        (output_depth_summary / max_predict_depth).cpu(),
                        colormap='viridis'),
                    torch.zeros(n_batch, 3, n_height, n_width, device=torch.device('cpu'))],
                    dim=3))

            # Log distribution of output depth
            summary_writer.add_histogram(tag + '_output_depth_distro', output_depth, global_step=step)

        if output_depth is not None and input_depth is not None:
            input_depth_summary = input_depth[0:n_display, ...]

            display_summary_depth_text += '_input_depth-error'

            # Compute output error w.r.t. input depth
            input_depth_error_summary = \
                torch.abs(output_depth_summary - input_depth_summary)

            input_depth_error_summary = torch.where(
                input_depth_summary > 0.0,
                input_depth_error_summary / (input_depth_summary + 1e-8),
                input_depth_summary)

            # Add to list of images to log
            input_depth_summary = log_utils.colorize(
                (input_depth_summary / max_predict_depth).cpu(),
                colormap='viridis')
            input_depth_error_summary = log_utils.colorize(
                (input_depth_error_summary / 0.05).cpu(),
                colormap='inferno')

            display_summary_depth.append(
                torch.cat([
                    input_depth_summary,
                    input_depth_error_summary],
                    dim=3))

            # Log distribution of input depth
            summary_writer.add_histogram(tag + '_input_depth_distro', input_depth, global_step=step)




        if output_depth is not None and input_response is not None:
            response_summary = input_response[0:n_display, ...]

            display_summary_depth_text += '_response'

            # Add to list of images to log
            response_summary = log_utils.colorize(
                response_summary.cpu(),
                colormap='inferno')

            display_summary_depth.append(
                torch.cat([
                    response_summary,
                    torch.zeros_like(response_summary)],
                    dim=3))

            # Log distribution of input depth
            summary_writer.add_histogram(tag + '_response_distro', input_depth, global_step=step)




        if output_depth is not None and ground_truth is not None:
            ground_truth = ground_truth[0:n_display, ...]
            ground_truth = torch.unsqueeze(ground_truth[:, 0, :, :], dim=1)

            ground_truth_summary = ground_truth[0:n_display]
            validity_map_summary = torch.where(
                ground_truth > 0,
                torch.ones_like(ground_truth),
                torch.zeros_like(ground_truth))

            display_summary_depth_text += '_ground_truth-error'

            # Compute output error w.r.t. ground truth
            ground_truth_error_summary = \
                torch.abs(output_depth_summary - ground_truth_summary)

            ground_truth_error_summary = torch.where(
                validity_map_summary == 1.0,
                (ground_truth_error_summary + 1e-8) / (ground_truth_summary + 1e-8),
                validity_map_summary)

            # Add to list of images to log
            ground_truth_summary = log_utils.colorize(
                (ground_truth_summary / max_predict_depth).cpu(),
                colormap='viridis')
            ground_truth_error_summary = log_utils.colorize(
                (ground_truth_error_summary / 0.05).cpu(),
                colormap='inferno')

            display_summary_depth.append(
                torch.cat([
                    ground_truth_summary,
                    ground_truth_error_summary],
                    dim=3))

            # Log distribution of ground truth
            summary_writer.add_histogram(tag + '_ground_truth_distro', ground_truth, global_step=step)

        # Log scalars to tensorboard
        for (name, value) in scalars.items():
            summary_writer.add_scalar(tag + '_' + name, value, global_step=step)

        # Log image summaries to tensorboard
        if len(display_summary_image) > 1:
            display_summary_image = torch.cat(display_summary_image, dim=2)

            summary_writer.add_image(
                display_summary_image_text,
                torchvision.utils.make_grid(display_summary_image, nrow=n_display),
                global_step=step)

        if len(display_summary_depth) > 1:
            display_summary_depth = torch.cat(display_summary_depth, dim=2)

            summary_writer.add_image(
                display_summary_depth_text,
                torchvision.utils.make_grid(display_summary_depth, nrow=n_display),
                global_step=step)
            
if __name__ == '__main__':
    train_root = '/home/rpng/datasets/splat_vins/table1'
    result_root = '/home/rpng/datasets/splat_vins/table1/results'
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    image_path = os.path.join(train_root, 'image')
    gt_path = os.path.join(train_root, 'ground_truth')
    sparse_depth_path = os.path.join(train_root, 'sparse_depth')    
    DepthModel = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
    
    train(
        # data load
        train_dataset_path = train_root,
        
        # train params
        learning_rates = [2e-4,1e-4],
        learning_schedule = [20,80],
        batch_size = 4,
        n_step_summary = 5,
        n_step_per_checkpoint = 100,
        
        # loss settings
        loss_func = 'smoothl1',
        w_smoothness = 0.0,
        loss_smoothness_kernel_size = -1,
        
        # model
        chkpt_path = '/home/rpng/Documents/sai_ws/splat_vins_repos_test/VI-Depth/weights/sml_model.dpredictor.dpt_hybrid.nsamples.150.ckpt',
        min_pred_depth = 0.1,
        max_pred_depth = 8.0,
        checkpoint_dir = os.path.join(result_root, 'checkpoints', current_time),
        n_threads = 3,
        DepthModel = DepthModel,
    )