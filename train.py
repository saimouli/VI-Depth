import argparse
import os, datetime
from tqdm import tqdm
import cv2
import numpy as np
import torch, torchvision
from torch.utils.tensorboard import SummaryWriter

from modules.midas.midas_net_custom import MidasNet_small_videpth
import modules.midas.transforms as transforms
import modules.midas.utils as utils
from modules.interpolator import Interpolator2D

import utils.log_utils as log_utils
from utils.common_op import resize_and_pad
from utils.loss import compute_loss, compute_consistency_loss  # , compute_metric_loss
from utils_eval import compute_ls_solution
from data.SML_dataset import SML_dataset
from data.SML_consistent_dataset import SML_consistent_dataset

import time
import matplotlib.pyplot as plt


def infer_depth(DepthModel, device, input_image, depth_model_transform):
    DepthModel.eval()
    DepthModel.to(device)

    input_height, input_width = input_image.shape[:2]

    sample = {"image": input_image}
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


def train_scale_consistency(
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

    depth_model_checkpoint_path = os.path.join(checkpoint_dir, "model-{}.pth")
    log_path = os.path.join(checkpoint_dir, "results.txt")
    event_path = os.path.join(checkpoint_dir, "events")

    log_utils.log_params(log_path, locals())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = SML_consistent_dataset(root=train_dataset_path)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=n_threads, drop_last=True
    )

    n_train_sample = len(train_dataset)
    n_train_step = learning_schedule[-1] * np.ceil(n_train_sample / batch_size).astype(
        np.int32
    )

    # n_train_step = learning_schedule[-1] * np.ceil(n_train_sample / batch_size).astype(np.int32)

    # transform
    model_transforms = transforms.get_transforms("dpt_hybrid", "void", "150")
    depth_model_transform = model_transforms["depth_model"]
    ScaleMapLearner_transform = model_transforms["sml_model"]

    # build SML model
    ScaleMapLearner = MidasNet_small_videpth(
        device=device,
        min_pred=min_pred_depth,
        max_pred=max_pred_depth,
    )

    """
    Train model
    """
    # init optim with learning rate
    learning_schedule_pos = 0
    learning_rate = learning_rates[0]

    # Initialize optimizer with starting learning rate
    parameters_model = list(ScaleMapLearner.parameters())
    optimizer = torch.optim.Adam([{"params": parameters_model}], lr=learning_rate)

    # Set up tensorboard summary writers
    train_summary_writer = SummaryWriter(event_path + "-train")

    if chkpt_path is not None and chkpt_path != "":
        ScaleMapLearner.load(chkpt_path)

    for g in optimizer.param_groups:
        g["lr"] = learning_rate

    time_start = time.time()

    print("Start training", log_path)
    for epoch in range(1, learning_schedule[-1] + 1):
        print("Epoch", epoch)

        # set learning rate
        if epoch > learning_schedule[learning_schedule_pos]:
            learning_schedule_pos = learning_schedule_pos + 1
            learning_rate = learning_rates[learning_schedule_pos]

            # update learning rate of all optimizers
            for g in optimizer.param_groups:
                g["lr"] = learning_rate

        # train the mode
        # tgt_img, tgt_gt_depth, tgt_ga_depth, tgt_interp, ref_img, \
        #             ref_ga_depth, ref_interp, ref_gt_depth, tgt_pose, ref_pose, intrinsics
        for train_step, batch_data in tqdm(enumerate(train_dataloader)):
            # Move each element in the batch to the device
            batch_data = [
                item.to(device) if torch.is_tensor(item) else item
                for item in batch_data
            ]

            (
                tgt_img,
                tgt_gt_depth,
                tgt_ga_depth,
                tgt_interp,
                ref_img,
                ref_ga_depth,
                ref_interp,
                ref_gt_depth,
                tgt_pose,
                ref_pose,
                intrinsics,
            ) = batch_data

            ref_img = [img.to(device) for img in ref_img]

            ## visualize the tgt_img, and ref_im
            tgt_img_test_1 = tgt_img[0].squeeze().cpu().numpy()
            tgt_img_test_2 = tgt_img[1].squeeze().cpu().numpy()

            ref_img0_test_1 = ref_img[0][0].squeeze().cpu().numpy()
            ref_img0_test_2 = ref_img[0][1].squeeze().cpu().numpy()

            ref_img1_test_1 = ref_img[1][0].squeeze().cpu().numpy()
            ref_img1_test_2 = ref_img[1][1].squeeze().cpu().numpy()

            # plt.figure(1)
            # plt.subplot(1, 3, 1)
            # plt.title("Tgt Img 1")
            # plt.imshow(tgt_img_test_1)
            # plt.subplot(1, 3, 2)
            # plt.title("ref0 Img 1")
            # plt.imshow(ref_img0_test_1)
            # plt.subplot(1, 3, 3)
            # plt.title("ref0 Img 2")
            # plt.imshow(ref_img0_test_2)

            # plt.figure(2)
            # plt.subplot(1, 3, 1)
            # plt.title("Tgt Img 2")
            # plt.imshow(tgt_img_test_2)
            # plt.subplot(1, 3, 2)
            # plt.title("ref1 Img 1")
            # plt.imshow(ref_img1_test_1)
            # plt.subplot(1, 3, 3)
            # plt.title("ref1 Img 2")
            # plt.imshow(ref_img1_test_2)
            # plt.show()

            # #visualize for each batch
            # for i in range(batch_size):
            #     tgt_img_test = tgt_img_test[i].squeeze()
            #     ref_img0_test = ref_img0_test[i].squeeze()
            #     ref_img1_test = ref_img1_test[i].squeeze()
            #     plt.figure()
            #     plt.title("Tgt Img")
            #     plt.imshow(tgt_img_test) #1
            #     plt.figure()
            #     plt.title("Ref 0 img")
            #     plt.imshow(ref_img0_test) #0
            #     plt.figure()
            #     plt.title("Ref 1 img")
            #     plt.imshow(ref_img1_test) #2
            #     plt.show()

            # TODO: below code assumes sequence of 3 modify to be general
            # each time empty batch
            batch_x_tgt = []
            batch_d_tgt = []
            batch_image_tgt = []
            batch_gt_tgt = []
            batch_x_ref0 = []
            batch_d_ref0 = []
            batch_image_ref0 = []
            batch_gt_ref0 = []
            batch_x_ref1 = []
            batch_d_ref1 = []
            batch_image_ref1 = []
            batch_gt_ref1 = []

            # ref0 (t-1), tgt_img (t), ref1 (t+1)
            for i in range(batch_size):
                sample_tgt = {
                    "image": tgt_img[i].squeeze().cpu().numpy(),
                    "gt_depth": tgt_gt_depth[i].squeeze().cpu().numpy(),
                    "int_depth": tgt_ga_depth[i].squeeze().cpu().numpy(),
                    "int_scales": tgt_interp[i].squeeze().cpu().numpy(),
                    "int_depth_no_tf": tgt_ga_depth[i].squeeze().cpu().numpy(),
                }

                sample_ref0 = {
                    "image": ref_img[0][i].squeeze().cpu().numpy(),
                    "gt_depth": ref_gt_depth[0][i].squeeze().cpu().numpy(),
                    "int_depth": ref_ga_depth[0][i].squeeze().cpu().numpy(),
                    "int_scales": ref_interp[0][i].squeeze().cpu().numpy(),
                    "int_depth_no_tf": ref_ga_depth[0][i].squeeze().cpu().numpy(),
                }

                sample_ref1 = {
                    "image": ref_img[1][i].squeeze().cpu().numpy(),
                    "gt_depth": ref_gt_depth[1][i].squeeze().cpu().numpy(),
                    "int_depth": ref_ga_depth[1][i].squeeze().cpu().numpy(),
                    "int_scales": ref_interp[1][i].squeeze().cpu().numpy(),
                    "int_depth_no_tf": ref_ga_depth[1][i].squeeze().cpu().numpy(),
                }

                sample_tgt = ScaleMapLearner_transform(sample_tgt)
                sample_ref0 = ScaleMapLearner_transform(sample_ref0)
                sample_ref1 = ScaleMapLearner_transform(sample_ref1)

                x_tgt = torch.cat(
                    (sample_tgt["int_depth"], sample_tgt["int_scales"]), dim=0
                )
                x_tgt = x_tgt.to(device)
                d_tgt = sample_tgt["int_depth_no_tf"].to(device)

                x_ref0 = torch.cat(
                    (sample_ref0["int_depth"], sample_ref0["int_scales"]), dim=0
                )
                x_ref0 = x_ref0.to(device)
                d_ref0 = sample_ref0["int_depth_no_tf"].to(device)

                x_ref1 = torch.cat(
                    (sample_ref1["int_depth"], sample_ref1["int_scales"]), dim=0
                )
                x_ref1 = x_ref1.to(device)
                d_ref1 = sample_ref1["int_depth_no_tf"].to(device)

                batch_x_tgt.append(x_tgt)
                batch_d_tgt.append(d_tgt)
                batch_image_tgt.append(sample_tgt["image"].to(device))
                batch_gt_tgt.append(sample_tgt["gt_depth"].to(device))

                batch_x_ref0.append(x_ref0)
                batch_d_ref0.append(d_ref0)
                batch_image_ref0.append(sample_ref0["image"].to(device))
                batch_gt_ref0.append(sample_ref0["gt_depth"].to(device))

                batch_x_ref1.append(x_ref1)
                batch_d_ref1.append(d_ref1)
                batch_image_ref1.append(sample_ref1["image"].to(device))
                batch_gt_ref1.append(sample_ref1["gt_depth"].to(device))

            x_tgt = torch.stack(batch_x_tgt, 0)
            x_ref0 = torch.stack(batch_x_ref0, 0)
            x_ref1 = torch.stack(batch_x_ref1, 0)

            d_tgt = torch.stack(batch_d_tgt, 0)
            d_ref0 = torch.stack(batch_d_ref0, 0)
            d_ref1 = torch.stack(batch_d_ref1, 0)

            batch_image_ref0_sml = torch.stack(batch_image_ref0, 0)
            batch_gt_ref0_sml = torch.stack(batch_gt_ref0, 0)
            batch_image_ref1_sml = torch.stack(batch_image_ref1, 0)
            batch_gt_ref1_sml = torch.stack(batch_gt_ref1, 0)
            batch_image_tgt_sml = torch.stack(batch_image_tgt, 0)
            batch_gt_tgt_sml = torch.stack(batch_gt_tgt, 0)
            # batch_image_tgt = torch.stack(batch_image_tgt, 0)
            # batch_gt_tgt = torch.stack(batch_gt_tgt, 0)
            # batch_image_ref = torch.stack(batch_image_ref0 + batch_image_ref1, dim=0)
            batch_gt_ref = torch.stack(batch_gt_ref0 + batch_gt_ref1, dim=0)

            # Perform forward pass
            sml_pred_tgt, sml_scales = ScaleMapLearner.forward(x_tgt, d_tgt)
            sml_pred_ref0, sml_scales_ref0 = ScaleMapLearner.forward(x_ref0, d_ref0)
            sml_pred_ref1, sml_scales_ref1 = ScaleMapLearner.forward(x_ref1, d_ref1)

            # Inverse depth to depth
            d_tgt, d_ref0, d_ref1 = 1.0 / d_tgt, 1.0 / d_ref0, 1.0 / d_ref1
            sml_pred_tgt, sml_pred_ref0, sml_pred_ref1 = (
                1.0 / sml_pred_tgt,
                1.0 / sml_pred_ref0,
                1.0 / sml_pred_ref1,
            )

            ref_output_depth = []
            ref_output_depth.append(sml_pred_ref0)
            ref_output_depth.append(sml_pred_ref1)
            tgt_output_depth = sml_pred_tgt

            # get poses from tgt to ref0, tgt to ref1
            pose_CttoG = tgt_pose.to(device)
            pose_Cref0toG = ref_pose[0].to(device)
            pose_Cref1toG = ref_pose[1].to(device)
            aux_mat = (
                torch.tensor([0, 0, 0, 1])
                .type_as(pose_CttoG)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, 1, 1)
            )
            pose_CttoG = torch.cat((pose_CttoG, aux_mat), dim=1)
            pose_Cref0toG = torch.cat((pose_Cref0toG, aux_mat), dim=1)
            pose_Cref1toG = torch.cat((pose_Cref1toG, aux_mat), dim=1)

            pose_GtoCref0 = torch.inverse(pose_Cref0toG)
            pose_GtoCref1 = torch.inverse(pose_Cref1toG)

            pose_CttoRef0 = torch.matmul(pose_GtoCref0, pose_CttoG)
            pose_CttoRef1 = torch.matmul(pose_GtoCref1, pose_CttoG)
            poses = [pose_CttoRef0, pose_CttoRef1]

            pose_CttoRef0_inv = torch.inverse(pose_CttoRef0)
            pose_CttoRef1_inv = torch.inverse(pose_CttoRef1)
            poses_inv = [pose_CttoRef0_inv, pose_CttoRef1_inv]

            # TODO: make sure the output depth sizes match
            ref_img_resize = [
                resize_and_pad(
                    img, (ref_output_depth[0].shape[-2], ref_output_depth[0].shape[-1])
                ).permute(0, 3, 1, 2)
                for img in ref_img
            ]
            tgt_img_resize = resize_and_pad(
                tgt_img, (tgt_output_depth.shape[-2], tgt_output_depth.shape[-1])
            ).permute(0, 3, 1, 2)
            # correct the batches before passing to the loss function
            photo_loss, geomentry_loss = compute_consistency_loss(
                ref_img_resize,
                tgt_img_resize,
                batch_gt_tgt,
                batch_gt_ref,  # gt_depth_tgt, gt_depth_ref
                poses,
                poses_inv,
                ref_output_depth,
                tgt_output_depth,
                intrinsics,
                loss_func,
                w_smoothness,
                loss_smoothness_kernel_size,
            )

            # prepare inputs
            # image = [batch_image_tgt, batch_image_ref]
            # output_depth = [sml_pred_tgt, sml_pred_ref0, sml_pred_ref1]
            # gt_depths = [batch_gt_tgt, batch_gt_ref]

            metric_loss_ref0, _ = compute_loss(
                batch_image_ref0_sml,
                sml_pred_ref0,
                batch_gt_ref0_sml,
                loss_func,
                w_smoothness,
                loss_smoothness_kernel_size,
            )

            metric_loss_ref1, _ = compute_loss(
                batch_image_ref1_sml,
                sml_pred_ref0,
                batch_gt_ref1_sml,
                loss_func,
                w_smoothness,
                loss_smoothness_kernel_size,
            )

            metric_loss_tgt, _ = compute_loss(
                batch_image_tgt_sml,
                sml_pred_tgt,
                batch_gt_tgt_sml,
                loss_func,
                w_smoothness,
                loss_smoothness_kernel_size,
            )
            metric_loss = metric_loss_ref0 + metric_loss_ref1 + metric_loss_tgt

            loss = 0.0 * photo_loss + 0.3 * geomentry_loss + 0.7 * metric_loss

            # print('{}/{} epoch:{}: {}'.format(train_step % n_train_step, n_train_step, epoch, loss.item()))

            # Compute gradient and backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (train_step % n_step_per_checkpoint) == 0:
                time_elapse = (time.time() - time_start) / 3600
                time_remain = (n_train_step - train_step) * time_elapse / train_step

                print(
                    "Step={:6}/{} Loss={:.5f} Time Elapsed={:.2f}h Time Remaining={:.2f}h".format(
                        train_step, n_train_step, loss.item(), time_elapse, time_remain
                    ),
                    log_path,
                )
                # Save chkpt
                ScaleMapLearner.save(depth_model_checkpoint_path.format(train_step))

    # save checkpoints
    ScaleMapLearner.save(depth_model_checkpoint_path.format(train_step))


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

    depth_model_checkpoint_path = os.path.join(checkpoint_dir, "model-{}.pth")
    log_path = os.path.join(checkpoint_dir, "results.txt")
    event_path = os.path.join(checkpoint_dir, "events")

    log_utils.log_params(log_path, locals())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(f"{train_dataset_path}/train_image.txt") as f:
        train_image_list = [train_dataset_path + "/" + line.rstrip() for line in f]

    train_gt_depth_list = [
        image_path.replace("image", "ground_truth") for image_path in train_image_list
    ]
    sparse_depth_list = [
        image_path.replace("image", "sparse_depth") for image_path in train_image_list
    ]

    n_train_sample = len(train_image_list)
    n_train_step = learning_schedule[-1] * np.ceil(n_train_sample / batch_size).astype(
        np.int32
    )

    train_dataloader = torch.utils.data.DataLoader(
        SML_dataset(
            image_paths=train_image_list,
            gt_depth_paths=train_gt_depth_list,
            sparse_paths=sparse_depth_list,
            depth_scale=1000.0,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_threads,
    )

    # transform
    model_transforms = transforms.get_transforms("dpt_hybrid", "void", "150")
    depth_model_transform = model_transforms["depth_model"]
    ScaleMapLearner_transform = model_transforms["sml_model"]

    # build SML model
    ScaleMapLearner = MidasNet_small_videpth(
        device=device,
        min_pred=min_pred_depth,
        max_pred=max_pred_depth,
    )

    """
    Train model
    """
    # init optim with learning rate
    learning_schedule_pos = 0
    learning_rate = learning_rates[0]

    # Initialize optimizer with starting learning rate
    parameters_model = list(ScaleMapLearner.parameters())
    optimizer = torch.optim.Adam([{"params": parameters_model}], lr=learning_rate)

    # Set up tensorboard summary writers
    train_summary_writer = SummaryWriter(event_path + "-train")

    # Start training
    train_step = 0

    if chkpt_path is not None and chkpt_path != "":
        ScaleMapLearner.load(chkpt_path)

    for g in optimizer.param_groups:
        g["lr"] = learning_rate

    time_start = time.time()

    print("Start training", log_path)
    for epoch in range(1, learning_schedule[-1] + 1):
        print("Epoch", epoch)

        # set learning rate
        if epoch > learning_schedule[learning_schedule_pos]:
            learning_schedule_pos = learning_schedule_pos + 1
            learning_rate = learning_rates[learning_schedule_pos]

            # update learning rate of all optimizers
            for g in optimizer.param_groups:
                g["lr"] = learning_rate

        # train the mode
        for batch_data in train_dataloader:
            train_step += 1
            batch_data = [in_.to(device) for in_ in batch_data]

            image, gt_depth, sparse_depth = batch_data

            # sparse depth
            sparse_depth_valid = (sparse_depth < max_pred_depth) * (
                sparse_depth > min_pred_depth
            )
            sparse_depth_valid = sparse_depth_valid.bool()
            sparse_depth[~sparse_depth_valid] = np.inf
            sparse_depth = 1.0 / sparse_depth

            batch_size = sparse_depth.shape[0]

            # each time empty batch
            batch_x = []
            batch_d = []
            batch_image = []
            batch_gt = []
            batch_sparse = []

            for i in range(batch_size):
                sparse_depth_i = sparse_depth[i].squeeze().cpu().numpy()
                sparse_depth_valid_i = sparse_depth_valid[i].squeeze().cpu().numpy()
                depth_pred_i = infer_depth(
                    DepthModel,
                    device,
                    image[i].squeeze().cpu().numpy(),
                    depth_model_transform,
                )

                int_depth_i, _, _ = compute_ls_solution(
                    depth_pred_i,
                    sparse_depth_i,
                    sparse_depth_valid_i,
                    min_pred_depth,
                    max_pred_depth,
                )
                ScaleMapInterpolator = Interpolator2D(
                    pred_inv=int_depth_i,
                    sparse_depth_inv=sparse_depth_i,
                    valid=sparse_depth_valid_i,
                )
                ScaleMapInterpolator.generate_interpolated_scale_map(
                    interpolate_method="linear", fill_corners=False
                )
                int_scales_i = ScaleMapInterpolator.interpolated_scale_map.astype(
                    np.float32
                )
                int_scales_i = utils.normalize_unit_range(int_scales_i)

                sample = {
                    "image": image[i].squeeze().cpu().numpy(),
                    "gt_depth": gt_depth[i].squeeze().cpu().numpy(),
                    "sparse_depth": sparse_depth[i].squeeze().cpu().numpy(),
                    "int_depth": int_depth_i,
                    "int_scales": int_scales_i,
                    "int_depth_no_tf": int_depth_i,
                }

                sample = ScaleMapLearner_transform(sample)

                x = torch.cat((sample["int_depth"], sample["int_scales"]), dim=0)
                x = x.to(device)
                d = sample["int_depth_no_tf"].to(device)
                batch_x.append(x)
                batch_d.append(d)
                batch_image.append(sample["image"].to(device))
                batch_gt.append(sample["gt_depth"].to(device))
                batch_sparse.append(sample["sparse_depth"].to(device))

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
                batch_gt > 0, torch.zeros_like(batch_gt), torch.ones_like(batch_gt)
            )

            loss, loss_info = compute_loss(
                image=batch_image,
                output_depth=sml_pred,
                ground_truth=batch_gt,
                loss_func=loss_func,  # smooth_l1 is less sensitive to error
                w_smoothness=w_smoothness,
                loss_smoothness_kernel_size=loss_smoothness_kernel_size,
            )

            print(
                "{}/{} epoch:{}: {}".format(
                    train_step % n_train_step, n_train_step, epoch, loss.item()
                )
            )

            # Compute gradient and backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (train_step % n_step_summary) == 0:
                with torch.no_grad():
                    log_summary(
                        summary_writer=train_summary_writer,
                        tag="train",
                        step=train_step,
                        max_predict_depth=max_pred_depth,
                        image=batch_image,
                        input_depth=d,
                        output_depth=sml_pred,
                        ground_truth=batch_gt,
                        scalars=loss_info,
                        n_display=min(4, batch_size),
                    )

            if (train_step % n_step_per_checkpoint) == 0:
                time_elapse = (time.time() - time_start) / 3600
                time_remain = (n_train_step - train_step) * time_elapse / train_step

                print(
                    "Step={:6}/{} Loss={:.5f} Time Elapsed={:.2f}h Time Remaining={:.2f}h".format(
                        train_step, n_train_step, loss.item(), time_elapse, time_remain
                    ),
                    log_path,
                )
                # Save chkpt
                ScaleMapLearner.save(depth_model_checkpoint_path.format(train_step))

    # save checkpoints
    ScaleMapLearner.save(depth_model_checkpoint_path.format(train_step))


def log_summary(
    summary_writer,
    tag,
    step,
    max_predict_depth,
    image=None,
    input_depth=None,
    input_response=None,
    output_depth=None,
    ground_truth=None,
    scalars={},
    n_display=4,
):
    with torch.no_grad():
        display_summary_image = []
        display_summary_depth = []

        display_summary_image_text = tag
        display_summary_depth_text = tag

        if image is not None:
            image_summary = image[0:n_display, ...]

            display_summary_image_text += "_image"
            display_summary_depth_text += "_image"

            # Add to list of images to log
            display_summary_image.append(
                torch.cat(
                    [
                        image_summary.cpu(),
                        torch.zeros_like(image_summary, device=torch.device("cpu")),
                    ],
                    dim=-1,
                )
            )

            display_summary_depth.append(display_summary_image[-1])

        if output_depth is not None:
            output_depth_summary = output_depth[0:n_display, ...]

            display_summary_depth_text += "_output_depth"

            # Add to list of images to log
            n_batch, _, n_height, n_width = output_depth_summary.shape

            display_summary_depth.append(
                torch.cat(
                    [
                        log_utils.colorize(
                            (output_depth_summary / max_predict_depth).cpu(),
                            colormap="viridis",
                        ),
                        torch.zeros(
                            n_batch, 3, n_height, n_width, device=torch.device("cpu")
                        ),
                    ],
                    dim=3,
                )
            )

            # Log distribution of output depth
            summary_writer.add_histogram(
                tag + "_output_depth_distro", output_depth, global_step=step
            )

        if output_depth is not None and input_depth is not None:
            input_depth_summary = input_depth[0:n_display, ...]

            display_summary_depth_text += "_input_depth-error"

            # Compute output error w.r.t. input depth
            input_depth_error_summary = torch.abs(
                output_depth_summary - input_depth_summary
            )

            input_depth_error_summary = torch.where(
                input_depth_summary > 0.0,
                input_depth_error_summary / (input_depth_summary + 1e-8),
                input_depth_summary,
            )

            # Add to list of images to log
            input_depth_summary = log_utils.colorize(
                (input_depth_summary / max_predict_depth).cpu(), colormap="viridis"
            )
            input_depth_error_summary = log_utils.colorize(
                (input_depth_error_summary / 0.05).cpu(), colormap="inferno"
            )

            display_summary_depth.append(
                torch.cat([input_depth_summary, input_depth_error_summary], dim=3)
            )

            # Log distribution of input depth
            summary_writer.add_histogram(
                tag + "_input_depth_distro", input_depth, global_step=step
            )

        if output_depth is not None and input_response is not None:
            response_summary = input_response[0:n_display, ...]

            display_summary_depth_text += "_response"

            # Add to list of images to log
            response_summary = log_utils.colorize(
                response_summary.cpu(), colormap="inferno"
            )

            display_summary_depth.append(
                torch.cat([response_summary, torch.zeros_like(response_summary)], dim=3)
            )

            # Log distribution of input depth
            summary_writer.add_histogram(
                tag + "_response_distro", input_depth, global_step=step
            )

        if output_depth is not None and ground_truth is not None:
            ground_truth = ground_truth[0:n_display, ...]
            ground_truth = torch.unsqueeze(ground_truth[:, 0, :, :], dim=1)

            ground_truth_summary = ground_truth[0:n_display]
            validity_map_summary = torch.where(
                ground_truth > 0,
                torch.ones_like(ground_truth),
                torch.zeros_like(ground_truth),
            )

            display_summary_depth_text += "_ground_truth-error"

            # Compute output error w.r.t. ground truth
            ground_truth_error_summary = torch.abs(
                output_depth_summary - ground_truth_summary
            )

            ground_truth_error_summary = torch.where(
                validity_map_summary == 1.0,
                (ground_truth_error_summary + 1e-8) / (ground_truth_summary + 1e-8),
                validity_map_summary,
            )

            # Add to list of images to log
            ground_truth_summary = log_utils.colorize(
                (ground_truth_summary / max_predict_depth).cpu(), colormap="viridis"
            )
            ground_truth_error_summary = log_utils.colorize(
                (ground_truth_error_summary / 0.05).cpu(), colormap="inferno"
            )

            display_summary_depth.append(
                torch.cat([ground_truth_summary, ground_truth_error_summary], dim=3)
            )

            # Log distribution of ground truth
            summary_writer.add_histogram(
                tag + "_ground_truth_distro", ground_truth, global_step=step
            )

        # Log scalars to tensorboard
        for name, value in scalars.items():
            summary_writer.add_scalar(tag + "_" + name, value, global_step=step)

        # Log image summaries to tensorboard
        if len(display_summary_image) > 1:
            display_summary_image = torch.cat(display_summary_image, dim=2)

            summary_writer.add_image(
                display_summary_image_text,
                torchvision.utils.make_grid(display_summary_image, nrow=n_display),
                global_step=step,
            )

        if len(display_summary_depth) > 1:
            display_summary_depth = torch.cat(display_summary_depth, dim=2)

            summary_writer.add_image(
                display_summary_depth_text,
                torchvision.utils.make_grid(display_summary_depth, nrow=n_display),
                global_step=step,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Train VI-Depth",
        description="What the program does",
        epilog="Text at the bottom of help",
    )

    parser.add_argument(
        "-dp",
        "--depth-predictor",
        type=str,
        default="dpt_hybrid",
        help="Name of depth predictor to use in pipeline.",
    )
    parser.add_argument(
        "-ns",
        "--nsamples",
        type=int,
        default=150,
        help="Number of sparse metric depth samples available.",
    )
    parser.add_argument(
        "-sm",
        "--sml-model-path",
        type=str,
        default="weights/sml_model.dpredictor.dpt_hybrid.nsamples.150.ckpt",
        help="Path to trained SML model weights.",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="/home/shuqi/dev/data/void/void_release/void_150/",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="output/train/classroom6",
    )

    # train_root = "/media/saimouli/Data6T/datasets/VOID_150_test"
    # '/media/saimouli/RPNG_FLASH_4/datasets/VOID_150'
    # result_root = "/media/saimouli/Data6T/datasets/VOID_150_test/results"  #'/media/vision/RPNG_FLASH_4/void_150_sample/results'
    args = parser.parse_args()

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # sml_ckt_path = "/home/saimouli/Documents/github/VI_Depth_sai/weights/sml_model.dpredictor.dpt_hybrid.nsamples.150.ckpt"

    image_path = os.path.join(args.train_dir, "image")
    gt_path = os.path.join(args.train_dir, "ground_truth")
    sparse_depth_path = os.path.join(args.train_dir, "sparse_depth")

    DepthModel = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")

    train_consistency = True

    if train_consistency == False:
        train(
            # data load
            train_dataset_path=args.train_dir,
            # train params
            learning_rates=[2e-4, 1e-4],
            learning_schedule=[20, 80],
            batch_size=4,
            n_step_summary=5,
            n_step_per_checkpoint=100,
            # loss settings
            loss_func="smoothl1",
            w_smoothness=0.0,
            loss_smoothness_kernel_size=-1,
            # model
            chkpt_path=args.sml_model_path,
            min_pred_depth=0.1,
            max_pred_depth=8.0,
            checkpoint_dir=os.path.join(args.train_dir, "checkpoints", current_time),
            n_threads=3,
            DepthModel=DepthModel,
        )
    else:
        train_scale_consistency(
            # data load
            train_dataset_path=args.train_dir,
            # train params
            learning_rates=[2e-4, 1e-4],
            learning_schedule=[20, 80],
            batch_size=3,
            n_step_summary=5,
            n_step_per_checkpoint=100,
            # loss settings
            loss_func="smoothl1",
            w_smoothness=0.0,
            loss_smoothness_kernel_size=-1,
            # model
            chkpt_path=args.sml_model_path,
            min_pred_depth=0.1,
            max_pred_depth=8.0,
            checkpoint_dir=os.path.join(args.train_dir, "checkpoints", current_time),
            n_threads=1,
            DepthModel=DepthModel,
        )
