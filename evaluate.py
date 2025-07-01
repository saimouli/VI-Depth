import os
import argparse

import torch
import imageio
import numpy as np
np.bool = np.bool_
from tqdm import tqdm
from PIL import Image

from modules.interpolator import PolynomialInterpolator2D, Interpolator2D
import modules.midas.utils as utils

import pipeline
import metrics

import matplotlib.pyplot as plt
from utils_eval import param_sweep_shift, param_sweep_scale, compute_ls_solution
from data.SML_dataset import SML_dataset
from model.main import midasNetModule

def get_ls_solution(depth_infer, input_sparse_depth, validity_map, min_pred, max_pred, max_depth, min_depth, mask, target_depth):

    input_sparse_depth_valid = (input_sparse_depth < max_depth) * (input_sparse_depth > min_depth)
    
    if validity_map is not None:
        input_sparse_depth_valid *= validity_map.astype(np.bool)

    input_sparse_depth_valid = input_sparse_depth_valid.astype(bool)
    input_sparse_depth[~input_sparse_depth_valid] = np.inf # set invalid depth
    input_sparse_depth = 1.0 / input_sparse_depth

    scaled_depth, scale_ls, shift_ls = compute_ls_solution(depth_infer, input_sparse_depth, input_sparse_depth_valid, min_pred, max_pred)
    
    error_w_int_depth_ls = metrics.ErrorMetrics()
    error_w_int_depth_ls.compute(scaled_depth, target_depth, mask.astype(bool))
    rmse_ls = error_w_int_depth_ls.rmse
    return rmse_ls, scale_ls, shift_ls

def evaluate_ddp(dataset_path, depth_predictor, nsamples, sml_model_path, device, rank, world_size):
    # Ranges for VOID
    min_depth, max_depth = 0.2, 5.0
    min_pred, max_pred = 0.1, 8.0

    # Instantiate method
    method = pipeline.VIDepth(
        depth_predictor, nsamples, sml_model_path, 
        min_pred, max_pred, min_depth, max_depth, device
    )

    # Get inputs
    with open(f"{dataset_path}/test_image.txt") as f:
        test_image_list = [line.rstrip() for line in f]

    # Split the dataset across processes
    test_image_list = test_image_list[rank::world_size]

    # Initialize error aggregators
    avg_error_w_int_depth = metrics.ErrorMetricsAverager_DDP(device)
    avg_error_w_pred = metrics.ErrorMetricsAverager_DDP(device)

    # Iterate through inputs list
    for i in tqdm(range(len(test_image_list)), desc=f"Rank {rank}"):
        # Image
        input_image_fp = os.path.join(dataset_path, test_image_list[i])
        input_image = utils.read_image(input_image_fp)

        # Sparse depth
        input_sparse_depth_fp = input_image_fp.replace("image", "sparse_depth")
        input_sparse_depth = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32) / 256.0
        input_sparse_depth[input_sparse_depth <= 0] = 0.0

        # Sparse depth validity map
        validity_map_fp = input_image_fp.replace("image", "validity_map")
        validity_map = np.array(Image.open(validity_map_fp), dtype=np.float32)
        assert np.all(np.unique(validity_map) == [0, 256])
        validity_map[validity_map > 0] = 1

        # Target (ground truth) depth
        target_depth_fp = input_image_fp.replace("image", "ground_truth")
        target_depth = np.array(Image.open(target_depth_fp), dtype=np.float32) / 256.0
        target_depth[target_depth <= 0] = 0.0

        # Target depth valid/mask
        mask = (target_depth < max_depth)
        if min_depth is not None:
            mask *= (target_depth > min_depth)
        target_depth[~mask] = np.inf  # Set invalid depth
        target_depth = 1.0 / target_depth  # Convert to inverse depth

        # Run pipeline
        output = method.run(input_image, input_sparse_depth, validity_map, device)

        # Convert outputs to tensors
        ga_depth = torch.from_numpy(output["ga_depth"]).to(device)
        sml_depth = torch.from_numpy(output["sml_depth"]).to(device)
        target_depth_tensor = torch.from_numpy(target_depth).to(device)
        mask_tensor = torch.from_numpy(mask.astype(np.bool)).to(device)

        # Compute error metrics using intermediate (globally aligned) depth
        error_w_int_depth = metrics.ErrorMetrics_DDP()
        error_w_int_depth.compute(ga_depth, target_depth_tensor, mask_tensor)

        # Compute error metrics using SML output depth
        error_w_pred = metrics.ErrorMetrics_DDP()
        error_w_pred.compute(sml_depth, target_depth_tensor, mask_tensor)

        # Accumulate error metrics
        avg_error_w_int_depth.accumulate(error_w_int_depth)
        avg_error_w_pred.accumulate(error_w_pred)

    # Synchronize and compute average metrics across all processes
    metrics_w_int_depth = avg_error_w_int_depth.get_metrics()
    metrics_w_pred = avg_error_w_pred.get_metrics()

    # Print results on rank 0
    if rank == 0:
        print("Averaging metrics for globally-aligned depth over {} samples".format(
            metrics_w_int_depth["total_count"]
        ))
        print("Averaging metrics for SML-aligned depth over {} samples".format(
            metrics_w_pred["total_count"]
        ))

        # Create a summary table
        from prettytable import PrettyTable
        summary_tb = PrettyTable()
        summary_tb.field_names = ["metric", "GA Only", "GA+SML"]

        summary_tb.add_row(["RMSE", f"{metrics_w_int_depth['rmse']:7.2f}", f"{metrics_w_pred['rmse']:7.2f}"])
        summary_tb.add_row(["MAE", f"{metrics_w_int_depth['mae']:7.2f}", f"{metrics_w_pred['mae']:7.2f}"])
        summary_tb.add_row(["AbsRel", f"{metrics_w_int_depth['absrel']:8.3f}", f"{metrics_w_pred['absrel']:8.3f}"])
        summary_tb.add_row(["iRMSE", f"{metrics_w_int_depth['inv_rmse']:7.2f}", f"{metrics_w_pred['inv_rmse']:7.2f}"])
        summary_tb.add_row(["iMAE", f"{metrics_w_int_depth['inv_mae']:7.2f}", f"{metrics_w_pred['inv_mae']:7.2f}"])
        summary_tb.add_row(["iAbsRel", f"{metrics_w_int_depth['inv_absrel']:8.3f}", f"{metrics_w_pred['inv_absrel']:8.3f}"])

        print(summary_tb)
        
def evaluate(dataset_path, depth_predictor, nsamples, sml_model_path, add_sparse_points=False, sparse_point_percentage=0,
             add_noise=False, noise_sigma=0.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # ranges for VOID
    min_depth, max_depth = 0.2, 5.0
    min_pred, max_pred = 0.1, 8.0

    # instantiate method
    method = pipeline.VIDepth(
        depth_predictor, nsamples, sml_model_path, 
        min_pred, max_pred, min_depth, max_depth, device
    )

    # get inputs
    #with open(f"{dataset_path}/void_{nsamples}/test_image.txt") as f: 
    with open(f"{dataset_path}/test_image.txt") as f:
        test_image_list = [line.rstrip() for line in f]
        
    # initialize error aggregators
    avg_error_w_int_depth = metrics.ErrorMetricsAverager()
    avg_error_w_pred = metrics.ErrorMetricsAverager()
    avg_error_w_poly = metrics.ErrorMetricsAverager()
    avg_error_w_linear = metrics.ErrorMetricsAverager()

    # Print experiment configuration
    if add_sparse_points:
        print(f"Adding {sparse_point_percentage}% additional sparse points from ground truth")
    if add_noise:
        print(f"Adding Gaussian noise with sigma={noise_sigma} to sparse points")
        
    sparse_count = []; rmse_val = []; mae_val = []; absrel_val = []
    # iterate through inputs list
    for i in tqdm(range(len(test_image_list))):
        
        # image
        input_image_fp = os.path.join(dataset_path, test_image_list[i])
        input_image = utils.read_image(input_image_fp)

        # sparse depth
        input_sparse_depth_fp = input_image_fp.replace("image", "sparse_depth")
        input_sparse_depth = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32) / 256.0
        input_sparse_depth[input_sparse_depth <= 0] = 0.0

        # sparse depth validity map
        validity_map_fp = input_image_fp.replace("image", "validity_map")
        validity_map = np.array(Image.open(validity_map_fp), dtype=np.float32)
        assert(np.all(np.unique(validity_map) == [0, 256]))
        validity_map[validity_map > 0] = 1
        
        ga_depth_fp = input_image_fp.replace("image", "ga_depth_inv").replace(".png", ".npy")
        ga_depth_inv = np.load(ga_depth_fp)
        
        interp_scale_fp = input_image_fp.replace("image", "interp_scale").replace(".png", ".npy")
        interp_scale = np.load(interp_scale_fp)
        
        #print("inpt img fp: ", input_image_fp)
        #print("ga depth fp: ", ga_depth_fp)
        
        
        # print("Before Pts: ", np.count_nonzero(validity_map))
        # reduce_pts = int(np.count_nonzero(validity_map) * 0.90)
        # nonzero_indices = np.argwhere(validity_map == 1)
        # remove_indices = np.random.choice(len(nonzero_indices), size=reduce_pts, replace=False)
        # points_to_remove = nonzero_indices[remove_indices]
        # for x, y in points_to_remove:
        #     validity_map[x, y] = 0
        # print("After Pts: ", np.count_nonzero(validity_map))
        
        sparse_count.append(np.count_nonzero(validity_map))

        # target (ground truth) depth
        target_depth_fp = input_image_fp.replace("image", "ground_truth")
        target_depth = np.array(Image.open(target_depth_fp), dtype=np.float32) / 256.0
        target_depth[target_depth <= 0] = 0.0

        original_sparse_count = np.count_nonzero(validity_map)

        # Add sparse points from ground truth if enabled
        if add_sparse_points and sparse_point_percentage > 0:
            # Calculate how many points to add
            current_points = np.count_nonzero(validity_map)
            points_to_add = int(current_points * sparse_point_percentage / 100)
            
            # Find valid ground truth points that aren't already in validity map
            valid_gt_mask = (target_depth > min_depth) & (target_depth < max_depth)
            candidate_points = valid_gt_mask & (validity_map == 0)
            candidate_indices = np.where(candidate_points)
            
            # If we have enough candidate points
            if len(candidate_indices[0]) >= points_to_add:
                print("Before Pts: ", np.count_nonzero(validity_map))
                # Randomly select points to add
                random_indices = np.random.choice(len(candidate_indices[0]), size=points_to_add, replace=False)
                selected_x = candidate_indices[0][random_indices]
                selected_y = candidate_indices[1][random_indices]
                
                # Add the points
                for x, y in zip(selected_x, selected_y):
                    validity_map[x, y] = 1
                    input_sparse_depth[x, y] = target_depth[x, y]
                
                #print(f"Added {points_to_add} sparse points ({sparse_point_percentage}%) to image {i}")
                #total how many
                total_points = np.count_nonzero(validity_map)
                print(f"Total points after addition: {total_points}")
            else:
                print(f"Warning: Not enough candidate points to add {points_to_add} points to image {i}")

        if add_noise and noise_sigma > 0:
            # Create a noise mask with same shape as input_sparse_depth
            noise = np.random.normal(0, noise_sigma, input_sparse_depth.shape)
            
            # Only add noise where validity_map is 1
            noise_mask = validity_map > 0
            
            # Apply noise only to valid sparse points
            input_sparse_depth[noise_mask] += noise[noise_mask]
            
            # Ensure all values remain positive and within valid range
            input_sparse_depth[input_sparse_depth <= 0] = min_depth
            input_sparse_depth[input_sparse_depth > max_depth] = max_depth
        
        # Store new sparse point count
        new_sparse_count = np.count_nonzero(validity_map)
        sparse_count.append(new_sparse_count)
        
        # target depth valid/mask
        mask = (target_depth < max_depth)
        if min_depth is not None:
            mask *= (target_depth > min_depth)
        target_depth[~mask] = np.inf  # set invalid depth
        target_depth = 1.0 / target_depth

        # run pipeline
        output = method.run(input_image, input_sparse_depth, validity_map, device, ga_depth_inv, interp_scale)
        
        input_sparse_depth[~validity_map.astype(np.bool)] = np.inf
        input_sparse_depth_inv = 1.0 / input_sparse_depth
        ScaleMapInterpolator = PolynomialInterpolator2D(pred_inv=output["ga_depth"], 
                                            sparse_depth_inv=input_sparse_depth_inv, 
                                            valid=validity_map.astype(np.bool))
        ScaleMapInterpolator.generate_polynomial_scale_map(degree=3, reg_lambda=5.0)
        poly_depth = ScaleMapInterpolator.apply_scale_map()
        
        ####### rbf/linear ######
        ScaleMapInterpolator = Interpolator2D(
            pred_inv = output["ga_depth"],
            sparse_depth_inv = input_sparse_depth_inv,
            valid=validity_map.astype(np.bool)
        )
        ScaleMapInterpolator.generate_interpolated_scale_map(
            interpolate_method='linear',
            fill_corners=False
        )
        int_scales = ScaleMapInterpolator.interpolated_scale_map.astype(np.float32)
        lin_depth = output["ga_depth"] * int_scales
        #################
        
        # Compute error metrics using intermediate (globally aligned) depth
        error_w_int_depth = metrics.ErrorMetrics()
        error_w_int_depth.compute(
            estimate = output["ga_depth"], 
            target = target_depth, 
            valid = mask.astype(np.bool),
        )

        # compute error metrics using SML output depth
        error_w_pred = metrics.ErrorMetrics()
        error_w_pred.compute(
            estimate = output["sml_depth"], 
            target = target_depth, 
            valid = mask.astype(np.bool),
        )
        
        error_w_poly = metrics.ErrorMetrics()
        error_w_poly.compute(
            estimate = poly_depth, 
            target = target_depth, 
            valid = mask.astype(np.bool),
        )

        error_w_lin = metrics.ErrorMetrics()
        error_w_lin.compute(
            estimate = lin_depth, 
            target = target_depth, 
            valid = mask.astype(np.bool),
        )
        # accumulate error metrics
        avg_error_w_int_depth.accumulate(error_w_int_depth)
        avg_error_w_pred.accumulate(error_w_pred)
        avg_error_w_poly.accumulate(error_w_poly)
        avg_error_w_linear.accumulate(error_w_lin)
        
        #rmse_val.append(error_w_pred.rmse)
        #mae_val.append(error_w_pred.mae)
        #absrel_val.append(error_w_pred.absrel)
    
    # plt.boxplot(sparse_count, vert=True, patch_artist=True)
    # plt.ylabel("Number of Sparse Points")
    # plt.show()
    
    if add_sparse_points or add_noise:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.boxplot(sparse_count, vert=True, patch_artist=True)
            plt.ylabel("Number of Sparse Points")
            plt.title("Distribution of Sparse Points Across Test Images")
            
            # If adding sparse points, show the percentage increase
            if add_sparse_points:
                plt.axhline(y=original_sparse_count, color='r', linestyle='--', 
                           label=f"Original count (avg: {original_sparse_count})")
                plt.axhline(y=original_sparse_count * (1 + sparse_point_percentage/100), 
                           color='g', linestyle='--', 
                           label=f"Target {sparse_point_percentage}% increase")
                plt.legend()
                
            #plt.savefig(f"sparse_points_distribution{'_with_noise' if add_noise else ''}.png")
            #plt.close()
            print(f"Saved sparse points distribution plot")
        except Exception as e:
            print(f"Could not create visualization: {e}")
            
    # compute average error metrics
    print("Averaging metrics for globally-aligned depth over {} samples".format(
        avg_error_w_int_depth.total_count
    ))
    avg_error_w_int_depth.average()

    print("Averaging metrics for SML-aligned depth over {} samples".format(
        avg_error_w_pred.total_count
    ))
    avg_error_w_pred.average()
    avg_error_w_poly.average()
    avg_error_w_linear.average()

    from prettytable import PrettyTable
    summary_tb = PrettyTable()
    
    title = "Evaluation Results"
    if add_sparse_points:
        title += f" with {sparse_point_percentage}% additional sparse points"
    if add_noise:
        title += f" and noise (sigma={noise_sigma})"
    summary_tb.title = title
    summary_tb.field_names = ["metric", "GA Only", "GA+SML", "Poly", "Lin"]

    summary_tb.add_row(["RMSE", f"{avg_error_w_int_depth.rmse_avg:7.2f}", f"{avg_error_w_pred.rmse_avg:7.2f}", f"{avg_error_w_poly.rmse_avg:7.2f}", f"{avg_error_w_linear.rmse_avg:7.2f}"])
    summary_tb.add_row(["MAE", f"{avg_error_w_int_depth.mae_avg:7.2f}", f"{avg_error_w_pred.mae_avg:7.2f}", f"{avg_error_w_poly.mae_avg:7.2f}", f"{avg_error_w_linear.mae_avg:7.2f}"])
    summary_tb.add_row(["AbsRel", f"{avg_error_w_int_depth.absrel_avg:8.3f}", f"{avg_error_w_pred.absrel_avg:8.3f}", f"{avg_error_w_poly.absrel_avg:8.3f}", f"{avg_error_w_linear.absrel_avg:8.3f}"])
    summary_tb.add_row(["iRMSE", f"{avg_error_w_int_depth.inv_rmse_avg:7.2f}", f"{avg_error_w_pred.inv_rmse_avg:7.2f}", f"{avg_error_w_poly.inv_rmse_avg:7.2f}", f"{avg_error_w_linear.inv_rmse_avg:7.2f}"])
    summary_tb.add_row(["iMAE", f"{avg_error_w_int_depth.inv_mae_avg:7.2f}", f"{avg_error_w_pred.inv_mae_avg:7.2f}", f"{avg_error_w_poly.inv_mae_avg:7.2f}", f"{avg_error_w_linear.inv_mae_avg:7.2f}"])
    summary_tb.add_row(["iAbsRel", f"{avg_error_w_int_depth.inv_absrel_avg:8.3f}", f"{avg_error_w_pred.inv_absrel_avg:8.3f}", f"{avg_error_w_poly.inv_absrel_avg:8.3f}", f"{avg_error_w_linear.inv_absrel_avg:8.3f}"])
    
    print(summary_tb)

def evaluate_custom(dataset_path, depth_predictor, nsamples, sml_model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # ranges for VOID
    min_depth, max_depth = 0.2, 5.0
    min_pred, max_pred = 0.1, 8.0

    # instantiate method
    method = pipeline.VIDepth(
        depth_predictor, nsamples, sml_model_path, 
        min_pred, max_pred, min_depth, max_depth, device
    )

    # get inputs
    with open(f"{dataset_path}/test_image.txt") as f: 
        test_image_list = [line.rstrip() for line in f]
        
    # initialize error aggregators
    avg_error_w_int_depth = metrics.ErrorMetricsAverager()
    avg_error_w_pred = metrics.ErrorMetricsAverager()
    ls_rmse = []; shift_optimized_rmse = []; scale_optimized_rmse = []; best_scale_shift_rmse = []
    # iterate through inputs list
    for i in tqdm(range(len(test_image_list))):
        
        # image
        input_image_fp = os.path.join(dataset_path, test_image_list[i])
        input_image = utils.read_image(input_image_fp)

        # sparse depth
        input_sparse_depth_fp = input_image_fp.replace("image", "sparse_depth")
        input_sparse_depth = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32) / 256.0
        input_sparse_depth[input_sparse_depth <= 0] = 0.0

        #plt.imshow(input_sparse_depth)
        #plt.show()

        # sparse depth validity map
        # validity_map_fp = input_image_fp.replace("image", "validity_map")
        # validity_map = np.array(Image.open(validity_map_fp), dtype=np.float32)
        # assert(np.all(np.unique(validity_map) == [0, 256]))
        # validity_map[validity_map > 0] = 1
        validity_map = None
        #plt.imshow(validity_map)
        #plt.show()

        # target (ground truth) depth
        target_depth_fp = input_image_fp.replace("image", "ground_truth")
        target_depth = np.array(Image.open(target_depth_fp), dtype=np.float32) / 256.0
        target_depth[target_depth <= 0] = 0.0

        #plt.imshow(target_depth)
        #plt.show()

        # target depth valid/mask
        mask = (target_depth < max_depth)
        if min_depth is not None:
            mask *= (target_depth > min_depth)
        target_depth[~mask] = np.inf  # set invalid depth
        target_depth = 1.0 / target_depth

        # run pipeline
        output = method.run(input_image, input_sparse_depth, validity_map, device)

        # run param sweep
        depth_infer = method.infer_depth(input_image)
        
        rmse_ls, scale_ls, shift_ls = get_ls_solution(depth_infer, input_sparse_depth, validity_map, min_pred, max_pred, max_depth, min_depth, mask.astype(bool), target_depth)
        ls_rmse.append(rmse_ls)
        print("Ls Shift: ", shift_ls, "Ls Scale: ", scale_ls, "Ls RMSE: ", rmse_ls)

        #optimize shift
        best_shift, best_shift_rmse = param_sweep_shift(shift_ls, scale_ls, depth_infer, target_depth, mask.astype(bool), rmse_ls, i)
        print(f"Optimizing Shift alone: {best_shift}, RMSE: {best_shift_rmse}")
        shift_optimized_rmse.append(best_shift_rmse)

        #optimize scale
        best_scale, best_scale_rmse = param_sweep_scale(scale_ls, shift_ls, depth_infer, target_depth, mask.astype(bool), rmse_ls, i)
        print(f"Optimizing Scale alone: {best_scale}, RMSE: {best_scale_rmse}")
        scale_optimized_rmse.append(best_scale_rmse)
        
        #optimize scale with best shift
        best_optimized_scale, best_optimized_scale_rmse = param_sweep_scale(scale_ls, best_shift, depth_infer, target_depth, mask, rmse_ls, i)
        print(f"Optimizing Scale with best shift: {best_optimized_scale}, RMSE: {best_optimized_scale_rmse}")
        best_scale_shift_rmse.append(best_optimized_scale_rmse)

        # compute error metrics using intermediate (globally aligned) depth
        error_w_int_depth = metrics.ErrorMetrics()
        error_w_int_depth.compute(
            estimate = output["ga_depth"], 
            target = target_depth, 
            valid = mask.astype(bool),
        )

        # compute error metrics using SML output depth
        error_w_pred = metrics.ErrorMetrics()
        error_w_pred.compute(
            estimate = output["sml_depth"], 
            target = target_depth, 
            valid = mask.astype(bool),
        )

        # accumulate error metrics
        avg_error_w_int_depth.accumulate(error_w_int_depth)
        avg_error_w_pred.accumulate(error_w_pred)

    ls_rmse = np.array(ls_rmse)
    shift_optimized_rmse = np.array(shift_optimized_rmse)
    scale_optimized_rmse = np.array(scale_optimized_rmse)
    best_scale_shift_rmse = np.array(best_scale_shift_rmse)

    # compute average error metrics
    print("Averaging metrics for globally-aligned depth over {} samples".format(
        avg_error_w_int_depth.total_count
    ))
    avg_error_w_int_depth.average()

    print("Averaging metrics for SML-aligned depth over {} samples".format(
        avg_error_w_pred.total_count
    ))
    avg_error_w_pred.average()

    from prettytable import PrettyTable
    summary_tb = PrettyTable()
    summary_tb.field_names = ["metric", "GA Only", "GA+SML", "LS", "Shift Opt.", "Scale Opt.", "Scale Opt. W best Shift"]

    summary_tb.add_row(["RMSE", f"{avg_error_w_int_depth.rmse_avg:7.2f}", f"{avg_error_w_pred.rmse_avg:7.2f}", f"{np.mean(ls_rmse):7.2f}", f"{np.mean(shift_optimized_rmse):7.2f}", f"{np.mean(scale_optimized_rmse):7.2f}", f"{np.mean(best_scale_shift_rmse):7.2f}"])
    summary_tb.add_row(["MAE", f"{avg_error_w_int_depth.mae_avg:7.2f}", f"{avg_error_w_pred.mae_avg:7.2f}", " ", " ", " ", " "])
    summary_tb.add_row(["AbsRel", f"{avg_error_w_int_depth.absrel_avg:8.3f}", f"{avg_error_w_pred.absrel_avg:8.3f}", " ", " ", " ", " "])
    summary_tb.add_row(["iRMSE", f"{avg_error_w_int_depth.inv_rmse_avg:7.2f}", f"{avg_error_w_pred.inv_rmse_avg:7.2f}", " ", " ", " ", " "])
    summary_tb.add_row(["iMAE", f"{avg_error_w_int_depth.inv_mae_avg:7.2f}", f"{avg_error_w_pred.inv_mae_avg:7.2f}", " ", " ", " ", " "])
    summary_tb.add_row(["iAbsRel", f"{avg_error_w_int_depth.inv_absrel_avg:8.3f}", f"{avg_error_w_pred.inv_absrel_avg:8.3f}", " ", " ", " ", " "])
    
    print(summary_tb)

def evaluate_sml(dataset_path, depth_predictor, nsamples, sml_model_path):
    min_depth, max_depth = 0.2, 5.0
    min_pred, max_pred = 0.1, 8.0
    
    avg_error_w_int_depth = metrics.ErrorMetricsAverager()
    avg_error_w_pred = metrics.ErrorMetricsAverager()
    
    dataset = SML_dataset(data_root=dataset_path, mode='val')
    dataloader = torch.utils.data.DataLoader(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = midasNetModule().load_from_checkpoint(sml_model_path)
    #model.load_from_checkpoint("/home/sai/Documents/VI-Depth/weights/total_loss=0.105.ckpt")
    model.eval()
    model.to(device)
    
    rmse_val = []; mae_val = []; absrel_val = []
    
    for batch_data in dataloader:
        batch_data = tuple(
            [item.to(device) if isinstance(item, torch.Tensor) else 
            [subitem.to(device) if isinstance(subitem, torch.Tensor) else subitem for subitem in item]
            if isinstance(item, list) else item
            for item in batch_data]
        )
        input_image, depth_gt_inv, input_sparse_depth, rel_depth_pred, ga_depth_inv, interp_scale, _,_ = batch_data
        
        sml_depth_inv = model(input_sparse_depth, input_image, rel_depth_pred, interp_scale, ga_depth_inv)
        sml_depth_inv = sml_depth_inv.detach().cpu()
        
        tgt_gt_depth = utils.inv2depth(depth_gt_inv)
        valid_mask = (tgt_gt_depth >= min_depth) & (tgt_gt_depth <= max_depth)
        
        error_w_int_depth = metrics.ErrorMetrics()
        valid_mask = valid_mask[0].squeeze(0).cpu().numpy()
        error_w_int_depth.compute(
            estimate = ga_depth_inv[0].cpu().numpy(), 
            target = depth_gt_inv[0].squeeze(0).cpu().numpy(), 
            valid = valid_mask.astype(np.bool),
        )

        # compute error metrics using SML output depth
        error_w_pred = metrics.ErrorMetrics()
        error_w_pred.compute(
            estimate = sml_depth_inv[0].cpu().squeeze(0).numpy(), 
            target = depth_gt_inv[0].squeeze(0).cpu().numpy(), 
            valid = valid_mask.astype(np.bool),
        )

        # accumulate error metrics
        avg_error_w_int_depth.accumulate(error_w_int_depth)
        avg_error_w_pred.accumulate(error_w_pred)
            
        rmse_val.append(error_w_pred.rmse)
        mae_val.append(error_w_pred.mae)
        absrel_val.append(error_w_pred.absrel)
        
    
    print("Averaging metrics for globally-aligned depth over {} samples".format(
        avg_error_w_int_depth.total_count
    ))
    avg_error_w_int_depth.average()

    print("Averaging metrics for SML-aligned depth over {} samples".format(
        avg_error_w_pred.total_count
    ))
    avg_error_w_pred.average()

    from prettytable import PrettyTable
    summary_tb = PrettyTable()
    summary_tb.field_names = ["metric", "GA Only", "GA+SML"]

    summary_tb.add_row(["RMSE", f"{avg_error_w_int_depth.rmse_avg:7.2f}", f"{avg_error_w_pred.rmse_avg:7.2f}"])
    summary_tb.add_row(["MAE", f"{avg_error_w_int_depth.mae_avg:7.2f}", f"{avg_error_w_pred.mae_avg:7.2f}"])
    summary_tb.add_row(["AbsRel", f"{avg_error_w_int_depth.absrel_avg:8.3f}", f"{avg_error_w_pred.absrel_avg:8.3f}"])
    summary_tb.add_row(["iRMSE", f"{avg_error_w_int_depth.inv_rmse_avg:7.2f}", f"{avg_error_w_pred.inv_rmse_avg:7.2f}"])
    summary_tb.add_row(["iMAE", f"{avg_error_w_int_depth.inv_mae_avg:7.2f}", f"{avg_error_w_pred.inv_mae_avg:7.2f}"])
    summary_tb.add_row(["iAbsRel", f"{avg_error_w_int_depth.inv_absrel_avg:8.3f}", f"{avg_error_w_pred.inv_absrel_avg:8.3f}"])
    
    print(summary_tb)
    
if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-ds', '--dataset-path', type=str, default='/media/saimouli/Data6T/datasets/void_150/testing',
                        help='Path to VOID release dataset.')
    parser.add_argument('-dp', '--depth-predictor', type=str, default='dpt_hybrid', 
                        help='Name of depth predictor to use in pipeline.')
    parser.add_argument('-ns', '--nsamples', type=int, default=150, 
                        help='Number of sparse metric depth samples available.')
    parser.add_argument('-sm', '--sml-model-path', type=str, default='weights/sml_model.dpredictor.dpt_hybrid.nsamples.150.ckpt', 
                        help='path')

    args = parser.parse_args()
    print(args)
    
    # evaluate(
    #     args.dataset_path,
    #     args.depth_predictor, 
    #     args.nsamples, 
    #     args.sml_model_path,
    #     add_sparse_points=False,
    #     sparse_point_percentage=50,
    #     add_noise=False,
    #     noise_sigma=0.00 # 4cm
    # )
    
    # evaluate_ddp(
    #     args.dataset_path,
    #     args.depth_predictor, 
    #     args.nsamples, 
    #     args.sml_model_path,
    #     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    #     rank=0,
    #     world_size=1,
    # )
    
    evaluate_sml(
        args.dataset_path,
        args.depth_predictor,
        args.nsamples,
        args.sml_model_path,
    )

    # to test on classroom
    #python3 evaluate.py -ds "/media/saimouli/RPNG_FLASH_4/data/VOID_small/classroom6" -sm /home/saimouli/Documents/github/VI_Depth_sai/weights/sml_model.dpredictor.midas_small.nsamples.150.ckpt

    # to test on table
    #python3 evaluate.py -ds "/media/saimouli/RPNG_FLASH_4/data/VOID_small/table1/" -dp dpt_beit_large_512 -sm /home/saimouli/Documents/github/VI_Depth_sai/weights/sml_model.dpredictor.dpt_beit_large_512.nsamples.150.ckpt