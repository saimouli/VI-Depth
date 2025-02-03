import os
import argparse

import torch
import imageio
import numpy as np

from tqdm import tqdm
from PIL import Image

import modules.midas.utils as utils

import pipeline
import metrics

import matplotlib.pyplot as plt
import seaborn as sns
from utils_eval import param_sweep_shift, param_sweep_scale, compute_ls_solution

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

def evaluate(dataset_path, depth_predictor, nsamples, sml_model_path):
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
        
        
        #print("Before Pts: ", np.count_nonzero(validity_map))
        # reduce_pts = int(np.count_nonzero(validity_map) * 0.10)
        # nonzero_indices = np.argwhere(validity_map == 1)
        # remove_indices = np.random.choice(len(nonzero_indices), size=reduce_pts, replace=False)
        # points_to_remove = nonzero_indices[remove_indices]
        # for x, y in points_to_remove:
        #     validity_map[x, y] = 0
        #print("After Pts: ", np.count_nonzero(validity_map))
        
        sparse_count.append(np.count_nonzero(validity_map))

        # target (ground truth) depth
        target_depth_fp = input_image_fp.replace("image", "ground_truth")
        target_depth = np.array(Image.open(target_depth_fp), dtype=np.float32) / 256.0
        target_depth[target_depth <= 0] = 0.0

        # target depth valid/mask
        mask = (target_depth < max_depth)
        if min_depth is not None:
            mask *= (target_depth > min_depth)
        target_depth[~mask] = np.inf  # set invalid depth
        target_depth = 1.0 / target_depth

        # run pipeline
        output = method.run(input_image, input_sparse_depth, validity_map, device)

        # compute error metrics using intermediate (globally aligned) depth
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

        # accumulate error metrics
        avg_error_w_int_depth.accumulate(error_w_int_depth)
        avg_error_w_pred.accumulate(error_w_pred)
        
        rmse_val.append(error_w_pred.rmse)
        mae_val.append(error_w_pred.mae)
        absrel_val.append(error_w_pred.absrel)
    
    plt.boxplot(sparse_count, vert=True, patch_artist=True)
    plt.ylabel("Number of Sparse Points")
    plt.show()
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
    summary_tb.field_names = ["metric", "GA Only", "GA+SML"]

    summary_tb.add_row(["RMSE", f"{avg_error_w_int_depth.rmse_avg:7.2f}", f"{avg_error_w_pred.rmse_avg:7.2f}"])
    summary_tb.add_row(["MAE", f"{avg_error_w_int_depth.mae_avg:7.2f}", f"{avg_error_w_pred.mae_avg:7.2f}"])
    summary_tb.add_row(["AbsRel", f"{avg_error_w_int_depth.absrel_avg:8.3f}", f"{avg_error_w_pred.absrel_avg:8.3f}"])
    summary_tb.add_row(["iRMSE", f"{avg_error_w_int_depth.inv_rmse_avg:7.2f}", f"{avg_error_w_pred.inv_rmse_avg:7.2f}"])
    summary_tb.add_row(["iMAE", f"{avg_error_w_int_depth.inv_mae_avg:7.2f}", f"{avg_error_w_pred.inv_mae_avg:7.2f}"])
    summary_tb.add_row(["iAbsRel", f"{avg_error_w_int_depth.inv_absrel_avg:8.3f}", f"{avg_error_w_pred.inv_absrel_avg:8.3f}"])
    
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


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-ds', '--dataset-path', type=str, default='/media/saimouli/Data6T/datasets/VOID_150/testing',
                        help='Path to VOID release dataset.')
    parser.add_argument('-dp', '--depth-predictor', type=str, default='dpt_hybrid', 
                        help='Name of depth predictor to use in pipeline.')
    parser.add_argument('-ns', '--nsamples', type=int, default=150, 
                        help='Number of sparse metric depth samples available.')
    parser.add_argument('-sm', '--sml-model-path', type=str, default='/home/saimouli/Documents/github/VI_Depth_sai/weights/sml_model.dpredictor.dpt_hybrid.nsamples.150.pretrained.ckpt', 
                        help='path')

    args = parser.parse_args()
    print(args)
    
    evaluate(
        args.dataset_path,
        args.depth_predictor, 
        args.nsamples, 
        args.sml_model_path,
    )

    # to test on classroom
    #python3 evaluate.py -ds "/media/saimouli/RPNG_FLASH_4/data/VOID_small/classroom6" -sm /home/saimouli/Documents/github/VI_Depth_sai/weights/sml_model.dpredictor.midas_small.nsamples.150.ckpt

    # to test on table
    #python3 evaluate.py -ds "/media/saimouli/RPNG_FLASH_4/data/VOID_small/table1/" -dp dpt_beit_large_512 -sm /home/saimouli/Documents/github/VI_Depth_sai/weights/sml_model.dpredictor.dpt_beit_large_512.nsamples.150.ckpt