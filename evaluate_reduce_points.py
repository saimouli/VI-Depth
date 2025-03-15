import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from prettytable import PrettyTable

from data.SML_consistent_dataset import SML_consistent_dataset
import metrics
import modules.midas.utils as utils
from utils.camera import Camera
from modules.interpolator import Interpolator2D
from model.main_consistent import midasNetConsistentModule
from utils_eval import compute_ls_solution

def run_evaluation(args):
    """Run evaluation with varying point reduction percentages"""
    
    # Parse arguments
    data_root = args.data_root
    sml_model_path = args.sml_model_path
    checkpoint_path = args.checkpoint_path
    experiment_name = args.experiment_name
    reduction_start = args.reduction_start
    reduction_end = args.reduction_end
    reduction_step = args.reduction_step
    output_dir = args.output_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    dataset = SML_consistent_dataset(data_root=data_root, mode='val')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    
    # Load model
    model = midasNetConsistentModule(sml_model_path=sml_model_path, useConvGRU=True, is_train=False)
    model = model.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)
    
    # Define constants
    min_depth, max_depth = 0.2, 5.0
    min_pred, max_pred = 0.1, 8.0
    
    # Generate reduction percentages
    reduction_percentages = np.arange(reduction_start, reduction_end + reduction_step, reduction_step)
    
    # Store results
    results = {
        'reduction_percentages': reduction_percentages.tolist(),
        'rmse_ga': [],
        'rmse_sml': [],
        'absrel_ga': [],
        'absrel_sml': [],
        'mae_ga': [],
        'mae_sml': [],
        'inv_rmse_ga': [],
        'inv_rmse_sml': [],
        'inv_mae_ga': [],
        'inv_mae_sml': [],
        'inv_absrel_ga': [],
        'inv_absrel_sml': []
    }
    
    # For each reduction percentage
    for reduction_percentage in tqdm(reduction_percentages, desc="Evaluating reduction percentages"):
        print(f"\n==== Evaluating with {reduction_percentage:.2f}% points removed ====")
        
        # Initialize error metrics averagers
        avg_error_w_int_depth = metrics.ErrorMetricsAverager()
        avg_error_w_pred = metrics.ErrorMetricsAverager()
        
        # Evaluate on dataset
        for batch_data in tqdm(dataloader, desc="Processing dataset", leave=False):
            batch_data = tuple(
                [
                    item.to(device) if isinstance(item, torch.Tensor) else
                    [
                        subitem.to(device) if isinstance(subitem, torch.Tensor) else subitem
                        for subitem in item
                    ] if isinstance(item, list) else item
                    for item in batch_data
                ]
            )
            
            # Unpack batch data
            tgt_img, tgt_gt_depth_inv, tgt_ga_depth, tgt_interp, tgt_sparse_depth, ref_img, \
            ref_ga_depth, ref_interp, ref_gt_depth, ref_sparse_depth, tgt_pose, ref_gt_pose, \
            intrinsics, _, ref_pose_perturbed, tgt_depth_pred = batch_data
            
            # Reduce points in tgt_sparse_depth based on percentage
            validity_map = (tgt_sparse_depth > 0).squeeze().cpu().numpy().astype(np.uint8)
            total_points = np.count_nonzero(validity_map)
            reduce_pts = int(total_points * reduction_percentage / 100)
            
            if reduce_pts > 0:
                nonzero_indices = np.argwhere(validity_map == 1)
                if len(nonzero_indices) > 0:  # Make sure there are points to remove
                    remove_indices = np.random.choice(len(nonzero_indices), size=min(reduce_pts, len(nonzero_indices)), replace=False)
                    points_to_remove = nonzero_indices[remove_indices]
                    for x, y in points_to_remove:
                        validity_map[x, y] = 0
            
            input_sparse_depth_valid = validity_map.astype(bool)
            tgt_sparse_depth_np = tgt_sparse_depth.squeeze().cpu().numpy()
            tgt_sparse_depth_np[~input_sparse_depth_valid] = np.inf
            tgt_sparse_depth_np = 1.0 / tgt_sparse_depth_np
            
            # Recompute tgt_ga_depth
            tgt_ga_depth_np = tgt_ga_depth.squeeze().cpu().numpy()
            tgt_ga_depth_np, _, _ = compute_ls_solution(
                tgt_depth_pred.squeeze().cpu().numpy(), 
                tgt_sparse_depth_np, 
                input_sparse_depth_valid,
                min_pred, max_pred
            )
            tgt_ga_depth = torch.from_numpy(tgt_ga_depth_np).unsqueeze(0).float().to(device)
            
            # Recompute tgt_interp
            ScaleMapInterpolator = Interpolator2D(
                pred_inv=tgt_ga_depth.squeeze().cpu().numpy(),
                sparse_depth_inv=tgt_sparse_depth_np,
                valid=input_sparse_depth_valid,
            )
            ScaleMapInterpolator.generate_interpolated_scale_map(
                interpolate_method='linear',
                fill_corners=False
            )
            int_scales = ScaleMapInterpolator.interpolated_scale_map.astype(np.float32)
            int_scales = utils.normalize_unit_range(int_scales)
            tgt_interp = torch.from_numpy(int_scales).unsqueeze(0).float().to(device)
            
            # Prepare relative poses
            ref_rel_poses = [tgt_pose.inverse() @ ref_p for ref_p in ref_pose_perturbed]
            
            # Process through model
            refined_depth_inv, refined_ref_poses = model(
                tgt_img, ref_img,
                tgt_ga_depth, ref_ga_depth,
                tgt_interp, ref_interp,
                tgt_pose,
                ref_rel_poses,
                intrinsics
            )
            
            # Compute error metrics
            tgt_gt_depth = utils.inv2depth(tgt_gt_depth_inv)
            valid_mask = (tgt_gt_depth >= min_depth) & (tgt_gt_depth <= max_depth)
            
            mask = valid_mask.squeeze(0).cpu().numpy()
            
            # Compute error with globally aligned depth
            error_w_int_depth = metrics.ErrorMetrics()
            error_w_int_depth.compute(
                estimate=tgt_ga_depth.cpu().detach().numpy(),
                target=tgt_gt_depth_inv.cpu().detach().squeeze(0).numpy(),
                valid=mask.astype(bool),
            )
            
            # Compute error with refined depth
            error_w_pred = metrics.ErrorMetrics()
            error_w_pred.compute(
                estimate=refined_depth_inv[-1].cpu().detach().numpy(),
                target=tgt_gt_depth_inv.cpu().detach().squeeze(0).numpy(),
                valid=mask.astype(bool),
            )
            
            # Accumulate error metrics
            avg_error_w_int_depth.accumulate(error_w_int_depth)
            avg_error_w_pred.accumulate(error_w_pred)
        
        # Average the accumulated metrics
        avg_error_w_int_depth.average()
        avg_error_w_pred.average()
        
        # Store results
        results['rmse_ga'].append(avg_error_w_int_depth.rmse_avg)
        results['rmse_sml'].append(avg_error_w_pred.rmse_avg)
        results['absrel_ga'].append(avg_error_w_int_depth.absrel_avg)
        results['absrel_sml'].append(avg_error_w_pred.absrel_avg)
        results['mae_ga'].append(avg_error_w_int_depth.mae_avg)
        results['mae_sml'].append(avg_error_w_pred.mae_avg)
        results['inv_rmse_ga'].append(avg_error_w_int_depth.inv_rmse_avg)
        results['inv_rmse_sml'].append(avg_error_w_pred.inv_rmse_avg)
        results['inv_mae_ga'].append(avg_error_w_int_depth.inv_mae_avg)
        results['inv_mae_sml'].append(avg_error_w_pred.inv_mae_avg)
        results['inv_absrel_ga'].append(avg_error_w_int_depth.inv_absrel_avg)
        results['inv_absrel_sml'].append(avg_error_w_pred.inv_absrel_avg)
        
        # Print current results
        summary_tb = PrettyTable()
        summary_tb.field_names = ["Metric", "GA Only", "GA+SML"]
        summary_tb.add_row(["RMSE", f"{avg_error_w_int_depth.rmse_avg:7.2f}", f"{avg_error_w_pred.rmse_avg:7.2f}"])
        summary_tb.add_row(["MAE", f"{avg_error_w_int_depth.mae_avg:7.2f}", f"{avg_error_w_pred.mae_avg:7.2f}"])
        summary_tb.add_row(["AbsRel", f"{avg_error_w_int_depth.absrel_avg:8.3f}", f"{avg_error_w_pred.absrel_avg:8.3f}"])
        summary_tb.add_row(["iRMSE", f"{avg_error_w_int_depth.inv_rmse_avg:7.2f}", f"{avg_error_w_pred.inv_rmse_avg:7.2f}"])
        summary_tb.add_row(["iMAE", f"{avg_error_w_int_depth.inv_mae_avg:7.2f}", f"{avg_error_w_pred.inv_mae_avg:7.2f}"])
        summary_tb.add_row(["iAbsRel", f"{avg_error_w_int_depth.inv_absrel_avg:8.3f}", f"{avg_error_w_pred.inv_absrel_avg:8.3f}"])
        print(summary_tb)
    
    # Save results to JSON file
    with open(os.path.join(output_dir, f"{experiment_name}_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Generate plots
    plot_results(results, output_dir, experiment_name)
    
    return results

def plot_results(results, output_dir, experiment_name):
    """Plot the evaluation results"""
    reduction_percentages = results['reduction_percentages']
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    
    # Plot RMSE
    axs[0, 0].plot(reduction_percentages, results['rmse_ga'], 'b-', label='GA Only')
    axs[0, 0].plot(reduction_percentages, results['rmse_sml'], 'r-', label='GA+SML')
    axs[0, 0].set_xlabel('Point Reduction Percentage (%)')
    axs[0, 0].set_ylabel('RMSE')
    axs[0, 0].set_title('RMSE vs. Point Reduction')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Plot MAE
    axs[0, 1].plot(reduction_percentages, results['mae_ga'], 'b-', label='GA Only')
    axs[0, 1].plot(reduction_percentages, results['mae_sml'], 'r-', label='GA+SML')
    axs[0, 1].set_xlabel('Point Reduction Percentage (%)')
    axs[0, 1].set_ylabel('MAE')
    axs[0, 1].set_title('MAE vs. Point Reduction')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Plot AbsRel
    axs[1, 0].plot(reduction_percentages, results['absrel_ga'], 'b-', label='GA Only')
    axs[1, 0].plot(reduction_percentages, results['absrel_sml'], 'r-', label='GA+SML')
    axs[1, 0].set_xlabel('Point Reduction Percentage (%)')
    axs[1, 0].set_ylabel('AbsRel')
    axs[1, 0].set_title('AbsRel vs. Point Reduction')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Plot iRMSE
    axs[1, 1].plot(reduction_percentages, results['inv_rmse_ga'], 'b-', label='GA Only')
    axs[1, 1].plot(reduction_percentages, results['inv_rmse_sml'], 'r-', label='GA+SML')
    axs[1, 1].set_xlabel('Point Reduction Percentage (%)')
    axs[1, 1].set_ylabel('iRMSE')
    axs[1, 1].set_title('iRMSE vs. Point Reduction')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    # Plot iMAE
    axs[2, 0].plot(reduction_percentages, results['inv_mae_ga'], 'b-', label='GA Only')
    axs[2, 0].plot(reduction_percentages, results['inv_mae_sml'], 'r-', label='GA+SML')
    axs[2, 0].set_xlabel('Point Reduction Percentage (%)')
    axs[2, 0].set_ylabel('iMAE')
    axs[2, 0].set_title('iMAE vs. Point Reduction')
    axs[2, 0].legend()
    axs[2, 0].grid(True)
    
    # Plot iAbsRel
    axs[2, 1].plot(reduction_percentages, results['inv_absrel_ga'], 'b-', label='GA Only')
    axs[2, 1].plot(reduction_percentages, results['inv_absrel_sml'], 'r-', label='GA+SML')
    axs[2, 1].set_xlabel('Point Reduction Percentage (%)')
    axs[2, 1].set_ylabel('iAbsRel')
    axs[2, 1].set_title('iAbsRel vs. Point Reduction')
    axs[2, 1].legend()
    axs[2, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{experiment_name}_plots.png"), dpi=300)
    
    # Also create individual plots
    metrics_pairs = [
        ('RMSE', 'rmse_ga', 'rmse_sml'),
        ('MAE', 'mae_ga', 'mae_sml'),
        ('AbsRel', 'absrel_ga', 'absrel_sml'),
        ('iRMSE', 'inv_rmse_ga', 'inv_rmse_sml'),
        ('iMAE', 'inv_mae_ga', 'inv_mae_sml'),
        ('iAbsRel', 'inv_absrel_ga', 'inv_absrel_sml')
    ]
    
    for title, ga_key, sml_key in metrics_pairs:
        plt.figure(figsize=(8, 6))
        plt.plot(reduction_percentages, results[ga_key], 'b-', label='GA Only')
        plt.plot(reduction_percentages, results[sml_key], 'r-', label='GA+SML')
        plt.xlabel('Point Reduction Percentage (%)')
        plt.ylabel(title)
        plt.title(f'{title} vs. Point Reduction')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{experiment_name}_{title.lower()}.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VI-Depth with varying point reduction")
    parser.add_argument("--data_root", type=str, default='/home/sai/Documents/void_small', 
                        help="Path to dataset")
    parser.add_argument("--sml_model_path", type=str, 
                        default="weights/sml_model.dpredictor.dpt_hybrid.nsamples.150.ckpt",
                        help="Path to SML model weights")
    parser.add_argument("--checkpoint_path", type=str, 
                        default="/home/sai/Downloads/v1_gru_pose_depth/total_loss=0.086.ckpt",
                        help="Path to model checkpoint")
    parser.add_argument("--experiment_name", type=str, default="gru_6_iter",
                        help="Name for this experiment")
    parser.add_argument("--reduction_start", type=float, default=0,
                        help="Starting point reduction percentage")
    parser.add_argument("--reduction_end", type=float, default=90,
                        help="Ending point reduction percentage")
    parser.add_argument("--reduction_step", type=float, default=10,
                        help="Step size for point reduction percentage")
    parser.add_argument("--output_dir", type=str, default="eval_results",
                        help="Directory to save results")
    
    args = parser.parse_args()
    run_evaluation(args)