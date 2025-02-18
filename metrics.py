import numpy as np
import torch
from torch import distributed as dist

def rmse(estimate, target):
    return np.sqrt(np.mean((estimate - target) ** 2))

def mae(estimate, target):
    return np.mean(np.abs(estimate - target))

def absrel(estimate, target):
    return np.mean(np.abs(estimate - target) / target)

def inv_rmse(estimate, target):
    return np.sqrt(np.mean((1.0/estimate - 1.0/target) ** 2))

def inv_mae(estimate, target):
    return np.mean(np.abs(1.0/estimate - 1.0/target))

def inv_absrel(estimate, target):
    return np.mean((np.abs(1.0/estimate - 1.0/target)) / (1.0/target))

class ErrorMetrics_DDP:
    def __init__(self):
        self.rmse = None
        self.mae = None
        self.absrel = None
        self.inv_rmse = None
        self.inv_mae = None
        self.inv_absrel = None
        
        self.trans_rmse = None
        self.rot_rmse = None
        

    def compute(self, estimate: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor):
        """Compute metrics using PyTorch tensors directly (inputs: inverse depth 1/m)"""
        valid_estimate_inv = estimate[valid_mask]
        valid_target_inv = target[valid_mask]

        if valid_estimate_inv.numel() == 0:
            # Handle empty valid mask case (matches non-DDP initialization)
            device = estimate.device
            self.rmse = torch.tensor(np.inf, device=device)
            self.mae = torch.tensor(np.inf, device=device)
            self.absrel = torch.tensor(np.inf, device=device)
            self.inv_rmse = torch.tensor(np.inf, device=device)
            self.inv_mae = torch.tensor(np.inf, device=device)
            self.inv_absrel = torch.tensor(np.inf, device=device)
            return

        # Convert inverse depth to depth (meters)
        valid_estimate_depth = 1.0 / valid_estimate_inv
        valid_target_depth = 1.0 / valid_target_inv

        # Depth metrics (converted to mm)
        diff_depth = 1000.0 * valid_estimate_depth - 1000.0 * valid_target_depth
        self.rmse = torch.sqrt(torch.mean(diff_depth ** 2))
        self.mae = torch.mean(torch.abs(diff_depth))
        self.absrel = torch.mean(torch.abs(diff_depth) / (1000.0 * valid_target_depth))

        # Inverse depth metrics (converted to 1/km)
        def irmse(estimate, target):
            return torch.sqrt(torch.mean((1.0/estimate - 1.0/target) ** 2))
        def imae(estimate, target):
            return torch.mean(torch.abs(1.0/estimate - 1.0/target))
        def iabsrel(estimate, target):
            return torch.mean((torch.abs(1.0/estimate - 1.0/target)) / (1.0/target))
        
        # inv_estimate_km = 0.001 * valid_estimate_inv  # Convert 1/m to 1/km
        # inv_target_km = 0.001 * valid_target_inv
        # inv_diff = inv_estimate_km - inv_target_km
        self.inv_rmse = irmse(0.001 * valid_estimate_depth, 0.001 * valid_target_depth)
        self.inv_mae = imae(0.001 * valid_estimate_depth, 0.001 * valid_target_depth)
        self.inv_absrel = iabsrel(0.001 * valid_estimate_depth, 0.001 * valid_target_depth)
        
    def compute_pose(self, pose_pred, pose_gt):
        """Compute pose errors between predicted and ground truth poses."""
        if pose_pred is None or pose_gt is None:
            # Handle empty pose case (matches non-DDP initialization)
            device = pose_pred.device if pose_pred is not None else 'cpu'
            self.trans_rmse = torch.tensor(np.inf, device=device)
            self.rot_rmse = torch.tensor(np.inf, device=device)
            return
        # Relative pose error
        rel_pose = torch.inverse(pose_pred) @ pose_gt

        # Translation error (meters)
        trans_error = torch.norm(rel_pose[:3, 3])
        self.trans_rmse = trans_error

        # Rotation error (degrees)
        trace = torch.trace(rel_pose[:3, :3])
        cos_theta = torch.clamp((trace - 1) / 2, -1.0, 1.0)
        rot_error = torch.acos(cos_theta) * 180 / torch.pi
        self.rot_rmse = rot_error

class ErrorMetricsAverager_DDP:
    def __init__(self, device):
        self.device = device
        self.rmse_sum = torch.tensor(0.0, device=device)
        self.mae_sum = torch.tensor(0.0, device=device)
        self.absrel_sum = torch.tensor(0.0, device=device)
        self.inv_rmse_sum = torch.tensor(0.0, device=device)
        self.inv_mae_sum = torch.tensor(0.0, device=device)
        self.inv_absrel_sum = torch.tensor(0.0, device=device)
        self.total_count = torch.tensor(0, device=device)
        self.total_pose_count = torch.tensor(0, device=device)
        
        self.trans_rmse_sum = torch.tensor(0.0, device=device)
        self.rot_rmse_sum = torch.tensor(0.0, device=device)

    def reset(self):
        """Reset all accumulators while maintaining device placement"""
        self.rmse_sum.zero_()
        self.mae_sum.zero_()
        self.absrel_sum.zero_()
        self.inv_rmse_sum.zero_()
        self.inv_mae_sum.zero_()
        self.inv_absrel_sum.zero_()
        self.total_count.zero_()
        self.total_pose_count.zero_()
        self.trans_rmse_sum.zero_()
        self.rot_rmse_sum.zero_()

    def accumulate(self, error_metrics):
        """Accumulate metrics from a single ErrorMetrics_DDP instance"""
        # Only accumulate valid batches (matches non-DDP behavior)
        if not torch.isinf(error_metrics.rmse):
            self.rmse_sum += error_metrics.rmse.detach().to(self.device)
            self.mae_sum += error_metrics.mae.detach().to(self.device)
            self.absrel_sum += error_metrics.absrel.detach().to(self.device)
            self.inv_rmse_sum += error_metrics.inv_rmse.detach().to(self.device)
            self.inv_mae_sum += error_metrics.inv_mae.detach().to(self.device)
            self.inv_absrel_sum += error_metrics.inv_absrel.detach().to(self.device)
            self.total_count += 1

    def accumulate_pose(self, error_metrics):
        if not torch.isinf(error_metrics.trans_rmse):
            self.total_pose_count += 1
            self.trans_rmse_sum += error_metrics.trans_rmse.detach().to(self.device)
            self.rot_rmse_sum += error_metrics.rot_rmse.detach().to(self.device)
    
    def get_pose_metrics(self):
        total_pose_count = self.all_gather(self.total_pose_count.detach()).sum()
        trans_rmse_sum = self.all_gather(self.trans_rmse_sum.detach()).sum()
        rot_rmse_sum = self.all_gather(self.rot_rmse_sum.detach()).sum()
        
        return {
            'trans_rmse': trans_rmse_sum / total_pose_count if total_pose_count > 0 else torch.tensor(np.inf),
            'rot_rmse': rot_rmse_sum / total_pose_count if total_pose_count > 0 else torch.tensor(np.inf),
            'total_pose_count': total_pose_count
        }
        
    def get_metrics(self):
        """Return synchronized metrics as a dictionary"""
        # Gather results from all processes
        rmse_sum = self.all_gather(self.rmse_sum.detach()).sum()
        mae_sum = self.all_gather(self.mae_sum.detach()).sum()
        absrel_sum = self.all_gather(self.absrel_sum.detach()).sum()
        inv_rmse_sum = self.all_gather(self.inv_rmse_sum.detach()).sum()
        inv_mae_sum = self.all_gather(self.inv_mae_sum.detach()).sum()
        inv_absrel_sum = self.all_gather(self.inv_absrel_sum.detach()).sum()
        total_count = self.all_gather(self.total_count.detach()).sum()

        # Handle edge case with no valid batches (matches non-DDP)
        if total_count == 0:
            return {
                'rmse': torch.tensor(np.inf, device=self.device),
                'mae': torch.tensor(np.inf, device=self.device),
                'absrel': torch.tensor(np.inf, device=self.device),
                'inv_rmse': torch.tensor(np.inf, device=self.device),
                'inv_mae': torch.tensor(np.inf, device=self.device),
                'inv_absrel': torch.tensor(np.inf, device=self.device),
                'trans_rmse': torch.tensor(np.inf, device=self.device),
                'rot_rmse': torch.tensor(np.inf, device=self.device),
                'total_count': total_count
            }

        return {
            'rmse': rmse_sum / total_count,
            'mae': mae_sum / total_count,
            'absrel': absrel_sum / total_count,
            'inv_rmse': inv_rmse_sum / total_count,
            'inv_mae': inv_mae_sum / total_count,
            'inv_absrel': inv_absrel_sum / total_count,
            'total_count': total_count,
        }

    def all_gather(self, tensor):
        """Helper function for gathering tensors across processes"""
        if dist.is_initialized() and dist.get_world_size() > 1:
            gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, tensor)

            # Ensure all tensors are at least 1D
            gathered = [t.unsqueeze(0) if t.dim() == 0 else t for t in gathered]

            return torch.cat(gathered)
        return tensor
    
class ErrorMetrics(object):
    def __init__(self):
        # initialize by setting to worst values
        self.rmse, self.mae, self.absrel = np.inf, np.inf, np.inf
        self.inv_rmse, self.inv_mae, self.inv_absrel = np.inf, np.inf, np.inf

    def compute(self, estimate, target, valid):
        # apply valid masks
        estimate = estimate[valid]
        target = target[valid]

        # estimate and target will be in inverse space, convert to regular
        estimate = 1.0/estimate
        target = 1.0/target

        # depth error, estimate in meters, convert units to mm
        self.rmse = rmse(1000.0*estimate, 1000.0*target)
        self.mae = mae(1000.0*estimate, 1000.0*target)
        self.absrel = absrel(1000.0*estimate, 1000.0*target)

        # inverse depth error, estimate in meters, convert units to 1/km
        self.inv_rmse = inv_rmse(0.001*estimate, 0.001*target)
        self.inv_mae = inv_mae(0.001*estimate, 0.001*target)
        self.inv_absrel = inv_absrel(0.001*estimate, 0.001*target)

class ErrorMetricsAverager(object):
    def __init__(self):
        # initialize avg accumulators to zero
        self.rmse_avg, self.mae_avg, self.absrel_avg = 0, 0, 0
        self.inv_rmse_avg, self.inv_mae_avg, self.inv_absrel_avg = 0, 0, 0
        self.total_count = 0

    def reset(self):
        # reset accumulators to zero
        self.rmse_avg, self.mae_avg, self.absrel_avg = 0, 0, 0
        self.inv_rmse_avg, self.inv_mae_avg, self.inv_absrel_avg = 0, 0, 0
        self.total_count = 0
        
    def accumulate(self, error_metrics):
        # adds to accumulators from ErrorMetrics object
        assert isinstance(error_metrics, ErrorMetrics)

        self.rmse_avg += error_metrics.rmse
        self.mae_avg += error_metrics.mae
        self.absrel_avg += error_metrics.absrel

        self.inv_rmse_avg += error_metrics.inv_rmse
        self.inv_mae_avg += error_metrics.inv_mae
        self.inv_absrel_avg += error_metrics.inv_absrel

        self.total_count += 1

    def average(self):
        print(f"Averaging depth metrics over {self.total_count} samples")
        self.rmse_avg = self.rmse_avg / self.total_count
        self.mae_avg = self.mae_avg / self.total_count
        self.absrel_avg = self.absrel_avg / self.total_count
        # print(f"Averaging inv depth metrics over {self.total_count} samples")
        self.inv_rmse_avg = self.inv_rmse_avg / self.total_count
        self.inv_mae_avg = self.inv_mae_avg / self.total_count
        self.inv_absrel_avg = self.inv_absrel_avg / self.total_count