import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dataset_path = "/media/saimouli/Data6T/datasets/VOID_150_test/testing/mechanical_lab3"
drift_percentage = 0.01  # 5% drift, adjust as needed

cam_K = np.loadtxt(dataset_path + "/K.txt")

with open(f"{dataset_path}/test_image.txt") as f: 
    test_image_list = [line.rstrip() for line in f]

gt_poses = []
gt_positions = []
for image_name in tqdm(test_image_list, desc="Loading GT poses"):
    input_image_fp = os.path.join(dataset_path, image_name)
    pose_fp = input_image_fp.replace("image", "absolute_pose").replace(".png", ".txt")
    pose_CtoG = np.loadtxt(pose_fp)
    gt_poses.append(pose_CtoG)
    gt_positions.append(pose_CtoG[:3, 3])
gt_positions = np.array(gt_positions)

# Calculate total path length
total_path_length = 0.0
for i in range(1, len(gt_positions)):
    delta = gt_positions[i] - gt_positions[i-1]
    total_path_length += np.linalg.norm(delta)
print(f"Total trajectory length: {total_path_length:.2f} meters")

# Compute total drift and direction
total_drift = drift_percentage * total_path_length
drift_direction = np.array([1.0, 0.0, 0.0])  # Drift along x-axis (unit vector)

# Generate drifted positions
N = len(gt_poses)
drifted_positions = []
for i in range(N):
    fraction = i / (N - 1) if N > 1 else 0.0
    current_drift = total_drift * fraction * drift_direction
    drifted_pos = gt_positions[i] + current_drift
    drifted_positions.append(drifted_pos)
drifted_positions = np.array(drifted_positions)

# Compute ATE/RMSE
errors = np.linalg.norm(drifted_positions - gt_positions, axis=1)
rmse = np.sqrt(np.mean(errors**2))
print(f"Absolute Trajectory Error (RMSE): {rmse:.4f} meters, {rmse*100:.2f} cm")

# Create output directory for drifted poses
output_dir = os.path.join(dataset_path, "absolute_drift_pose")
os.makedirs(output_dir, exist_ok=True)

# Save drifted poses
for i in tqdm(range(len(test_image_list)), desc="Saving drifted poses"):
    image_name = test_image_list[i]
    input_image_fp = os.path.join(dataset_path, image_name)
    pose_fp = input_image_fp.replace("image", "absolute_pose").replace(".png", ".txt")
    
    # Construct output pose path
    output_pose_fp = pose_fp.replace("absolute_pose", "absolute_drift_pose")
    drifted_pose = np.eye(4)
    drifted_pose[:3, :3] = gt_poses[i][:3, :3]  # Original rotation
    drifted_pose[:3, 3] = drifted_positions[i]  # Drifted translation
    np.savetxt(output_pose_fp, drifted_pose)

# Plot trajectories
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 
        label='Ground Truth', linewidth=2, c='blue')
ax.plot(drifted_positions[:, 0], drifted_positions[:, 1], drifted_positions[:, 2],
        label=f'Drifted ({drift_percentage*100:.0f}%)', linewidth=2, c='red', linestyle='--')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.legend()
plt.title(f"Trajectory Comparison\nATE RMSE: {rmse:.4f} meters")
#plt.savefig(os.path.join(dataset_path, "trajectory_comparison.png"))
#plt.show()

#2d plot xy
plt.figure(figsize=(8, 6))
plt.plot(gt_positions[:, 0], gt_positions[:, 1], label='Ground Truth', linewidth=2, c='blue')
plt.plot(drifted_positions[:, 0], drifted_positions[:, 1], label=f'Drifted ({drift_percentage*100:.0f}%)', linewidth=2, c='red', linestyle='--')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.legend()
plt.title(f"XY Trajectory Comparison\nATE RMSE: {rmse:.4f} meters")
plt.show()