import matplotlib.pyplot as plt
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(script_dir, os.pardir))


batch_gt_ref0_sml = np.load(
    "output/debug/batch_gt_ref0_sml.npy"
)  # .shape = (3, 1, 288, 384)
batch_image_ref0_sml = np.load("output/debug/batch_image_ref0_sml.npy")
sml_pred_ref0 = np.load("output/debug/sml_pred_ref0.npy")  # .shape = (3, 1, 288, 384)


# Function to visualize a single depth image
def visualize_depth(ax, depth_image, title):
    im = ax.imshow(depth_image[0, 0], cmap="viridis")
    ax.set_title(title)
    return im


# Function to visualize RGB or grayscale image
def visualize_image(ax, image, title):
    if image.shape[1] == 3:  # RGB image
        im = ax.imshow(np.transpose(image[0], (1, 2, 0)))
    else:  # Grayscale image
        im = ax.imshow(image[0, 0], cmap="gray")
    ax.set_title(title)
    return im


# Function to compute and visualize pixel-wise L1 loss
def visualize_l1_loss(ax, gt, pred):
    l1_loss = np.abs(gt - pred)
    im = ax.imshow(l1_loss[0, 0], cmap="hot")
    ax.set_title("Pixel-wise L1 Loss")
    return im


# Create a figure with subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 15))

# Visualize the depth images and the reference image
im1 = visualize_depth(axs[0, 0], batch_gt_ref0_sml, "Ground Truth Depth")
im2 = visualize_image(axs[0, 1], batch_image_ref0_sml, "Reference Image")
im3 = visualize_depth(axs[1, 0], sml_pred_ref0, "Predicted Depth")

# Visualize the pixel-wise L1 loss
im4 = visualize_l1_loss(axs[1, 1], batch_gt_ref0_sml, sml_pred_ref0)

# Add colorbars (except for RGB image)
fig.colorbar(im1, ax=axs[0, 0])
if batch_image_ref0_sml.shape[1] == 1:  # Only add colorbar for grayscale
    fig.colorbar(im2, ax=axs[0, 1])
fig.colorbar(im3, ax=axs[1, 0])
fig.colorbar(im4, ax=axs[1, 1])

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
