# type: ignore

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from torch.distributions import MixtureSameFamily, MultivariateNormal, Categorical
from mpl_toolkits.mplot3d import Axes3D
import os

from config import info, DEV
from gmm import GaussMix

def visualize_gmm_3d_on_image(gmm: GaussMix, jitter: callable, image_tensor: torch.Tensor, samples=None, title="GMM Distribution", save_path=None):
    """
    Visualize the GMM distribution in 3D overlaid on the input image.
    """
    image_tensor.squeeze_(0)
    _, height, width = image_tensor.shape
    # Create higher resolution grid for visualization
    resolution = 500  # Higher resolution for better visualization
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    positions = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), dtype=torch.float32).unsqueeze(0)
    # info(positions.shape)
    
    # Evaluate log probabilities and convert to probabilities
    Z = gmm.log_prob(positions.to(DEV)).detach().cpu().numpy()
    # info(Z.shape)
    Z = np.exp(Z) + jitter(positions.to(DEV) * 250).cpu().numpy()
    Z = Z.reshape(X.shape)
    
    # Find the peak for verification
    max_idx = np.argmax(Z)
    max_x, max_y = X.flatten()[max_idx], Y.flatten()[max_idx]
    # print(f"Visualization peak at: ({max_x:.4f}, {max_y:.4f}) with density {Z.flatten()[max_idx]:.6e}")
    
    # Create figure with 2 subplots
    fig = plt.figure(figsize=(16, 8))
    
    # 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.7)
    ax1.set_xlabel('X coordinate (normalized)')
    ax1.set_ylabel('Y coordinate (normalized)')
    ax1.set_zlabel('Probability Density')
    ax1.set_title(f'3D GMM Distribution - {title}')
    
    # Add a colorbar
    cbar = fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    cbar.set_label('Probability Density')
    
    # Set reasonable z-axis limits
    max_z = np.max(Z)
    ax1.set_zlim(0, max_z * 1.1)
    
    # 2D contour plot overlaid on the image
    ax2 = fig.add_subplot(122)
    
    # Convert image tensor to numpy and display
    if isinstance(image_tensor, torch.Tensor):
        img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    else:
        img_np = image_tensor
    
    img_np = np.flip(img_np, axis=0)
    ax2.imshow(img_np, origin='lower')
    
    # Add contour plot
    contour = ax2.contour(X * width, Y * height, Z, levels=10, cmap='viridis', alpha=0.7)
    cbar2 = fig.colorbar(contour, ax=ax2)
    cbar2.set_label('Probability Density')
    
    # If sample points are provided, plot them
    if samples is not None:
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()
            
        if len(samples.shape) == 1:
            samples = samples.reshape(1, -1)
            
        # Scale samples to image dimensions
        x_samples = samples[:, 0] * width
        y_samples = samples[:, 1] * height
        ax2.scatter(x_samples, y_samples, c='red', marker='x', s=100, linewidths=2, label='Samples')
        ax2.legend()
    
    ax2.set_title(f'GMM Overlaid on Image - {title}')
    ax2.set_xlabel('X coordinate (pixels)')
    ax2.set_ylabel('Y coordinate (pixels)')
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig