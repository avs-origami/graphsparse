# type: ignore

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from torch.distributions import MixtureSameFamily, MultivariateNormal, Categorical
from mpl_toolkits.mplot3d import Axes3D
import os

from config import info

def initialize_flat_gmm(num_components=8):
    """
    Initialize GMM to approximate a flat (uniform) distribution across the environment.
    
    Args:
        num_components: Number of GMM components to use
        
    Returns:
        means, logstds, weights_logits: Initial GMM parameters
    """
    # Arrange components in a grid to cover the space evenly
    grid_size = int(math.ceil(math.sqrt(num_components)))
    
    # Calculate spacing between components
    spacing = 1.0 / grid_size
    
    means = torch.zeros(num_components, 2)
    
    # Position components evenly
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if idx < num_components:
                # Center of each grid cell
                means[idx, 0] = spacing * (j + 0.5)
                means[idx, 1] = spacing * (i + 0.5)
                idx += 1
    
    # Set wide standard deviations to create overlap
    # The standard deviation should be large enough to create significant overlap
    # but not so large that it extends far beyond neighboring components
    std_value = spacing * 0.75  # This creates substantial overlap
    stds = torch.ones(num_components, 2) * std_value
    logstds = torch.log(stds)
    
    # Equal weights for all components
    weights = torch.ones(num_components) / num_components
    weights_logits = torch.log(weights)
    
    return means, logstds, weights_logits

def initialize_gmm_components(num_components=8):
    """
    Initialize GMM components to evenly cover the environment space.
    Places each component in a different sector with appropriate means, stds, and weights.
    """
    # Determine grid layout (2x4 grid for 8 components)
    grid_h, grid_w = 2, 4
    
    # Calculate cell size
    cell_h, cell_w = 1.0/grid_h, 1.0/grid_w
    
    # Initialize tensor for means
    means = torch.zeros(num_components, 2)
    
    # Set means to the center of each grid cell
    component_idx = 0
    for i in range(grid_h):
        for j in range(grid_w):
            # Center of this grid cell
            center_x = (j + 0.5) * cell_w
            center_y = (i + 0.5) * cell_h
            
            # Set the mean for this component
            means[component_idx, 0] = center_x
            means[component_idx, 1] = center_y
            
            component_idx += 1
    
    # Initialize standard deviations - make them proportional to cell size
    # but not too large to avoid excessive overlap
    stds = torch.ones(num_components, 2) * min(cell_w, cell_h) * 0.3
    logstds = torch.log(stds)
    
    # Initialize weights to be equal
    weights = torch.ones(num_components) / num_components
    weights_logits = torch.log(weights)
    
    return means, logstds, weights_logits

def initialize_gmm_components_with_noise(num_components=8, noise_scale=0.05, std_size=0.3):
    """Initialize GMM with slight noise around sector centers"""
    pwr = math.log2(num_components)
    grid_h, grid_w = int(2 ** (pwr // 2)), int(2 ** (pwr // 2))
    if pwr % 2 != 0:
        grid_w *= 2

    info(grid_h, grid_w)
    
    cell_h, cell_w = 1.0/grid_h, 1.0/grid_w
    
    means = torch.zeros(num_components, 2)
    
    component_idx = 0
    for i in range(grid_h):
        for j in range(grid_w):
            # Center of this grid cell
            center_x = (j + 0.5) * cell_w
            center_y = (i + 0.5) * cell_h
            
            # Add small random noise
            center_x += torch.randn(1).item() * noise_scale * cell_w
            center_y += torch.randn(1).item() * noise_scale * cell_h
            
            # Clamp to ensure it stays in the right sector
            center_x = max(j * cell_w + 0.1 * cell_w, min((j+1) * cell_w - 0.1 * cell_w, center_x))
            center_y = max(i * cell_h + 0.1 * cell_h, min((i+1) * cell_h - 0.1 * cell_h, center_y))
            
            # Set the mean for this component
            means[component_idx, 0] = center_x
            means[component_idx, 1] = center_y
            
            component_idx += 1
    
    # Initialize standard deviations - vary slightly for each component
    stds = torch.ones(num_components, 2) * min(cell_w, cell_h) * std_size
    stds *= (1 + torch.randn(num_components, 2) * 0.1)  # Add 10% variation
    logstds = torch.log(stds)
    
    # Initialize weights with slight variation
    weights = torch.ones(num_components) / num_components
    weights *= (1 + torch.randn(num_components) * 0.1).clamp(0.5, 1.5)  # Add variation
    weights = weights / weights.sum()  # Normalize
    weights_logits = torch.log(weights)
    
    return means, logstds, weights_logits

def visualize_gmm_3d_on_image(means, logstds, weights, image_tensor, samples=None, title="GMM Distribution", save_path=None):
    """
    Visualize the GMM distribution in 3D overlaid on the input image.
    """
    # Convert to numpy for matplotlib
    if isinstance(means, torch.Tensor):
        means = means.detach().cpu().numpy()
    if isinstance(logstds, torch.Tensor):
        logstds = logstds.detach().cpu().numpy()
    if isinstance(weights, torch.Tensor):
        # Make sure weights are properly converted from logits to probabilities
        weights = torch.softmax(weights, dim=0).detach().cpu().numpy()
    
    # For debugging, print the actual components
    # print("Means shape:", means.shape)
    # print("Means:", means)
    # print("LogStds:", np.exp(logstds))  # Print actual standard deviations, not logs
    # print("Weights (after softmax):", weights)
    
    # Extract image dimensions
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor[0]  # Take first batch if batched
    _, height, width = image_tensor.shape
    
    # Create the component distributions
    component_means = torch.tensor(means, dtype=torch.float32)
    component_covs = torch.diag_embed(torch.tensor(np.exp(logstds) ** 2, dtype=torch.float32))
    component_weights = torch.tensor(weights, dtype=torch.float32)
    
    # Create higher resolution grid for visualization
    resolution = 300  # Higher resolution for better visualization
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    positions = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), dtype=torch.float32)
    
    # Create the mixture distribution (similar to how it's done in the model)
    component_distribution = MultivariateNormal(loc=component_means, scale_tril=torch.diag_embed(torch.exp(torch.tensor(logstds))))
    mixture_distribution = Categorical(probs=component_weights)  # Using properly normalized weights
    gmm = MixtureSameFamily(mixture_distribution=mixture_distribution, component_distribution=component_distribution)
    
    # Evaluate log probabilities and convert to probabilities
    Z = gmm.log_prob(positions).detach().cpu().numpy()
    Z = np.exp(Z).reshape(X.shape)
    
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