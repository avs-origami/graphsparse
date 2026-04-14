import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import MixtureSameFamily, Normal

from typing import Callable, Tuple

from config import DEV, info
from nets import LinHead


class GMMHead(nn.Module):
    def __init__(self, d_input: int, d_hidden: int, num_hidden: int, num_components: int, mode: str = "train"):
        super().__init__()

        # Policy networks for means, stds, and weights
        self.mean_net = LinHead("actor", d_input, d_hidden, num_hidden, num_components * 5)
      
        # Log standard deviations
        # self.logstd_net = LinHead("actor", d_input, d_hidden, num_hidden, num_components * 2)
        
        # # Component weights
        # self.weight_net = LinHead("actor", d_input, d_hidden, num_hidden, num_components)

        self.mean_std = nn.Parameter(torch.zeros(1, num_components, 2) - 2)
        self.logstd_std = nn.Parameter(torch.zeros(1, num_components, 2) - 2)
        self.weights_std = nn.Parameter(torch.zeros(1, num_components) - 2)

        self.num_components = num_components

        self.initialize_parameters()

    def initialize_parameters(self):
        # Get initial parameter values
        init_means, init_logstds, init_weight_logits = initialize_gmm_components_with_noise(num_components=self.num_components, noise_scale=0.3, std_size=0.7)
        
        # Initialize the output layers of each network to produce these values by default
        # This is a bit tricky - we need to set the bias terms to produce our desired outputs
        # when the inputs are zero (or close to zero at the start of training)
        
        out_layer = self.mean_net.layers[-1]

        # For means network, convert means to logits suitable for sigmoid activation
        means = torch.logit(init_means.clamp(0.01, 0.99)).reshape(-1)
        
        # For logstd network
        logstds = init_logstds.reshape(-1)
        
        # For weight network
        weights = init_weight_logits
        
        # Concat the things and set output layers
        out_data = torch.cat([means, logstds, weights])
        out_layer.bias.data = out_data
        out_layer.weight.data.zero_()


    def forward(self, x: Tensor, stochastic: bool = True) -> Tuple[Tensor, Tensor, Tensor, Tensor, Normal, Normal, Normal]:
        # Get means (constrained to [0,1] range)
        thingy = self.mean_net(x)
        means = torch.sigmoid(thingy[..., :(self.num_components * 2)])
        # info(means.shape)
        means = means.view(-1, self.num_components, 2)
        
        # Get log standard deviations (with constraints)
        logstds = thingy[..., (self.num_components * 2):(self.num_components * 4)]
        # info(logstds.shape)
        logstds = (3.5 * torch.sigmoid(logstds)) - 4.0 # Constrain reasonable range [-4.0, -0.5]
        logstds = logstds.view(-1, self.num_components, 2)
        # logstds = torch.zeros([x.size(0), self.num_components, 2]).to(DEV) - 3.5
        
        # Get component weights
        # TODO: sigmoid this???
        weights_logits = torch.sigmoid(thingy[..., (self.num_components * 4):(self.num_components * 5)] + 2.0)
        
        # weights_logits = torch.ones([x.size(0), self.num_components]).to(DEV)
        # info("WL", weights_logits.shape)
        mean_dist = Normal(means, self.mean_std.expand_as(means).exp())
        logstd_dist = Normal(logstds, self.logstd_std.expand_as(logstds).exp())
        weights_dist = Normal(weights_logits, self.weights_std.expand_as(weights_logits).exp())

        if stochastic:
            means = mean_dist.sample()
            logstds = logstd_dist.sample()
            weights_logits = weights_dist.sample()

        return means, logstds, weights_logits, thingy, mean_dist, logstd_dist, weights_dist


def find_maximum_probability_point(mixture, resolution=200):
    """
    Find the point with maximum probability in a GMM using a grid search.
    This function should be used for both sampling and visualization to ensure consistency.
    
    Args:
        mixture: Mixture distribution
        resolution: Grid resolution
        
    Returns:
        max_point: Tensor of shape [2] representing the maximum probability point
        max_prob: Maximum probability value
    """
    device = mixture.component_distribution.loc.device
    
    # Create grid
    x = torch.linspace(0, 1, resolution, device=device)
    y = torch.linspace(0, 1, resolution, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    
    # Compute raw (not log) probabilities
    with torch.no_grad():
        log_probs = mixture.log_prob(grid_points)
        probs = torch.exp(log_probs)
    
    # Find the maximum
    max_idx = torch.argmax(probs)
    max_point = grid_points[max_idx]
    max_prob = probs[max_idx]
    
    # Print for debugging
    # print(f"Maximum probability found at ({max_point[0].item():.4f}, {max_point[1].item():.4f}) with value {max_prob.item():.6f}")
    
    return max_point, max_prob


class GaussMix:
    def __init__(self, mixture: list[MixtureSameFamily]):
        self.mixture = mixture

    def sample(self, n=1):
        """Sample random points from GMMs"""
        all_samples = []
        for mix in self.mixture:
            samples = []
            for _ in range(n):
                sample = torch.tensor([-1.0, -1.0])
                for _ in range(100):
                    sample = mix.sample()
                    if ((sample >= 0).all() and (sample <= 1).all()):
                        break
                    
                samples.append(sample.clamp(0, 1).unsqueeze(0))

            all_samples.append(torch.cat(samples))

        return torch.stack(all_samples)

    def sample_max_probability(self, n=1):
        """Sample highest probability points from GMMs"""
        all_samples = []
        for mix in self.mixture:
            samples = []
            for _ in range(n):
                max_point, _ = find_maximum_probability_point(mix)
                samples.append(max_point.unsqueeze(0))

            all_samples.append(torch.cat(samples))

        return torch.stack(all_samples)
        
    def log_prob(self, actions):
        """Calculate log probability of actions under GMMs"""
        log_probs = []
        for i, mix in enumerate(self.mixture):
            log_probs.append(mix.log_prob(actions[i]))
        return torch.stack(log_probs)
            
    def entropy(self):
        """Calculate entropy as the sum of the entropy of the components."""
        entropies = []
        
        for mix in self.mixture:
            # Access the component distributions
            component_dist = mix.component_distribution
            
            # For each component in the mixture
            k = component_dist.loc.size(-1)  # Get dimensionality
            
            # Get means and covariances from the component distribution
            means = component_dist.loc  # Shape: [num_components, dim]
            covariances = component_dist.covariance_matrix  # Shape: [num_components, dim, dim]
            
            # Calculate entropy for each component
            component_entropies = []
            for i in range(covariances.size(0)):
                # Get single component covariance
                cov = covariances[i]
                
                # Calculate determinant of covariance matrix
                det_cov = torch.linalg.det(cov)
                
                # Calculate entropy: H = k/2 + k/2*log(2π) + 1/2*log(det(Σ))
                entropy_val = k/2 * (1.0 + torch.log(2 * torch.tensor(math.pi))) + 0.5 * torch.log(det_cov)
                component_entropies.append(entropy_val)
            
            # Average the component entropies weighted by the mixture weights
            mixture_weights = mix.mixture_distribution.probs
            weighted_entropy = torch.sum(mixture_weights * torch.stack(component_entropies))
            
            entropies.append(weighted_entropy)
        
        return torch.stack(entropies)

 
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