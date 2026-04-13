import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import MixtureSameFamily, Normal

from typing import Callable, Tuple

from config import DEV, info
from nets import LinHead
from utils import initialize_gmm_components_with_noise, initialize_flat_gmm


class GMMHead(nn.Module):
    def __init__(self, d_input: int, d_hidden: int, num_hidden: int, num_components: int, mode: str = "train"):
        super().__init__()

        # Policy networks for means, stds, and weights
        self.mean_net = LinHead("actor", d_input, d_hidden, num_hidden, num_components * 5)
      
        # Log standard deviations
        # self.logstd_net = LinHead("actor", d_input, d_hidden, num_hidden, num_components * 2)
        
        # # Component weights
        # self.weight_net = LinHead("actor", d_input, d_hidden, num_hidden, num_components)

        self.mean_std = nn.Parameter(torch.zeros(1, num_components, 2))
        self.logstd_std = nn.Parameter(torch.zeros(1, num_components, 2))
        self.weights_std = nn.Parameter(torch.zeros(1, num_components))

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


    def forward(self, x: Tensor, stochastic: bool = True) -> Tuple[Tensor, Tensor, Tensor, Normal, Normal, Normal]:
        # Get means (constrained to [0,1] range)
        thingy = self.mean_net(x)
        means = torch.sigmoid(thingy[..., :(self.num_components * 2)])
        # info(means.shape)
        means = means.view(-1, self.num_components, 2)
        
        # Get log standard deviations (with constraints)
        logstds = thingy[..., (self.num_components * 2):(self.num_components * 4)]
        # info(logstds.shape)
        logstds = (torch.sigmoid(logstds) * 3.5) - 4.0 # Constrain reasonable range
        logstds = logstds.view(-1, self.num_components, 2)
        
        # Get component weights
        # TODO: sigmoid this???
        weights_logits = thingy[..., (self.num_components * 4):(self.num_components * 5)]
        # info(weights_logits)
        # info(weights_logits.shape)
        mean_dist = Normal(means, self.mean_std.expand_as(means).exp())
        logstd_dist = Normal(logstds, self.logstd_std.expand_as(logstds).exp())
        weights_dist = Normal(weights_logits, self.weights_std.expand_as(weights_logits).exp())

        if stochastic:
            means = mean_dist.sample()
            logstds = logstd_dist.sample()
            weights_logits = weights_dist.sample()
            # info(weights_logits)

        weights_logits = torch.where(weights_logits < -1, -1e10, weights_logits)
        # info(weights_logits)
        return means, logstds, weights_logits, mean_dist, logstd_dist, weights_dist


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