##! Implementation of ViT style image patching and tokenization.

import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch import Tensor

from config import DEV

class PatchLayer(nn.Module):
    def __init__(self, patch_size: int):
        """Create a new patching layer."""
        super().__init__()
        self.patch_size = patch_size


    def forward(self, images: Tensor) -> Tensor:
        """Given a tensor of image data with shape `(B, C, W, H)` divide into patches
        of size `(C, P, P)` and stack into tensor of shape `(B, N, C, P, P)`."""

        batch = images.size(0)
        chan = images.size(1)

        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)

        # Now tensor has shape (B, C, H/patch_size, W/patch_size, patch_size, patch_size)
        # Reshape to (batches, num_patches, channels, patch_size, patch_size)
        grid_h = images.size(3) // self.patch_size
        grid_w = images.size(2) // self.patch_size
        patches = patches.permute([0, 2, 3, 1, 4, 5]).reshape([batch, grid_h * grid_w, chan, self.patch_size, self.patch_size])

        return patches
    

    def save_patches(self, patches: Tensor):
        """Given a tensor of shape `(B, N, C, P, P)` save each patch `(C, P, P)` as a png."""

        base_path = "tmp/patch_"
        for i in range(patches.size(0)):
            batch = patches[i]
            for j in range(patches.size(1)):
                patch = batch[j]
                vutils.save_image(patch, base_path + str(i) + "_" + str(j) + ".png")


class PatchTokenizer(nn.Module):
    def __init__(self, n_patches: int, d_patch: int, d_project: int):
        """Create a new patch encoding layer."""
        super().__init__()
        self.n_patches = n_patches
        self.d_patch = d_patch
        self.d_project = d_project
        self.proj = nn.Linear(d_patch, d_project)
        self.pos_embed = nn.Embedding(n_patches, d_project)


    def forward(self, patches: Tensor) -> Tensor:
        """Given a tensor of patches with shape `(B, N, C, P, P)` flatten the patches
        and apply a linear layer to get a tensor of shape `(B, N, D)`."""

        # patches shape: [batch, n_patch, chan, patch_w, patch_h]
        batch_size = patches.size(0)

        # Flatten each patch: combine chan, patch_w, patch_h into a single dimension
        patch_dim = patches.size(2) * patches.size(3) * patches.size(4)
        assert patch_dim == self.d_patch, f"size of each patch differs from expectation! LHS {patch_dim}, RHS {self.d_patch}; patches {patches.shape}"
        assert patches.size(1) == self.n_patches, f"patch dimension doesn't match expected size! LHS {patches.size(1)}, {self.n_patches}; patches {patches.shape}"

        # shape: [batch, n_patch, chan*patch_w*patch_h]
        patches = patches.view([batch_size, self.n_patches, -1])

        # Create position indices and expand to match batch size
        # shape: [1, n_patch]
        positions = torch.arange(0, self.n_patches, dtype=torch.int64, device=DEV).unsqueeze(0)
        # shape: [batch, n_patch]
        positions = positions.expand([batch_size, -1])

        # Project the flattened patches; these are the tokens
        # shape: [batch, n_patch, projection_dim]
        projected_patches = self.proj(patches)

        # Get positional embeddings
        # shape: [batch, n_patch, projection_dim]
        pos_embeddings = self.pos_embed(positions)

        # Add positional embeddings to projected patches
        # shape: [batch, n_patch, projection_dim]
        return projected_patches + pos_embeddings
        