##! Implementation of the PPO algorithm for training the model.
##! Based on https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py

import dataclasses
import os
import random
import time
from dataclasses import dataclass, fields
from typing import Callable, Optional, Tuple
import threading

import numpy as np
import matplotlib.pyplot as plt
import torch
import tensorhue
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.distributions import Normal, MixtureSameFamily, MultivariateNormal, Categorical
from torch.utils.tensorboard.writer import SummaryWriter
from PIL import Image
from tensorviz import tviz

from config import DEV, info
from gmm import GMMHead, GaussMix
from gtrxl import StableTransformerXL
from nets import LinHead, layer_init
from vit import PatchLayer, PatchTokenizer
from utils import visualize_gmm_3d_on_image

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""

    seed: Optional[int] = 1
    """seed of the experiment"""

    torch_deterministic: bool = False
    """if toggled, `torch.backends.cudnn.deterministic=False`"""

    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""

    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""

    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""

    # =========================================
    # =======   Training Loop Options   =======
    # =========================================
    
    episodes: int = 3750
    """Number of training episodes"""

    num_steps: int = 512
    """Number of timesteps in each episode"""

    num_moves: int = 50
    """Number of robot moves before resetting environment"""

    num_envs: int = 1
    """Number of parallel environments"""
    
    ppo_epochs: int = 10
    """Number of epochs to train on each batch of data"""
    
    num_minis: int = 32
    """Number of minibatches to train on"""
    
    warmup: int = 5
    """Number of robot moves for context collection"""

    num_prune: int = 25
    """Number of nodes to prune per timestep"""

    prune_frac: float = 0.5
    """Total fraction of nodes to prune between robot moves"""

    # =========================================
    # =======   Model Hyperparameters   =======
    # =========================================
    
    hidden_size: int = 256
    """Number of neurons in the hidden layers of the linear heads"""
    
    num_hidden: int = 5
    """Number of hidden layers in the linear heads"""
    
    num_actions: int = 2
    """Number of actions"""

    d_model: int = 128
    """Size of GTrXL input / output"""
    
    gtrxl_layers: int = 3
    """Number of encoder layers for the GTrXL"""
    
    num_heads: int = 8
    """Number of attention heads for the GTrXL"""
    
    d_head_inner: int = 64
    """Size of attention heads for GTrXL"""
    
    d_ff_inner: int = 128
    """Size of FF layers for GTrXL"""

    d_comp: int = 64
    """Compressed size of each token"""
    
    gtrxl_mem_len: int = 102 * 16
    """Number of hidden states to store in GTrXL memory"""

    img_size: int = 250
    """Image dimensions (assumes square images)"""
    
    patch_size: int = 25
    """Patch size for GTrXL input"""

    img_chan: int = 3
    """Image channels"""

    num_gauss: int = 8
    """Number of gaussian components for GMM"""
    
    # =========================================
    # ========   PPO Hyperparameters   ========
    # =========================================

    anneal_lr: bool = True
    """Whether to anneal the learning rate"""
    
    norm_adv: bool = True
    """Whether to normalize advantages"""
    
    clip_vloss: bool = True
    """Whether to clip the value loss"""
    
    early_stop: bool = True
    """Whether to check for KL divergence"""

    stochastic: str = "On"
    """Whether to sample GMM parameters stochastically (train time) or not (eval time)."""

    lr: float = 3e-4
    """Learning rate"""
    
    gamma: float = 0.99
    """Discount factor"""
    
    gae_lambda: float = 0.95
    """Lambda for GAE estimation"""
    
    clip_coef: float = 0.2
    """PPO clip parameter"""
    
    vf_coef: float = 0.5
    """Value function coefficient in the loss"""
    
    ent_coef: float = 0.01
    """Entropy coefficient in the loss"""
    
    max_grad_norm: float = 0.5
    """Maximum gradient norm for clipping"""
    
    target_kl: float = 0.03
    """Target KL divergence threshold"""

    max_explore: float = 0.5
    """Maximum exploration rate during training"""

    min_explore: float = 0.05
    """Minimum exploration rate during training"""

    # =========================================
    # =======   Miscellaneous Options   =======
    # =========================================
    
    render: bool = False
    """Whether to render the simulation"""
    
    dbg: bool = False
    """Whether to show debugging info"""
    
    save_dir: str = "models"
    """Directory to save model files"""

    test: bool = False,
    """Don't train the model, just test"""

    no_prune: bool = False,
    """Disable pruning. Only takes effect when used with --test"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""

    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""

    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    def update_from_dict(self, config_data: dict) -> None:
        """
        Update this config instance with values from a dictionary,
        but only for options that exist in both the dictionary and dataclass.
        """
        # Get all valid field names for this dataclass
        valid_fields = {f.name for f in fields(self)}
        
        # Update only fields that exist in both the dataclass and config_data
        for key, value in config_data.items():
            if key in valid_fields:
                setattr(self, key, value)


# def evaluate(
#     model_path: str,
#     make_env: Callable,
#     env_id: str,
#     eval_episodes: int,
#     run_name: str,
#     Model: torch.nn.Module,
#     capture_video: bool = True,
#     gamma: float = 0.99,
# ):
#     envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, run_name, gamma)])
#     agent = Model(envs).to(DEV)
#     agent.load_state_dict(torch.load(model_path, map_location=DEV))
#     agent.eval()

#     obs, _ = envs.reset()
#     episodic_returns = []
#     while len(episodic_returns) < eval_episodes:
#         actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(DEV))
#         next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
#         if "final_info" in infos:
#             for info in infos["final_info"]:
#                 if "episode" not in info:
#                     continue
#                 print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
#                 episodic_returns += [info["episode"]["r"]]
#         obs = next_obs

#     return episodic_returns

PREV = None

class Agent(nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        self.args = args
        self.patch = PatchLayer(args.patch_size)
        self.token = PatchTokenizer(
            int(args.img_size / args.patch_size) ** 2,
            args.img_chan * args.patch_size * args.patch_size,
            args.d_model
        )

        # self.atok = nn.Linear(args.num_gauss * 2, args.d_model)
        # self.rtok = nn.Linear(1, args.d_model)
        # self.pemb = nn.Embedding(2, args.d_model)

        self.gtrxl = StableTransformerXL(
            args.d_model,
            args.gtrxl_layers,
            args.num_heads,
            args.d_head_inner,
            args.d_ff_inner,
            mem_len=args.gtrxl_mem_len
        )

        self.comp = nn.Sequential(
            nn.Linear(args.d_model, args.d_comp * 4),
            nn.Linear(args.d_comp * 4, args.d_comp),
        )

        self.critic = LinHead("critic", args.d_comp * 100, args.hidden_size, args.num_hidden, 1).to(DEV)
        self.gauss = GMMHead(args.d_comp * 100, args.hidden_size, args.num_hidden, args.num_gauss).to(DEV)
        self.sincos = LinHead("actor", args.d_comp * 100, args.hidden_size, args.num_hidden, 2).to(DEV)
        self.sincos_std = nn.Parameter(torch.zeros(1, 2))

        self.stochastic = (args.stochastic == "On")

    def tform(self, x: Tensor) -> Tensor:
        # Split the input image (3x250x250) into 100 patches (3x25x25)
        patches = self.patch.forward(x)
        pdim = patches.shape

        # Attach the robot's x and y position as two additional patches. This is
        # okay since the x and y are in the range [0, 250], just like the color
        # channels on the original input images.
        # rob_x = torch.full([1, 1, pdim[-3], pdim[-2], pdim[-1]], pos[0] / 255.0, device=DEV)
        # rob_y = torch.full([1, 1, pdim[-3], pdim[-2], pdim[-1]], pos[1] / 255.0, device=DEV)
        # patches = torch.cat([patches, rob_x, rob_y], dim=1)

        # Convert the patches to tokens (positional embeddings are applied here
        # as well).
        tokens = self.token.forward(patches)
        # global PREV
        # if PREV is not None and tokens.shape[0] == 1:
        #     tviz(tokens.squeeze(0) - PREV.squeeze(0))
        #     print()
        
        # if tokens.shape[0] != 1:
        #     PREV = None
        # else:
        #     PREV = tokens

        # Tokenize the context and combine it with the patch tokens.
        # atok = self.atok.forward(ctx[0])
        # rtok = self.rtok.forward(ctx[1])

        # if len(rtok.shape) == 1:
        #     rtok = rtok.unsqueeze(0)

        # posn = torch.arange(0, 2, dtype=torch.int64, device=DEV).unsqueeze(0)
        # posn = posn.expand([patches.size(0), -1])
        # pemb = self.pemb.forward(posn)

        # if atok.size(0) != 1:
        #     info(atok.shape, rtok.shape, tokens.shape, pemb.shape)

        # context = torch.stack([atok, rtok], dim=1)
        # context += pemb

        # tokens = torch.cat([tokens, context], dim=1)

        # Pass the tokens through the GTrXL.
        tform: dict[str, Tensor] = self.gtrxl.forward(tokens.permute(1, 0, 2))
        out = tform["logits"]  # Shape: [n_patches, batch, d_model]
        out = out.permute(1, 0, 2)

        # Compress the size of each token.
        batch = out.size(0)
        patch = out.size(1)
        out = out.flatten(0, 1)
        out = self.comp.forward(out)
        out = out.view(batch, patch, self.args.d_comp)

        return out

    def get_value(self, x: Tensor):
        out = self.tform(x)
        return self.critic(out.flatten(1))

    def get_action_and_value(self, x: Tensor, action: tuple[Tensor, Tensor, Tensor, Tensor] | None = None, stoch=None):
        out = self.tform(x)
        if stoch is not None:
            if not stoch:
                # tensorhue.viz(x.cpu()[0][0] + x.cpu()[0][1] + x.cpu()[0][2])
                # tensorhue.viz(out.cpu().squeeze(0).view(50, 256), vmin=-1, vmax=1)
                # tviz(out.squeeze(0).view(50, 256))
                pass
        
        # Get GMM parameters directly
        if stoch is None:
            means, logstds, weights_logits, thingy, md, ld, wd = self.gauss.forward(out.flatten(1), stochastic=self.stochastic)
        else:
            means, logstds, weights_logits, thingy, md, ld, wd = self.gauss.forward(out.flatten(1), stochastic=stoch)
            if not stoch and not self.args.test:
                global PREV
                if PREV is not None:
                    tviz((means.squeeze(0) - PREV[0].squeeze(0)).t())
                    tviz((logstds.squeeze(0) - PREV[1].squeeze(0)).t())
                    tviz(weights_logits - PREV[2])
                    print()
                
                PREV = (means, logstds, weights_logits)
                # tviz(means.squeeze(0).t())
                # tviz(logstds.squeeze(0).t())
                # tviz(weights_logits)
        
        # Create batch of GMMs
        batch_size = means.shape[0]
        gmm_list = []
        
        for i in range(batch_size):
            mlogstds = logstds.clone()
            mlogstds[..., 0] = torch.where(weights_logits < 0.5, 13, mlogstds[..., 0])
            mlogstds[..., 1] = torch.where(weights_logits < 0.5, 13, mlogstds[..., 1])
            mweights_logits = torch.where(weights_logits < 0.5, -1e10, weights_logits)
            # Create component distribution
            component_dist = MultivariateNormal(
                loc=means[i],
                scale_tril=torch.diag_embed(torch.exp(mlogstds[i]))
            )
            
            # Create mixture distribution 
            mixture_dist = Categorical(logits=mweights_logits[i])
            # print(mixture_dist.logits)
            # print(torch.cat([weights_logits[i], torch.full((1,), 10).to(DEV)], dim=0))
            
            # Create the GMM
            gmm = MixtureSameFamily(
                mixture_distribution=mixture_dist,
                component_distribution=component_dist
            )
            
            gmm_list.append(gmm)
        
        gmm = GaussMix(gmm_list)

        sincos = self.sincos.forward(out.flatten(1))
        scd = Normal(sincos, self.sincos_std.exp())
        if stoch is None:
            stoch = self.stochastic

        if stoch:
            sincos = scd.sample()
        
        # Sample actions
        if action is None:
            action = (means, logstds, weights_logits, sincos)
        
        # Calculate log probabilities and entropy of the GMM
        log_prob = md.log_prob(action[0]).sum(-1).sum(-1) + ld.log_prob(action[1]).sum(-1).sum(-1) + wd.log_prob(action[2]).sum(-1) + scd.log_prob(action[3]).sum(-1)
        entropy = md.entropy().sum(-1).sum(-1) + ld.entropy().sum(-1).sum(-1) + wd.entropy().sum(-1) + scd.entropy().sum(-1)
        
        params = (means, logstds, weights_logits, sincos)
        return thingy, params, gmm, log_prob, entropy, self.critic.forward(out.flatten(1))
    
    def get_logprobs(self, dist: Normal, means: Tensor) -> Tensor:
        log_prob = dist.log_prob(means).sum(-1).sum(-1)
        return log_prob

    
    def viz_3d(self, params: Tuple[Tensor, Tensor, Tensor, Normal, Normal, Normal], gmm: GaussMix, jitter, image_tensor, step=0, save_path=None):
        """Create a 3D visualization of the GMM PDF overlaid on the input image."""
        with torch.no_grad():
            # Save visualization
            def thing():
                if save_path:
                    os.makedirs(save_path, exist_ok=True)
                    
                fig = visualize_gmm_3d_on_image(
                    gmm,
                    jitter,
                    image_tensor,
                    # samples=greedy_sample_np,
                    title=f"GMM Distribution at step {step}",
                    save_path=f"{save_path}/gmm_3d_overlay_{step}.png" if save_path else None
                )
                
                if not save_path:
                    return fig
                
            t = threading.Thread(target=thing)
            t.start()
            # thing()