##! This file contains the runner, an object used to contain everything related
##! to the training loop. This also provides the functions that will be called
##! from Rust through the socket bridge.
##!
##! Training logic adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py

import os
import random
import time
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms

from config import DEV, info
from ppo import Agent, Args

class Runner:
    def __init__(self, args: Args):
        """Create a new runner."""

        self.args = args
        self.args.batch_size = int(args.num_envs * args.num_steps)
        self.args.minibatch_size = int(args.batch_size // args.num_minis)
        self.args.num_iterations = (args.episodes * args.num_steps) // args.batch_size
        
        self.run_name = f"{args.exp_name}_{int(time.time())}"
        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=self.run_name,
                monitor_gym=True,
                save_code=True,
            )

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        self.agent = Agent(args).to(DEV)
        self.optimizer = optim.AdamW(self.agent.parameters(), lr=args.lr, eps=1e-5)
        
        self.obs = torch.zeros((args.num_steps, args.num_envs, args.img_chan, args.img_size, args.img_size)).to(DEV)
        self.actions = torch.zeros((args.num_steps, args.num_envs, args.num_prune, 2)).to(DEV)
        self.logprobs = torch.zeros((args.num_steps, args.num_envs, 1)).to(DEV)
        self.rewards = torch.zeros((args.num_steps, args.num_envs)).to(DEV)
        self.dones = torch.zeros((args.num_steps, args.num_envs)).to(DEV)
        self.values = torch.zeros((args.num_steps, args.num_envs)).to(DEV)

        self.param_mean = torch.zeros((args.num_steps, args.num_envs, args.num_gauss, 2)).to(DEV)
        self.param_std = torch.zeros((args.num_steps, args.num_envs, args.num_gauss, 2)).to(DEV)
        self.param_weight = torch.zeros((args.num_steps, args.num_envs, args.num_gauss)).to(DEV)

        self.global_step = 0
        self.start_time = time.time()

        images = []
        for i in range(args.num_envs):
            image = Image.open(f"{i}.png")
            transform = transforms.ToTensor()
            images.append(transform(image))#[0].unsqueeze(0))

        self.next_obs = torch.stack(images, dim=0).to(DEV)
        self.eval_obs = torch.stack(images, dim=0).to(DEV)
        self.next_done = torch.zeros(args.num_envs).to(DEV)

        info(f"Parameters: {sum(p.numel() for p in self.agent.parameters() if p.requires_grad) / 1_000_000}M")
        info(f"  GTrXL: {sum(p.numel() for p in self.agent.gtrxl.parameters() if p.requires_grad) / 1_000_000}M")
        info(f"  GMM: {sum(p.numel() for p in self.agent.gauss.parameters() if p.requires_grad) / 1_000_000}M")
        info(f"  Critic: {sum(p.numel() for p in self.agent.critic.parameters() if p.requires_grad) / 1_000_000}M")

    def init_writer(self):
        self.writer = SummaryWriter(f"{self.args.save_dir}/{self.run_name}")
    
    def step(self, step: int, tree: dict) -> Tuple[list[int], list[float]]:
        """Collect an experience."""

        self.global_step += self.args.num_envs
        self.obs[step] = self.next_obs
        self.dones[step] = self.next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, params, gmm, logprob, ent, value = self.agent.get_action_and_value(self.next_obs)
            # info(params[0].shape, params[1].shape, params[2].shape, logprob.shape, value.shape)
            self.values[step] = value.flatten()
            self.param_mean[step] = params[0]
            self.param_std[step] = params[1]
            self.param_weight[step] = params[2]
            # self.actions[step] = action
            self.logprobs[step] = logprob
            # print(logprob.shape)

        if self.global_step % 200 == 0:
            img = self.next_obs
            # Use the 2D visualization instead of 3D
            self.agent.viz_3d(
                params,
                gmm,
                image_tensor=img,
                step=self.global_step,
                save_path=f"{self.args.save_dir}/{self.run_name}/viz"
            )

        # need to return probs for each (x, y) coord
        coords = torch.tensor(list(tree.values())).to(DEV).unsqueeze(0)
        probs = gmm.log_prob(coords).to(DEV).exp()
        # action.squeeze_(0)
        return ([int(num) for num in list(tree.keys())], probs.squeeze(0).tolist())
    

    def step_eval(self, step: int, tree: dict) -> Tuple[list[int], list[float]]:
        """Just eval the model."""
        with torch.no_grad():
            action, params, gmm, logprob, ent, value = self.agent.get_action_and_value(self.eval_obs, stoch=False)
            coords = torch.tensor(list(tree.values())).to(DEV).unsqueeze(0)
            probs = gmm.log_prob(coords).to(DEV).exp()
            return ([int(num) for num in list(tree.keys())], probs.squeeze(0).tolist())
    
    
    def next(self, step: int, rewards: List[float], terms: List[bool]):
        """Collect remaining experience info after robot has taken an action in Rust."""

        images = []
        for i in range(self.args.num_envs):
            image = Image.open(f"{i}.png")
            transform = transforms.ToTensor()
            images.append(transform(image))#[0].unsqueeze(0))

        self.next_obs = torch.stack(images, dim=0).to(DEV)
        self.next_done = torch.tensor(terms).to(DEV).view(-1)
        self.rewards[step] = torch.tensor(rewards).to(DEV).view(-1)


    def next_eval(self, step: int, rewards: List[float], terms: List[bool]):
        images = []
        for i in range(self.args.num_envs):
            image = Image.open(f"{i+1}.png")
            transform = transforms.ToTensor()
            images.append(transform(image))#[0].unsqueeze(0))

        self.eval_obs = torch.stack(images, dim=0).to(DEV)


    def train(self, iteration: int):
        """Perform a PPO policy update."""

        if self.args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / self.args.num_iterations
            lrnow = frac * self.args.lr
            self.optimizer.param_groups[0]["lr"] = lrnow

        with torch.no_grad():
            next_value = self.agent.get_value(self.next_obs).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards).to(DEV)
            lastgaelam = 0
            for t in reversed(range(self.args.num_steps)):
                if t == self.args.num_steps - 1:
                    nextnonterminal = 1.0 - self.next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + self.args.gamma * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.values

        # flatten the batch
        b_obs = self.obs.reshape((-1, self.args.img_chan, self.args.img_size, self.args.img_size))
        b_logprobs = self.logprobs.reshape(-1)
        # b_actions = self.actions.reshape((-1, 2))
        b_means = self.param_mean.reshape((-1, self.args.num_gauss, 2))
        b_stds = self.param_std.reshape((-1, self.args.num_gauss, 2))
        b_weights = self.param_weight.reshape((-1, self.args.num_gauss))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        # info(b_obs.shape, b_logprobs.shape, b_means.shape, b_stds.shape, b_weights.shape, b_advantages.shape, b_returns.shape, b_values.shape)

        # Optimizing the policy and value network
        b_inds = np.arange(self.args.batch_size)
        clipfracs = []
        approx_kl = v_loss = pg_loss = entropy_loss = old_approx_kl = approx_kl = torch.empty(0)

        for epoch in range(self.args.ppo_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.args.batch_size, self.args.minibatch_size):
                end = start + self.args.minibatch_size
                mb_inds = b_inds[start:end]

                _, _, _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], (b_means[mb_inds], b_stds[mb_inds], b_weights[mb_inds]))
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                # info(b_obs[mb_inds].shape, b_means[mb_inds].shape, b_stds[mb_inds].shape, b_weights[mb_inds].shape, newlogprob.shape, entropy.shape, newvalue.shape, ratio.shape)

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss + v_loss * self.args.vf_coef + entropy_loss * self.args.ent_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

            if self.args.early_stop and approx_kl > self.args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
        self.writer.add_scalar("losses/value_loss", v_loss.detach().item(), self.global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss.detach().item(), self.global_step)
        self.writer.add_scalar("losses/entropy", entropy_loss.detach().item(), self.global_step)
        self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.global_step)
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.global_step)
        self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.global_step)
        self.writer.add_scalar("losses/explained_variance", explained_var, self.global_step)
        info("SPS:", int(self.global_step / (time.time() - self.start_time)))
        self.writer.add_scalar("charts/SPS", int(self.global_step / (time.time() - self.start_time)), self.global_step)


    def plot(self, r: float, c: float, er: float, ec: float):
        self.writer.add_scalar("rewards/rewards", r, self.global_step)
        self.writer.add_scalar("rewards/coverage", c, self.global_step)
        self.writer.add_scalar("rewards/eval_rewards", er, self.global_step)
        self.writer.add_scalar("rewards/eval_coverage", ec, self.global_step)
        if self.args.test:
            self.global_step += 1


    def save(self):
        """Save the model."""

        model_path = f"{self.args.save_dir}/{self.run_name}/{self.args.exp_name}{self.global_step // 6400}.pt"
        torch.save(self.agent.state_dict(), model_path)
        info(f"model saved to {model_path}")

    
    def load(self, run_name: str, checkpoint: int):
        """Load the model."""

        model_path = f"{self.args.save_dir}/{run_name}/{self.args.exp_name}{checkpoint}.pt"
        self.agent.load_state_dict(torch.load(model_path))
        info(f"model loaded from {model_path}")

    def rs(self, amt: float, start: int, end: int):
        """Scale rewards for a simulation based on final coverage achieved."""
        self.rewards[start : end] += amt