##! This file contains the runner, an object used to contain everything related
##! to the training loop. This also provides the functions that will be called
##! from Rust through the socket bridge.
##!
##! Training logic adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py

import os
from pathlib import Path
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
from welford import WelfordNorm


def _agent_raw(agent):
    """Peel `torch.compile` wrappers so we can reach the real modules.
    torch.compile puts the wrapped module at `._orig_mod`; peel repeatedly for
    the (rare) doubly-compiled case that can happen if we wrap tform then wrap
    the whole agent."""
    m = agent
    for _ in range(4):
        if hasattr(m, "_orig_mod"):
            m = m._orig_mod
        else:
            break
    return m


@torch.no_grad()
def probe_sensitivity(agent, obs, writer=None, step=None):
    """Compare the agent's GMM parameters on `obs` vs an all-zero baseline.
    Ratio near 0 means the output does not depend on the input; ratio ~1 means
    input drives a big fraction of the output magnitude."""
    obs2 = torch.zeros_like(obs)
    _, p1, _, _, _, _ = agent.get_action_and_value(obs,  stoch=False)
    _, p2, _, _, _, _ = agent.get_action_and_value(obs2, stoch=False)
    results = {}
    for name, a, b in [
        ("means",   p1[0], p2[0]),
        ("logstds", p1[1], p2[1]),
        ("weights", p1[2], p2[2]),
    ]:
        rel = ((a - b).abs().mean() / (a.abs().mean() + 1e-8)).item()
        results[name] = rel
        info(f"sensitivity/{name}: |delta|/|.| = {rel:.4f}")
        if writer is not None:
            writer.add_scalar(f"sensitivity/{name}", rel, step)
    return results


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

        # random.seed(args.seed)
        # np.random.seed(args.seed)
        # torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        # Enable TF32 on Ampere+ GPUs.
        torch.set_float32_matmul_precision("high")
        # Let dynamo trace `.item()` on scalar tensors instead of breaking the
        # graph. Needed because gtrxl.py's attention mask uses
        # `if mask.any().item():` as Python-side control flow.
        torch._dynamo.config.capture_scalar_outputs = True

        self.agent = Agent(args).to(DEV)
        self.optimizer = optim.AdamW(self.agent.parameters(), lr=args.lr, eps=1e-5)

        # Compile the per-step tform (patchify + GTrXL + compress), which is where
        # the vast majority of forward-pass time lives. Batch shape varies between
        # rollout (num_envs) and PPO update (minibatch), so `dynamic=True`.
        if hasattr(torch, "compile"):
            try:
                self.agent.tform = torch.compile(self.agent.tform, dynamic=True)
                info("\x1b[93mtorch.compile enabled on Agent.tform\x1b[0m")
            except Exception as e:
                info(f"\x1b[92mwarning: torch.compile disabled:\x1b[0m {e}")
        
        self.obs = torch.zeros((args.num_steps, args.num_envs, args.img_chan, args.img_size, args.img_size)).to(DEV)
        self.actions = torch.zeros((args.num_steps, args.num_envs, args.num_prune, 2)).to(DEV)
        self.logprobs = torch.zeros((args.num_steps, args.num_envs, 1)).to(DEV)
        self.rewards = torch.zeros((args.num_steps, args.num_envs)).to(DEV)
        self.dones = torch.zeros((args.num_steps, args.num_envs)).to(DEV)
        self.values = torch.zeros((args.num_steps, args.num_envs)).to(DEV)

        # Welford running-stats normalizer on the incoming reward stream.
        # Keeps returns in a unit-variance range so `v_loss * vf_coef` doesn't
        # dwarf `pg_loss` on the shared GTrXL. Frozen during eval via tmode/emode.
        self.welford = WelfordNorm(1)

        self.param_mean = torch.zeros((args.num_steps, args.num_envs, args.num_gauss, 2)).to(DEV)
        self.param_std = torch.zeros((args.num_steps, args.num_envs, args.num_gauss, 2)).to(DEV)
        self.param_weight = torch.zeros((args.num_steps, args.num_envs, args.num_gauss)).to(DEV)

        # Per-step snapshot of the GTrXL input memory. During PPO update we
        # replay each sample under the exact memory context the actor consumed
        # at collection time, ensuring newlogprob is correct. Shape per layer:
        #   (num_steps, num_envs, mem_len, d_model)
        # Left-padded with zeros for warmup steps where actual memory length
        # was < mem_len. Allocated lazily on the first Runner.step call once
        # we can read the layer count off self.agent.gtrxl_memory.
        self.saved_mems: list[torch.Tensor] | None = None

        self.global_step = 0
        self.start_time = time.time()

        # images = []
        # for i in range(args.num_envs):
        #     image = Image.open(f"{i}.png")
        #     transform = transforms.ToTensor()
        #     images.append(transform(image))#[0].unsqueeze(0))

        # New 07-01-2026: reduce latency of transferring state from Rust
        images = []
        for i in range(args.num_envs):
            with open(f"/dev/shm/sim_obs_{i}.bin", "rb") as f:
                data = np.frombuffer(f.read(), dtype=np.uint8).reshape(self.args.img_size, self.args.img_size, self.args.img_chan)
            images.append(torch.tensor(data, dtype=torch.float32).permute(2, 0, 1).div_(255.0))

        self.next_obs = torch.stack(images, dim=0).to(DEV)
        self.eval_obs = torch.stack(images, dim=0).to(DEV)
        self.next_done = torch.zeros(args.num_envs).to(DEV)

        info(f"Parameters: {sum(p.numel() for p in self.agent.parameters() if p.requires_grad) / 1_000_000}M")
        info(f"  GTrXL: {sum(p.numel() for p in self.agent.gtrxl.parameters() if p.requires_grad) / 1_000_000}M")
        info(f"  GMM: {sum(p.numel() for p in self.agent.gauss.parameters() if p.requires_grad) / 1_000_000}M")
        info(f"  Critic: {sum(p.numel() for p in self.agent.critic.parameters() if p.requires_grad) / 1_000_000}M")

        # if not args.test:
        #     info("Compiling agent with mode=\"reduce-overhead\"")
        #     self.agent = torch.compile(self.agent, mode="reduce-overhead")

    def init_writer(self):
        self.writer = SummaryWriter(f"{self.args.save_dir}/{self.run_name}")

    def _snapshot_mem(self, step: int):
        """
        Copy the current input memory of self.agent into self.saved_mems[step].

        Each layer's memory is a tensor of shape (L, num_envs, d) with
        0 <= L <= mem_len (grows from 0 up to mem_len during warmup and after
        py.b()/py.eb() resets). We left-pad with zeros to length mem_len so
        every step's snapshot has the same shape and can be batch-gathered
        during PPO update.
        """
        cur = _agent_raw(self.agent).gtrxl_memory
        if self.saved_mems is None:
            mem_len = self.args.gtrxl_mem_len
            d = self.args.d_model
            # Kept on CPU (pinned). This buffer is too large to maintain in VRAM without
            # affecting PPO updates. We move the needed slice back to VRAM as-needed.
            self.saved_mems = [
                torch.zeros(
                    (self.args.num_steps, self.args.num_envs, mem_len, d),
                    device="cpu",
                    pin_memory=torch.cuda.is_available(),
                )
                for _ in cur
            ]

        mem_len = self.args.gtrxl_mem_len
        for li, m in enumerate(cur):
            L = m.size(0)
            if L == 0:
                self.saved_mems[li][step].zero_()
                continue
            # m: (L, num_envs, d). Place at the tail so the newest entries sit
            # against the current-input boundary — this matches how GTrXL
            # extends its memory when a real (unpadded) L-length memory is
            # given.
            pad = mem_len - L
            self.saved_mems[li][step, :, :pad].zero_()
            # Move the live GPU memory to the CPU buffer.
            self.saved_mems[li][step, :, pad:].copy_(
                m.permute(1, 0, 2).detach(), non_blocking=True
            )
    
    def step(self, step: int, tree: dict) -> Tuple[list[int], list[float]]:
        """Collect an experience."""

        self.global_step += self.args.num_envs
        self.obs[step] = self.next_obs
        self.dones[step] = self.next_done

        # Snapshot the memory the policy is about to consume at this step.
        # tform will advance self.agent.gtrxl_memory during forward pass.
        self._snapshot_mem(step)

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
            # info("visualizing")
            img = self.next_obs.clone()
            _, dparams, dgmm, _, _, _ = self.agent.get_action_and_value(self.next_obs, stoch=False)
            # Use the 2D visualization instead of 3D
            self.agent.viz_3d(
                dparams,
                dgmm,
                image_tensor=img,
                step=self.global_step,
                save_path=f"{self.args.save_dir}/{self.run_name}/viz",
                suffix="d"
            )
        elif self.global_step % 200 == 100:
            # info("visualizing")
            img = self.next_obs.clone()
            # _, dparams, dgmm, _, _, _ = self.agent.get_action_and_value(self.next_obs, stoch=False)
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
            action, params, gmm, logprob, ent, value = self.agent.get_action_and_value(self.eval_obs, stoch=False, eval=True)
            coords = torch.tensor(list(tree.values())).to(DEV).unsqueeze(0)
            probs = gmm.log_prob(coords).to(DEV).exp()
            return ([int(num) for num in list(tree.keys())], probs.squeeze(0).tolist())
    
    
    def next(self, step: int, rewards: List[float], terms: List[bool]):
        """Collect remaining experience info after robot has taken an action in Rust."""

        # images = []
        # for i in range(self.args.num_envs):
        #     image = Image.open(f"{i}.png")
        #     transform = transforms.ToTensor()
        #     images.append(transform(image))#[0].unsqueeze(0))

        # New 07-01-2026: reduce latency of transferring state from Rust
        images = []
        for i in range(self.args.num_envs):
            with open(f"/dev/shm/sim_obs_{i}.bin", "rb") as f:
                data = np.frombuffer(f.read(), dtype=np.uint8).reshape(self.args.img_size, self.args.img_size, self.args.img_chan)
            images.append(torch.tensor(data, dtype=torch.float32).permute(2, 0, 1).div_(255.0))

        self.next_obs = torch.stack(images, dim=0).to(DEV)
        self.next_done = torch.tensor(terms).to(DEV).view(-1)
        self.rewards[step] = self.welford.norm(torch.tensor(rewards)).to(DEV).view(-1)


    def next_eval(self, step: int, rewards: List[float], terms: List[bool]):
        # images = []
        # for i in range(self.args.num_envs):
        #     image = Image.open(f"{i+1}.png")
        #     transform = transforms.ToTensor()
        #     images.append(transform(image))#[0].unsqueeze(0))

        # New 07-01-2026: reduce latency of transferring state from Rust
        images = []
        for i in range(self.args.num_envs):
            with open(f"/dev/shm/sim_obs_{i+1}.bin", "rb") as f:
                data = np.frombuffer(f.read(), dtype=np.uint8).reshape(self.args.img_size, self.args.img_size, self.args.img_chan)
            images.append(torch.tensor(data, dtype=torch.float32).permute(2, 0, 1).div_(255.0))

        self.eval_obs = torch.stack(images, dim=0).to(DEV)

    def boundary(self):
        # Rust has just done reset+step+export_state(0). Refresh next_obs from
        # shmem so the first step of the new episode sees the post-reset image,
        # not the pre-reset one that py.next() latched.
        self.agent.reset_mem()
        images = []
        for i in range(self.args.num_envs):
            with open(f"/dev/shm/sim_obs_{i}.bin", "rb") as f:
                data = np.frombuffer(f.read(), dtype=np.uint8).reshape(self.args.img_size, self.args.img_size, self.args.img_chan)
            images.append(torch.tensor(data, dtype=torch.float32).permute(2, 0, 1).div_(255.0))
        self.next_obs = torch.stack(images, dim=0).to(DEV)

    def eboundary(self):
        # Same as boundary() but for the eval buffer.
        self.agent.reset_eval_mem()
        images = []
        for i in range(self.args.num_envs):
            with open(f"/dev/shm/sim_obs_{i+1}.bin", "rb") as f:
                data = np.frombuffer(f.read(), dtype=np.uint8).reshape(self.args.img_size, self.args.img_size, 1)
            images.append(torch.tensor(data, dtype=torch.float32).permute(2, 0, 1).div_(255.0))
        self.eval_obs = torch.stack(images, dim=0).to(DEV)

    def train(self, iteration: int):
        """Perform a PPO policy update."""

        if self.args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / (2 * self.args.num_iterations)
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

                # Gather the input memory each of these mb_inds samples
                # actually consumed during rollout. saved_mems lives on CPU
                # (pinned); we transfer just this minibatch's slice per step.
                # Each saved_mems[layer] is (num_steps, num_envs, mem_len, d);
                # we want the GTrXL `memory` layout (mem_len, batch, d) where
                # batch = mb_size * num_envs, matching the flattened b_obs[mb_inds] shape.
                mb_inds_cpu = torch.as_tensor(mb_inds, device="cpu")
                mem_override = [
                    m.index_select(0, mb_inds_cpu)     # (mb, envs, mem_len, d)
                     .to(DEV, non_blocking=True)
                     .reshape(-1, m.size(2), m.size(3)) # (mb*envs, mem_len, d)
                     .permute(1, 0, 2)                  # (mem_len, mb*envs, d)
                     .contiguous()
                    for m in self.saved_mems
                ]

                _, _, _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                    b_obs[mb_inds],
                    (b_means[mb_inds], b_stds[mb_inds], b_weights[mb_inds]),
                    mem_override=mem_override
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                #info(b_obs[mb_inds].shape, b_means[mb_inds].shape, b_stds[mb_inds].shape, b_weights[mb_inds].shape, newlogprob.shape, entropy.shape, newvalue.shape, ratio.shape)

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

                if iteration < self.args.warmup:
                    loss = v_loss * self.args.vf_coef
                else:
                    loss = pg_loss + v_loss * self.args.vf_coef - entropy_loss * self.args.ent_coef

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
        self.writer.add_scalar("charts/SPS", int(self.global_step / (time.time() - self.start_time)), self.global_step)

        info("charts/SPS =", int(self.global_step / (time.time() - self.start_time)))
        info("losses/value_loss", v_loss.detach().item(), self.global_step)
        info("losses/entropy =", entropy_loss.detach().item())
        info("losses/approx_kl =", approx_kl.item())
        info("losses/clipfrac =", np.mean(clipfracs))
        info("losses/explained_variance =", explained_var)

        # Sensitivity probe every 10 iters: swap input with all-zero and see how
        # much the deterministic GMM parameters actually shift.
        iter_idx = self.global_step // max(self.args.batch_size, 1)
        if iter_idx % 10 == 0:
            probe_sensitivity(self.agent, self.next_obs,
                              writer=self.writer, step=self.global_step)


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

    
    def load(self, run_name: str, checkpoint: str):
        """Load the model."""

        model_path_a = f"{self.args.save_dir}/{run_name}/{self.args.exp_name}{checkpoint}.pt"
        model_path_b = f"{self.args.save_dir}/{run_name}/{checkpoint}.pt"

        if Path(model_path_a).exists():
            self.agent.load_state_dict(torch.load(model_path_a))
            info(f"model loaded from {model_path_a}")
        elif Path(model_path_b).exists():
            self.agent.load_state_dict(torch.load(model_path_b))
            info(f"model loaded from {model_path_b}")
        else:
            raise FileNotFoundError(f"Specified model checkpoint not found. Tried:\n  - {model_path_a}\n  - {model_path_b}")


    def rs(self, amt: float, start: int, end: int):
        """Scale rewards for a simulation based on final coverage achieved."""
        self.rewards[start : end] += amt