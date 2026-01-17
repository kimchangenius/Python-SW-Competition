import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def mlp(in_dim: int, out_dim: int, hidden: int = 128) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = mlp(obs_dim, action_dim, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = mlp(obs_dim, 1, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Trajectory:
    obs: List[torch.Tensor]
    actions: List[torch.Tensor]
    log_probs: List[torch.Tensor]
    rewards: List[float]
    dones: List[float]
    values: List[torch.Tensor]
    agent_ids: List[str]


class MAPPOAgent:
    def __init__(self, agent_ids: List[str], obs_dim: int, action_dim: int, config):
        self.agent_ids = agent_ids
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cfg = config

        self.actor = Actor(obs_dim, action_dim)
        self.critic = Critic(obs_dim)

        self.optim_actor = optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)

    # --------- 저장/로드 ----------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "config": self.cfg.__dict__,
                "agent_ids": self.agent_ids,
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
            },
            path,
        )
        print(f"모델 저장 완료: {path}")

    def select_actions(self, obs_dict: Dict[str, np.ndarray]) -> Tuple[Dict[str, int], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        actions = {}
        logps = {}
        values = {}

        with torch.no_grad():
            for aid in self.agent_ids:
                obs = torch.tensor(obs_dict[aid], dtype=torch.float32).to(self.device)
                logits = self.actor(obs)
                probs = torch.distributions.Categorical(logits=logits)
                act = probs.sample()

                actions[aid] = int(act.item())
                logps[aid] = probs.log_prob(act)
                values[aid] = self.critic(obs)

        return actions, logps, values

    def _calc_gae(self, rewards, values, dones):
        advantages = []
        gae = 0.0
        next_value = 0.0
        for step in reversed(range(len(rewards))):
            mask = 1.0 - dones[step]
            delta = rewards[step] + self.cfg.gamma * next_value * mask - values[step]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * mask * gae
            advantages.insert(0, gae)
            next_value = values[step]
        return advantages

    def update(self, traj: Trajectory) -> Dict[str, float]:
        # stack
        obs = torch.stack(traj.obs).to(self.device)
        actions = torch.stack(traj.actions).to(self.device)
        old_logps = torch.stack(traj.log_probs).to(self.device)
        values = torch.stack(traj.values).squeeze(-1).to(self.device)

        rewards = torch.tensor(traj.rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(traj.dones, dtype=torch.float32).to(self.device)

        adv = torch.tensor(self._calc_gae(rewards.cpu().numpy(), values.detach().cpu().numpy(), dones.cpu().numpy()), dtype=torch.float32).to(self.device)
        returns = adv + values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        batch_size = len(obs)
        idxs = np.arange(batch_size)
        metrics = {}

        for _ in range(self.cfg.train_epochs):
            np.random.shuffle(idxs)
            splits = np.array_split(idxs, self.cfg.mini_batch)
            for split in splits:
                b_idx = torch.tensor(split, dtype=torch.long).to(self.device)
                b_obs = obs[b_idx]
                b_actions = actions[b_idx]
                b_old_logps = old_logps[b_idx]
                b_adv = adv[b_idx]
                b_returns = returns[b_idx]

                # actor
                logits = self.actor(b_obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_logps = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratio = (new_logps - b_old_logps).exp()
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.cfg.eps_clip, 1 + self.cfg.eps_clip) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean() - self.cfg.entropy_coef * entropy

                # critic
                # critic는 전체 obs를 입력으로 해야 하지만, 여기서는 간단히 동일 obs 사용
                value_pred = self.critic(b_obs).squeeze(-1)
                critic_loss = (b_returns - value_pred).pow(2).mean() * self.cfg.value_coef

                loss = actor_loss + critic_loss

                self.optim_actor.zero_grad()
                self.optim_critic.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.optim_actor.step()
                self.optim_critic.step()

        metrics["actor_loss"] = float(actor_loss.item())
        metrics["critic_loss"] = float(critic_loss.item())
        metrics["entropy"] = float(entropy.item())
        return metrics
