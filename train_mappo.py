from typing import Dict, List
import csv
from pathlib import Path

import numpy as np
import torch

import config as cfg
from env import SumoEnv, set_seed
from mappo_agent import MAPPOAgent, Trajectory


def flatten_obs(obs_dict: Dict[str, np.ndarray], agent_order: List[str]) -> torch.Tensor:
    return torch.cat([torch.tensor(obs_dict[aid], dtype=torch.float32) for aid in agent_order], dim=0)


def main():
    conf = cfg.MAPPOConfig()
    set_seed(conf.seed)

    env = SumoEnv(use_gui=cfg.USE_GUI_DEFAULT, step_length=cfg.DEFAULT_STEP_LENGTH)
    # 관측 차원은 설정된 차로 수로 직접 계산하여 초기 reset을 한 번으로 제한
    obs_dim = len(cfg.OBS_LANES_PER_TLS[cfg.TLS_IDS[0]])
    action_dim = len(conf.action_phases)

    agent = MAPPOAgent(agent_ids=cfg.TLS_IDS, obs_dim=obs_dim, action_dim=action_dim, config=conf)
    best_reward = -float("inf")
    save_path = cfg.SITE_ROOT / "models" / "mappo_latest.pth"
    log_path = cfg.SITE_ROOT / "logs" / "mappo_train.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_episode(ep_idx: int, steps: int, reward: float, actor_loss: float, critic_loss: float, best_reward_val: float) -> None:
        """에피소드별 학습 결과를 CSV로 기록."""
        write_header = not log_path.exists()
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["episode", "steps", "reward", "actor_loss", "critic_loss", "best_reward"])
            writer.writerow([ep_idx, steps, reward, actor_loss, critic_loss, best_reward_val])

    for ep in range(conf.total_episodes):
        obs = env.reset()
        traj_obs = []
        traj_actions = []
        traj_logps = []
        traj_values = []
        traj_rewards = []
        traj_dones = []
        traj_agent_ids = []

        ep_reward = 0.0
        done = False
        steps = 0

        while not done and steps < conf.rollout_steps:
            actions, logps, values = agent.select_actions(obs)
            next_obs, rewards, done = env.step(actions)

            # 모든 에이전트 관측/행동/가치/보상을 순회 저장 (길이 정합 유지)
            for aid in cfg.TLS_IDS:
                traj_obs.append(torch.tensor(obs[aid], dtype=torch.float32))
                traj_actions.append(torch.tensor(actions[aid]))
                traj_logps.append(logps[aid])
                traj_values.append(values[aid])
                # 모든 에이전트에 동일 보상/종료 플래그 사용
                shared_rew = np.mean(list(rewards.values()))
                traj_rewards.append(shared_rew)
                traj_dones.append(float(done))
                traj_agent_ids.append(aid)

            ep_reward += np.mean(list(rewards.values()))
            obs = next_obs
            steps += 1

        # 업데이트
        traj = Trajectory(
            obs=traj_obs,
            actions=traj_actions,
            log_probs=traj_logps,
            rewards=traj_rewards,
            dones=traj_dones,
            values=traj_values,
            agent_ids=traj_agent_ids,
        )
        metrics = agent.update(traj)

        print(
            f"[EP {ep+1}/{conf.total_episodes}] "
            f"steps={steps} reward={ep_reward:.2f} "
            f"actor_loss={metrics['actor_loss']:.4f} critic_loss={metrics['critic_loss']:.4f}"
        )
        
        log_episode(ep, steps, ep_reward, metrics['actor_loss'], metrics['critic_loss'], ep_reward)
        
        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(save_path)


    env.close()


if __name__ == "__main__":
    main()
