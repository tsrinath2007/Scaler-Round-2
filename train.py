"""
train.py — Train a PPO agent on LifeSupportEnv using stable-baselines3.

RECOMMENDED USAGE:
  Local (fast demo, ~2 min):
    python train.py --task task_medium --timesteps 300000

  HuggingFace (proper training):
    python train.py --task task_hard --timesteps 2000000

WHY task_medium?
  task_easy only lasts 24 steps — random agents almost always survive it,
  so the trained agent has nothing to learn above the baseline.
  task_medium (168 steps, 5 crew) is where random agents regularly fail,
  making the PPO improvement visible and meaningful.

Output:
  models/<task_id>/ppo_lifesupport.zip  — trained model
  training_curve.png                    — episode reward over training
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from gym_wrapper import LifeSupportGymEnv


class RewardLoggerCallback(BaseCallback):
    """Records per-episode reward and survival length."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self._ep_reward = 0.0
        self._ep_len    = 0

    def _on_step(self) -> bool:
        self._ep_reward += self.locals["rewards"][0]
        self._ep_len    += 1
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._ep_reward)
            self.episode_lengths.append(self._ep_len)
            self._ep_reward = 0.0
            self._ep_len    = 0
        return True


def make_env(task_id: str, seed: int = 0):
    def _init():
        env = LifeSupportGymEnv(task_id=task_id, seed=seed)
        return Monitor(env)
    return _init


def plot_training_curve(rewards, lengths, save_path: str, task_id: str):
    if not rewards:
        print("No episodes completed — skipping plot.")
        return

    window = max(1, len(rewards) // 20)
    smooth_r = np.convolve(rewards, np.ones(window) / window, mode="valid")
    smooth_l = np.convolve(lengths, np.ones(window) / window, mode="valid")
    x_smooth = range(window - 1, len(rewards))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax1.plot(rewards,  alpha=0.25, color="steelblue")
    ax1.plot(x_smooth, smooth_r,   color="steelblue", linewidth=2,
             label=f"Smoothed (window={window})")
    ax1.set_ylabel("Episode Reward")
    ax1.set_title(f"PPO Training — {task_id}")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(lengths,  alpha=0.25, color="coral")
    ax2.plot(x_smooth, smooth_l,   color="coral", linewidth=2,
             label=f"Smoothed (window={window})")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps Survived")
    ax2.set_title("Survival Length per Episode")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Training curve saved -> {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="task_medium",
                        choices=["task_easy", "task_medium", "task_hard"],
                        help="task_medium is recommended — easy is trivially solved by random agents")
    parser.add_argument("--timesteps", type=int, default=300_000,
                        help="300k is minimum for task_medium. Use 2M on HuggingFace for task_hard.")
    parser.add_argument("--save-dir", default="models")
    parser.add_argument("--n-envs",   type=int, default=4,
                        help="Parallel envs for faster training (4 is good for most machines)")
    args = parser.parse_args()

    os.makedirs(f"{args.save_dir}/{args.task}", exist_ok=True)
    model_path = f"{args.save_dir}/{args.task}/ppo_lifesupport"
    curve_path = f"training_curve_{args.task}.png"

    print(f"\n{'='*62}")
    print(f"  Life Support RL Trainer — Round 2")
    print(f"  Task      : {args.task}")
    print(f"  Timesteps : {args.timesteps:,}")
    print(f"  Parallel  : {args.n_envs} envs")
    print(f"{'='*62}\n")

    if args.task == "task_easy":
        print("  NOTE: task_easy only lasts 24 steps and random agents almost")
        print("  always survive it. For a meaningful comparison use task_medium.\n")

    # Vectorised training envs
    vec_env = DummyVecEnv([make_env(args.task, seed=i) for i in range(args.n_envs)])

    # Separate eval env (no Monitor wrapping — we handle it ourselves)
    eval_env = LifeSupportGymEnv(task_id=args.task, seed=999)

    callback = RewardLoggerCallback()

    # Check if tensorboard is available
    try:
        import tensorboard  # noqa
        tb_log = f"./tb_logs/{args.task}/"
    except ImportError:
        tb_log = None

    # PPO hyperparams tuned for this env's reward scale and episode length
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        device="cpu",                 # MlpPolicy runs faster on CPU
        learning_rate=2e-4,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=tb_log,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        progress_bar=True,
    )

    model.save(model_path)
    print(f"\n  Model saved -> {model_path}.zip")

    plot_training_curve(callback.episode_rewards, callback.episode_lengths,
                        curve_path, args.task)

    if callback.episode_rewards:
        ep = callback.episode_rewards
        tail = ep[max(0, len(ep) * 4 // 5):]
        print(f"\n  Training summary:")
        print(f"    Episodes completed   : {len(ep)}")
        print(f"    Mean reward (all)    : {np.mean(ep):.2f}")
        print(f"    Mean reward (last 20%): {np.mean(tail):.2f}")
        print(f"    Best episode         : {max(ep):.2f}")
        print(f"    Worst episode        : {min(ep):.2f}")
        if np.mean(tail) > np.mean(ep[:len(ep)//5 or 1]):
            print(f"\n  ✓ Agent improved over training!")
        else:
            print(f"\n  ↻ Agent needs more timesteps — try doubling --timesteps")

    print(f"\n  Next step: python evaluate.py --task {args.task}\n")


if __name__ == "__main__":
    main()
