"""
evaluate.py — Compare Random Agent vs Trained PPO Agent side-by-side.

RECOMMENDED USAGE:
  python evaluate.py --task task_medium --episodes 20

WHY NOT task_easy?
  task_easy only lasts 24 steps. Because initial O2/CO2 start near-safe,
  random actions almost always keep the crew alive for 24 steps — so the
  random baseline is already near-perfect, and there's no room to show
  improvement. Use task_medium (168 steps) or task_hard (720 steps).

Usage:
  python evaluate.py --task task_medium --episodes 20
  python evaluate.py --task task_hard   --episodes 10
"""

import argparse
import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from gym_wrapper import LifeSupportGymEnv


def make_fresh_env(task_id: str) -> LifeSupportGymEnv:
    """Create a plain (unwrapped) env for evaluation."""
    return LifeSupportGymEnv(task_id=task_id)


def run_episodes(env, policy, n_episodes: int, label: str):
    """Run n_episodes, collect detailed stats."""
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        failure = None
        green_streak_max = 0
        fire_total = 0
        min_health = 1.0

        while not done:
            if policy == "random":
                action = env.action_space.sample()
            else:
                # deterministic=True uses the learned policy mean, not sampling
                action, _ = policy.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            green_streak_max = max(green_streak_max, info.get("green_streak", 0))
            fire_total += 1 if info.get("fire_active", False) else 0
            min_health = min(min_health, info.get("reward_breakdown", {}).get("health_component", 1.0))
            if done and info.get("failure_reason"):
                failure = info["failure_reason"]

        survived = "✓" if not failure else "✗"
        print(f"  [{label}] Ep {ep+1:2d}: reward={total_reward:7.2f}  "
              f"steps={steps:4d}  streak={green_streak_max:3d}  {survived}")
        if failure:
            print(f"           → FAIL: {failure}")

        results.append({
            "total_reward": total_reward,
            "steps_survived": steps,
            "success": failure is None,
            "green_streak_max": green_streak_max,
            "fire_steps": fire_total,
            "min_health": min_health,
        })

    return results


def print_table(random_res, trained_res, max_steps: int):
    def stats(results):
        return {
            "mean_r":   np.mean([r["total_reward"] for r in results]),
            "std_r":    np.std( [r["total_reward"] for r in results]),
            "mean_s":   np.mean([r["steps_survived"] for r in results]),
            "win_rate": np.mean([r["success"] for r in results]) * 100,
            "streak":   np.mean([r["green_streak_max"] for r in results]),
            "fire":     np.mean([r["fire_steps"] for r in results]),
            "health":   np.mean([r["min_health"] for r in results]),
        }

    r = stats(random_res)
    t = stats(trained_res)

    print("\n" + "="*72)
    print(f"  {'Metric':<32} {'Random Agent':>16} {'Trained PPO':>16}")
    print("-"*72)
    print(f"  {'Mean Total Reward':<32} {r['mean_r']:>16.2f} {t['mean_r']:>16.2f}")
    print(f"  {'Reward Std Dev':<32} {r['std_r']:>16.2f} {t['std_r']:>16.2f}")
    print(f"  {'Mean Steps Survived / '+str(max_steps):<32} {r['mean_s']:>16.1f} {t['mean_s']:>16.1f}")
    print(f"  {'Mission Success Rate':<32} {r['win_rate']:>15.1f}% {t['win_rate']:>15.1f}%")
    print(f"  {'Avg Max Green Streak':<32} {r['streak']:>16.1f} {t['streak']:>16.1f}")
    print(f"  {'Avg Fire-Risk Steps':<32} {r['fire']:>16.1f} {t['fire']:>16.1f}")
    print(f"  {'Avg Min Health Score':<32} {r['health']:>16.3f} {t['health']:>16.3f}")
    print("="*72)

    reward_delta = t["mean_r"] - r["mean_r"]
    pct = reward_delta / (abs(r["mean_r"]) + 1e-9) * 100
    win_delta = t["win_rate"] - r["win_rate"]

    print(f"\n  Reward improvement    : {reward_delta:+.2f}  ({pct:+.1f}%)")
    print(f"  Success rate delta    : {win_delta:+.1f}%")
    if t["streak"] > r["streak"]:
        print(f"  Trained agent maintains GREEN streak {t['streak']:.0f} vs {r['streak']:.0f} steps on avg")
    if t["fire"] < r["fire"]:
        print(f"  Trained agent triggers {r['fire'] - t['fire']:.1f} fewer fire-risk steps on avg")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="task_medium",
                        choices=["task_easy", "task_medium", "task_hard"])
    parser.add_argument("--episodes",  type=int, default=20)
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    if args.task == "task_easy":
        print("\n  NOTE: task_easy (24 steps) is almost always solved by random agents.")
        print("  The trained model may not show meaningful improvement here.")
        print("  Consider:  python evaluate.py --task task_medium\n")

    model_path = f"{args.model_dir}/{args.task}/ppo_lifesupport.zip"
    if not os.path.exists(model_path):
        print(f"\n  ERROR: No model found at {model_path}")
        print(f"  Run:  python train.py --task {args.task} --timesteps 300000\n")
        return

    max_steps = {"task_easy": 24, "task_medium": 168, "task_hard": 720}[args.task]

    print(f"\n{'='*62}")
    print(f"  Life Support Evaluation — {args.task}")
    print(f"  Episodes : {args.episodes}  |  Max steps : {max_steps}")
    print(f"{'='*62}\n")

    # ── Random agent ─────────────────────────────────────────────────────────
    env = make_fresh_env(args.task)
    print("[ Random Agent ]")
    random_results = run_episodes(env, "random", args.episodes, "RND")

    # ── Trained PPO agent ─────────────────────────────────────────────────────
    # Load into a fresh env — do NOT pass env= to PPO.load (causes double-wrapping)
    print("\n[ Trained PPO Agent ]")
    env2   = make_fresh_env(args.task)
    model  = PPO.load(model_path)   # no env= argument — avoids SB3 auto-wrapping
    trained_results = run_episodes(env2, model, args.episodes, "PPO")

    print_table(random_results, trained_results, max_steps)


if __name__ == "__main__":
    main()
