"""
generate_expert_data.py — Generate training data for LLM fine-tuning
====================================================================
Uses the trained PPO agent to play episodes and records
(observation → action) pairs in chat format for SFT.

Usage:
  python generate_expert_data.py --task task_medium --episodes 150
  python generate_expert_data.py --task task_hard --episodes 100

Output:
  expert_data/expert_train.jsonl  — training split (90%)
  expert_data/expert_eval.jsonl   — evaluation split (10%)
"""

import argparse
import json
import os
import random
import textwrap
import numpy as np

from stable_baselines3 import PPO
from gym_wrapper import LifeSupportGymEnv

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI agent controlling a space habitat life support system.
    You receive sensor readings and must output control actions to keep the crew alive.

    CRITICAL THRESHOLDS:
    - O2 must stay between 19.5% and 23.5%
    - CO2 must stay below 1000 ppm
    - Water must stay above 5 liters
    - Food must stay above 0 kg
    - Crew health is your primary objective (keep above 0.8)

    ACTIONS (all floats in given ranges):
    - increase_plant_growth [0-1]
    - recycle_water [0-1]
    - adjust_oxygen [-1 to +1]
    - ration_food [0-1]
    - crew_activity [0-1]

    Respond ONLY with a valid JSON object. No explanation:
    {"increase_plant_growth": 0.7, "recycle_water": 0.6, "adjust_oxygen": 0.1, "ration_food": 1.0, "crew_activity": 0.8}
""").strip()

ACTION_KEYS = [
    "increase_plant_growth",
    "recycle_water",
    "adjust_oxygen",
    "ration_food",
    "crew_activity",
]


def obs_to_text(obs_array, step, max_steps):
    """Format observation array into the same text format inference.py uses."""
    return (
        f"Step {step}/{max_steps}\n"
        f"O2: {obs_array[0]:.2f}% | CO2: {obs_array[1]:.0f}ppm | "
        f"Water: {obs_array[2]:.1f}L | Food: {obs_array[3]:.2f}kg\n"
        f"Crew: {int(obs_array[4])} | Plant growth: {obs_array[5]:.2f} | "
        f"Water recycling: {obs_array[6]:.2f}\n"
        f"Day: {int(obs_array[7])} | Crew health: {obs_array[8]:.3f} | "
        f"Power budget: {obs_array[9]:.2f}"
    )


def action_to_json(action_array):
    """Convert PPO action array to JSON string matching inference.py format."""
    action_dict = {}
    for i, key in enumerate(ACTION_KEYS):
        action_dict[key] = round(float(action_array[i]), 4)
    return json.dumps(action_dict)


def generate_episodes(task_id, model_path, n_episodes, min_survival_frac=0.5):
    """
    Run PPO agent for n_episodes, collect successful trajectories.
    Only keeps episodes where the agent survived > min_survival_frac of max_steps.
    """
    max_steps_map = {"task_easy": 24, "task_medium": 168, "task_hard": 720}
    max_steps = max_steps_map[task_id]

    model = PPO.load(model_path)
    samples = []
    good_episodes = 0
    total_episodes = 0

    for ep in range(n_episodes):
        seed = random.randint(0, 100000)
        env = LifeSupportGymEnv(task_id=task_id, seed=seed)
        obs, _ = env.reset()

        episode_samples = []
        step = 0
        done = False

        while not done:
            step += 1
            action, _ = model.predict(obs, deterministic=True)

            # Record this (observation, action) pair
            obs_text = obs_to_text(obs, step, max_steps)
            action_json = action_to_json(action)

            episode_samples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": obs_text},
                    {"role": "assistant", "content": action_json},
                ]
            })

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        total_episodes += 1
        survival_frac = step / max_steps

        # Only keep good episodes (agent survived long enough)
        if survival_frac >= min_survival_frac:
            samples.extend(episode_samples)
            good_episodes += 1
            print(f"  Episode {ep+1:3d}: {step:4d}/{max_steps} steps "
                  f"({survival_frac:.0%}) [OK] ({len(episode_samples)} samples)")
        else:
            print(f"  Episode {ep+1:3d}: {step:4d}/{max_steps} steps "
                  f"({survival_frac:.0%}) [SKIP]")

    print(f"\n  Kept {good_episodes}/{total_episodes} episodes "
          f"({len(samples)} total samples)")
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate expert training data from PPO agent for LLM fine-tuning"
    )
    parser.add_argument("--task", default="task_medium",
                        choices=["task_easy", "task_medium", "task_hard"])
    parser.add_argument("--episodes", type=int, default=150,
                        help="Number of episodes to run")
    parser.add_argument("--model-dir", default="models",
                        help="Directory containing trained PPO models")
    parser.add_argument("--output-dir", default="expert_data",
                        help="Output directory for JSONL files")
    parser.add_argument("--min-survival", type=float, default=0.5,
                        help="Minimum survival fraction to keep an episode (0.0-1.0)")
    parser.add_argument("--train-split", type=float, default=0.9,
                        help="Fraction of data for training (rest is eval)")
    args = parser.parse_args()

    model_path = f"{args.model_dir}/{args.task}/ppo_lifesupport.zip"
    if not os.path.exists(model_path):
        print(f"\n  ERROR: No trained model at {model_path}")
        print(f"  Train first: python train.py --task {args.task} --timesteps 300000\n")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  Expert Data Generator")
    print(f"  Task     : {args.task}")
    print(f"  Episodes : {args.episodes}")
    print(f"  Model    : {model_path}")
    print(f"{'='*62}\n")

    samples = generate_episodes(
        args.task, model_path, args.episodes,
        min_survival_frac=args.min_survival
    )

    if not samples:
        print("\n  No samples generated! Check if your model is trained well enough.")
        return

    # Shuffle and split
    random.shuffle(samples)
    split_idx = int(len(samples) * args.train_split)
    train_samples = samples[:split_idx]
    eval_samples = samples[split_idx:]

    # Save
    train_path = os.path.join(args.output_dir, "expert_train.jsonl")
    eval_path = os.path.join(args.output_dir, "expert_eval.jsonl")

    with open(train_path, "w") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")

    with open(eval_path, "w") as f:
        for s in eval_samples:
            f.write(json.dumps(s) + "\n")

    print(f"\n  Saved {len(train_samples)} training samples -> {train_path}")
    print(f"  Saved {len(eval_samples)} eval samples     -> {eval_path}")
    print(f"\n  Next: python finetune_llm.py\n")


if __name__ == "__main__":
    main()
