"""
train_full_pipeline.py — End-to-end: PPO → Expert Data → LLM Fine-tuning
=========================================================================
Complete pipeline that:
1. Trains PPO agent on the life support environment
2. Generates expert trajectory data from the trained PPO agent
3. Fine-tunes a small LLM on the expert data using LoRA
4. Pushes the fine-tuned model to HuggingFace Hub

USAGE (on HuggingFace Spaces with T4 GPU):
  HF_TOKEN=your_token python train_full_pipeline.py \\
      --hub-repo YOUR_USERNAME/lifesupport-llm

USAGE (local):
  python train_full_pipeline.py --skip-push
"""

import argparse
import os
import subprocess
import sys


def run_cmd(cmd, description):
    """Run a command and stream output."""
    print(f"\n{'='*62}")
    print(f"  PIPELINE STEP: {description}")
    print(f"  Command: {cmd}")
    print(f"{'='*62}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n  ✗ FAILED: {description}")
        print(f"  Return code: {result.returncode}")
        sys.exit(1)
    print(f"\n  ✓ COMPLETE: {description}")


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: PPO training → Expert data → LLM fine-tuning"
    )
    parser.add_argument("--task", default="task_medium",
                        choices=["task_easy", "task_medium", "task_hard"])
    parser.add_argument("--ppo-timesteps", type=int, default=500000,
                        help="PPO training timesteps")
    parser.add_argument("--ppo-envs", type=int, default=4,
                        help="Parallel environments for PPO")
    parser.add_argument("--expert-episodes", type=int, default=150,
                        help="Episodes to generate for expert data")
    parser.add_argument("--llm-base", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Base LLM to fine-tune")
    parser.add_argument("--llm-epochs", type=int, default=3,
                        help="LLM fine-tuning epochs")
    parser.add_argument("--hub-repo", default=None,
                        help="HuggingFace Hub repo (e.g., 'username/lifesupport-llm')")
    parser.add_argument("--skip-ppo", action="store_true",
                        help="Skip PPO training (use existing model)")
    parser.add_argument("--skip-push", action="store_true",
                        help="Skip pushing to HuggingFace Hub")
    parser.add_argument("--use-4bit", action="store_true",
                        help="Use 4-bit quantization for LLM fine-tuning")
    args = parser.parse_args()

    print(f"\n{'#'*62}")
    print(f"  Life Support — Full Training Pipeline")
    print(f"  Task       : {args.task}")
    print(f"  PPO steps  : {args.ppo_timesteps:,}")
    print(f"  Expert eps : {args.expert_episodes}")
    print(f"  LLM base   : {args.llm_base}")
    print(f"  LLM epochs : {args.llm_epochs}")
    print(f"  Hub repo   : {args.hub_repo or 'not set'}")
    print(f"{'#'*62}")

    # ── Step 1: PPO Training ─────────────────────────────────────────────────
    if not args.skip_ppo:
        run_cmd(
            f"python train.py --task {args.task} "
            f"--timesteps {args.ppo_timesteps} --n-envs {args.ppo_envs}",
            f"PPO Training ({args.task}, {args.ppo_timesteps:,} timesteps)"
        )
    else:
        model_path = f"models/{args.task}/ppo_lifesupport.zip"
        if not os.path.exists(model_path):
            print(f"\n  ERROR: --skip-ppo but no model at {model_path}")
            sys.exit(1)
        print(f"\n  Skipping PPO training — using existing model at {model_path}")

    # ── Step 2: Generate Expert Data ─────────────────────────────────────────
    run_cmd(
        f"python generate_expert_data.py --task {args.task} "
        f"--episodes {args.expert_episodes}",
        f"Expert Data Generation ({args.expert_episodes} episodes)"
    )

    # ── Step 3: Fine-tune LLM ────────────────────────────────────────────────
    ft_cmd = (
        f"python finetune_llm.py "
        f"--base-model {args.llm_base} "
        f"--epochs {args.llm_epochs}"
    )
    if args.use_4bit:
        ft_cmd += " --use-4bit"
    if not args.skip_push and args.hub_repo:
        ft_cmd += f" --push-to-hub --hub-repo {args.hub_repo}"

    run_cmd(ft_cmd, f"LLM Fine-tuning ({args.llm_base}, {args.llm_epochs} epochs)")

    # ── Step 4: Run Evaluation ───────────────────────────────────────────────
    run_cmd(
        f"python evaluate.py --task {args.task} --episodes 10",
        f"PPO Evaluation ({args.task})"
    )

    # ── Done ─────────────────────────────────────────────────────────────────
    print(f"\n{'#'*62}")
    print(f"  ✓ PIPELINE COMPLETE!")
    print(f"")
    print(f"  Outputs:")
    print(f"    PPO model:      models/{args.task}/ppo_lifesupport.zip")
    print(f"    Expert data:    expert_data/expert_train.jsonl")
    print(f"    LLM (LoRA):     lifesupport-llm-lora/")
    print(f"    LLM (merged):   lifesupport-llm/")
    if args.hub_repo:
        print(f"    HF Hub:         https://huggingface.co/{args.hub_repo}")
    print(f"")
    print(f"  To run inference with your fine-tuned LLM:")
    if args.hub_repo:
        print(f"    MODEL_NAME={args.hub_repo} python inference.py")
    else:
        print(f"    python inference.py  (update MODEL_NAME in the script)")
    print(f"{'#'*62}\n")


if __name__ == "__main__":
    main()
