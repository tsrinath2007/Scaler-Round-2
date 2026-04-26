"""
train_grpo.py — GRPO Training for Life Support LLM Agent
=========================================================
Trains Qwen3-1.7B using Group Relative Policy Optimization (GRPO) via TRL
to control a space habitat life support system. Mirrors the OpenEnv Wordle
GRPO approach (04-training.md) but adapted for the life support environment.

INSTALL:
  pip install -Uq git+https://github.com/huggingface/trl.git vllm==0.10.2 bitsandbytes trackio

USAGE (HuggingFace Spaces / A100):
  HF_TOKEN=your_token python train_grpo.py --push-to-hub --hub-repo YOUR_USER/lifesupport-grpo

USAGE (local, no vLLM):
  python train_grpo.py --no-vllm --task task_easy --dataset-size 200
"""

import argparse
import json
import re
import sys

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions

sys.path.insert(0, ".")
from env.environment import LifeSupportEnv
from env.models import Action


# ── Safe-range constants (mirrors env/environment.py) ────────────────────────

O2_SAFE_MIN   = 19.5
O2_SAFE_MAX   = 23.5
CO2_SAFE_MAX  = 1000
WATER_WARN    = 20.0
FOOD_WARN     = 0.5


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI agent controlling a space habitat life support system on the Artemis Moon Mission.
Each hour you receive sensor readings and must output exactly one JSON action to keep the crew alive.

CRITICAL THRESHOLDS:
- O2: 19.5–23.5% (below 19.5 = hypoxia; above 25 = fire risk)
- CO2: below 1000 ppm (above 3000 = lethal)
- Water: stay above 20L
- Food: stay above 0.5 kg
- Crew health (0–1): keep above 0.8

ACTION FIELDS (include all six):
- increase_plant_growth [0-1]: boosts O2 production and food; costs power + water
- recycle_water [0-1]: reclaims waste water; costs power
- adjust_oxygen [-1 to +1]: positive releases O2; negative scrubs CO2
- ration_food [0-1]: 1.0 = full rations; lower to conserve food
- crew_activity [0-1]: controls O2 + water consumption (lower to conserve)
- route_power: "balanced" | "life_support" | "shields" | "hydroponics" | "emergency"

STRATEGY:
- O2 low → adjust_oxygen > 0, increase_plant_growth high
- CO2 high → adjust_oxygen < 0, crew_activity low
- Water low → recycle_water high, increase_plant_growth low
- Food low → increase_plant_growth high (food harvests when biomass > 30 kg)
- Dust storm / lunar night → reduce power-heavy actions
- Solar flare → route_power = "shields"
- Meteor impact (O2 leak) → adjust_oxygen > 0

Respond ONLY with a single-line JSON object, no explanation:
{"increase_plant_growth": 0.6, "recycle_water": 0.5, "adjust_oxygen": 0.0, "ration_food": 1.0, "crew_activity": 0.8, "route_power": "balanced"}"""

DEFAULT_ACTION = {
    "increase_plant_growth": 0.5,
    "recycle_water": 0.5,
    "adjust_oxygen": 0.0,
    "ration_food": 0.9,
    "crew_activity": 0.7,
    "route_power": "balanced",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_action(text: str) -> tuple[Action, bool]:
    """
    Extract JSON action from model output.
    Returns (Action, valid) — valid=False means the model produced garbage output.
    """
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            d = json.loads(match.group())
            return Action(
                increase_plant_growth=float(d.get("increase_plant_growth", 0.5)),
                recycle_water=float(d.get("recycle_water", 0.5)),
                adjust_oxygen=float(d.get("adjust_oxygen", 0.0)),
                ration_food=float(d.get("ration_food", 0.9)),
                crew_activity=float(d.get("crew_activity", 0.7)),
                route_power=str(d.get("route_power", "balanced")),
            ), True
        except (json.JSONDecodeError, ValueError):
            pass
    return Action(**DEFAULT_ACTION), False


def format_observation(obs, step: int, max_steps: int) -> str:
    """Compact one-screen summary of the current sensor state."""
    lines = [
        f"Step {step}/{max_steps}",
        f"O2:{obs.o2_percent:.1f}% | CO2:{obs.co2_ppm:.0f}ppm | "
        f"H2O:{obs.water_liters:.0f}L | Food:{obs.food_kg:.1f}kg",
        f"Health:{obs.crew_health:.3f} | Crew:{obs.crew_size} | "
        f"Solar:{obs.solar_panel_health:.0%} | Power:{obs.power_budget:.0%}",
    ]
    if obs.event_name:
        lines.append(
            f"EVENT: {obs.event_name} [{obs.event_severity}] "
            f"{obs.event_turns_remaining}t left"
        )
    if len(obs.active_events) > 1:
        lines.append(f"MULTI-EVENT: {', '.join(obs.active_events)}")
    if obs.radiation_level > 0.1:
        lines.append(
            f"RADIATION:{obs.radiation_level:.0%} | "
            f"Shield:{obs.shield_integrity:.0%} | "
            f"Cumulative:{obs.cumulative_radiation:.3f}"
        )
    if obs.crew_injured > 0:
        lines.append(f"INJURED: {obs.crew_injured} crew incapacitated")
    return "\n".join(lines)


# ── Single-episode rollout ─────────────────────────────────────────────────────

def rollout_once(trainer, env: LifeSupportEnv, tokenizer):
    """
    Run one full episode, interleaving model generations with env steps.
    Returns accumulated token ids, logprobs, and episode-level reward signals.
    """
    obs = env.reset()
    max_steps = env.config["max_steps"]

    prompt_ids      = []
    completion_ids  = []
    logprobs        = []
    step_rewards    = []
    health_scores   = []
    safe_steps      = 0
    valid_actions   = 0   # counts steps with valid JSON output
    total_steps     = 0
    failure_reason  = None

    for step in range(1, max_steps + 1):
        obs_text = format_observation(obs, step, max_steps)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": obs_text},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

        out = generate_rollout_completions(trainer, [prompt_text])[0]
        prompt_ids.extend(out["prompt_ids"])
        completion_ids.extend(out["completion_ids"])
        logprobs.extend(out["logprobs"])

        completion_text = out.get("text") or tokenizer.decode(
            out["completion_ids"], skip_special_tokens=True
        )

        action, is_valid = parse_action(completion_text)
        if is_valid:
            valid_actions += 1

        obs, reward, done, info = env.step(action)
        total_steps += 1

        step_rewards.append(reward)
        health_scores.append(obs.crew_health)

        all_safe = (
            O2_SAFE_MIN <= obs.o2_percent <= O2_SAFE_MAX
            and obs.co2_ppm < CO2_SAFE_MAX
            and obs.water_liters > WATER_WARN
            and obs.food_kg > FOOD_WARN
        )
        if all_safe:
            safe_steps += 1

        if done:
            failure_reason = info.get("failure_reason")
            break

    survived     = failure_reason is None
    avg_health   = sum(health_scores) / len(health_scores) if health_scores else 0.0
    safe_frac    = safe_steps / total_steps if total_steps > 0 else 0.0
    avg_step_rew = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0
    format_frac  = valid_actions / total_steps if total_steps > 0 else 0.0

    # Bipolar survival: +1 survived, -1 died — 2× gradient vs 0/1
    survival_bipolar = 1.0 if survived else -1.0

    return {
        "prompt_ids":      prompt_ids,
        "completion_ids":  completion_ids,
        "logprobs":        logprobs,
        "survival_reward": survival_bipolar,
        "health_reward":   avg_health,
        "safe_reward":     safe_frac,
        "step_reward":     avg_step_rew,
        "format_reward":   format_frac,   # 1.0 = all JSON valid, 0.0 = all garbage
    }


# ── Batch rollout function (called by GRPOTrainer) ────────────────────────────

def make_rollout_func(task_id: str, tokenizer):
    def rollout_func(prompts, trainer=None):
        batch = {
            "prompt_ids":      [],
            "completion_ids":  [],
            "logprobs":        [],
            "survival_reward": [],
            "health_reward":   [],
            "safe_reward":     [],
            "step_reward":     [],
            "format_reward":   [],
        }
        for _ in prompts:
            env = LifeSupportEnv(task_id=task_id)
            ep  = rollout_once(trainer=trainer, env=env, tokenizer=tokenizer)
            for k in batch:
                batch[k].append(ep[k])
        return batch
    return rollout_func


# ── Reward functions (called per-completion by GRPOTrainer) ──────────────────

def reward_survival(completions, **kwargs):
    """1.0 if crew survived the episode, 0.0 otherwise."""
    r = kwargs.get("survival_reward")
    return [float(x) for x in r] if r is not None else [0.0] * len(completions)


def reward_health(completions, **kwargs):
    """Average crew health across the episode [0, 1]."""
    r = kwargs.get("health_reward")
    return [float(x) for x in r] if r is not None else [0.0] * len(completions)


def reward_safe_environment(completions, **kwargs):
    """Fraction of steps where all parameters were within safe ranges [0, 1]."""
    r = kwargs.get("safe_reward")
    return [float(x) for x in r] if r is not None else [0.0] * len(completions)


def reward_step(completions, **kwargs):
    """Average per-step shaped reward returned by the environment."""
    r = kwargs.get("step_reward")
    return [float(x) for x in r] if r is not None else [0.0] * len(completions)


def reward_format(completions, **kwargs):
    """
    Fraction of steps where the model output valid JSON [0, 1].
    Penalises garbage output — the most common early mistake.
    Reaching 1.0 means the model has learned the correct response format.
    """
    r = kwargs.get("format_reward")
    return [float(x) for x in r] if r is not None else [0.0] * len(completions)


# ── Training curve callback ───────────────────────────────────────────────────

TRACKED_METRICS = [
    "rewards/reward_survival",
    "rewards/reward_health",
    "rewards/reward_safe_environment",
    "rewards/reward_step",
    "rewards/reward_format",
    "loss",
]

class RewardLogger(TrainerCallback):
    """Captures per-step metrics and saves a training curve PNG at the end."""

    def __init__(self, output_dir: str, task_id: str, hub_repo: str | None = None):
        self.output_dir = output_dir
        self.task_id    = task_id
        self.hub_repo   = hub_repo
        self.history: dict[str, list] = {m: [] for m in TRACKED_METRICS}
        self.steps: list[int] = []

    def on_log(
        self,
        _args: TrainingArguments,
        state: TrainerState,
        _control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ):
        if not logs:
            return
        self.steps.append(state.global_step)
        for m in TRACKED_METRICS:
            self.history[m].append(logs.get(m, float("nan")))

    def on_train_end(
        self,
        _args: TrainingArguments,
        _state: TrainerState,
        _control: TrainerControl,
        **kwargs,
    ):
        self._save_plot()

    def _save_plot(self):
        reward_keys = [m for m in TRACKED_METRICS if m != "loss"]
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(
            f"GRPO Training — {self.task_id}",
            fontsize=14, fontweight="bold",
        )

        # Top panel: reward signals
        ax_r = axes[0]
        labels = {
            "rewards/reward_survival":         "Survival (+1/-1)",
            "rewards/reward_health":            "Avg crew health",
            "rewards/reward_safe_environment":  "Safe-step fraction",
            "rewards/reward_step":              "Avg env reward",
            "rewards/reward_format":            "Valid JSON fraction",
        }
        for key in reward_keys:
            vals = self.history[key]
            if any(v == v for v in vals):  # skip if all NaN
                ax_r.plot(self.steps, vals, label=labels.get(key, key), linewidth=1.5)
        ax_r.set_ylabel("Reward")
        ax_r.set_ylim(-1.1, 1.15)  # survival is now bipolar -1/+1
        ax_r.legend(fontsize=9, loc="lower right")
        ax_r.grid(True, alpha=0.3)
        ax_r.axhline(0.8, color="green", linestyle="--", linewidth=0.8,
                     label="Health target (0.8)", alpha=0.6)

        # Bottom panel: training loss
        ax_l = axes[1]
        loss_vals = self.history["loss"]
        if any(v == v for v in loss_vals):
            ax_l.plot(self.steps, loss_vals, color="tab:red",
                      label="Training loss", linewidth=1.5)
        ax_l.set_xlabel("Training step")
        ax_l.set_ylabel("Loss")
        ax_l.legend(fontsize=9)
        ax_l.grid(True, alpha=0.3)

        plt.tight_layout()
        path = f"training_curve_{self.task_id}_grpo.png"
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"\nTraining curve saved → {path}")

        if self.hub_repo:
            try:
                HfApi().upload_file(
                    path_or_fileobj=path,
                    path_in_repo=path,
                    repo_id=self.hub_repo,
                    repo_type="model",
                )
                print(f"Training curve uploaded → https://huggingface.co/{self.hub_repo}/blob/main/{path}")
            except Exception as e:
                print(f"Warning: could not upload training curve to Hub: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GRPO training for life support agent")
    parser.add_argument("--task", default="task_easy",
                        choices=["task_easy", "task_medium", "task_hard"],
                        help="Environment difficulty (default: task_easy, 24 steps)")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B",
                        help="Base model to fine-tune")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset-size", type=int, default=1000,
                        help="Number of training episodes (dataset rows)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo", default=None,
                        help="HuggingFace Hub repo, e.g. username/lifesupport-grpo")
    parser.add_argument("--no-vllm", action="store_true",
                        help="Disable vLLM (for small-GPU or CPU testing)")
    args = parser.parse_args()

    model_slug = args.model.split("/")[-1]
    output_dir = args.output_dir or f"lifesupport-grpo-{model_slug}-{args.task}"

    print(f"\n{'='*62}")
    print(f"  Life Support GRPO Training")
    print(f"  Task       : {args.task}")
    print(f"  Model      : {args.model}")
    print(f"  Dataset    : {args.dataset_size} episodes")
    print(f"  Output dir : {output_dir}")
    print(f"  vLLM       : {'disabled' if args.no_vllm else 'enabled (colocate)'}")
    print(f"{'='*62}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_dict({
        "prompt": [
            f"Manage the life support system for {args.task}. Keep the crew alive."
        ] * args.dataset_size
    })

    rollout_func = make_rollout_func(args.task, tokenizer)

    use_vllm = not args.no_vllm
    grpo_config = GRPOConfig(
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        gradient_accumulation_steps=64,
        per_device_train_batch_size=1,
        warmup_steps=20,
        num_generations=4,  # compare 4 rollouts per prompt — much stronger GRPO signal
        max_completion_length=96,   # enough for one JSON action line
        max_prompt_length=512,      # compact observation fits easily
        use_vllm=use_vllm,
        vllm_mode="colocate" if use_vllm else None,
        vllm_gpu_memory_utilization=0.1,
        output_dir=output_dir,
        report_to=["trackio", "tensorboard"],
        trackio_space_id=output_dir,
        logging_steps=1,
        save_steps=10,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        push_to_hub=args.push_to_hub,
    )

    hub_repo = args.hub_repo if args.push_to_hub else None
    reward_logger = RewardLogger(output_dir=output_dir, task_id=args.task, hub_repo=hub_repo)

    trainer = GRPOTrainer(
        model=args.model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_format,           # learned first: teaches correct JSON format
            reward_survival,         # bipolar +1/-1: strong gradient for staying alive
            reward_health,           # dense: avg crew health across episode
            reward_safe_environment, # dense: fraction of steps all params in safe range
            reward_step,             # dense: avg per-step env reward
        ],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
        callbacks=[reward_logger],
    )

    # Memory stats before training
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        mem_start = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        mem_total = round(gpu.total_memory / 1024**3, 3)
        print(f"GPU = {gpu.name}. Max memory = {mem_total} GB.")
        print(f"{mem_start} GB of memory reserved before training.\n")

    trainer_stats = trainer.train()

    # Memory stats after training
    if torch.cuda.is_available():
        mem_used = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        mem_train = round(mem_used - mem_start, 3)
        print(f"\n{trainer_stats.metrics['train_runtime']:.1f}s training time "
              f"({trainer_stats.metrics['train_runtime']/60:.1f} min)")
        print(f"Peak reserved memory = {mem_used} GB ({mem_train} GB for training)")

    trainer.save_model(output_dir)
    if args.push_to_hub:
        repo = args.hub_repo or output_dir
        trainer.push_to_hub(repo)

    print(f"\nDone. Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
