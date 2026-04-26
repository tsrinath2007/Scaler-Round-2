"""
Gradio app — Life Support RL Trainer
Two training modes:
  1. PPO (Stable-Baselines3) — fast, classical RL
  2. TRL GRPO — LLM fine-tuned with environment reward signal
"""
import threading
import json
import re
import random
import numpy as np
import gradio as gr
import plotly.graph_objects as go

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from gym_wrapper import LifeSupportGymEnv
from env.environment import LifeSupportEnv
from env.models import Action

# ── Shared chart builder (Plotly — zoomable, pannable, hoverable) ────────────

def _make_chart(data, x_label, y_label, title):
    """Return an interactive Plotly figure from a list of (x, y) tuples."""
    if not data:
        fig = go.Figure()
        fig.update_layout(
            title=dict(text=title, font=dict(color="#e0e0e0", size=14)),
            plot_bgcolor="#111", paper_bgcolor="#0d0d0d",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            annotations=[dict(text="Waiting for data…", xref="paper", yref="paper",
                              x=0.5, y=0.5, showarrow=False,
                              font=dict(color="#444", size=14))],
        )
        return fig

    xs = [d[0] for d in data]
    ys = [d[1] for d in data]

    # Rolling average
    w  = max(1, len(ys) // 8)
    ra = np.convolve(ys, np.ones(w) / w, mode="valid").tolist()
    rx = xs[w - 1:]

    fig = go.Figure()

    # Raw values — thin, semi-transparent
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", name="Raw",
        line=dict(color="rgba(249,115,22,0.25)", width=1),
        hovertemplate=f"{x_label}: %{{x}}<br>Reward: %{{y:.4f}}<extra></extra>",
    ))

    # Rolling average — thick orange, filled
    fig.add_trace(go.Scatter(
        x=rx, y=ra, mode="lines", name=f"Avg (w={w})",
        line=dict(color="#f97316", width=2.8),
        fill="tozeroy", fillcolor="rgba(249,115,22,0.09)",
        hovertemplate=f"{x_label}: %{{x}}<br>Avg reward: %{{y:.4f}}<extra></extra>",
    ))

    # Current value marker
    fig.add_trace(go.Scatter(
        x=[xs[-1]], y=[ys[-1]], mode="markers+text",
        marker=dict(color="#f97316", size=9, symbol="circle",
                    line=dict(color="#fff", width=1.5)),
        text=[f"  {ys[-1]:.3f}"], textposition="middle right",
        textfont=dict(color="#f97316", size=11),
        name="Latest", showlegend=False,
        hovertemplate=f"Latest: %{{y:.4f}}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(color="#e0e0e0", size=14), x=0.02),
        plot_bgcolor="#111",
        paper_bgcolor="#0d0d0d",
        font=dict(color="#888", size=11),
        xaxis=dict(
            title=x_label, titlefont=dict(color="#666", size=11),
            gridcolor="#1e1e1e", zerolinecolor="#222",
            tickfont=dict(color="#666"), showspikes=True,
            spikecolor="#f97316", spikethickness=1, spikedash="dot",
        ),
        yaxis=dict(
            title=y_label, titlefont=dict(color="#666", size=11),
            gridcolor="#1e1e1e", zerolinecolor="#222",
            tickfont=dict(color="#666"), showspikes=True,
            spikecolor="#f97316", spikethickness=1, spikedash="dot",
        ),
        legend=dict(
            bgcolor="rgba(17,17,17,0.8)", bordercolor="#2a2a2a",
            font=dict(color="#aaa"), orientation="h",
            yanchor="bottom", y=1.01, xanchor="right", x=1,
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1a1a1a", bordercolor="#f97316",
                        font=dict(color="#e0e0e0")),
        margin=dict(l=60, r=30, t=55, b=50),
        dragmode="zoom",   # default to zoom-box on drag
    )
    return fig


# ── Shared state — PPO ───────────────────────────────────────────────────────
_ppo_rewards  = []
_ppo_logs     = []
_ppo_status   = "Idle — press Start PPO Training to begin."
_ppo_running  = False

# ── Shared state — TRL ───────────────────────────────────────────────────────
_trl_rewards  = []
_trl_logs     = []
_trl_status   = "Idle — press Start TRL Training to begin."
_trl_running  = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PPO training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TASK_STEPS = {"task_easy": 100_000, "task_medium": 300_000, "task_hard": 500_000}


class PPOCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self._rollout = 0

    def _on_rollout_end(self):
        self._rollout += 1
        steps = self.num_timesteps
        if len(self.model.ep_info_buffer) > 0:
            ep_rewards = [ep["r"] for ep in self.model.ep_info_buffer]
            ep_lengths = [ep["l"] for ep in self.model.ep_info_buffer]
            mean_r = float(np.mean(ep_rewards))
            mean_l = float(np.mean(ep_lengths))
            _ppo_rewards.append((self._rollout, mean_r))
            _ppo_logs.append(
                f"[Rollout {self._rollout:>4}]  Steps: {steps:>8,}  |  "
                f"Mean reward: {mean_r:>8.3f}  |  Mean ep len: {mean_l:>6.1f}"
            )
        return True

    def _on_step(self):
        return True


def _run_ppo(task_id, total_steps, n_envs):
    global _ppo_status, _ppo_running
    _ppo_running = True
    _ppo_rewards.clear()
    _ppo_logs.clear()
    try:
        _ppo_logs.append(f"Setting up {n_envs} envs for {task_id}...")
        _ppo_status = f"Training PPO on {task_id}..."
        env = make_vec_env(lambda: LifeSupportGymEnv(task_id=task_id), n_envs=n_envs)
        model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048,
                    batch_size=256, n_epochs=10, gamma=0.99, verbose=0)
        model.learn(total_timesteps=total_steps, callback=PPOCallback())
        model.save(f"ppo_{task_id}")
        final = f"{_ppo_rewards[-1][1]:.3f}" if _ppo_rewards else "N/A"
        _ppo_logs.append(f"✅ Done! Saved ppo_{task_id}.zip  |  Final mean reward: {final}")
        _ppo_status = f"Done! Final mean reward: {final}"
    except Exception as e:
        _ppo_logs.append(f"❌ Error: {e}")
        _ppo_status = f"Error: {e}"
    finally:
        _ppo_running = False


def start_ppo(task_id):
    if _ppo_running:
        return "Already running!", _ppo_log_text(), _ppo_chart()
    threading.Thread(target=_run_ppo,
                     args=(task_id, TASK_STEPS[task_id], 4),
                     daemon=True).start()
    return f"Started PPO on {task_id}...", "", None


def poll_ppo():
    return _ppo_status, _ppo_log_text(), _ppo_chart()


def _ppo_log_text():
    return "\n".join(_ppo_logs[-200:])


def _ppo_chart():
    return _make_chart(
        _ppo_rewards,
        x_label="Rollout",
        y_label="Mean Reward",
        title="PPO — Mean Episode Reward per Rollout",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRL GRPO training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYSTEM_PROMPT = (
    "You are an AI controller for a space habitat life support system. "
    "Keep the crew alive by choosing the right actions. "
    "Respond with ONLY a valid JSON object."
)
ACTION_SCHEMA = (
    '{"adjust_oxygen": <-1 to 1>, "increase_plant_growth": <0 to 1>, '
    '"recycle_water": <0 to 1>, "ration_food": <0 to 1>, "crew_activity": <0 to 1>}'
)


def _obs_to_text(obs):
    o2_s  = "✅ SAFE" if 19.5 <= obs.o2_percent <= 23.5 else "⚠️ DANGER"
    co2_s = "✅ SAFE" if obs.co2_ppm <= 1000 else "⚠️ DANGER"
    return (
        f"Day {obs.day} — O2: {obs.o2_percent:.2f}% {o2_s} | "
        f"CO2: {obs.co2_ppm:.0f}ppm {co2_s} | "
        f"Water: {obs.water_liters:.1f}L | Food: {obs.food_kg:.2f}kg | "
        f"Health: {obs.crew_health:.3f} | Power: {obs.power_budget:.2f}\n"
        f"Respond with ONLY this JSON:\n{ACTION_SCHEMA}"
    )


def _parse_action(text):
    match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if not match:
        return None
    try:
        d = json.loads(match.group())
        return Action(
            adjust_oxygen=float(d.get("adjust_oxygen", 0.0)),
            increase_plant_growth=float(d.get("increase_plant_growth", 0.5)),
            recycle_water=float(d.get("recycle_water", 0.5)),
            ration_food=float(d.get("ration_food", 1.0)),
            crew_activity=float(d.get("crew_activity", 0.5)),
        )
    except Exception:
        return None


def _trl_reward_fn(completions, **kwargs):
    rewards = []
    for text in completions:
        action = _parse_action(text)
        if action is None:
            rewards.append(-1.0)
            continue
        try:
            env = LifeSupportEnv(task_id="task_easy", seed=0)
            env.reset()
            _, reward, _, _ = env.step(action)
            rewards.append(float(reward))
        except Exception:
            rewards.append(-1.0)
    return rewards


def _run_trl():
    global _trl_status, _trl_running
    _trl_running = True
    _trl_rewards.clear()
    _trl_logs.clear()
    try:
        _trl_logs.append("Loading TRL + Unsloth...")
        _trl_status = "Loading model..."

        from unsloth import FastLanguageModel
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
        import torch

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen2.5-3B-Instruct",
            max_seq_length=512,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16, lora_dropout=0, bias="none",
            use_gradient_checkpointing="unsloth",
        )
        _trl_logs.append("✅ Model loaded: Qwen2.5-3B-Instruct (4-bit)")

        # Build dataset
        rng = random.Random(42)
        prompts = []
        for i in range(300):
            task = ["task_easy", "task_medium"][i % 2]
            env = LifeSupportEnv(task_id=task, seed=i)
            obs = env.reset()
            for _ in range(rng.randint(0, 4)):
                try:
                    obs, _, done, _ = env.step(Action(
                        adjust_oxygen=rng.uniform(-1, 1),
                        increase_plant_growth=rng.uniform(0, 1),
                        recycle_water=rng.uniform(0, 1),
                        ration_food=rng.uniform(0, 1),
                        crew_activity=rng.uniform(0, 1),
                    ))
                    if done:
                        obs = env.reset()
                except Exception:
                    obs = env.reset()

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _obs_to_text(obs)},
            ]
            prompts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))

        dataset = Dataset.from_dict({"prompt": prompts})
        _trl_logs.append(f"✅ Dataset: {len(dataset)} prompts")
        _trl_status = "Training TRL GRPO..."

        class TRLLogCallback:
            def __init__(self):
                self._step = 0

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "reward" in logs:
                    self._step += 1
                    r = logs["reward"]
                    _trl_rewards.append((state.global_step, r))
                    _trl_logs.append(
                        f"[Step {state.global_step:>4}]  Reward: {r:.4f}"
                    )

        from transformers import TrainerCallback

        class _Cb(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "reward" in logs:
                    r = logs["reward"]
                    _trl_rewards.append((state.global_step, float(r)))
                    _trl_logs.append(f"[Step {state.global_step:>4}]  Reward: {float(r):.4f}")

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=_trl_reward_fn,
            args=GRPOConfig(
                output_dir="life_support_trl",
                num_train_epochs=3,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                learning_rate=5e-6,
                max_prompt_length=256,
                max_completion_length=100,
                num_generations=4,
                temperature=0.9,
                logging_steps=5,
                save_steps=100,
                report_to="none",
                fp16=True,
            ),
            train_dataset=dataset,
            callbacks=[_Cb()],
        )

        trainer.train()
        model.save_pretrained("life_support_trl")
        tokenizer.save_pretrained("life_support_trl")

        final = f"{_trl_rewards[-1][1]:.4f}" if _trl_rewards else "N/A"
        _trl_logs.append(f"✅ Done! Model saved → life_support_trl/  |  Final reward: {final}")
        _trl_status = f"Done! Final reward: {final}"

    except Exception as e:
        import traceback
        _trl_logs.append(f"❌ Error: {traceback.format_exc()}")
        _trl_status = f"Error: {e}"
    finally:
        _trl_running = False


def start_trl():
    if _trl_running:
        return "Already running!", _trl_log_text(), _trl_chart()
    if _ppo_running:
        return "PPO is running — wait for it to finish first.", _trl_log_text(), _trl_chart()
    threading.Thread(target=_run_trl, daemon=True).start()
    return "Started TRL GRPO training (Qwen2.5-3B)...", "", None


def poll_trl():
    return _trl_status, _trl_log_text(), _trl_chart()


def _trl_log_text():
    return "\n".join(_trl_logs[-200:])


def _trl_chart():
    return _make_chart(
        _trl_rewards,
        x_label="Step",
        y_label="Reward",
        title="TRL GRPO — Mean Reward per Training Step",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Gradio UI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with gr.Blocks(title="Life Support RL Trainer") as demo:
    gr.Markdown(
        "# 🚀 Life Support RL Trainer\n"
        "Train agents to keep a space crew alive. "
        "Choose **PPO** (fast classical RL) or **TRL GRPO** (LLM fine-tuning)."
    )

    with gr.Tabs():

        # ── PPO Tab ──────────────────────────────────────────────────────────
        with gr.Tab("🤖 PPO Training"):
            with gr.Row():
                ppo_task = gr.Dropdown(
                    ["task_easy", "task_medium", "task_hard"],
                    value="task_easy", label="Task")
                ppo_btn = gr.Button("▶ Start PPO Training", variant="primary")
            ppo_status = gr.Textbox(label="Status", interactive=False,
                                    value=_ppo_status)
            with gr.Tabs():
                with gr.Tab("📈 Reward Chart"):
                    ppo_chart = gr.Plot(label="", show_label=False)
                with gr.Tab("📋 Logs"):
                    ppo_logs = gr.Textbox(label="Training Logs", interactive=False,
                                          lines=20, max_lines=20)

            ppo_btn.click(fn=start_ppo, inputs=[ppo_task],
                          outputs=[ppo_status, ppo_logs, ppo_chart])
            gr.Timer(value=3).tick(fn=poll_ppo,
                                   outputs=[ppo_status, ppo_logs, ppo_chart])

        # ── TRL Tab ──────────────────────────────────────────────────────────
        with gr.Tab("🧠 TRL GRPO (LLM)"):
            gr.Markdown(
                "Fine-tunes **Qwen2.5-3B-Instruct** using GRPO with the "
                "life support environment as the reward signal. "
                "The LLM learns to output valid JSON actions that keep the crew alive."
            )
            trl_btn = gr.Button("▶ Start TRL Training", variant="primary")
            trl_status = gr.Textbox(label="Status", interactive=False,
                                    value=_trl_status)
            with gr.Tabs():
                with gr.Tab("📈 Reward Chart"):
                    trl_chart = gr.Plot(label="", show_label=False)
                with gr.Tab("📋 Logs"):
                    trl_logs = gr.Textbox(label="Training Logs", interactive=False,
                                          lines=20, max_lines=20)

            trl_btn.click(fn=start_trl, inputs=[],
                          outputs=[trl_status, trl_logs, trl_chart])
            gr.Timer(value=3).tick(fn=poll_trl,
                                   outputs=[trl_status, trl_logs, trl_chart])

    gr.Markdown(
        "**Tip:** Run PPO first (fast), then TRL GRPO. "
        "TRL training takes ~20–30 min on T4. "
        "Trained models are saved in the Space files."
    )

if __name__ == "__main__":
    demo.launch()