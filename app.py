"""
Gradio app — Life Support RL Trainer (Submission Edition)
Optimized for HF Spaces startup and matching the user-requested UI.
"""
import sys
sys.path.insert(0, ".")

import threading
import json
import numpy as np
import gradio as gr
import plotly.graph_objects as go

# ── Shared state ─────────────────────────────────────────────────────────────
_ppo_rewards = []
_ppo_logs = []
_ppo_status = "Idle — press Start Training to begin."
_ppo_running = False

# ── PPO Training Logic (Lazy loaded) ─────────────────────────────────────────

TASK_STEPS = {"task_easy": 100_000, "task_medium": 300_000, "task_hard": 500_000}

def _run_ppo(task_id, total_steps, n_envs):
    global _ppo_status, _ppo_running
    _ppo_running = True
    _ppo_rewards.clear()
    _ppo_logs.clear()
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.callbacks import BaseCallback
        from gym_wrapper import LifeSupportGymEnv

        class _PPOCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self._rollout = 0

            def _on_rollout_end(self):
                self._rollout += 1
                steps = self.num_timesteps
                if len(self.model.ep_info_buffer) > 0:
                    ep_rewards = [ep["r"] for ep in self.model.ep_info_buffer]
                    mean_r = float(np.mean(ep_rewards))
                    _ppo_rewards.append((self._rollout, mean_r))
                    _ppo_logs.append(f"[Rollout {self._rollout:>3}] Steps: {steps:>7,} | Mean Reward: {mean_r:>8.3f}")
                return True

            def _on_step(self): return True

        _ppo_logs.append(f"Starting {task_id} with {n_envs} envs...")
        _ppo_status = f"Training PPO on {task_id}..."
        
        env = make_vec_env(lambda: LifeSupportGymEnv(task_id=task_id), n_envs=n_envs)
        model = PPO("MlpPolicy", env, learning_rate=3e-4, verbose=0)
        model.learn(total_timesteps=total_steps, callback=_PPOCallback())
        model.save(f"ppo_{task_id}")
        
        final_val = _ppo_rewards[-1][1] if _ppo_rewards else 0.0
        _ppo_status = f"Done! Final mean reward: {final_val:.3f}"
        _ppo_logs.append(f"✅ Training complete. Model saved as ppo_{task_id}.zip")
    except Exception as e:
        import traceback
        _ppo_logs.append(f"❌ Error: {traceback.format_exc()}")
        _ppo_status = f"Error: {e}"
    finally:
        _ppo_running = False

def start_training(task_id):
    if _ppo_running: return "Already running!", "\n".join(_ppo_logs), None
    threading.Thread(target=_run_ppo, args=(task_id, TASK_STEPS[task_id], 4), daemon=True).start()
    return f"Started training on {task_id}...", "Initializing environment...", None

def poll_updates():
    global _ppo_status, _ppo_logs, _ppo_rewards
    # Generate Chart
    if not _ppo_rewards:
        fig = go.Figure()
        fig.update_layout(
            plot_bgcolor="#111", paper_bgcolor="#0d0d0d",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            annotations=[dict(text="Waiting for data...", showarrow=False, font=dict(color="#444"))]
        )
    else:
        xs = [d[0] for d in _ppo_rewards]
        ys = [d[1] for d in _ppo_rewards]
        fig = go.Figure(go.Scatter(
            x=xs, y=ys, mode="lines+markers", 
            line=dict(color="#f97316", width=2), 
            marker=dict(size=6, color="#f97316", line=dict(width=1, color="#0d0d0d"))
        ))
        fig.update_layout(
            title=dict(text="Mean Episode Reward per Rollout", font=dict(color="#888", size=14), x=0.5),
            plot_bgcolor="#111", paper_bgcolor="#0d0d0d", 
            font=dict(color="#666"),
            xaxis=dict(title="Rollout", gridcolor="#1e1e1e", zeroline=False, tickfont=dict(size=10)),
            yaxis=dict(title="Mean Reward", gridcolor="#1e1e1e", zeroline=False, tickfont=dict(size=10)),
            margin=dict(l=60, r=30, t=60, b=60),
        )

    return _ppo_status, "\n".join(_ppo_logs[-200:]), fig

# ── UI Construction ──────────────────────────────────────────────────────────

CSS = """
.gradio-container { background-color: #0d0d0d !important; color: white !important; }
.start-btn { background-color: #f97316 !important; color: white !important; font-weight: bold !important; }
.status-box { background-color: #1a1a1a !important; border: 1px solid #333 !important; }
"""

with gr.Blocks(theme=gr.themes.Base(primary_hue="orange", neutral_hue="slate"), css=CSS) as demo:
    gr.Markdown("# 🚀 Life Support RL Trainer")
    gr.Markdown("Train a PPO agent to keep your crew alive in a space habitat.")
    
    with gr.Row():
        task_input = gr.Dropdown(["task_easy", "task_medium", "task_hard"], value="task_easy", label="Task", scale=4)
        start_btn = gr.Button("▶ Start Training", variant="primary", scale=1, elem_classes=["start-btn"])
        
    status_output = gr.Textbox(label="Status", value=_ppo_status, interactive=False, elem_classes=["status-box"])
    
    with gr.Tabs():
        with gr.Tab("📈 Reward Chart"):
            chart_output = gr.Plot(label="")
        with gr.Tab("📋 Training Logs"):
            logs_output = gr.Textbox(label="", lines=15, interactive=False)
            
    gr.Markdown("Tip: Switch to the 📋 Training Logs tab to see step-by-step progress. The trained model is saved as `ppo_<task>.zip` in your Space files.")

    start_btn.click(fn=start_training, inputs=[task_input], outputs=[status_output, logs_output, chart_output])
    
    # Auto-poll
    timer = gr.Timer(value=2)
    timer.tick(fn=poll_updates, outputs=[status_output, logs_output, chart_output])

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860)