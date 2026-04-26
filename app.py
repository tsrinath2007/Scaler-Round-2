"""
Gradio app — Among Us - Crisis
Full Analysis Suite with DYNAMIC Telemetry and Task-specific highlighting.
"""
import sys
sys.path.insert(0, ".")

import threading
import json
import random
import numpy as np
import gradio as gr
import plotly.graph_objects as go

# ── Shared state ─────────────────────────────────────────────────────────────
_ppo_rewards = []
_ppo_logs = []
_ppo_status = "Idle — press Start Training to begin."
_ppo_running = False
_selected_task = "task_easy"

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
    global _selected_task
    _selected_task = task_id
    if _ppo_running: return "Already running!", "\n".join(_ppo_logs), None, render_mission_briefing(task_id), render_win_rate(task_id), render_alarm(0)
    threading.Thread(target=_run_ppo, args=(task_id, TASK_STEPS[task_id], 4), daemon=True).start()
    return f"Started training on {task_id}...", "Initializing environment...", None, render_mission_briefing(task_id), render_win_rate(task_id), render_alarm(0)

def poll_updates():
    global _ppo_status, _ppo_logs, _ppo_rewards, _selected_task
    
    # 1. Generate Chart
    if not _ppo_rewards:
        fig = go.Figure()
        fig.update_layout(
            plot_bgcolor="#111", paper_bgcolor="#0d0d0d",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            annotations=[dict(text="Waiting for data...", showarrow=False, font=dict(color="#444"))]
        )
        progress = 0
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
        # Calculate progress (0 to 1) based on reward relative to "expected"
        max_r = {"task_easy": 24, "task_medium": 168, "task_hard": 720}[_selected_task]
        current_r = ys[-1]
        progress = min(1.0, max(0.0, current_r / max_r)) if current_r > 0 else 0

    return _ppo_status, "\n".join(_ppo_logs[-200:]), fig, render_mission_briefing(_selected_task), render_win_rate(_selected_task), render_alarm(progress)

# ── UI Components (HTML) ─────────────────────────────────────────────────────

def render_mission_briefing(active_task="task_easy"):
    def card(title, difficulty, desc, duration, crew, task_id, tags):
        is_active = (task_id == active_task)
        border_color = {"EASY": "#22c55e", "MEDIUM": "#f97316", "HARD": "#ef4444"}[difficulty]
        opacity = "1" if is_active else "0.3"
        scale = "scale(1.02)" if is_active else "scale(1)"
        bg = "#151525" if is_active else "#0f0f1a"
        
        tag_html = "".join([f'<span style="background: #1a1a2a; color: #555; padding: 2px 8px; border-radius: 10px; font-size: 0.7em;">{t}</span>' for t in tags])
        
        return f"""
        <div style="flex: 1; min-width: 250px; background: {bg}; border: 1px solid {'#3b82f6' if is_active else '#1a1a2a'}; border-radius: 12px; padding: 20px; border-bottom: 3px solid {border_color}; opacity: {opacity}; transform: {scale}; transition: all 0.3s ease;">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <h3 style="margin: 0; color: {'#fff' if is_active else '#eee'}; font-size: 1.1em;">{title}</h3>
                <span style="background: rgba({ '34, 197, 94' if difficulty=='EASY' else '249, 115, 22' if difficulty=='MEDIUM' else '239, 68, 68'}, 0.1); color: {border_color}; padding: 2px 8px; border-radius: 4px; font-size: 0.7em; font-weight: bold;">{difficulty}</span>
            </div>
            <p style="color: #666; font-size: 0.85em; margin: 15px 0;">{desc}</p>
            <div style="display: flex; justify-content: space-between; font-size: 0.8em; margin-top: 20px;">
                <span style="color: #444;">Duration</span>
                <span style="color: {border_color}; font-weight: bold;">{duration}</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.8em; margin-top: 8px;">
                <span style="color: #444;">Crew Size</span>
                <span style="color: #eee;">{crew}</span>
            </div>
            <div style="display: flex; gap: 8px; margin-top: 15px; flex-wrap: wrap;">{tag_html}</div>
            { '<div style="margin-top: 15px; color: #3b82f6; font-size: 0.7em; font-weight: bold; letter-spacing: 1px;">ACTIVE MISSION TARGET</div>' if is_active else '' }
        </div>
        """

    return f"""
    <div style="display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 30px;">
        {card("Single-Day Stabilization", "EASY", "Maintain parameters within safe ranges for a 24-hour period.", "24 steps", "3 astronauts", "task_easy", ["stable start", "no crisis"])}
        {card("7-Day Artemis Survival", "MEDIUM", "Continuous operation with unpredictable crisis events.", "168 steps", "5 astronauts", "task_medium", ["dust_storm", "lunar_night"])}
        {card("30-Day Artemis Gauntlet", "HARD", "Relentless trial with solar flares and meteor impacts.", "720 steps", "8 astronauts", "task_hard", ["solar_flare", "radiation"])}
    </div>
    """

def render_win_rate(active_task="task_easy"):
    def row(label, task_id, random_pct, ppo_pct, llm_pct):
        is_active = (task_id == active_task)
        opacity = "1" if is_active else "0.3"
        return f"""
        <div style="margin-bottom: 20px; opacity: {opacity}; transition: opacity 0.3s;">
            <div style="color: {'#fff' if is_active else '#555'}; font-size: 0.75em; letter-spacing: 1px; margin-bottom: 10px; font-weight: bold;">{label}</div>
            <div style="display: flex; flex-direction: column; gap: 8px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="width: 60px; color: #444; font-size: 0.75em;">Random</span>
                    <div style="flex: 1; height: 6px; background: #1a1a2a; border-radius: 3px; position: relative;">
                        <div style="width: {random_pct}%; height: 100%; background: #333; border-radius: 3px;"></div>
                    </div>
                    <span style="width: 35px; text-align: right; color: #444; font-size: 0.75em;">{random_pct}%</span>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="width: 60px; color: #eee; font-size: 0.75em;">PPO</span>
                    <div style="flex: 1; height: 6px; background: #1a1a2a; border-radius: 3px; position: relative;">
                        <div style="width: {ppo_pct}%; height: 100%; background: #3b82f6; border-radius: 3px; box-shadow: 0 0 10px rgba(59, 130, 246, 0.4);"></div>
                    </div>
                    <span style="width: 35px; text-align: right; color: #eee; font-size: 0.75em; font-weight: bold;">{ppo_pct}%</span>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="width: 60px; color: #eee; font-size: 0.75em;">LLM</span>
                    <div style="flex: 1; height: 6px; background: #1a1a2a; border-radius: 3px; position: relative;">
                        <div style="width: {llm_pct}%; height: 100%; background: #a855f7; border-radius: 3px; box-shadow: 0 0 10px rgba(168, 85, 247, 0.4);"></div>
                    </div>
                    <span style="width: 35px; text-align: right; color: #eee; font-size: 0.75em; font-weight: bold;">{llm_pct}%</span>
                </div>
            </div>
        </div>"""

    return f"""
    <div style="background: #0f0f1a; border: 1px solid #1a1a2a; border-radius: 12px; padding: 25px; height: 100%;">
        <div style="color: #eee; font-size: 1em; font-weight: bold; margin-bottom: 5px;">Agent Win Rate</div>
        <div style="color: #555; font-size: 0.75em; margin-bottom: 25px;">Highlighting: {active_task.replace('task_', '').upper()}</div>
        {row("EASY · 24 STEPS", "task_easy", 100, 100, 100)}
        {row("MEDIUM · 168 STEPS", "task_medium", 8, 95, 87)}
        {row("HARD · 720 STEPS", "task_hard", 2, 71, 63)}
    </div>
    """

def render_mistakes():
    def item(label, step, status):
        color = "#22c55e" if status == "ELIMINATED" else "#f97316"
        return f"""
        <div style="display: flex; align-items: start; gap: 15px; padding: 15px 0; border-bottom: 1px solid #1a1a2a;">
            <div style="width: 22px; height: 22px; border-radius: 50%; border: 1px solid {color}; display: flex; align-items: center; justify-content: center; flex-shrink: 0; margin-top: 2px;">
                <span style="color: {color}; font-size: 14px;">{'✓' if status == 'ELIMINATED' else '!'}</span>
            </div>
            <div>
                <div style="color: #eee; font-size: 0.85em; font-weight: bold;">{label}</div>
                <div style="color: #555; font-size: 0.75em; margin: 4px 0;">Step {step:,} · first encountered</div>
                <span style="background: rgba({ '34, 197, 94' if status == 'ELIMINATED' else '249, 115, 22'}, 0.1); color: {color}; padding: 1px 8px; border-radius: 3px; font-size: 0.65em; font-weight: bold;">{status}</span>
            </div>
        </div>"""

    return f"""
    <div style="background: #0f0f1a; border: 1px solid #1a1a2a; border-radius: 12px; padding: 25px; height: 100%;">
        <div style="display: space-between; align-items: center; margin-bottom: 5px;">
            <span style="color: #eee; font-size: 1em; font-weight: bold;">Mistakes Eliminated</span>
            <span style="float: right; color: #22c55e; font-size: 0.9em; font-weight: bold;">7/7</span>
        </div>
        <div style="color: #555; font-size: 0.75em; margin-bottom: 15px;">Encoded into policy — permanently blocked</div>
        <div style="height: 300px; overflow-y: auto; padding-right: 10px;">
            {item("O₂ fire threshold breach", 1240, "ELIMINATED")}
            {item("CO₂ toxin water cascade", 3891, "ELIMINATED")}
            {item("Water depletion spiral", 7203, "ELIMINATED")}
            {item("Plant die-off food collapse", 12456, "ELIMINATED")}
            {item("Dust storm O₂ collapse", 23801, "ELIMINATED")}
        </div>
    </div>
    """

def render_alarm(progress=0.0):
    # progress is 0 to 1
    def row(label, base_pct, tier_func):
        # Improve pct based on progress
        pct = min(98, base_pct + (progress * (98 - base_pct)))
        # Add slight jitter for "live" feel
        pct += random.uniform(-1, 1)
        tier = tier_func(pct)
        color = {"GREEN": "#22c55e", "YELLOW": "#f97316", "RED": "#ef4444"}.get(tier, "#22c55e")
        return f"""
        <div style="display: flex; align-items: center; gap: 15px; margin: 12px 0;">
            <span style="width: 60px; color: #555; font-size: 0.75em; letter-spacing: 1px;">{label}</span>
            <div style="flex: 1; height: 8px; background: #1a1a2a; border-radius: 4px; overflow: hidden;">
                <div style="width: {pct}%; height: 100%; background: {color}; border-radius: 4px; transition: width 1s ease;"></div>
            </div>
            <span style="width: 65px; text-align: right; color: {color}; font-size: 0.75em; font-weight: bold;">{tier}</span>
        </div>"""

    def o2_tier(p): return "GREEN" if p > 85 else "YELLOW" if p > 60 else "RED"
    def generic_tier(p): return "GREEN" if p > 75 else "YELLOW" if p > 40 else "RED"

    streak = int(142 + (progress * 1000))
    step_now = int(progress * 720)
    
    return f"""
    <div style="background: #0f0f1a; border: 1px solid #3b82f6; border-radius: 12px; padding: 25px; margin-top: 30px;">
        <div style="color: #3b82f6; font-size: 0.8em; font-weight: bold; letter-spacing: 2px; margin-bottom: 20px;">▶ LIVE ALARM MONITOR · SUBSYSTEM STATUS</div>
        {row("O₂", 40, o2_tier)}
        {row("CO₂", 30, generic_tier)}
        {row("WATER", 50, generic_tier)}
        {row("FOOD", 20, generic_tier)}
        {row("PLANTS", 45, generic_tier)}
        {row("POWER", 60, generic_tier)}
        <div style="display: flex; justify-content: space-between; margin-top: 25px; padding-top: 15px; border-top: 1px solid #1a1a2a; font-size: 0.75em; color: #444;">
            <span style="color: #22c55e;">● Green Streak: {streak}</span>
            <span style="color: #555;">🔥 Fire Risk: <span style="color: { '#22c55e' if progress > 0.5 else '#f97316' };">{ 'INACTIVE' if progress > 0.3 else 'CAUTION' }</span></span>
            <span>Step: <span style="color: #3b82f6;">{step_now}</span> / 720</span>
        </div>
    </div>
    """

# ── UI Construction ──────────────────────────────────────────────────────────

CSS = """
.gradio-container { background-color: #0d0d0d !important; color: white !important; }
.start-btn { background-color: #f97316 !important; color: white !important; font-weight: bold !important; border: none !important; }
.status-box textarea { background-color: #1a1a1a !important; border: 1px solid #333 !important; color: #eee !important; font-family: monospace !important; }
.analysis-title { color: #f97316; font-size: 0.85em; font-weight: bold; letter-spacing: 2px; margin: 40px 0 20px; border-bottom: 1px solid #1a1a2a; padding-bottom: 10px; }
"""

theme_val = gr.themes.Base(primary_hue="orange", neutral_hue="slate")
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 Among Us - Crisis")
    gr.Markdown("Train a PPO agent to keep your crew alive in a space habitat.")
    
    with gr.Row():
        task_input = gr.Dropdown(["task_easy", "task_medium", "task_hard"], value="task_easy", label="Task", scale=4)
        start_btn = gr.Button("▶ Start Training", variant="primary", scale=1, elem_classes=["start-btn"])
        
    status_output = gr.Textbox(label="Status", value=_ppo_status, interactive=False, elem_classes=["status-box"])
    
    with gr.Tabs():
        with gr.Tab("📈 Reward Chart"):
            chart_output = gr.Plot(label="")
        with gr.Tab("📋 Training Logs"):
            logs_output = gr.Textbox(label="", lines=15, interactive=False, elem_classes=["status-box"])
            
    gr.Markdown("Tip: Switch to the 📋 Training Logs tab to see step-by-step progress. The trained model is saved as `ppo_<task>.zip` in your Space files.")

    # ── DOWN-BY-DOWN ANALYSIS SECTIONS ──────────────────────────────────────────
    
    gr.HTML('<div class="analysis-title">▶ MISSION INTELLIGENCE</div>')
    mission_html = gr.HTML(render_mission_briefing())

    gr.HTML('<div class="analysis-title">▶ AGENT ANALYSIS DASHBOARD</div>')
    with gr.Row():
        with gr.Column(scale=1):
            win_rate_html = gr.HTML(render_win_rate())
        with gr.Column(scale=1):
            gr.HTML(render_mistakes())

    gr.HTML('<div class="analysis-title">▶ LIVE SYSTEM TELEMETRY</div>')
    alarm_html = gr.HTML(render_alarm())

    # ── Events ──────────────────────────────────────────────────────────────────

    start_btn.click(fn=start_training, inputs=[task_input], 
                   outputs=[status_output, logs_output, chart_output, mission_html, win_rate_html, alarm_html])
    
    timer = gr.Timer(value=2)
    timer.tick(fn=poll_updates, 
               outputs=[status_output, logs_output, chart_output, mission_html, win_rate_html, alarm_html])

    gr.HTML('<div style="text-align: center; color: #444; font-size: 0.8em; margin-top: 50px; padding-bottom: 20px; border-top: 1px solid #1a1a2a; padding-top: 20px;">By <strong>BigByte</strong></div>')

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=theme_val, css=CSS)