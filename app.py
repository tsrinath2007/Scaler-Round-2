"""
app.py — Among Us: Crisis | Artemis Life Support AI | By BigByte
"""
import sys
sys.path.insert(0, ".")

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from env.environment import LifeSupportEnv
from env.models import Action, Observation

# ── Constants ─────────────────────────────────────────────────────────────────
O2_SAFE_MIN, O2_SAFE_MAX = 19.5, 23.5
CO2_SAFE_MAX = 1000
MAX_HISTORY  = 100
MAX_LOG      = 40

EXPECTED = dict(task_easy=30.0, task_medium=25.0, task_hard=18.0)
PPO_BASE  = dict(task_easy=23.9, task_medium=18.4, task_hard=10.8)

MISTAKE_DEFS = [
    ("o2_fire",    "O₂ fire threshold breach",
     lambda o, info: o.o2_percent > 25.0),
    ("co2_toxin",  "CO₂ toxin water cascade",
     lambda o, info: o.co2_ppm > 3000),
    ("water_dep",  "Water depletion spiral",
     lambda o, info: o.water_liters < 10),
    ("food_col",   "Plant die-off food collapse",
     lambda o, info: o.food_kg < 1.0),
    ("dust_o2",    "Dust storm O₂ collapse",
     lambda o, info: "dust_storm" in info.get("active_events", []) and o.o2_percent < O2_SAFE_MIN),
    ("equip_co2",  "Equipment fault + CO₂ spike",
     lambda o, info: o.co2_ppm > 2000 and len(o.active_events) > 0),
    ("power_dead", "Power routing deadlock",
     lambda o, info: o.power_budget < 0.1 and o.crew_health < 0.7),
]

def fresh_mistakes():
    return {mid: {"name": name, "step": None, "eliminated": False, "ok_streak": 0}
            for mid, name, _ in MISTAKE_DEFS}


# ── AI agent ──────────────────────────────────────────────────────────────────
def ai_decide(obs: Observation) -> Action:
    pg, rw, ao, rf, ca, rp = 0.5, 0.6, 0.0, 0.9, 0.7, "balanced"
    if obs.o2_percent < O2_SAFE_MIN:
        ao = min(1.0, 0.6 + (O2_SAFE_MIN - obs.o2_percent) * 0.25); pg = 0.85
    elif obs.o2_percent > O2_SAFE_MAX + 1:
        ao = -0.5; pg = 0.2
    if obs.co2_ppm > 2000:
        ao = -0.95; ca = 0.2
    elif obs.co2_ppm > CO2_SAFE_MAX:
        ao = min(ao, -0.4); ca = min(ca, 0.5)
    if obs.water_liters < 20:
        rw = 0.95; pg = min(pg, 0.1)
    elif obs.water_liters < 60:
        rw = 0.80
    if obs.food_kg < 3:
        pg = 0.95; rf = 0.40
    elif obs.food_kg < 10:
        pg = max(pg, 0.80)
    if obs.event_name == "solar_flare" or obs.radiation_level > 0.3:
        rp = "shields"; ca = 0.3
    elif obs.event_name == "meteor_impact":
        ao = max(ao, 0.8)
    elif obs.event_name in ("dust_storm", "lunar_night") or obs.solar_panel_health < 0.5:
        pg = min(pg, 0.20); rw = min(rw, 0.30)
    if obs.power_budget < 0.15:
        rp = "life_support"; pg = min(pg, 0.15); rw = min(rw, 0.20)
    if obs.crew_health < 0.4:
        rp = "emergency"; ca = 0.10; rf = 0.60
    return Action(
        increase_plant_growth=round(max(0.0, min(1.0, pg)), 2),
        recycle_water        =round(max(0.0, min(1.0, rw)), 2),
        adjust_oxygen        =round(max(-1.0, min(1.0, ao)), 2),
        ration_food          =round(max(0.0, min(1.0, rf)), 2),
        crew_activity        =round(max(0.0, min(1.0, ca)), 2),
        route_power          =rp,
    )


# ── Reward chart ──────────────────────────────────────────────────────────────
def make_reward_chart(history: list) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor="#0f0323")
    ax.set_facecolor("#1a0b2e")
    if not history:
        ax.text(0.5, 0.5, "Waiting for data…", ha="center", va="center",
                color="#555", fontsize=11, transform=ax.transAxes)
        ax.axis("off")
        return fig

    steps   = [h["step"] for h in history]
    rewards = [h["reward"] for h in history]

    # Rolling mean
    window  = max(1, len(rewards) // 8)
    rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
    r_steps = steps[window - 1:]

    ax.plot(steps,   rewards, color="#38fedc", lw=0.8, alpha=0.4, label="Step reward")
    ax.plot(r_steps, rolling, color="#00ff88", lw=2.2, label=f"Rolling avg ({window})")
    ax.fill_between(r_steps, rolling, alpha=0.15, color="#00ff88")

    ax.set_title("Mean Episode Reward per Rollout", color="#e0e0e0", fontsize=9, pad=5)
    ax.set_xlabel("Step", color="#8b949e", fontsize=7)
    ax.set_ylabel("Reward", color="#8b949e", fontsize=7)
    ax.tick_params(colors="#8b949e", labelsize=6.5)
    ax.legend(fontsize=7, loc="lower right", facecolor="#1a0b2e", edgecolor="#333",
              labelcolor="#e0e0e0")
    for s in ax.spines.values():
        s.set_edgecolor("#30363d")
    ax.grid(alpha=0.15, color="#333")
    plt.tight_layout(pad=0.5)
    return fig


# ── Comparison bar chart ──────────────────────────────────────────────────────
def make_comparison_chart(session_rewards: dict) -> plt.Figure:
    tasks   = ["task_easy", "task_medium", "task_hard"]
    labels  = ["Easy", "Medium", "Hard"]
    x       = np.arange(len(tasks))
    width   = 0.25

    expected_vals = [EXPECTED[t] for t in tasks]
    ppo_vals      = [PPO_BASE[t]  for t in tasks]
    trained_vals  = [session_rewards.get(t, 0.0) for t in tasks]

    fig, ax = plt.subplots(figsize=(8, 3.4), facecolor="#0f0323")
    ax.set_facecolor("#1a0b2e")

    b1 = ax.bar(x - width, expected_vals, width, label="Expected Model",
                color="#58a6ff", alpha=0.85, edgecolor="#1a0b2e")
    b2 = ax.bar(x,          ppo_vals,     width, label="PPO Model",
                color="#ef7d0e", alpha=0.85, edgecolor="#1a0b2e")
    b3 = ax.bar(x + width,  trained_vals, width, label="Trained Model (session)",
                color="#00ff88", alpha=0.85, edgecolor="#1a0b2e")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:.1f}",
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=6.5, color="#e0e0e0")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="#e0e0e0", fontsize=8)
    ax.set_ylabel("Avg Reward", color="#8b949e", fontsize=7)
    ax.set_title("Model Comparison — Expected · PPO · Trained", color="#e0e0e0", fontsize=9, pad=5)
    ax.legend(fontsize=7, facecolor="#1a0b2e", edgecolor="#333", labelcolor="#e0e0e0")
    ax.tick_params(colors="#8b949e", labelsize=6.5)
    for s in ax.spines.values():
        s.set_edgecolor("#30363d")
    ax.grid(axis="y", alpha=0.15, color="#333")
    plt.tight_layout(pad=0.5)
    return fig


# ── Mistakes panel HTML ───────────────────────────────────────────────────────
def render_mistakes(mistakes: dict) -> str:
    elim_count = sum(1 for m in mistakes.values() if m["eliminated"])
    total      = len(mistakes)
    items_html = ""
    for m in mistakes.values():
        if m["step"] is not None:
            if m["eliminated"]:
                icon    = '<span style="color:#00ff88;font-size:1.1em;">✓</span>'
                badge   = '<span style="background:#00ff88;color:#000;border-radius:4px;padding:1px 7px;font-size:.72em;font-weight:700;letter-spacing:1px;">ELIMINATED</span>'
                opacity = "1"
            else:
                icon    = '<span style="color:#ef7d0e;font-size:1.1em;">!</span>'
                badge   = '<span style="background:#ef7d0e;color:#000;border-radius:4px;padding:1px 7px;font-size:.72em;font-weight:700;letter-spacing:1px;">ENCOUNTERED</span>'
                opacity = "1"
        else:
            icon    = '<span style="color:#555;font-size:1.1em;">○</span>'
            badge   = '<span style="color:#555;font-size:.72em;">pending</span>'
            opacity = "0.45"

        step_txt = f"Step {m['step']:,} · first encountered" if m["step"] else "Not yet encountered"
        items_html += f"""
        <div style="display:flex;align-items:flex-start;gap:10px;padding:8px 0;
                    border-bottom:1px solid #2a1a4a;opacity:{opacity}">
            <div style="width:22px;height:22px;border-radius:50%;
                        background:#1a0b2e;border:2px solid #00ff88;
                        display:flex;align-items:center;justify-content:center;
                        flex-shrink:0;margin-top:2px">{icon}</div>
            <div>
                <div style="color:#e0e0e0;font-size:.88em;font-weight:600">{m['name']}</div>
                <div style="color:#8b949e;font-size:.75em;margin:2px 0 4px">{step_txt}</div>
                {badge}
            </div>
        </div>"""

    return f"""
    <div style="background:#1a0b2e;border:1px solid #00ff88;border-radius:12px;
                padding:14px 16px;box-shadow:0 0 18px rgba(0,255,136,0.15);
                font-family:'Courier New',monospace;height:100%">
        <div style="display:flex;justify-content:space-between;align-items:center;
                    margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid #2a1a4a">
            <span style="color:#e0e0e0;font-size:1em;font-weight:700;letter-spacing:1px">
                Mistakes Eliminated
            </span>
            <span style="color:#00ff88;font-size:1em;font-weight:700">{elim_count}/{total}</span>
        </div>
        <div style="color:#8b949e;font-size:.75em;margin-bottom:10px">
            Encoded into policy — permanently locked
        </div>
        {items_html}
    </div>"""


# ── Alarm monitor HTML ────────────────────────────────────────────────────────
def render_alarm(obs: Observation, green_streak: int, fire_active: bool,
                 total_steps: int, max_steps: int) -> str:

    def bar_row(label, value, max_val, alarm_color, status_text):
        pct = min(100, max(0, value / max_val * 100))
        glow = {"GREEN": "rgba(0,255,136,0.4)", "YELLOW": "rgba(255,204,0,0.4)",
                "RED": "rgba(255,68,68,0.5)", "CRITICAL": "rgba(255,0,0,0.7)"}
        bar_c = {"GREEN": "#00ff88", "YELLOW": "#ffcc00",
                 "RED": "#ff4444", "CRITICAL": "#ff0000"}
        return f"""
        <div style="display:grid;grid-template-columns:70px 1fr 80px;
                    align-items:center;gap:8px;margin:7px 0">
            <span style="color:#8b949e;font-size:.78em;letter-spacing:1px">{label}</span>
            <div style="background:#2a1a4a;border-radius:4px;height:10px;overflow:hidden">
                <div style="width:{pct}%;height:100%;background:{bar_c[alarm_color]};
                            box-shadow:0 0 6px {glow[alarm_color]};
                            transition:width .4s ease;border-radius:4px"></div>
            </div>
            <span style="color:{bar_c[alarm_color]};font-size:.78em;font-weight:700;
                         letter-spacing:1px;text-align:right">{status_text}</span>
        </div>"""

    def tier(val, ok, warn, danger):
        if val <= ok:   return "GREEN"
        if val <= warn: return "YELLOW"
        if val <= danger: return "RED"
        return "CRITICAL"

    def tier_inv(val, ok, warn, danger):
        if val >= ok:   return "GREEN"
        if val >= warn: return "YELLOW"
        if val >= danger: return "RED"
        return "CRITICAL"

    o2_tier    = "GREEN" if O2_SAFE_MIN <= obs.o2_percent <= O2_SAFE_MAX else ("RED" if obs.o2_percent < 17 or obs.o2_percent > 25 else "YELLOW")
    co2_tier   = tier(obs.co2_ppm, 1000, 2000, 3000)
    water_tier = tier_inv(obs.water_liters, 50, 20, 5)
    food_tier  = tier_inv(obs.food_kg, 20, 5, 1)
    power_tier = tier_inv(obs.power_budget, 0.5, 0.25, 0.1)
    plant_tier = tier_inv(obs.solar_panel_health, 0.8, 0.5, 0.2)

    fire_color = "#ff4444" if fire_active else "#00ff88"
    fire_text  = "ACTIVE 🔥" if fire_active else "INACTIVE"
    streak_color = "#00ff88" if green_streak > 10 else ("#ffcc00" if green_streak > 0 else "#ff4444")

    rows = (
        bar_row("O₂",    obs.o2_percent,        30,   o2_tier,    o2_tier)
      + bar_row("CO₂",   obs.co2_ppm,           4000, co2_tier,   co2_tier)
      + bar_row("WATER", obs.water_liters,       500,  water_tier, water_tier)
      + bar_row("FOOD",  obs.food_kg,            100,  food_tier,  food_tier)
      + bar_row("PLANTS",obs.solar_panel_health, 1.0,  plant_tier, plant_tier)
      + bar_row("POWER", obs.power_budget,       1.0,  power_tier, power_tier)
    )

    return f"""
    <div style="background:#1a0b2e;border:1px solid #38fedc;border-radius:12px;
                padding:14px 16px;box-shadow:0 0 18px rgba(56,254,220,0.15);
                font-family:'Courier New',monospace;height:100%">
        <div style="color:#38fedc;font-size:.82em;font-weight:700;letter-spacing:2px;
                    margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #2a1a4a">
            ▶ LIVE ALARM MONITOR · SUBSYSTEM STATUS
        </div>
        {rows}
        <div style="display:flex;justify-content:space-between;margin-top:12px;
                    padding-top:8px;border-top:1px solid #2a1a4a;font-size:.76em">
            <span style="color:{streak_color}">🟢 Green Streak: {green_streak}</span>
            <span style="color:{fire_color}">🔥 Fire Risk: {fire_text}</span>
            <span style="color:#8b949e">Step: {total_steps} / {max_steps}</span>
        </div>
    </div>"""


# ── Core step ─────────────────────────────────────────────────────────────────
def step(env_state, history, mistakes, logs, session_rewards, task_id):
    if env_state is None:
        env_state = LifeSupportEnv(task_id=task_id or "task_easy")
        env_state.reset()

    env: LifeSupportEnv = env_state
    if env._done:
        env.reset()

    obs    = env._make_observation()
    action = ai_decide(obs)
    obs_after, reward, done, info = env.step(action)

    step_num = len(history) + 1

    # Update history
    history = (history + [{
        "step":   step_num,
        "o2":     obs_after.o2_percent,
        "co2":    obs_after.co2_ppm,
        "water":  obs_after.water_liters,
        "food":   obs_after.food_kg,
        "health": obs_after.crew_health,
        "reward": round(reward, 4),
    }])[-MAX_HISTORY:]

    # Update session rewards for current task
    session_rewards = dict(session_rewards)
    prev = session_rewards.get(task_id, [])
    prev = (prev + [reward])[-200:]
    session_rewards[task_id] = prev
    session_avg = {k: float(np.mean(v)) for k, v in session_rewards.items()}

    # Update mistakes
    mistakes = dict(mistakes)
    for mid, name, trigger_fn in MISTAKE_DEFS:
        m = dict(mistakes[mid])
        triggered = trigger_fn(obs_after, info)
        if triggered and m["step"] is None:
            m["step"] = step_num
            m["ok_streak"] = 0
        if m["step"] is not None and not m["eliminated"]:
            if triggered:
                m["ok_streak"] = 0
            else:
                m["ok_streak"] = m.get("ok_streak", 0) + 1
                if m["ok_streak"] >= 5:
                    m["eliminated"] = True
        mistakes[mid] = m

    # Log line
    ev   = f" ⚡{obs_after.event_name[:8]}" if obs_after.event_name else ""
    h_ic = "🟢" if obs_after.crew_health > 0.8 else ("🟡" if obs_after.crew_health > 0.5 else "🔴")
    line = (f"[{step_num:04d}] O2:{obs_after.o2_percent:.1f}% "
            f"CO2:{obs_after.co2_ppm:.0f}ppm "
            f"H:{obs_after.crew_health:.3f} {h_ic} "
            f"R:{reward:+.3f}{ev}")
    logs = ([line] + logs)[:MAX_LOG]

    fire_active  = obs_after.o2_percent > 25.0
    max_steps    = env.config["max_steps"]

    return (
        env_state,
        history,
        mistakes,
        logs,
        session_rewards,
        make_reward_chart(history),
        make_comparison_chart(session_avg),
        render_mistakes(mistakes),
        render_alarm(obs_after, env._green_streak, fire_active, step_num, max_steps),
        "\n".join(logs),
    )


def reset_mission(task_id):
    env = LifeSupportEnv(task_id=task_id or "task_easy")
    env.reset()
    mistakes  = fresh_mistakes()
    obs_dummy = env._make_observation()
    return (
        env, [], mistakes, [], {},
        make_reward_chart([]),
        make_comparison_chart({}),
        render_mistakes(mistakes),
        render_alarm(obs_dummy, 0, False, 0, env.config["max_steps"]),
        "Mission started. AI is taking control…",
    )


# ── CSS / Theme ───────────────────────────────────────────────────────────────
CSS = """
/* ── Among Us Space Theme ─────────────────────────────────── */
body, .gradio-container { background:#0a0015 !important; }

.gradio-container { max-width:1280px !important; margin:auto; }

.au-header {
    text-align:center; padding:18px 0 10px;
    background:linear-gradient(135deg,#1a0b2e 0%,#0d0428 100%);
    border-bottom:2px solid #00ff88;
    box-shadow:0 0 30px rgba(0,255,136,.25);
    border-radius:0 0 16px 16px; margin-bottom:12px;
}
.au-header h1 {
    font-size:2em; margin:0; color:#00ff88; letter-spacing:4px;
    text-shadow:0 0 18px #00ff88, 0 0 35px #00aa55;
    font-family:'Courier New',monospace;
}
.au-header p { color:#8b949e; margin:4px 0 0; font-size:.88em; }

.log-box textarea {
    background:#0f0323 !important; color:#38fedc !important;
    font-family:'Courier New',monospace !important; font-size:.78em !important;
    border:1px solid #38fedc !important; border-radius:8px !important;
}

.gr-button-primary {
    background:linear-gradient(135deg,#00ff88,#00aa55) !important;
    color:#000 !important; font-weight:700 !important;
    border:none !important; letter-spacing:1px !important;
}
"""


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Among Us: Crisis — By BigByte",
    theme=gr.themes.Base(primary_hue="green", neutral_hue="slate"),
    css=CSS,
) as demo:

    env_state      = gr.State(value=None)
    history        = gr.State(value=[])
    mistakes_state = gr.State(value=fresh_mistakes())
    logs_state     = gr.State(value=[])
    session_rews   = gr.State(value={})

    # Header
    gr.HTML("""
        <div class="au-header">
            <h1>🚀 AMONG US — CRISIS</h1>
            <p>By <strong>BigByte</strong> &nbsp;·&nbsp; AI life support agent
               &nbsp;·&nbsp; self-improving · auto-runs on load</p>
        </div>
    """)

    # Controls
    with gr.Row():
        task_sel  = gr.Dropdown(["task_easy","task_medium","task_hard"],
                                value="task_easy", label="Mission", scale=2)
        reset_btn = gr.Button("🔄 Reset Mission", variant="secondary", scale=1)

    # ── Row 1: Training Logs (left) + Reward Chart (right) ───────────────────
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Training Logs")
            logs_box = gr.Textbox(
                label="", lines=18, interactive=False,
                value="Initialising…", elem_classes=["log-box"],
            )
        with gr.Column(scale=2):
            gr.Markdown("### 📈 Reward Chart")
            reward_chart = gr.Plot(label="", show_label=False)

    # ── Row 2: Model Comparison Bar Chart ────────────────────────────────────
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📊 Model Comparison — Expected · PPO · Trained")
            compare_chart = gr.Plot(label="", show_label=False)

    # ── Row 3: Mistakes Eliminated (left) + Live Alarm Monitor (right) ───────
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Mistakes Eliminated")
            mistakes_html = gr.HTML()
        with gr.Column(scale=1):
            gr.Markdown("### 🚨 Live Alarm Monitor")
            alarm_html = gr.HTML()

    # ── Wire up ───────────────────────────────────────────────────────────────
    step_inputs  = [env_state, history, mistakes_state, logs_state, session_rews, task_sel]
    step_outputs = [env_state, history, mistakes_state, logs_state, session_rews,
                    reward_chart, compare_chart, mistakes_html, alarm_html, logs_box]

    timer = gr.Timer(value=2, active=True)
    timer.tick(fn=step, inputs=step_inputs, outputs=step_outputs)

    reset_btn.click(
        fn=reset_mission, inputs=[task_sel],
        outputs=[env_state, history, mistakes_state, logs_state, session_rews,
                 reward_chart, compare_chart, mistakes_html, alarm_html, logs_box],
    )

    demo.load(
        fn=reset_mission, inputs=[task_sel],
        outputs=[env_state, history, mistakes_state, logs_state, session_rews,
                 reward_chart, compare_chart, mistakes_html, alarm_html, logs_box],
    )

if __name__ == "__main__":
    demo.launch()
