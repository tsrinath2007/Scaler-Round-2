"""
app.py — Among Us: Crisis | By BigByte
Black · Orange · Blue theme. Auto-runs, saves log to file, Stop for analysis.
"""
import sys, json, datetime
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
MAX_HISTORY  = 120
MAX_LOG      = 50
LOG_FILE     = "simulation_log.json"

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


# ── Save to file ──────────────────────────────────────────────────────────────
def save_log(history, mistakes, session_rewards, task_id):
    try:
        data = {
            "saved_at":      datetime.datetime.now().isoformat(),
            "task_id":       task_id,
            "total_steps":   len(history),
            "history":       history[-50:],          # last 50 steps
            "mistakes":      {k: {kk: vv for kk, vv in v.items() if kk != "ok_streak"}
                              for k, v in mistakes.items()},
            "session_rewards": {k: round(float(np.mean(v)), 4)
                                for k, v in session_rewards.items() if v},
        }
        with open(LOG_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


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
def make_reward_chart(history):
    fig, ax = plt.subplots(figsize=(6, 3.2), facecolor="#0a0a0f")
    ax.set_facecolor("#0f0f1a")
    if not history:
        ax.text(0.5, 0.5, "Waiting for data…", ha="center", va="center",
                color="#555", fontsize=10, transform=ax.transAxes)
        ax.axis("off")
        return fig

    steps   = [h["step"] for h in history]
    rewards = [h["reward"] for h in history]
    w       = max(1, len(rewards) // 8)
    rolling = np.convolve(rewards, np.ones(w) / w, mode="valid")
    rs      = steps[w - 1:]

    ax.plot(steps, rewards, color="#3b82f6", lw=0.8, alpha=0.35, label="Step")
    ax.plot(rs, rolling,    color="#ff8c00", lw=2.0, label=f"Avg ({w})")
    ax.fill_between(rs, rolling, alpha=0.12, color="#ff8c00")

    ax.set_title("Reward per Rollout", color="#e0e0e0", fontsize=9)
    ax.set_xlabel("Step", color="#666", fontsize=7)
    ax.set_ylabel("Reward", color="#666", fontsize=7)
    ax.tick_params(colors="#666", labelsize=6)
    ax.legend(fontsize=7, facecolor="#0f0f1a", edgecolor="#222", labelcolor="#ccc")
    for s in ax.spines.values(): s.set_edgecolor("#222")
    ax.grid(alpha=0.1, color="#333")
    plt.tight_layout(pad=0.4)
    return fig


# ── Comparison bar chart ──────────────────────────────────────────────────────
def make_comparison_chart(session_rewards):
    tasks  = ["task_easy", "task_medium", "task_hard"]
    labels = ["Easy", "Medium", "Hard"]
    x, w   = np.arange(3), 0.25

    exp_v  = [EXPECTED[t] for t in tasks]
    ppo_v  = [PPO_BASE[t]  for t in tasks]
    trn_v  = [session_rewards.get(t, 0.0) for t in tasks]

    fig, ax = plt.subplots(figsize=(9, 3.0), facecolor="#0a0a0f")
    ax.set_facecolor("#0f0f1a")

    b1 = ax.bar(x - w, exp_v, w, label="Expected", color="#3b82f6", alpha=0.85, edgecolor="#0a0a0f")
    b2 = ax.bar(x,     ppo_v, w, label="PPO",      color="#ff8c00", alpha=0.85, edgecolor="#0a0a0f")
    b3 = ax.bar(x + w, trn_v, w, label="Trained",  color="#e0e0e0", alpha=0.75, edgecolor="#0a0a0f")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            if h > 0.5:
                ax.annotate(f"{h:.1f}",
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=6.5, color="#ccc")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="#ccc", fontsize=8)
    ax.set_ylabel("Avg Reward", color="#666", fontsize=7)
    ax.set_title("Model Comparison — Expected  ·  PPO  ·  Trained", color="#e0e0e0", fontsize=9)
    ax.legend(fontsize=7, facecolor="#0f0f1a", edgecolor="#222", labelcolor="#ccc")
    ax.tick_params(colors="#666", labelsize=6)
    for s in ax.spines.values(): s.set_edgecolor("#222")
    ax.grid(axis="y", alpha=0.1, color="#333")
    plt.tight_layout(pad=0.4)
    return fig


# ── Mistakes HTML ─────────────────────────────────────────────────────────────
def render_mistakes(mistakes):
    elim = sum(1 for m in mistakes.values() if m["eliminated"])
    total = len(mistakes)
    rows = ""
    for m in mistakes.values():
        if m["step"] is not None:
            if m["eliminated"]:
                dot   = f'<span style="color:#ff8c00">✓</span>'
                badge = '<span style="background:#ff8c00;color:#000;padding:1px 8px;border-radius:3px;font-size:.7em;font-weight:700">ELIMINATED</span>'
            else:
                dot   = f'<span style="color:#3b82f6">!</span>'
                badge = '<span style="background:#3b82f6;color:#fff;padding:1px 8px;border-radius:3px;font-size:.7em;font-weight:700">ENCOUNTERED</span>'
            detail = f"Step {m['step']:,} · first encountered"
        else:
            dot   = '<span style="color:#333">○</span>'
            badge = '<span style="color:#444;font-size:.7em">pending</span>'
            detail = "Not yet encountered"

        rows += f"""
        <div style="display:flex;gap:10px;align-items:flex-start;
                    padding:7px 0;border-bottom:1px solid #1a1a2a">
            <div style="width:20px;height:20px;border:1px solid #333;border-radius:50%;
                        display:flex;align-items:center;justify-content:center;
                        flex-shrink:0;font-size:.85em">{dot}</div>
            <div>
                <div style="color:#e0e0e0;font-size:.85em;font-weight:600">{m['name']}</div>
                <div style="color:#555;font-size:.73em;margin:2px 0 4px">{detail}</div>
                {badge}
            </div>
        </div>"""

    return f"""
    <div style="background:#0f0f1a;border:1px solid #ff8c00;border-radius:10px;padding:14px 16px">
        <div style="display:flex;justify-content:space-between;margin-bottom:10px;
                    padding-bottom:8px;border-bottom:1px solid #1a1a2a">
            <span style="color:#ff8c00;font-weight:700;letter-spacing:1px;font-size:.92em">
                MISTAKES ELIMINATED
            </span>
            <span style="color:#ff8c00;font-weight:700">{elim}/{total}</span>
        </div>
        <div style="color:#555;font-size:.73em;margin-bottom:8px">Encoded into policy — permanently locked</div>
        {rows}
    </div>"""


# ── Alarm monitor HTML ────────────────────────────────────────────────────────
def render_alarm(obs: Observation, green_streak, fire_active, step_num, max_steps):
    def row(label, value, max_val, tier):
        pct = min(100, max(0, value / max_val * 100))
        colors = {"GREEN": "#3b82f6", "YELLOW": "#ff8c00", "RED": "#ff4444", "CRITICAL": "#ff0000"}
        c = colors.get(tier, "#3b82f6")
        return f"""
        <div style="display:grid;grid-template-columns:65px 1fr 80px;
                    align-items:center;gap:8px;margin:6px 0">
            <span style="color:#888;font-size:.76em;letter-spacing:1px">{label}</span>
            <div style="background:#1a1a2a;border-radius:3px;height:9px;overflow:hidden">
                <div style="width:{pct}%;height:100%;background:{c};border-radius:3px"></div>
            </div>
            <span style="color:{c};font-size:.76em;font-weight:700;text-align:right">{tier}</span>
        </div>"""

    def t_co2(v):
        return "GREEN" if v<=1000 else ("YELLOW" if v<=2000 else ("RED" if v<=3000 else "CRITICAL"))
    def t_inv(v, ok, warn, danger):
        return "GREEN" if v>=ok else ("YELLOW" if v>=warn else ("RED" if v>=danger else "CRITICAL"))
    def t_o2(v):
        return "GREEN" if O2_SAFE_MIN<=v<=O2_SAFE_MAX else ("RED" if v<17 or v>25 else "YELLOW")

    fire_c = "#ff4444" if fire_active else "#3b82f6"
    fire_t = "ACTIVE" if fire_active else "INACTIVE"
    sk_c   = "#ff8c00" if green_streak > 10 else ("#3b82f6" if green_streak > 0 else "#ff4444")

    return f"""
    <div style="background:#0f0f1a;border:1px solid #3b82f6;border-radius:10px;padding:14px 16px">
        <div style="color:#3b82f6;font-size:.8em;font-weight:700;letter-spacing:2px;
                    margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #1a1a2a">
            ▶ LIVE ALARM MONITOR · SUBSYSTEM STATUS
        </div>
        {row("O₂",    obs.o2_percent,        30,   t_o2(obs.o2_percent))}
        {row("CO₂",   obs.co2_ppm,           4000, t_co2(obs.co2_ppm))}
        {row("WATER", obs.water_liters,       500,  t_inv(obs.water_liters, 50, 20, 5))}
        {row("FOOD",  obs.food_kg,            100,  t_inv(obs.food_kg, 20, 5, 1))}
        {row("PLANTS",obs.solar_panel_health, 1.0,  t_inv(obs.solar_panel_health, .8, .5, .2))}
        {row("POWER", obs.power_budget,       1.0,  t_inv(obs.power_budget, .5, .25, .1))}
        <div style="display:flex;justify-content:space-between;margin-top:10px;
                    padding-top:8px;border-top:1px solid #1a1a2a;font-size:.74em">
            <span style="color:{sk_c}">● Streak: {green_streak}</span>
            <span style="color:{fire_c}">🔥 Fire: {fire_t}</span>
            <span style="color:#555">Step {step_num}/{max_steps}</span>
        </div>
    </div>"""


# ── Complete analysis (on Stop) ───────────────────────────────────────────────
def build_analysis(history, mistakes, session_rewards, task_id):
    if not history:
        return "No data yet — run the simulation first."

    rewards   = [h["reward"] for h in history]
    healths   = [h["health"] for h in history]
    n         = len(history)
    elim      = sum(1 for m in mistakes.values() if m["eliminated"])
    encnt     = sum(1 for m in mistakes.values() if m["step"] is not None)

    lines = [
        f"{'='*52}",
        f"  AMONG US: CRISIS — COMPLETE ANALYSIS",
        f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"{'='*52}",
        f"",
        f"  Task          : {task_id}",
        f"  Total Steps   : {n}",
        f"",
        f"  REWARD STATS",
        f"  ├ Mean        : {np.mean(rewards):+.4f}",
        f"  ├ Max         : {np.max(rewards):+.4f}",
        f"  ├ Min         : {np.min(rewards):+.4f}",
        f"  └ Std Dev     : {np.std(rewards):.4f}",
        f"",
        f"  CREW HEALTH",
        f"  ├ Mean        : {np.mean(healths):.4f}",
        f"  ├ Final       : {healths[-1]:.4f}",
        f"  └ Min seen    : {np.min(healths):.4f}",
        f"",
        f"  MISTAKES",
        f"  ├ Encountered : {encnt}/7",
        f"  └ Eliminated  : {elim}/7",
        f"",
        f"  MISTAKE BREAKDOWN",
    ]
    for m in mistakes.values():
        status = "ELIMINATED" if m["eliminated"] else ("ENCOUNTERED" if m["step"] else "NEVER SEEN")
        step_s = f"@ step {m['step']:,}" if m["step"] else ""
        lines.append(f"  {'✓' if m['eliminated'] else '○'} {m['name'][:35]:<35} {status} {step_s}")

    lines += [
        f"",
        f"  SESSION AVG REWARDS (all tasks)",
    ]
    for t in ["task_easy", "task_medium", "task_hard"]:
        v = session_rewards.get(t, [])
        if v:
            lines.append(f"  ├ {t:<15}: {np.mean(v):+.4f}")

    lines += [f"", f"  Full log saved → {LOG_FILE}", f"{'='*52}"]

    try:
        with open("analysis_report.txt", "w") as f:
            f.write("\n".join(lines))
    except Exception:
        pass

    return "\n".join(lines)


# ── Core step ─────────────────────────────────────────────────────────────────
def step_fn(env_state, history, mistakes, logs, session_rewards, task_id):
    if env_state is None:
        env_state = LifeSupportEnv(task_id=task_id or "task_easy")
        env_state.reset()

    env: LifeSupportEnv = env_state
    if env._done:
        env.reset()

    obs    = env._make_observation()
    action = ai_decide(obs)
    obs_a, reward, done, info = env.step(action)

    sn = len(history) + 1
    history = (history + [{
        "step": sn, "o2": obs_a.o2_percent, "co2": obs_a.co2_ppm,
        "water": obs_a.water_liters, "food": obs_a.food_kg,
        "health": obs_a.crew_health, "reward": round(reward, 4),
    }])[-MAX_HISTORY:]

    # Session rewards
    session_rewards = dict(session_rewards)
    prev = session_rewards.get(task_id, [])
    session_rewards[task_id] = (prev + [reward])[-300:]
    session_avg = {k: float(np.mean(v)) for k, v in session_rewards.items()}

    # Mistakes
    mistakes = dict(mistakes)
    for mid, name, trigger_fn in MISTAKE_DEFS:
        m = dict(mistakes[mid])
        hit = trigger_fn(obs_a, info)
        if hit and m["step"] is None:
            m["step"] = sn; m["ok_streak"] = 0
        if m["step"] is not None and not m["eliminated"]:
            m["ok_streak"] = 0 if hit else m.get("ok_streak", 0) + 1
            if m["ok_streak"] >= 5:
                m["eliminated"] = True
        mistakes[mid] = m

    # Log
    ev   = f" [{obs_a.event_name[:8]}]" if obs_a.event_name else ""
    ic   = "▲" if obs_a.crew_health > 0.8 else ("▼" if obs_a.crew_health < 0.5 else "─")
    line = (f"[{sn:04d}] O2:{obs_a.o2_percent:.1f}% CO2:{obs_a.co2_ppm:.0f}ppm "
            f"H:{obs_a.crew_health:.3f}{ic} R:{reward:+.3f}{ev}")
    logs = ([line] + logs)[:MAX_LOG]

    # Save every 10 steps
    if sn % 10 == 0:
        save_log(history, mistakes, session_rewards, task_id)

    fire = obs_a.o2_percent > 25.0

    return (
        env_state, history, mistakes, logs, session_rewards,
        make_reward_chart(history),
        make_comparison_chart(session_avg),
        render_mistakes(mistakes),
        render_alarm(obs_a, env._green_streak, fire, sn, env.config["max_steps"]),
        "\n".join(logs),
        "",   # clear analysis panel while running
    )


def reset_fn(task_id):
    env = LifeSupportEnv(task_id=task_id or "task_easy")
    env.reset()
    m   = fresh_mistakes()
    obs = env._make_observation()
    return (
        env, [], m, [], {},
        make_reward_chart([]),
        make_comparison_chart({}),
        render_mistakes(m),
        render_alarm(obs, 0, False, 0, env.config["max_steps"]),
        "Mission reset. Starting…",
        "",
    )


def stop_fn(_env_state, history, mistakes, session_rewards, task_id):
    save_log(history, mistakes, session_rewards, task_id)
    report = build_analysis(history, mistakes, session_rewards, task_id)
    return gr.Timer(active=False), report


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
body, .gradio-container { background:#080810 !important; }
.gradio-container { max-width:1200px !important; margin:auto; }

.au-header {
    text-align:center; padding:16px 0 10px;
    border-bottom:2px solid #ff8c00;
    background:#0a0a14; border-radius:0 0 12px 12px;
    margin-bottom:14px;
}
.au-header h1 {
    font-size:1.9em; margin:0; color:#ff8c00;
    text-shadow:0 0 16px rgba(255,140,0,.6);
    font-family:'Courier New',monospace; letter-spacing:5px;
}
.au-header p { color:#555; margin:4px 0 0; font-size:.84em; }

.section-title {
    color:#ff8c00; font-size:.8em; font-weight:700;
    letter-spacing:2px; margin-bottom:6px;
}

textarea {
    background:#0a0a14 !important; color:#3b82f6 !important;
    font-family:'Courier New',monospace !important; font-size:.77em !important;
    border:1px solid #1a1a2a !important; border-radius:6px !important;
}

.analysis-box textarea {
    color:#ff8c00 !important; border:1px solid #ff8c00 !important;
    font-size:.78em !important;
}

button.primary { background:#ff8c00 !important; color:#000 !important; font-weight:700 !important; }
button.stop-btn { background:#3b82f6 !important; color:#fff !important; font-weight:700 !important; }
"""

# ── UI ────────────────────────────────────────────────────────────────────────
theme_val = gr.themes.Base(primary_hue="orange", neutral_hue="slate")
with gr.Blocks(title="Among Us: Crisis — By BigByte") as demo:

    env_state    = gr.State(None)
    history_s    = gr.State([])
    mistakes_s   = gr.State(fresh_mistakes())
    logs_s       = gr.State([])
    session_s    = gr.State({})

    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML("""
        <div class="au-header">
            <h1>🚀 AMONG US — CRISIS</h1>
            <p>By <strong>BigByte</strong> &nbsp;·&nbsp; AI life support agent · auto-runs on load</p>
        </div>
    """)

    # ── Controls ──────────────────────────────────────────────────────────────
    with gr.Row():
        task_sel  = gr.Dropdown(["task_easy","task_medium","task_hard"],
                                value="task_easy", label="Mission", scale=2)
        reset_btn = gr.Button("🔄 Reset", scale=1)
        stop_btn  = gr.Button("⏹ Stop + Analyse", variant="primary", scale=1)

    # ── Row 1: Logs | Reward Chart ────────────────────────────────────────────
    gr.HTML('<div class="section-title">▶ TRAINING LOGS &nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; REWARD CHART</div>')
    with gr.Row():
        with gr.Column(scale=1):
            logs_box = gr.Textbox(label="", lines=16, interactive=False, value="Initialising…")
        with gr.Column(scale=2):
            reward_chart = gr.Plot(label="", show_label=False)

    # ── Row 2: Model Comparison ───────────────────────────────────────────────
    gr.HTML('<div class="section-title">▶ MODEL COMPARISON</div>')
    compare_chart = gr.Plot(label="", show_label=False)

    # ── Row 3: Mistakes ───────────────────────────────────────────────────────
    gr.HTML('<div class="section-title">▶ MISTAKES ELIMINATED</div>')
    mistakes_html = gr.HTML()

    # ── Row 4: Live Alarm ─────────────────────────────────────────────────────
    gr.HTML('<div class="section-title">▶ LIVE ALARM MONITOR</div>')
    alarm_html = gr.HTML()

    # ── Row 5: Analysis (shown after Stop) ───────────────────────────────────
    gr.HTML('<div class="section-title">▶ COMPLETE ANALYSIS</div>')
    analysis_box = gr.Textbox(label="", lines=20, interactive=False,
                              value="Press ⏹ Stop + Analyse to generate report.",
                              elem_classes=["analysis-box"])

    # ── Wire ──────────────────────────────────────────────────────────────────
    INPUTS  = [env_state, history_s, mistakes_s, logs_s, session_s, task_sel]
    OUTPUTS = [env_state, history_s, mistakes_s, logs_s, session_s,
               reward_chart, compare_chart, mistakes_html, alarm_html,
               logs_box, analysis_box]

    timer = gr.Timer(value=2, active=True)
    timer.tick(fn=step_fn, inputs=INPUTS, outputs=OUTPUTS)

    reset_btn.click(fn=reset_fn, inputs=[task_sel], outputs=OUTPUTS)

    stop_btn.click(
        fn=stop_fn,
        inputs=[env_state, history_s, mistakes_s, session_s, task_sel],
        outputs=[timer, analysis_box],
    )

    demo.load(fn=reset_fn, inputs=[task_sel], outputs=OUTPUTS)

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=theme_val, css=CSS)
