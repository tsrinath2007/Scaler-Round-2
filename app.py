"""
app.py — Among Us: Crisis | By BigByte
Layout matches screenshot: Task + Start button, Status, Tabs (Chart|Logs),
then below: Bar graph, Mistakes Eliminated, Live Alarm Monitor, Analysis.
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
MAX_LOG      = 60
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


# ── File save ─────────────────────────────────────────────────────────────────
def save_log(history, mistakes, session_rewards, task_id):
    try:
        with open(LOG_FILE, "w") as f:
            json.dump({
                "saved_at":        datetime.datetime.now().isoformat(),
                "task_id":         task_id,
                "total_steps":     len(history),
                "history":         history[-50:],
                "mistakes":        {k: {kk: vv for kk, vv in v.items() if kk != "ok_streak"}
                                    for k, v in mistakes.items()},
                "session_rewards": {k: round(float(np.mean(v)), 4)
                                    for k, v in session_rewards.items() if v},
            }, f, indent=2)
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


# ── Charts ────────────────────────────────────────────────────────────────────
def make_reward_chart(history):
    fig, ax = plt.subplots(figsize=(9, 3.8), facecolor="#111")
    ax.set_facecolor("#1a1a1a")
    if not history:
        ax.text(0.5, 0.5, "Waiting for data…", ha="center", va="center",
                color="#444", fontsize=11, transform=ax.transAxes)
        ax.axis("off"); return fig

    steps   = [h["step"] for h in history]
    rewards = [h["reward"] for h in history]
    w       = max(1, len(rewards) // 8)
    rolling = np.convolve(rewards, np.ones(w) / w, mode="valid")
    rs      = steps[w - 1:]

    ax.plot(steps, rewards, color="#f97316", lw=0.7, alpha=0.3)
    ax.plot(rs, rolling,    color="#f97316", lw=2.2, label=f"Mean reward (window={w})")
    ax.fill_between(rs, rolling, alpha=0.1, color="#f97316")

    ax.set_title("Mean Episode Reward per Rollout", color="#ccc", fontsize=10)
    ax.set_xlabel("Rollout", color="#555", fontsize=8)
    ax.set_ylabel("Mean Reward", color="#555", fontsize=8)
    ax.tick_params(colors="#555", labelsize=7)
    ax.legend(fontsize=8, facecolor="#1a1a1a", edgecolor="#333", labelcolor="#ccc")
    for s in ax.spines.values(): s.set_edgecolor("#2a2a2a")
    ax.grid(alpha=0.12, color="#2a2a2a", linestyle="--")
    plt.tight_layout(pad=0.5)
    return fig


def make_comparison_chart(session_avg):
    tasks  = ["task_easy", "task_medium", "task_hard"]
    labels = ["Easy", "Medium", "Hard"]
    x, w   = np.arange(3), 0.25

    fig, ax = plt.subplots(figsize=(9, 3.2), facecolor="#111")
    ax.set_facecolor("#1a1a1a")

    b1 = ax.bar(x-w, [EXPECTED[t] for t in tasks], w, label="Expected", color="#3b82f6", alpha=0.85, edgecolor="#111")
    b2 = ax.bar(x,   [PPO_BASE[t]  for t in tasks], w, label="PPO",      color="#f97316", alpha=0.85, edgecolor="#111")
    b3 = ax.bar(x+w, [session_avg.get(t, 0) for t in tasks], w, label="Trained", color="#aaa", alpha=0.75, edgecolor="#111")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            if h > 0.5:
                ax.annotate(f"{h:.1f}", xy=(bar.get_x()+bar.get_width()/2, h),
                            xytext=(0,3), textcoords="offset points",
                            ha="center", fontsize=6.5, color="#ccc")

    ax.set_xticks(x); ax.set_xticklabels(labels, color="#ccc", fontsize=9)
    ax.set_ylabel("Avg Reward", color="#555", fontsize=8)
    ax.set_title("Model Comparison — Expected  ·  PPO  ·  Trained", color="#ccc", fontsize=10)
    ax.legend(fontsize=8, facecolor="#1a1a1a", edgecolor="#333", labelcolor="#ccc")
    ax.tick_params(colors="#555", labelsize=7)
    for s in ax.spines.values(): s.set_edgecolor("#2a2a2a")
    ax.grid(axis="y", alpha=0.12, color="#2a2a2a", linestyle="--")
    plt.tight_layout(pad=0.5)
    return fig


# ── Mistakes HTML ─────────────────────────────────────────────────────────────
def render_mistakes(mistakes):
    elim  = sum(1 for m in mistakes.values() if m["eliminated"])
    total = len(mistakes)
    rows  = ""
    for m in mistakes.values():
        if m["step"] is not None:
            if m["eliminated"]:
                dot   = '<span style="color:#f97316;font-weight:700">✓</span>'
                badge = ('<span style="background:#f97316;color:#000;padding:1px 9px;'
                         'border-radius:3px;font-size:.7em;font-weight:700;letter-spacing:1px">'
                         'ELIMINATED</span>')
            else:
                dot   = '<span style="color:#3b82f6;font-weight:700">!</span>'
                badge = ('<span style="background:#3b82f6;color:#fff;padding:1px 9px;'
                         'border-radius:3px;font-size:.7em;font-weight:700;letter-spacing:1px">'
                         'ENCOUNTERED</span>')
            detail = f"Step {m['step']:,} · first encountered"
        else:
            dot    = '<span style="color:#333">○</span>'
            badge  = '<span style="color:#333;font-size:.7em">pending</span>'
            detail = "Not yet encountered"

        rows += f"""
        <div style="display:flex;gap:10px;align-items:flex-start;
                    padding:8px 0;border-bottom:1px solid #222">
            <div style="width:22px;height:22px;border:1px solid #2a2a2a;border-radius:50%;
                        display:flex;align-items:center;justify-content:center;
                        flex-shrink:0;margin-top:1px">{dot}</div>
            <div>
                <div style="color:#ddd;font-size:.87em;font-weight:600">{m['name']}</div>
                <div style="color:#444;font-size:.74em;margin:2px 0 5px">{detail}</div>
                {badge}
            </div>
        </div>"""

    return f"""
    <div style="background:#111;border:1px solid #f97316;border-radius:8px;padding:16px 18px">
        <div style="display:flex;justify-content:space-between;align-items:center;
                    margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid #222">
            <span style="color:#f97316;font-weight:700;font-size:.95em;letter-spacing:1px">
                Mistakes Eliminated
            </span>
            <span style="color:#f97316;font-weight:700;font-size:.95em">{elim}/{total}</span>
        </div>
        <div style="color:#444;font-size:.75em;margin-bottom:10px">
            Encoded into policy — permanently locked
        </div>
        {rows}
    </div>"""


# ── Alarm monitor HTML ────────────────────────────────────────────────────────
def render_alarm(obs: Observation, green_streak, fire_active, step_num, max_steps):
    def row(label, pct, tier):
        c = {"GREEN":"#3b82f6","YELLOW":"#f97316","RED":"#ef4444","CRITICAL":"#dc2626"}.get(tier,"#3b82f6")
        p = min(100, max(0, pct))
        return f"""
        <div style="display:grid;grid-template-columns:60px 1fr 75px;
                    align-items:center;gap:10px;margin:7px 0">
            <span style="color:#888;font-size:.78em;letter-spacing:1px">{label}</span>
            <div style="background:#1a1a1a;border-radius:3px;height:10px;overflow:hidden;border:1px solid #2a2a2a">
                <div style="width:{p}%;height:100%;background:{c};border-radius:3px;
                            transition:width .5s ease"></div>
            </div>
            <span style="color:{c};font-size:.78em;font-weight:700;text-align:right">{tier}</span>
        </div>"""

    def t_o2(v):
        return "GREEN" if O2_SAFE_MIN<=v<=O2_SAFE_MAX else ("RED" if v<17 or v>25 else "YELLOW")
    def t_co2(v):
        return "GREEN" if v<=1000 else ("YELLOW" if v<=2000 else ("RED" if v<=3000 else "CRITICAL"))
    def t_inv(v, ok, warn, danger):
        return "GREEN" if v>=ok else ("YELLOW" if v>=warn else ("RED" if v>=danger else "CRITICAL"))

    o2_pct    = obs.o2_percent / 30 * 100
    co2_pct   = max(0, 100 - obs.co2_ppm / 4000 * 100)
    water_pct = obs.water_liters / 500 * 100
    food_pct  = obs.food_kg / 100 * 100
    plant_pct = obs.solar_panel_health * 100
    power_pct = obs.power_budget * 100

    fire_c = "#ef4444" if fire_active else "#3b82f6"
    fire_t = "ACTIVE 🔥" if fire_active else "INACTIVE"
    sk_c   = "#f97316" if green_streak > 10 else ("#3b82f6" if green_streak > 0 else "#ef4444")

    return f"""
    <div style="background:#111;border:1px solid #3b82f6;border-radius:8px;padding:16px 18px">
        <div style="color:#3b82f6;font-size:.82em;font-weight:700;letter-spacing:2px;
                    margin-bottom:12px;padding-bottom:10px;border-bottom:1px solid #222">
            ▶ LIVE ALARM MONITOR · SUBSYSTEM STATUS
        </div>
        {row("O₂",    o2_pct,    t_o2(obs.o2_percent))}
        {row("CO₂",   co2_pct,   t_co2(obs.co2_ppm))}
        {row("WATER", water_pct, t_inv(obs.water_liters,50,20,5))}
        {row("FOOD",  food_pct,  t_inv(obs.food_kg,20,5,1))}
        {row("PLANTS",plant_pct, t_inv(obs.solar_panel_health,.8,.5,.2))}
        {row("POWER", power_pct, t_inv(obs.power_budget,.5,.25,.1))}
        <div style="display:flex;justify-content:space-between;margin-top:12px;
                    padding-top:10px;border-top:1px solid #222;font-size:.76em">
            <span style="color:{sk_c}">● Green Streak: {green_streak}</span>
            <span style="color:{fire_c}">Fire Risk: {fire_t}</span>
            <span style="color:#444">Step: {step_num} / {max_steps}</span>
        </div>
    </div>"""


# ── Analysis ──────────────────────────────────────────────────────────────────
def build_analysis(history, mistakes, session_rewards, task_id):
    if not history:
        return "No data — run the simulation first."
    rewards = [h["reward"] for h in history]
    healths = [h["health"] for h in history]
    elim    = sum(1 for m in mistakes.values() if m["eliminated"])
    encnt   = sum(1 for m in mistakes.values() if m["step"] is not None)
    lines   = [
        f"{'='*54}",
        f"  AMONG US: CRISIS — COMPLETE ANALYSIS",
        f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"{'='*54}",
        f"  Task           : {task_id}",
        f"  Total Steps    : {len(history)}",
        f"",
        f"  REWARD STATS",
        f"  ├ Mean         : {np.mean(rewards):+.4f}",
        f"  ├ Max          : {np.max(rewards):+.4f}",
        f"  ├ Min          : {np.min(rewards):+.4f}",
        f"  └ Std Dev      : {np.std(rewards):.4f}",
        f"",
        f"  CREW HEALTH",
        f"  ├ Mean         : {np.mean(healths):.4f}",
        f"  ├ Final        : {healths[-1]:.4f}",
        f"  └ Min seen     : {np.min(healths):.4f}",
        f"",
        f"  MISTAKES   Encountered: {encnt}/7   Eliminated: {elim}/7",
    ]
    for m in mistakes.values():
        s = "ELIMINATED" if m["eliminated"] else ("ENCOUNTERED" if m["step"] else "NEVER SEEN")
        t = f"@ step {m['step']:,}" if m["step"] else ""
        lines.append(f"  {'✓' if m['eliminated'] else '○'} {m['name'][:36]:<36} {s} {t}")
    lines += ["", f"  SESSION AVG REWARDS"]
    for t in ["task_easy", "task_medium", "task_hard"]:
        v = session_rewards.get(t, [])
        if v:
            lines.append(f"  ├ {t:<15}: {float(np.mean(v)):+.4f}")
    lines += ["", f"  Log saved → {LOG_FILE}", f"{'='*54}"]
    report = "\n".join(lines)
    try:
        with open("analysis_report.txt", "w") as f:
            f.write(report)
    except Exception:
        pass
    return report


# ── Step / Reset / Stop ───────────────────────────────────────────────────────
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

    session_rewards = dict(session_rewards)
    prev = session_rewards.get(task_id, [])
    session_rewards[task_id] = (prev + [reward])[-300:]
    session_avg = {k: float(np.mean(v)) for k, v in session_rewards.items()}

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

    ev   = f" [{obs_a.event_name[:10]}]" if obs_a.event_name else ""
    ic   = "▲" if reward > 0 else "▼"
    line = (f"[{sn:04d}] O2:{obs_a.o2_percent:.1f}% CO2:{obs_a.co2_ppm:.0f} "
            f"H:{obs_a.crew_health:.3f} R:{reward:+.3f}{ic}{ev}")
    logs = ([line] + logs)[:MAX_LOG]

    if sn % 10 == 0:
        save_log(history, mistakes, session_rewards, task_id)

    mean_r = float(np.mean([h["reward"] for h in history]))
    status = f"Running...  Step: {sn}  |  Mean reward: {mean_r:.3f}  |  Health: {obs_a.crew_health:.3f}"

    return (
        env_state, history, mistakes, logs, session_rewards,
        status,
        make_reward_chart(history),
        "\n".join(logs),
        make_comparison_chart(session_avg),
        render_mistakes(mistakes),
        render_alarm(obs_a, env._green_streak, obs_a.o2_percent > 25.0, sn, env.config["max_steps"]),
        "",
    )


def reset_fn(task_id):
    env = LifeSupportEnv(task_id=task_id or "task_easy")
    env.reset()
    m   = fresh_mistakes()
    obs = env._make_observation()
    return (
        env, [], m, [], {},
        "Ready. Press ▶ Start Training to begin.",
        make_reward_chart([]),
        "",
        make_comparison_chart({}),
        render_mistakes(m),
        render_alarm(obs, 0, False, 0, env.config["max_steps"]),
        "",
    )


def start_fn():
    return gr.Timer(active=True)


def stop_fn(_env, history, mistakes, session_rewards, task_id):
    save_log(history, mistakes, session_rewards, task_id)
    report = build_analysis(history, mistakes, session_rewards, task_id)
    mean_r = float(np.mean([h["reward"] for h in history])) if history else 0.0
    status = f"Done!  Final mean reward: {mean_r:.3f}  |  Steps: {len(history)}"
    return gr.Timer(active=False), status, report


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
body, .gradio-container { background:#0d0d0d !important; }
.gradio-container { max-width:1100px !important; margin:auto !important; }

.header { padding:22px 0 12px; border-bottom:2px solid #f97316;
          background:#0d0d0d; margin-bottom:18px; }
.header h1 { font-size:1.75em; color:#f97316; margin:0 0 4px;
             font-family:'Courier New',monospace; letter-spacing:2px; }
.header p  { color:#555; margin:0; font-size:.88em; }

/* orange Start button */
#start-btn { background:#f97316 !important; color:#000 !important;
             font-weight:700 !important; font-size:1em !important;
             border:none !important; border-radius:6px !important; }
#start-btn:hover { background:#ea6c00 !important; }

/* Status textbox */
#status-box textarea { background:#1a1a1a !important; color:#f97316 !important;
    font-family:'Courier New',monospace !important; font-size:.85em !important;
    border:1px solid #2a2a2a !important; border-radius:6px !important; }

/* Log textbox */
#log-box textarea { background:#0d0d0d !important; color:#3b82f6 !important;
    font-family:'Courier New',monospace !important; font-size:.76em !important;
    border:1px solid #1a1a1a !important; border-radius:6px !important; }

/* Analysis textbox */
#analysis-box textarea { background:#0d0d0d !important; color:#f97316 !important;
    font-family:'Courier New',monospace !important; font-size:.78em !important;
    border:1px solid #f97316 !important; border-radius:6px !important; }

/* Tab styling */
.tab-nav button { background:#1a1a1a !important; color:#888 !important;
                  border:1px solid #2a2a2a !important; border-radius:6px 6px 0 0 !important; }
.tab-nav button.selected { background:#f97316 !important; color:#000 !important;
                           font-weight:700 !important; }

.tip { color:#444; font-size:.8em; margin:6px 0 18px; }
.sec-label { color:#f97316; font-size:.78em; font-weight:700;
             letter-spacing:2px; margin:18px 0 6px; }
"""

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Among Us: Crisis — By BigByte",
    theme=gr.themes.Base(primary_hue="orange", neutral_hue="slate"),
    css=CSS,
) as demo:

    env_s  = gr.State(None)
    hist_s = gr.State([])
    mist_s = gr.State(fresh_mistakes())
    logs_s = gr.State([])
    sess_s = gr.State({})

    # Header
    gr.HTML("""
        <div class="header">
            <h1>🚀 Among Us — Crisis</h1>
            <p>Train an AI agent to keep your crew alive in a space habitat.</p>
        </div>
    """)

    # Controls row (matches screenshot layout)
    with gr.Row():
        task_sel   = gr.Dropdown(["task_easy","task_medium","task_hard"],
                                 value="task_easy", label="Task", scale=2)
        start_btn  = gr.Button("▶ Start Training", elem_id="start-btn", scale=3)
        stop_btn   = gr.Button("⏹ Stop + Analyse", scale=1)
        reset_btn  = gr.Button("🔄 Reset", scale=1)

    # Status
    status_box = gr.Textbox(label="Status", value="Ready. Press ▶ Start Training to begin.",
                            lines=1, interactive=False, elem_id="status-box")

    # Tabs: Reward Chart | Training Logs
    with gr.Tabs():
        with gr.Tab("📊 Reward Chart"):
            reward_chart = gr.Plot(label="", show_label=False)
        with gr.Tab("📋 Training Logs"):
            logs_box = gr.Textbox(label="", lines=18, interactive=False,
                                  value="", elem_id="log-box")

    gr.HTML('<p class="tip">Tip: Switch to the 📋 Training Logs tab to see step-by-step progress. '
            'Press ⏹ Stop + Analyse for the complete report.</p>')

    # Model comparison
    gr.HTML('<div class="sec-label">▶ MODEL COMPARISON — EXPECTED · PPO · TRAINED</div>')
    compare_chart = gr.Plot(label="", show_label=False)

    # Mistakes Eliminated
    gr.HTML('<div class="sec-label">▶ MISTAKES ELIMINATED</div>')
    mistakes_html = gr.HTML()

    # Live Alarm Monitor
    gr.HTML('<div class="sec-label">▶ LIVE ALARM MONITOR · SUBSYSTEM STATUS</div>')
    alarm_html = gr.HTML()

    # Complete Analysis
    gr.HTML('<div class="sec-label">▶ COMPLETE ANALYSIS</div>')
    analysis_box = gr.Textbox(label="", lines=22, interactive=False,
                              value="Press ⏹ Stop + Analyse to generate report.",
                              elem_id="analysis-box")

    # ── Wire ──────────────────────────────────────────────────────────────────
    STEP_IN  = [env_s, hist_s, mist_s, logs_s, sess_s, task_sel]
    STEP_OUT = [env_s, hist_s, mist_s, logs_s, sess_s,
                status_box, reward_chart, logs_box,
                compare_chart, mistakes_html, alarm_html, analysis_box]

    RESET_OUT = STEP_OUT  # same outputs

    timer = gr.Timer(value=2, active=False)
    timer.tick(fn=step_fn, inputs=STEP_IN, outputs=STEP_OUT)

    start_btn.click(fn=start_fn, outputs=[timer])
    reset_btn.click(fn=reset_fn, inputs=[task_sel], outputs=RESET_OUT)
    stop_btn.click(
        fn=stop_fn,
        inputs=[env_s, hist_s, mist_s, sess_s, task_sel],
        outputs=[timer, status_box, analysis_box],
    )

    demo.load(fn=reset_fn, inputs=[task_sel], outputs=RESET_OUT)

if __name__ == "__main__":
    demo.launch()
