"""
app.py — Artemis Moon Mission By BigByte
Auto-running life support simulation. The AI agent starts immediately on
page load and continuously manages the habitat — no user input required.
"""

import sys
sys.path.insert(0, ".")

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from env.environment import LifeSupportEnv
from env.models import Action, Observation

O2_SAFE_MIN,  O2_SAFE_MAX = 19.5, 23.5
CO2_SAFE_MAX  = 1000
MAX_HISTORY   = 80


# ── AI agent ──────────────────────────────────────────────────────────────────

def ai_decide(obs: Observation) -> Action:
    pg, rw, ao, rf, ca, rp = 0.5, 0.6, 0.0, 0.9, 0.7, "balanced"

    if obs.o2_percent < O2_SAFE_MIN:
        ao = min(1.0, 0.6 + (O2_SAFE_MIN - obs.o2_percent) * 0.25)
        pg = 0.85
    elif obs.o2_percent > O2_SAFE_MAX + 1.0:
        ao = -0.5
        pg = 0.2

    if obs.co2_ppm > 2000:
        ao = -0.95
        ca = 0.2
    elif obs.co2_ppm > CO2_SAFE_MAX:
        ao  = min(ao, -0.4)
        ca  = min(ca, 0.5)

    if obs.water_liters < 20:
        rw = 0.95
        pg = min(pg, 0.1)
    elif obs.water_liters < 60:
        rw = 0.80

    if obs.food_kg < 3:
        pg = 0.95
        rf = 0.40
    elif obs.food_kg < 10:
        pg = max(pg, 0.80)

    if obs.event_name == "solar_flare" or obs.radiation_level > 0.3:
        rp = "shields"
        ca = 0.3
    elif obs.event_name == "meteor_impact":
        ao = max(ao, 0.8)
    elif obs.event_name in ("dust_storm", "lunar_night") or obs.solar_panel_health < 0.5:
        pg = min(pg, 0.20)
        rw = min(rw, 0.30)

    if obs.power_budget < 0.15:
        rp = "life_support"
        pg = min(pg, 0.15)
        rw = min(rw, 0.20)

    if obs.crew_health < 0.4:
        rp = "emergency"
        ca = 0.10
        rf = 0.60

    return Action(
        increase_plant_growth = round(max(0.0, min(1.0, pg)), 2),
        recycle_water         = round(max(0.0, min(1.0, rw)), 2),
        adjust_oxygen         = round(max(-1.0, min(1.0, ao)), 2),
        ration_food           = round(max(0.0, min(1.0, rf)), 2),
        crew_activity         = round(max(0.0, min(1.0, ca)), 2),
        route_power           = rp,
    )


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_plot(history: list) -> plt.Figure:
    fig = plt.figure(figsize=(12, 5), facecolor="#0d1117")

    if not history:
        ax = fig.add_subplot(111)
        ax.set_facecolor("#0d1117")
        ax.text(0.5, 0.5, "Initialising mission…",
                ha="center", va="center", color="#8b949e", fontsize=14,
                transform=ax.transAxes)
        ax.axis("off")
        return fig

    steps = [h["step"] for h in history]
    gs    = gridspec.GridSpec(2, 3, figure=fig, hspace=0.65, wspace=0.38)
    BG, SP = "#161b22", "#30363d"

    def sub(pos, key, label, color, smin=None, smax=None, yfloor=None, ytop=None):
        ax   = fig.add_subplot(pos)
        vals = [h[key] for h in history]
        ax.set_facecolor(BG)
        ax.plot(steps, vals, color=color, lw=2.2, zorder=3)
        ax.fill_between(steps, vals, alpha=0.2, color=color, zorder=2)
        if smin is not None:
            ax.axhline(smin, color="#2ea043", ls="--", lw=0.9, alpha=0.65)
        if smax is not None:
            ax.axhline(smax, color="#f85149", ls="--", lw=0.9, alpha=0.65)
        ax.set_title(label, color="#e6edf3", fontsize=8.5, pad=4, fontweight="bold")
        ax.tick_params(colors="#8b949e", labelsize=6.5)
        for s in ax.spines.values():
            s.set_edgecolor(SP)
        if yfloor is not None:
            ax.set_ylim(bottom=yfloor)
        if ytop is not None:
            ax.set_ylim(top=ytop)
        x_lo = max(steps[0], steps[-1] - MAX_HISTORY + 1)
        ax.set_xlim(left=x_lo)
        if vals:
            ax.annotate(f"{vals[-1]:.1f}",
                        (steps[-1], vals[-1]),
                        xytext=(0, 6), textcoords="offset points",
                        color=color, fontsize=8, fontweight="bold", ha="center")

    sub(gs[0, 0], "o2",     "🌬  O2 (%)",       "#3fb950", O2_SAFE_MIN, O2_SAFE_MAX)
    sub(gs[0, 1], "co2",    "☁  CO2 (ppm)",     "#f85149", smax=CO2_SAFE_MAX, yfloor=0)
    sub(gs[0, 2], "health", "❤  Crew Health",   "#ffa657", smin=0.8, yfloor=0, ytop=1.05)
    sub(gs[1, 0], "water",  "💧  Water (L)",     "#58a6ff", yfloor=0)
    sub(gs[1, 1], "food",   "🍎  Food (kg)",     "#da77f2", yfloor=0)
    sub(gs[1, 2], "reward", "⭐  Step Reward",   "#4dabf7")

    fig.suptitle("🚀  Artemis Moon Mission — Live AI Performance  ·  BigByte",
                 color="#e6edf3", fontsize=11, y=1.01, fontweight="bold")
    plt.tight_layout()
    return fig


# ── Step logic ────────────────────────────────────────────────────────────────

def step(env_state, history):
    if env_state is None:
        env_state = LifeSupportEnv(task_id="task_easy")
        env_state.reset()

    env: LifeSupportEnv = env_state
    if env._done:
        env.reset()

    obs    = env._make_observation()
    action = ai_decide(obs)
    obs_after, reward, done, info = env.step(action)

    history = (history + [{
        "step":   len(history) + 1,
        "o2":     obs_after.o2_percent,
        "co2":    obs_after.co2_ppm,
        "water":  obs_after.water_liters,
        "food":   obs_after.food_kg,
        "health": obs_after.crew_health,
        "reward": round(reward, 4),
    }])[-MAX_HISTORY:]

    h_icon  = "🟢" if obs_after.crew_health > 0.8 else ("🟡" if obs_after.crew_health > 0.5 else "🔴")
    o2_ok   = "✅" if O2_SAFE_MIN <= obs_after.o2_percent <= O2_SAFE_MAX else "⚠️"
    co2_ok  = "✅" if obs_after.co2_ppm < CO2_SAFE_MAX else "⚠️"
    ev_line = f"\n⚡ Event  : {obs_after.event_name} ({obs_after.event_severity})" if obs_after.event_name else ""

    status = (
        f"{h_icon} Health  : {obs_after.crew_health:.3f}\n"
        f"{o2_ok} O2      : {obs_after.o2_percent:.2f}%\n"
        f"{co2_ok} CO2     : {obs_after.co2_ppm:.0f} ppm\n"
        f"💧 Water   : {obs_after.water_liters:.1f} L\n"
        f"🍎 Food    : {obs_after.food_kg:.2f} kg\n"
        f"☀️ Solar   : {obs_after.solar_panel_health:.0%}\n"
        f"📅 Step    : {len(history)}  |  Reward: {reward:+.3f}"
        f"{ev_line}"
    )

    act_txt = (
        f"🌱 Plant Growth : {action.increase_plant_growth:.2f}\n"
        f"♻️  Recycle Water: {action.recycle_water:.2f}\n"
        f"🌬 Adj. Oxygen  : {action.adjust_oxygen:+.2f}\n"
        f"🍎 Ration Food  : {action.ration_food:.2f}\n"
        f"👷 Crew Activity: {action.crew_activity:.2f}\n"
        f"⚡ Route Power  : {action.route_power}"
    )

    return env_state, history, make_plot(history), status, act_txt


def reset_mission(task_id):
    env = LifeSupportEnv(task_id=task_id)
    env.reset()
    return env, [], make_plot([]), "Starting…", "Starting…"


# ── UI ────────────────────────────────────────────────────────────────────────

css = """
.header     { text-align:center; padding:10px 0 6px; }
.header h1  { font-size:1.6em; margin:0; color:#3fb950; }
.header p   { color:#8b949e; margin:3px 0 0; font-size:.88em; }
.mono       { font-family:monospace !important; font-size:.84em !important; }
"""

with gr.Blocks(
    title="Artemis Moon Mission — By BigByte",
    theme=gr.themes.Base(primary_hue="green", neutral_hue="slate"),
    css=css,
) as demo:

    env_state = gr.State(value=None)
    history   = gr.State(value=[])

    gr.HTML("""
        <div class="header">
            <h1>🚀 Welcome to Artemis Moon Mission</h1>
            <p>By <strong>BigByte</strong> — AI self-improving life support agent · auto-runs on load</p>
        </div>
    """)

    with gr.Row():
        task_sel  = gr.Dropdown(
            ["task_easy", "task_medium", "task_hard"],
            value="task_easy", label="Mission Difficulty", scale=3,
        )
        reset_btn = gr.Button("🔄 Reset", variant="secondary", scale=1)

    with gr.Row():
        with gr.Column(scale=1, min_width=230):
            gr.Markdown("### 📡 Live Status")
            status_box = gr.Textbox(label="", lines=8,  interactive=False,
                                    value="Starting…", elem_classes=["mono"])
            gr.Markdown("### 🤖 AI Decision")
            act_box    = gr.Textbox(label="", lines=7,  interactive=False,
                                    value="Starting…", elem_classes=["mono"])

        with gr.Column(scale=3):
            plot_out = gr.Plot(label="", show_label=False)

    step_outputs = [env_state, history, plot_out, status_box, act_box]

    # Auto-runs every 2 seconds from the moment the page loads
    timer = gr.Timer(value=2, active=True)
    timer.tick(fn=step, inputs=[env_state, history], outputs=step_outputs)

    reset_btn.click(
        fn=reset_mission,
        inputs=[task_sel],
        outputs=step_outputs,
    )

    # Kick off immediately on load
    demo.load(
        fn=reset_mission,
        inputs=[task_sel],
        outputs=step_outputs,
    )

if __name__ == "__main__":
    demo.launch()
