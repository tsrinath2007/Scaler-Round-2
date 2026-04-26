"""
app.py — Artemis Moon Mission Live Dashboard
Welcome to Artemis Moon Mission By BigByte

Interactive Gradio app: adjust sliders to inject crisis conditions,
the AI agent reacts instantly and the live graph updates every step.
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
CO2_SAFE_MAX = 1000
MAX_HISTORY  = 60


# ── Rule-based AI agent ───────────────────────────────────────────────────────

def ai_decide(obs: Observation) -> Action:
    """Adaptive rule-based agent — reacts to every sensor reading."""
    pg, rw, ao, rf, ca, rp = 0.5, 0.6, 0.0, 0.9, 0.7, "balanced"

    # O2 management
    if obs.o2_percent < O2_SAFE_MIN:
        ao = min(1.0, 0.6 + (O2_SAFE_MIN - obs.o2_percent) * 0.25)
        pg = 0.85
    elif obs.o2_percent > O2_SAFE_MAX + 1.0:
        ao = -0.5
        pg = 0.2

    # CO2 management
    if obs.co2_ppm > 2000:
        ao = -0.95
        ca = 0.2
    elif obs.co2_ppm > CO2_SAFE_MAX:
        ao  = min(ao, -0.4)
        ca  = min(ca, 0.5)

    # Water
    if obs.water_liters < 20:
        rw = 0.95
        pg = min(pg, 0.1)
    elif obs.water_liters < 60:
        rw = 0.80

    # Food
    if obs.food_kg < 3:
        pg = 0.95
        rf = 0.40
    elif obs.food_kg < 10:
        pg = max(pg, 0.80)

    # Events
    if obs.event_name == "solar_flare" or obs.radiation_level > 0.3:
        rp = "shields"
        ca = 0.3
    elif obs.event_name == "meteor_impact":
        ao = max(ao, 0.8)
    elif obs.event_name in ("dust_storm", "lunar_night") or obs.solar_panel_health < 0.5:
        pg = min(pg, 0.20)
        rw = min(rw, 0.30)

    # Power crisis
    if obs.power_budget < 0.15:
        rp = "life_support"
        pg = min(pg, 0.15)
        rw = min(rw, 0.20)

    # Health critical → emergency mode
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


# ── Live plot ─────────────────────────────────────────────────────────────────

def make_plot(history: list) -> plt.Figure:
    if not history:
        fig, ax = plt.subplots(figsize=(11, 4.5), facecolor="#0d1117")
        ax.set_facecolor("#0d1117")
        ax.text(0.5, 0.5,
                "Press  ▶ Step  or enable  ⏱ Auto-Run  to start",
                ha="center", va="center", color="#8b949e", fontsize=13,
                transform=ax.transAxes)
        ax.axis("off")
        return fig

    steps = [h["step"] for h in history]
    fig   = plt.figure(figsize=(11, 4.8), facecolor="#0d1117")
    gs    = gridspec.GridSpec(2, 2, figure=fig, hspace=0.62, wspace=0.38)
    BG    = "#161b22"
    SP    = "#30363d"

    def sub(pos, key, label, color, smin=None, smax=None, yfloor=None):
        ax   = fig.add_subplot(pos)
        vals = [h[key] for h in history]
        ax.set_facecolor(BG)
        ax.plot(steps, vals, color=color, lw=2.2, zorder=3)
        ax.fill_between(steps, vals, alpha=0.18, color=color, zorder=2)
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
        x_lo = max(steps[0], steps[-1] - MAX_HISTORY)
        ax.set_xlim(left=x_lo)
        if vals:
            ax.annotate(
                f"{vals[-1]:.1f}",
                (steps[-1], vals[-1]),
                xytext=(0, 6), textcoords="offset points",
                color=color, fontsize=8, fontweight="bold", ha="center",
            )

    sub(gs[0, 0], "o2",     "🌬  O2 (%)",       "#3fb950", O2_SAFE_MIN, O2_SAFE_MAX)
    sub(gs[0, 1], "co2",    "☁  CO2 (ppm)",     "#f85149", smax=CO2_SAFE_MAX, yfloor=0)
    sub(gs[1, 0], "water",  "💧  Water (L)",     "#58a6ff", yfloor=0)
    sub(gs[1, 1], "health", "❤  Crew Health",   "#ffa657", smin=0.8, yfloor=0)

    fig.suptitle(
        "🚀  Artemis Live Metrics  ·  BigByte",
        color="#e6edf3", fontsize=10.5, y=1.0, fontweight="bold",
    )
    plt.tight_layout()
    return fig


# ── Core step logic ───────────────────────────────────────────────────────────

def do_step(env_state, history, o2_val, co2_val, water_val, food_val, inject):
    """One simulation tick. Injects slider values if inject=True, AI decides, env steps."""
    if env_state is None:
        env_state = LifeSupportEnv(task_id="task_easy")
        env_state.reset()

    env: LifeSupportEnv = env_state

    if env._done:
        env.reset()

    # Override env internals with slider values when inject is on
    if inject:
        env._o2_percent = float(o2_val)
        env._co2_ppm    = float(co2_val)
        env._water      = float(water_val)
        env._food       = float(food_val)

    obs    = env._make_observation()
    action = ai_decide(obs)

    obs_after, reward, done, info = env.step(action)

    entry = {
        "step":   len(history) + 1,
        "o2":     obs_after.o2_percent,
        "co2":    obs_after.co2_ppm,
        "water":  obs_after.water_liters,
        "food":   obs_after.food_kg,
        "health": obs_after.crew_health,
        "reward": round(reward, 4),
    }
    history = (history + [entry])[-MAX_HISTORY:]

    # Build status text
    h_icon  = "🟢" if obs_after.crew_health > 0.8 else ("🟡" if obs_after.crew_health > 0.5 else "🔴")
    o2_icon = "✅" if O2_SAFE_MIN <= obs_after.o2_percent <= O2_SAFE_MAX else "⚠️"
    c2_icon = "✅" if obs_after.co2_ppm < CO2_SAFE_MAX else "⚠️"
    ev_line = f"\n⚡ Event: {obs_after.event_name} ({obs_after.event_severity})" if obs_after.event_name else ""

    status = (
        f"{h_icon} Health  : {obs_after.crew_health:.3f}\n"
        f"{o2_icon} O2      : {obs_after.o2_percent:.2f}%\n"
        f"{c2_icon} CO2     : {obs_after.co2_ppm:.0f} ppm\n"
        f"💧 Water   : {obs_after.water_liters:.1f} L\n"
        f"🍎 Food    : {obs_after.food_kg:.2f} kg\n"
        f"☀️ Solar   : {obs_after.solar_panel_health:.0%}\n"
        f"📅 Step    : {entry['step']}  |  Reward: {reward:+.3f}"
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
    return (
        env, [],
        make_plot([]),
        "Mission reset. Press ▶ Step to begin.",
        "Waiting for first step…",
    )


def on_slider_change(env_state, history, o2, co2, water, food, inject):
    """When a slider moves and inject is on, immediately step."""
    if inject and env_state is not None:
        return do_step(env_state, history, o2, co2, water, food, inject)
    return env_state, history, make_plot(history), gr.update(), gr.update()


# ── Gradio UI ─────────────────────────────────────────────────────────────────

css = """
.header        { text-align:center; padding:8px 0 4px; }
.header h1     { font-size:1.55em; margin:0; color:#3fb950; letter-spacing:.5px; }
.header p      { color:#8b949e; margin:2px 0 0; font-size:.88em; }
.info-box      { font-family:monospace !important; font-size:.85em !important; }
"""

with gr.Blocks(
    title="Artemis Moon Mission — By BigByte",
    theme=gr.themes.Base(primary_hue="green", neutral_hue="slate"),
    css=css,
) as demo:

    env_state = gr.State(value=None)
    history   = gr.State(value=[])

    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML("""
        <div class="header">
            <h1>🚀 Welcome to Artemis Moon Mission</h1>
            <p>By <strong>BigByte</strong> — AI-controlled closed-loop life support simulator</p>
        </div>
    """)

    # ── Top controls ──────────────────────────────────────────────────────────
    with gr.Row():
        task_sel  = gr.Dropdown(
            ["task_easy", "task_medium", "task_hard"],
            value="task_easy", label="Mission Difficulty", scale=2,
        )
        reset_btn = gr.Button("🔄 Reset Mission", variant="secondary", scale=1)
        auto_chk  = gr.Checkbox(label="⏱ Auto-Run (2 s)", value=False, scale=1)

    # ── Main layout ───────────────────────────────────────────────────────────
    with gr.Row():

        # Left column — controls
        with gr.Column(scale=1, min_width=260):

            gr.Markdown("### 🎛 Inject Crisis Conditions")
            gr.Markdown(
                "*Enable the toggle, drag a slider — the AI reacts and the graph updates instantly.*"
            )
            inject_chk = gr.Checkbox(label="⚠️ Inject Values into Simulation", value=False)

            o2_sl    = gr.Slider(14.0, 28.0, value=21.0, step=0.1,  label="O2 (%)")
            co2_sl   = gr.Slider(200,  4500,  value=400,  step=10,   label="CO2 (ppm)")
            water_sl = gr.Slider(0,    500,   value=200,  step=5,    label="Water (L)")
            food_sl  = gr.Slider(0,    100,   value=30,   step=0.5,  label="Food (kg)")

            step_btn = gr.Button("▶ Step", variant="primary", size="lg")

            gr.Markdown("### 🤖 AI Last Action")
            act_box = gr.Textbox(
                label="", lines=7, interactive=False,
                value="Waiting for first step…",
                elem_classes=["info-box"],
            )

            gr.Markdown("### 📡 Live Status")
            status_box = gr.Textbox(
                label="", lines=9, interactive=False,
                value="Press ▶ Step to begin.",
                elem_classes=["info-box"],
            )

        # Right column — graph
        with gr.Column(scale=2):
            plot_out = gr.Plot(label="", show_label=False)

    # ── Wire inputs/outputs ───────────────────────────────────────────────────
    step_inputs  = [env_state, history, o2_sl, co2_sl, water_sl, food_sl, inject_chk]
    step_outputs = [env_state, history, plot_out, status_box, act_box]

    step_btn.click(fn=do_step,        inputs=step_inputs, outputs=step_outputs)
    reset_btn.click(fn=reset_mission, inputs=[task_sel],  outputs=[env_state, history, plot_out, status_box, act_box])

    # Slider → instant step when inject is on
    for sl in [o2_sl, co2_sl, water_sl, food_sl]:
        sl.change(fn=on_slider_change, inputs=step_inputs, outputs=step_outputs)

    # Auto-run timer
    timer = gr.Timer(value=2, active=False)
    auto_chk.change(
        fn=lambda on: gr.Timer(active=on),
        inputs=[auto_chk], outputs=[timer],
    )
    timer.tick(fn=do_step, inputs=step_inputs, outputs=step_outputs)

    # Init on page load
    demo.load(
        fn=reset_mission,
        inputs=[task_sel],
        outputs=[env_state, history, plot_out, status_box, act_box],
    )


if __name__ == "__main__":
    demo.launch()
