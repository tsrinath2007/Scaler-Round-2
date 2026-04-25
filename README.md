# Life Support ENV — Round 2 (RL + LLM Edition)

> OpenEnv Hackathon — Round 2 Submission  
> PPO agent learns to manage a cascading space habitat life support system,  
> then its expertise is distilled into a fine-tuned LLM for inference.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Training Pipeline                      │
│                                                          │
│   LifeSupportEnv ──→ PPO Agent ──→ Expert Trajectories   │
│                        (RL)            (obs → action)    │
│                                            │             │
│                                            ▼             │
│                                   Qwen2.5-1.5B-Instruct │
│                                   + LoRA Fine-tuning     │
│                                            │             │
│                                            ▼             │
│                                    lifesupport-llm       │
│                                   (HuggingFace Hub)      │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   Inference Pipeline                     │
│                                                          │
│   Sensor Readings ──→ Fine-tuned LLM ──→ Control Actions │
│   (O2, CO2, etc.)     (via HF API)       (JSON output)  │
└─────────────────────────────────────────────────────────┘
```

---

## Quickstart

### Option 1: Full Pipeline (recommended)
```bash
pip install -r requirements.txt

# Run everything: PPO → Expert Data → LLM Fine-tuning
python train_full_pipeline.py --task task_medium --ppo-timesteps 500000
```

### Option 2: Step by Step
```bash
pip install -r requirements.txt

# 1. Train PPO agent
python train.py --task task_medium --timesteps 500000

# 2. Generate expert data from trained PPO
python generate_expert_data.py --task task_medium --episodes 150

# 3. Fine-tune LLM on expert data
python finetune_llm.py --base-model Qwen/Qwen2.5-1.5B-Instruct --epochs 3

# 4. Evaluate PPO agent
python evaluate.py --task task_medium --episodes 20
```

### HuggingFace Training (with $30 credits)
```bash
# On HF Space with T4 GPU ($0.60/hr):
python train_full_pipeline.py \
    --task task_hard \
    --ppo-timesteps 2000000 \
    --expert-episodes 200 \
    --llm-epochs 3 \
    --hub-repo YOUR_USERNAME/lifesupport-llm
```

---

## Why task_medium, not task_easy?

`task_easy` lasts only 24 steps and starts with near-safe O2/CO2 values.
Random agents almost always survive it (100% win rate for random).
`task_medium` runs for 168 steps with 5 crew — random agents regularly fail
here, giving the trained PPO room to show real improvement.

---

## What's New vs Round 1: Cascading Interconnections

| Trigger | Cascade Effect |
|---|---|
| O2 > 25% (fire risk) | Plants scorch → O2 production drops → food chain weakens |
| CO2 > 1000 ppm | Water recycling becomes toxic → less clean water recovered |
| Water < 20 L | Crew dehydration → food metabolised less efficiently |
| Plants die | Manual O2 needed → more power consumed → less for recycling |

## Gamification
- **Alarm tiers** per subsystem: `GREEN / YELLOW / RED / CRITICAL`
- **Green streak**: reward compounds for sustained safety
- **Trend bonus**: rewarded for actively correcting a bad O2/CO2 trend
- **Terminal**: `+5` mission complete, `−5` crew loss

---

## LLM Fine-tuning (Expert Distillation)

The trained PPO agent's knowledge is distilled into a small LLM:

1. **PPO generates expert trajectories** — observation/action pairs from successful episodes
2. **SFT with LoRA** — Qwen2.5-1.5B-Instruct is fine-tuned on these pairs
3. **Inference via HF API** — the fine-tuned model outputs control actions as JSON

This approach combines the best of both worlds:
- **RL** learns the optimal policy through millions of environment interactions
- **LLM** provides natural language understanding and generalizable decision-making

---

## Files

```
env/environment.py        — LifeSupportEnv with cascades + gamification
env/models.py             — Pydantic models (Observation, Action, Reward)
gym_wrapper.py            — Gymnasium wrapper for SB3 / any RL library
train.py                  — PPO training pipeline + training_curve.png
evaluate.py               — Random vs Trained comparison table
generate_expert_data.py   — Generate LLM training data from PPO agent
finetune_llm.py           — LoRA fine-tuning of LLM on expert data
train_full_pipeline.py    — End-to-end: PPO → Expert Data → LLM
inference.py              — LLM-based inference (supports fine-tuned model)
server/app.py             — FastAPI server for OpenEnv API
tasks/graders.py          — Episode graders for easy/medium/hard
```

---

## API

```python
from env.environment import LifeSupportEnv
from env.models import Action

env = LifeSupportEnv(task_id="task_medium")
obs = env.reset()

obs, reward, done, info = env.step(Action(
    increase_plant_growth=0.6,
    recycle_water=0.8,
    adjust_oxygen=0.0,
    ration_food=1.0,
    crew_activity=0.7,
))

print(info["alarm_tiers"])    # {'o2': 'GREEN', 'co2': 'YELLOW', 'water': 'GREEN', 'food': 'GREEN'}
print(info["fire_active"])    # True if O2 > 25%
print(info["green_streak"])   # consecutive steps all-GREEN
print(info["co2_toxin_factor"])  # how much CO2 is degrading water recycling
```

---

## Cost Estimate (HuggingFace, T4 GPU @ $0.60/hr)

| Step | Time | Cost |
|------|------|------|
| PPO Training (task_hard, 2M steps) | ~45–60 min | ~$0.60 |
| Expert Data Generation (200 episodes) | ~5 min | ~$0.05 |
| LLM Fine-tuning (3 epochs, LoRA) | ~30–45 min | ~$0.40 |
| **Total** | **~1.5–2 hrs** | **~$1.05** |
