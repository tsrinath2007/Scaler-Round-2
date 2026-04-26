# 🌙 Among Us - Crisis: The Artemis Life Support Challenge
> **A 3-Part Engineering Deep Dive into Reinforcement Learning, LLM Distillation, and Autonomous Space Habitats.**

---

## 🛰️ The Mission Briefing
We didn't just build a game; we built a simulation of the most critical problem in space exploration: **Survival.** 

Inspired by NASA's **Artemis program**, we asked: *Can AI manage a lunar habitat's life support autonomously?* The answer led us down a rabbit hole of cascading failures, imitation learning, and high-stakes deployment.

---

## 🤖 Part 1: The RL Genesis
**By Srinath — RL Engineer**

### The Spark: 250,000 Miles from Help
Life support isn't glamorous. It's CO₂ scrubbers, oxygen electrolysis, and power budgets. If a solar flare hits 250,000 miles from Earth, you can't wait for mission control. The habitat has to think for itself.

### The Challenge: A Domino Effect
I built the **LifeSupportEnv** to mirror real lunar habitat dynamics. One bad decision at Step 10 — like neglecting water recycling — triggers a dehydration cascade that kills the crew at Step 50. It’s a "long-horizon" problem where early actions have distant, fatal consequences.

### Training the Expert (PPO)
I used **Proximal Policy Optimization (PPO)** to train our "expert" agent.
- **Goal**: Keep 3–8 crew members alive.
- **Horizon**: Up to 720 steps of relentless crisis management.
- **Results**: Explained variance hit **0.97**—meaning the agent could almost perfectly predict survival outcomes from any state.

> [!TIP]
> **What the Agent Learned**: It discovered a hierarchy of survival. Life support > Water > Greenhouse. Nobody told it that; it learned by "dying" hundreds of times in simulation.

---

## 🧠 Part 2: The LLM Brain Translocation
**By Pushkar — LLM Engineer**

### Teaching a Language Model to Reason
Part 1 gave us a robot that could act. But ask a PPO agent *why* it cut power to the greenhouse? It can't answer. For the Artemis mission, we need **Explainable AI**.

### Behavioral Cloning Pipeline
I built a bridge between raw numbers and human language.
1. **Expert Observation**: Recorded 20,000 steps of the PPO agent's best runs.
2. **Translation**: Converted sensor data (O₂, Power, etc.) into a structured reasoning prompt.
3. **Fine-Tuning**: Used **Qwen2.5-1.5B** with **LoRA (Low-Rank Adaptation)**.

| Metric | Value |
| :--- | :--- |
| **Base Model** | Qwen2.5-1.5B-Instruct |
| **Training Samples** | 19,385 Expert Steps |
| **Technique** | 4-bit LoRA (r=16, alpha=32) |
| **Reasoning** | Structured JSON Policy |

### The Result
The LLM didn't just copy the PPO agent; it generalized. It learned to prioritize life-critical subsystems while remaining within the strict JSON action schema required by the hardware.

---

## 🚀 Part 3: Shifting to High Gear (Deployment)
**By Nikhil — Deployment Engineer**

### From Artifact to Habitat
My teammates handed me weights; I had to build the habitat. To make this "Artemis-ready," I deployed the entire stack on **Hugging Face Spaces**.

### The Tech Stack
- **Backend**: FastAPI for ultra-low latency inference.
- **Frontend**: Gradio (optimized with lazy-loading for 3-second startup).
- **Environment**: Docker-containerized RL simulation.

### The "Live" Realization
The most satisfying moment? Seeing the **Live Alarm Monitor** move. Watching the O₂ bars fluctuate and turn green as the LLM successfully mitigates a meteor strike or a power surge.

> [!IMPORTANT]
> **Artemis Data Roadmap**: Our pipeline is data-ready. The next step is replacing synthetic parameters with real NASA telemetry from the ISS and future lunar surface sensors.

---

## 🏆 Final Summary: Team BigByte
| Role | Contributor |
| :--- | :--- |
| **Team Lead / RL** | **Koppeti Pushkar** |
| **RL & Env** | **Thota Sai Eswar Srinath** |
| **LLM & UI** | **Nikhil sai kadiri** |

---
### 🔗 Try the Simulation
- 🎮 **[Live Demo on Hugging Face](https://huggingface.co/spaces/tsrinath/Scaler-Round-2)**
- 💻 **[Full Source Code on GitHub](https://github.com/tsrinath2007/Scaler-Round-2)**

*Built for the OpenEnv Hackathon Grand Finale · Bangalore 2026*
