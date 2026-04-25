"""
FastAPI server for the Closed-Loop Life Support OpenEnv environment.
Exposes step() / reset() / state() over HTTP as required by the OpenEnv spec.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uuid
import time

from env.environment import LifeSupportEnv
from env.models import Action, Observation
from tasks.graders import grade_episode, GradeResult

app = FastAPI(
    title="Closed-Loop Life Support OpenEnv",
    description=(
        "OpenEnv-compliant space habitat life support simulator. "
        "An AI agent manages oxygen, water, and food for a crew aboard a space station."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (keyed by session_id)
_sessions: Dict[str, Dict[str, Any]] = {}


# ── Request / Response schemas ───────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task_easy"
    seed: Optional[int] = None

class ResetResponse(BaseModel):
    session_id: str
    task_id: str
    observation: Dict[str, Any]
    info: Dict[str, Any]

class StepRequest(BaseModel):
    session_id: str
    action: Action

class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]

class StateResponse(BaseModel):
    session_id: str
    state: Dict[str, Any]

class GradeRequest(BaseModel):
    session_id: str
    task_id: str

class GradeResponse(BaseModel):
    score: float
    passed: bool
    breakdown: Dict[str, float]
    feedback: str


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "welcome": "Welcome to Artemis Moon Mission By BigByte",
        "name": "Closed-Loop Life Support OpenEnv",
        "version": "1.0.0",
        "tasks": ["task_easy", "task_medium", "task_hard"],
        "endpoints": ["/reset", "/step", "/state", "/grade", "/health"],
        "docs": "/docs",
    }

@app.get("/health")
def health():
    return {"status": "ok", "sessions": len(_sessions), "timestamp": time.time()}

@app.post("/reset", response_model=ResetResponse)
def reset(req: Optional[ResetRequest] = None):
    """Start a new episode. Returns initial observation and a session_id."""
    if req is None:
        req = ResetRequest()
    try:
        env = LifeSupportEnv(task_id=req.task_id, seed=req.seed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    obs = env.reset()
    session_id = str(uuid.uuid4())

    _sessions[session_id] = {
        "env": env,
        "task_id": req.task_id,
        "trajectory": [],
        "created_at": time.time(),
    }

    return ResetResponse(
        session_id=session_id,
        task_id=req.task_id,
        observation=obs.dict(),
        info={"task_config": env.config, "max_steps": env.config["max_steps"]},
    )

@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    """Apply an action and advance the environment by one step."""
    if req.session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")

    session = _sessions[req.session_id]
    env: LifeSupportEnv = session["env"]

    try:
        obs, reward, done, info = env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Record trajectory for grading
    session["trajectory"].append({
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "done_reason": info.get("failure_reason"),
        "step": info["step"],
    })

    return StepResponse(
        observation=obs.dict(),
        reward=reward,
        done=done,
        info=info,
    )

@app.get("/state/{session_id}", response_model=StateResponse)
def state(session_id: str):
    """Return the full internal environment state."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    env: LifeSupportEnv = _sessions[session_id]["env"]
    env_state = env.state()

    return StateResponse(
        session_id=session_id,
        state=env_state.dict(),
    )

@app.post("/grade", response_model=GradeResponse)
def grade(req: Optional[GradeRequest] = None):
    """Grade a completed episode trajectory."""
    if req is None:
        raise HTTPException(status_code=400, detail="Missing session_id in grade request.")
    if req.session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    session = _sessions[req.session_id]
    trajectory = session["trajectory"]

    if not trajectory:
        raise HTTPException(status_code=400, detail="No steps recorded yet.")

    result: GradeResult = grade_episode(req.task_id, trajectory)

    return GradeResponse(
        score=result.score,
        passed=result.passed,
        breakdown=result.breakdown,
        feedback=result.feedback,
    )

@app.get("/tasks")
def list_tasks():
    """List all available tasks with descriptions."""
    return {
        "tasks": [
            {
                "id": "task_easy",
                "name": "Single-Day Stabilization",
                "difficulty": "easy",
                "crew_size": 3,
                "max_steps": 24,
                "description": "Keep all life support parameters within safe ranges for 24 hours.",
            },
            {
                "id": "task_medium",
                "name": "7-Day Artemis Survival",
                "difficulty": "medium",
                "crew_size": 5,
                "max_steps": 168,
                "events": ["dust_storm", "equipment_fault", "supply_pod", "lunar_night"],
                "description": "7-day mission with dust storms, equipment faults, and supply drops. Manage crises while keeping crew alive.",
            },
            {
                "id": "task_hard",
                "name": "30-Day Artemis Gauntlet",
                "difficulty": "hard",
                "crew_size": 8,
                "max_steps": 720,
                "events": ["solar_flare", "meteor_impact", "crew_emergency", "equipment_catastrophic", "dust_storm", "supply_pod", "lunar_night"],
                "description": "30-day Artemis Moon survival with solar flares, meteor impacts, radiation, cascading failures, power routing, and crew dynamics. Only the best agents survive.",
            },
        ]
    }
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
