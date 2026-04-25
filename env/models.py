"""
Pydantic models for the Closed-Loop Life Support OpenEnv environment.

Round 2 v2 — Adds tiered event system fields (medium & hard).
All new fields have safe defaults so easy-mode is unchanged.
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class Observation(BaseModel):
    """Sensor readings from the habitat at each timestep."""
    co2_ppm: float = Field(..., ge=0, le=5000, description="CO2 in parts per million")
    o2_percent: float = Field(..., ge=0, le=30, description="O2 percentage in atmosphere")
    water_liters: float = Field(..., ge=0, le=500, description="Potable water available (L)")
    food_kg: float = Field(..., ge=0, le=100, description="Food supply (kg)")
    crew_size: int = Field(..., ge=1, le=20, description="Active crew members")
    plant_growth_rate: float = Field(..., ge=0, le=1, description="Normalized plant growth rate")
    water_recycling_rate: float = Field(..., ge=0, le=1, description="Water recycling efficiency")
    day: int = Field(..., ge=1, le=365, description="Mission day")
    crew_health: float = Field(..., ge=0, le=1, description="Crew health score (0=critical, 1=optimal)")
    power_budget: float = Field(..., ge=0, le=1, description="Remaining power budget fraction")

    # ── Event system fields (medium & hard) ──────────────────────────────────
    solar_panel_health: float = Field(1.0, ge=0, le=1, description="Solar panel condition (0=destroyed, 1=perfect)")
    radiation_level: float = Field(0.0, ge=0, le=1, description="Ambient radiation (0=safe, 1=lethal)")
    shield_integrity: float = Field(1.0, ge=0, le=1, description="Habitat radiation shielding")
    event_name: Optional[str] = Field(None, description="Name of active crisis event")
    event_severity: Optional[str] = Field(None, description="minor / major / catastrophic")
    event_turns_remaining: int = Field(0, ge=0, description="Steps until current event ends")
    crew_injured: int = Field(0, ge=0, description="Number of incapacitated crew members")
    active_events: List[str] = Field(default_factory=list, description="All currently active event names")
    power_routing: str = Field("balanced", description="Current power routing mode")
    cumulative_radiation: float = Field(0.0, ge=0, description="Total radiation exposure so far")


class Action(BaseModel):
    """Control inputs to the life support subsystems."""
    increase_plant_growth: float = Field(0.5, ge=0, le=1, description="Photosynthesis boost (uses power + water)")
    recycle_water: float = Field(0.5, ge=0, le=1, description="Water reclamation intensity")
    adjust_oxygen: float = Field(0.0, ge=-1, le=1, description="O2 release (+) or CO2 scrub (-)")
    ration_food: float = Field(1.0, ge=0, le=1, description="Food ration level")
    crew_activity: float = Field(0.7, ge=0, le=1, description="Permitted crew activity level")
    # Power routing (hard mode) — "balanced"/"life_support"/"shields"/"hydroponics"/"emergency"
    route_power: str = Field("balanced", description="Power routing priority mode")


class Reward(BaseModel):
    """Reward signal with breakdown for interpretability."""
    total: float = Field(..., description="Total shaped reward")
    health_component: float = Field(..., description="Reward from crew health")
    resource_component: float = Field(..., description="Reward from resource sustainability")
    efficiency_component: float = Field(..., description="Reward from power efficiency")
    penalty: float = Field(..., description="Penalty for critical failures")


class EnvironmentState(BaseModel):
    """Full internal state for reproducibility and debugging."""
    observation: Observation
    step_count: int
    episode_done: bool
    task_id: str
    total_reward: float
    failure_reason: Optional[str] = None
    # Internal simulation state
    co2_scrubber_efficiency: float = 0.85
    plant_biomass: float = 10.0  # kg of growing plants
    waste_water_buffer: float = 50.0  # liters of recoverable waste
