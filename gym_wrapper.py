"""
gym_wrapper.py — Gymnasium wrapper around LifeSupportEnv.

Bridges OpenEnv format to stable-baselines3 / any standard RL library.
Round 2 v2: expanded observation space for events + power routing action.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.environment import LifeSupportEnv
from env.models import Action


# Observation order (must stay consistent with _obs_to_array)
OBS_KEYS = [
    "o2_percent", "co2_ppm", "water_liters", "food_kg",
    "crew_size", "plant_growth_rate", "water_recycling_rate",
    "day", "crew_health", "power_budget",
    # New event fields
    "solar_panel_health", "radiation_level", "shield_integrity",
    "crew_injured", "cumulative_radiation",
]

OBS_LOW  = np.array([0,    0,    0,   0,  1, 0, 0,  1, 0, 0,   0, 0, 0, 0, 0], dtype=np.float32)
OBS_HIGH = np.array([30, 5000, 500, 100, 20, 1, 1, 365, 1, 1,   1, 1, 1, 10, 5], dtype=np.float32)

# Power routing modes encoded as floats
ROUTING_MODES = ["balanced", "life_support", "shields", "hydroponics", "emergency"]


class LifeSupportGymEnv(gym.Env):
    """
    Gymnasium-compatible wrapper for LifeSupportEnv.

    Observation space: Box(15,) — normalised sensor readings + event fields
    Action space:      Box(6,)  — continuous controls [0,1] + power routing
    """

    metadata = {"render_modes": []}

    def __init__(self, task_id: str = "task_easy", seed: int = None):
        super().__init__()
        self._env  = LifeSupportEnv(task_id=task_id, seed=seed)
        self._task = task_id

        self.observation_space = spaces.Box(low=OBS_LOW, high=OBS_HIGH, dtype=np.float32)

        # [plant_growth, recycle_water, adjust_oxygen, ration_food, crew_activity, power_routing]
        # power_routing: 0-0.2=balanced, 0.2-0.4=life_support, 0.4-0.6=shields, 0.6-0.8=hydroponics, 0.8-1.0=emergency
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0,  1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs_model = self._env.reset()
        return self._to_array(obs_model), {}

    def step(self, action: np.ndarray):
        # Decode power routing from continuous value
        routing_idx = int(np.clip(action[5] * 5, 0, 4))
        routing = ROUTING_MODES[routing_idx]

        act = Action(
            increase_plant_growth=float(action[0]),
            recycle_water=float(action[1]),
            adjust_oxygen=float(action[2]),
            ration_food=float(action[3]),
            crew_activity=float(action[4]),
            route_power=routing,
        )
        obs_model, reward, done, info = self._env.step(act)
        obs_array = self._to_array(obs_model)
        truncated = False  # we use 'done' for both termination and truncation
        return obs_array, float(reward), done, truncated, info

    def _to_array(self, obs) -> np.ndarray:
        return np.array([
            obs.o2_percent, obs.co2_ppm, obs.water_liters, obs.food_kg,
            obs.crew_size, obs.plant_growth_rate, obs.water_recycling_rate,
            obs.day, obs.crew_health, obs.power_budget,
            obs.solar_panel_health, obs.radiation_level, obs.shield_integrity,
            float(obs.crew_injured), obs.cumulative_radiation,
        ], dtype=np.float32)
