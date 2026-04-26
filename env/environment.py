"""
Closed-Loop Life Support Environment — Round 2 v2 (Artemis Survival Edition)
OpenEnv-compliant simulation with cascading interconnections + tiered events.

KEY UPGRADES:
  - Tiered event system: medium gets Tier 1, hard gets Tier 1+2
  - Power routing (hard): agent allocates power during crises
  - Cascading failures: events compound and spiral
  - Crew dynamics: supply pods bring crew, injuries reduce effective crew
  - Radiation tracking (hard): cumulative exposure affects health
"""

import random
from typing import Tuple, Dict, Any, Optional

from env.models import Observation, Action, Reward, EnvironmentState
from env.events import EventSystem


# ── Physical constants ──────────────────────────────────────────────────────
O2_SAFE_MIN   = 19.5
O2_SAFE_MAX   = 23.5
O2_FIRE_RISK  = 25.0
CO2_SAFE_MAX  = 1000
CO2_CRITICAL  = 3000
WATER_PER_CREW_PER_STEP   = 0.25
FOOD_PER_CREW_PER_DAY     = 0.7
O2_PER_CREW_PER_STEP      = 0.04
CO2_PER_CREW_PER_STEP     = 0.035
PLANT_O2_PER_KG_PER_STEP  = 0.008
PLANT_CO2_CONSUME         = 0.006
WATER_FROM_PLANTS         = 0.02
MAX_PLANT_BIOMASS         = 50.0
POWER_COST_PLANT          = 0.30
POWER_COST_RECYCLE        = 0.25
POWER_COST_O2_ADJUST      = 0.15

# Cascade constants
FIRE_DAMAGE_RATE       = 0.08
CO2_WATER_TOXIN_FACTOR = 0.0003

# Power routing multipliers
POWER_ROUTING = {
    "balanced":      {"life_support": 1.0, "shields": 1.0, "hydroponics": 1.0},
    "life_support":  {"life_support": 1.5, "shields": 0.5, "hydroponics": 0.3},
    "shields":       {"life_support": 0.7, "shields": 2.0, "hydroponics": 0.4},
    "hydroponics":   {"life_support": 0.7, "shields": 0.4, "hydroponics": 2.0},
    "emergency":     {"life_support": 1.8, "shields": 0.2, "hydroponics": 0.1},
}


class LifeSupportEnv:
    """Round 2 v2 environment with tiered events and power routing."""

    TASK_CONFIGS = {
        "task_easy": {
            "crew_size": 3, "max_steps": 24,
            "initial_water": 200.0, "initial_food": 30.0,
            "initial_o2": 21.0, "initial_co2": 400.0,
            "initial_plant_biomass": 15.0,
            "description": "Keep all parameters safe for 24h with 3-person crew.",
            "events_enabled": False,
            "event_tier": None,
            "max_active_events": 0,
            "power_routing_enabled": False,
            "radiation_enabled": False,
        },
        "task_medium": {
            "crew_size": 5, "max_steps": 168,
            "initial_water": 250.0, "initial_food": 50.0,
            "initial_o2": 21.0, "initial_co2": 450.0,
            "initial_plant_biomass": 20.0,
            "description": "Sustain 5-person crew for 7 days with dust storms and equipment faults.",
            "events_enabled": True,
            "event_tier": "medium",
            "max_active_events": 1,
            "power_routing_enabled": False,
            "radiation_enabled": False,
        },
        "task_hard": {
            "crew_size": 8, "max_steps": 720,
            "initial_water": 300.0, "initial_food": 60.0,
            "initial_o2": 21.0, "initial_co2": 400.0,
            "initial_plant_biomass": 25.0,
            "description": "30-day Artemis survival with solar flares, meteors, radiation, and power routing.",
            "events_enabled": True,
            "event_tier": "hard",
            "max_active_events": 3,
            "power_routing_enabled": True,
            "radiation_enabled": True,
        },
    }

    def __init__(self, task_id: str = "task_easy", seed: Optional[int] = None):
        if task_id not in self.TASK_CONFIGS:
            raise ValueError(f"Unknown task_id: {task_id}. Options: {list(self.TASK_CONFIGS.keys())}")
        self.task_id = task_id
        self.config  = self.TASK_CONFIGS[task_id]
        self.rng     = random.Random(seed)
        self._init_state()

    # ── OpenEnv API ──────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Deep-randomised reset to prevent overfitting."""
        self._init_state()
        cfg = self.config
        self._o2_percent    = max(15.0, min(28.0, cfg["initial_o2"]  + self.rng.uniform(-2.0, 2.0)))
        self._co2_ppm       = max(200.0, min(3500.0, cfg["initial_co2"] + self.rng.uniform(-200, 500)))
        self._water         = cfg["initial_water"] * self.rng.uniform(0.7, 1.0)
        self._food          = cfg["initial_food"]  * self.rng.uniform(0.6, 1.0)
        self._plant_biomass = min(MAX_PLANT_BIOMASS, max(0.0,
                                  cfg["initial_plant_biomass"] * self.rng.uniform(0.5, 1.2)))
        self._prev_o2  = self._o2_percent
        self._prev_co2 = self._co2_ppm
        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """One-hour simulation tick with cascade physics and events."""
        if self._done:
            raise RuntimeError("Episode done. Call reset() first.")

        self._step_count += 1
        prev_o2  = self._o2_percent
        prev_co2 = self._co2_ppm

        # ── Event system tick ────────────────────────────────────────────────
        event_effects = {
            "solar_panel_mult": 1.0, "radiation": 0.0,
            "co2_scrubber_mult": 1.0, "water_recycler_mult": 1.0,
            "o2_gen_mult": 1.0, "o2_leak": 0.0,
            "crew_injured": 0, "crew_added": 0,
            "water_bonus": 0.0, "food_bonus": 0.0, "event_names": [],
        }

        if self._event_system is not None:
            new_events = self._event_system.tick()
            event_effects = self._event_system.apply_effects(self)

            # Apply supply pod bonuses
            if event_effects["water_bonus"] > 0:
                self._water = min(500.0, self._water + event_effects["water_bonus"])
            if event_effects["food_bonus"] > 0:
                self._food = min(100.0, self._food + event_effects["food_bonus"])
            if event_effects["crew_added"] > 0:
                self._crew_added += event_effects["crew_added"]

            # Track crew injuries
            self._crew_injured = min(self._effective_crew(), event_effects["crew_injured"])

            # Solar panel degradation
            self._solar_panel_health = max(0.1, event_effects["solar_panel_mult"])

            # O2 leak from meteor impacts
            if event_effects["o2_leak"] > 0:
                self._o2_percent -= event_effects["o2_leak"]

        # ── Power routing (hard mode) ────────────────────────────────────────
        route = "balanced"
        if self.config["power_routing_enabled"]:
            route = action.route_power if action.route_power in POWER_ROUTING else "balanced"
            if self._event_system:
                self._event_system.track_routing(route)
        self._current_routing = route
        pr = POWER_ROUTING.get(route, POWER_ROUTING["balanced"])

        # ── Radiation (hard mode) ────────────────────────────────────────────
        if self.config.get("radiation_enabled") and event_effects["radiation"] > 0:
            shield_factor = pr["shields"] * self._shield_integrity
            effective_rad = event_effects["radiation"] * max(0.1, 1.0 - shield_factor * 0.6)
            self._radiation_level = effective_rad
            self._cumulative_radiation += effective_rad * 0.01
            # Radiation health drain
            self._crew_health = max(0.0, self._crew_health - effective_rad * 0.015)
            # Shield degrades under radiation
            self._shield_integrity = max(0.0, self._shield_integrity - event_effects["radiation"] * 0.005)
        else:
            self._radiation_level = max(0.0, self._radiation_level - 0.05)

        # Clamp actions
        pg = max(0.0, min(1.0, action.increase_plant_growth))
        rw = max(0.0, min(1.0, action.recycle_water))
        ao = max(-1.0, min(1.0, action.adjust_oxygen))
        rf = max(0.0, min(1.0, action.ration_food))
        ca = max(0.0, min(1.0, action.crew_activity))

        # Baseline metabolism: crew consumes resources even at zero activity
        effective_ca = 0.4 + 0.6 * ca

        crew = self._effective_crew()

        # Power budget — affected by solar panel health and routing
        available_power = self._solar_panel_health
        power_used = (pg * POWER_COST_PLANT * (1.0 / max(0.3, pr["hydroponics"])) +
                      rw * POWER_COST_RECYCLE * (1.0 / max(0.3, pr["life_support"])) +
                      abs(ao) * POWER_COST_O2_ADJUST * (1.0 / max(0.3, pr["life_support"])))
        if power_used > available_power:
            s = available_power / power_used
            pg *= s; rw *= s; ao *= s
            power_used = available_power
        power_budget = max(0.0, available_power - power_used)

        # ── CASCADE 1: High O2 → fire risk → plant damage ────────────────────
        fire_active = self._o2_percent > O2_FIRE_RISK
        if fire_active:
            self._plant_biomass = max(0.0,
                self._plant_biomass - FIRE_DAMAGE_RATE * (self._o2_percent - O2_FIRE_RISK))
            self._fire_steps += 1
        else:
            self._fire_steps = max(0, self._fire_steps - 1)

        # Plant simulation — boosted by hydroponics routing
        hydro_mult = pr.get("hydroponics", 1.0)
        water_for_plants = min(self._water * 0.05, pg * 2.0 * hydro_mult)
        growth_delta = pg * 0.5 * hydro_mult * (water_for_plants / max(water_for_plants, 0.01))
        self._plant_biomass = min(MAX_PLANT_BIOMASS, max(0.0,
                                   self._plant_biomass + growth_delta - 0.1))
        self._plant_growth_rate = pg
        if self._plant_biomass > 30.0:
            harvest = (self._plant_biomass - 30.0) * 0.1
            self._food = min(100.0, self._food + harvest * 0.8)
            self._plant_biomass -= harvest

        # O2 / CO2 dynamics — affected by equipment status
        o2_gen_mult = event_effects["o2_gen_mult"]
        co2_scrub_mult = event_effects["co2_scrubber_mult"]

        plant_o2  = self._plant_biomass * PLANT_O2_PER_KG_PER_STEP * o2_gen_mult
        plant_co2 = self._plant_biomass * PLANT_CO2_CONSUME * 1000
        scrubber_eff = self._co2_scrubber_efficiency * co2_scrub_mult * pr.get("life_support", 1.0)
        co2_scrubbed = self._co2_ppm * scrubber_eff * abs(min(0.0, ao)) * 0.3

        self._o2_percent += plant_o2 - crew * O2_PER_CREW_PER_STEP * effective_ca + ao * 0.5
        self._co2_ppm    += crew * CO2_PER_CREW_PER_STEP * effective_ca * 1000 - plant_co2 - co2_scrubbed

        self._o2_percent = max(0.0, min(30.0, self._o2_percent))
        self._co2_ppm    = max(0.0, min(5000.0, self._co2_ppm))

        # ── CASCADE 2: High CO2 → water toxicity ────────────────────────────
        co2_toxin_factor = 1.0 - min(0.4, CO2_WATER_TOXIN_FACTOR * max(0.0, self._co2_ppm - CO2_SAFE_MAX))
        water_recycle_mult = event_effects["water_recycler_mult"]
        effective_rw = rw * co2_toxin_factor * water_recycle_mult

        # Water dynamics
        crew_water_used = crew * WATER_PER_CREW_PER_STEP
        self._waste_water_buffer += crew_water_used * 0.7 + self._plant_biomass * WATER_FROM_PLANTS
        recycled = self._waste_water_buffer * effective_rw * 0.9
        self._water = max(0.0, min(500.0, self._water - crew_water_used + recycled))
        self._waste_water_buffer = max(0.0, self._waste_water_buffer - recycled)
        self._water_recycling_rate = rw

        # ── CASCADE 3: Low water → food efficiency drop ──────────────────────
        dehydration_factor = max(0.5, self._water / 20.0) if self._water < 20.0 else 1.0
        if self._water < 20.0:
            self._dehydration_steps += 1
        else:
            self._dehydration_steps = max(0, self._dehydration_steps - 1)

        # Food dynamics
        total_crew_for_food = self.config["crew_size"] + self._crew_added
        self._food = max(0.0, self._food - total_crew_for_food * (FOOD_PER_CREW_PER_DAY / 24.0) * rf * effective_ca * dehydration_factor)

        # Crew health
        h = 0.0
        if self._o2_percent < O2_SAFE_MIN:
            h -= 0.05 * (O2_SAFE_MIN - self._o2_percent)
        elif self._o2_percent > O2_SAFE_MAX:
            h -= 0.01 * (self._o2_percent - O2_SAFE_MAX)
        if self._co2_ppm > CO2_SAFE_MAX:
            h -= 0.03 * (self._co2_ppm - CO2_SAFE_MAX) / 1000
        if self._co2_ppm > CO2_CRITICAL:
            h -= 0.1
        if self._water < 10.0:
            h -= 0.04 * (10.0 - self._water) / 10.0
        
        # Penalize health for extreme rationing (prevents hoarding food forever)
        if rf < 0.5:
            h -= 0.02 * (0.5 - rf)
        if self._food <= 0.0:
            h -= 0.05
        if fire_active:
            h -= 0.02 * (self._o2_percent - O2_FIRE_RISK)

        # Radiation long-term damage
        if self._cumulative_radiation > 0.1:
            h -= self._cumulative_radiation * 0.02

        # EVA radiation exposure (high crew activity during radiation)
        if ca > 0.9 and self._radiation_level > 0.2:
            h -= self._radiation_level * 0.01

        all_green = (O2_SAFE_MIN <= self._o2_percent <= O2_SAFE_MAX and
                     self._co2_ppm < CO2_SAFE_MAX and
                     self._water > 20.0 and self._food > 0.5)
        if all_green:
            h += 0.005
            self._green_streak += 1
        else:
            self._green_streak = 0

        self._crew_health = max(0.0, min(1.0, self._crew_health + h))
        self._cumulative_health += self._crew_health

        delta_o2  = self._o2_percent - prev_o2
        delta_co2 = self._co2_ppm    - prev_co2

        reward_obj = self._compute_reward(power_budget, delta_o2, delta_co2, all_green, event_effects)
        reward = reward_obj.total
        self._total_reward += reward

        # Terminal conditions
        done = False; failure_reason = None
        if self._crew_health <= 0.0:
            done = True; failure_reason = "Crew health reached zero"; reward -= 5.0
        elif self._o2_percent < 15.0:
            done = True; failure_reason = "O2 below survivable threshold"; reward -= 5.0
        elif self._co2_ppm > 4500:
            done = True; failure_reason = "CO2 lethal concentration"; reward -= 5.0
        elif self._step_count >= self.config["max_steps"]:
            done = True; reward += 5.0

        self._done = done
        self._failure_reason = failure_reason

        obs  = self._make_observation(power_budget)
        primary_event = self._event_system.get_primary_event() if self._event_system else None
        info = {
            "step": self._step_count,
            "total_reward": self._total_reward,
            "failure_reason": failure_reason,
            "plant_biomass": round(self._plant_biomass, 2),
            "waste_water_buffer": round(self._waste_water_buffer, 2),
            "fire_active": fire_active,
            "fire_steps": self._fire_steps,
            "dehydration_steps": self._dehydration_steps,
            "green_streak": self._green_streak,
            "co2_toxin_factor": round(co2_toxin_factor, 3),
            "delta_o2": round(delta_o2, 4),
            "delta_co2": round(delta_co2, 2),
            "alarm_tiers": self._alarm_tiers(),
            "reward_breakdown": reward_obj.dict(),
            "active_events": event_effects["event_names"],
            "radiation_level": round(self._radiation_level, 3),
            "solar_panel_health": round(self._solar_panel_health, 3),
            "crew_injured": self._crew_injured,
            "crew_added": self._crew_added,
            "power_routing": self._current_routing,
        }
        return obs, reward, done, info

    def state(self) -> EnvironmentState:
        return EnvironmentState(
            observation=self._make_observation(),
            step_count=self._step_count,
            episode_done=self._done,
            task_id=self.task_id,
            total_reward=self._total_reward,
            failure_reason=self._failure_reason,
            co2_scrubber_efficiency=self._co2_scrubber_efficiency,
            plant_biomass=self._plant_biomass,
            waste_water_buffer=self._waste_water_buffer,
        )

    # ── Internals ─────────────────────────────────────────────────────────────

    def _effective_crew(self) -> int:
        total = self.config["crew_size"] + self._crew_added
        return max(1, total - self._crew_injured)

    def _init_state(self):
        self._step_count = 0; self._done = False
        self._total_reward = 0.0; self._failure_reason = None
        self._cumulative_health = 0.0
        self._co2_ppm = 400.0; self._o2_percent = 21.0
        self._water = 200.0; self._food = 30.0
        self._plant_biomass = 15.0; self._waste_water_buffer = 40.0
        self._crew_health = 1.0
        self._water_recycling_rate = 0.5; self._plant_growth_rate = 0.5
        self._co2_scrubber_efficiency = 0.85
        self._green_streak = 0; self._fire_steps = 0; self._dehydration_steps = 0
        self._prev_o2 = 21.0; self._prev_co2 = 400.0

        # Event system state
        self._solar_panel_health = 1.0
        self._radiation_level = 0.0
        self._cumulative_radiation = 0.0
        self._shield_integrity = 1.0
        self._crew_injured = 0
        self._crew_added = 0
        self._current_routing = "balanced"

        # Initialize event system based on difficulty
        cfg = self.config
        if cfg.get("events_enabled"):
            self._event_system = EventSystem(
                tier=cfg["event_tier"],
                rng=self.rng,
                max_active=cfg.get("max_active_events", 1),
            )
        else:
            self._event_system = None

    def _make_observation(self, power_budget: float = 1.0) -> Observation:
        primary_event = self._event_system.get_primary_event() if self._event_system else None
        active_names = [e.name for e in self._event_system.active] if self._event_system else []

        return Observation(
            co2_ppm=round(self._co2_ppm, 2),
            o2_percent=round(self._o2_percent, 3),
            water_liters=round(self._water, 2),
            food_kg=round(self._food, 3),
            crew_size=self.config["crew_size"] + self._crew_added,
            plant_growth_rate=round(self._plant_growth_rate, 3),
            water_recycling_rate=round(self._water_recycling_rate, 3),
            day=max(1, (self._step_count // 24) + 1),
            crew_health=round(self._crew_health, 4),
            power_budget=round(power_budget, 3),
            solar_panel_health=round(self._solar_panel_health, 3),
            radiation_level=round(self._radiation_level, 3),
            shield_integrity=round(self._shield_integrity, 3),
            event_name=primary_event.name if primary_event else None,
            event_severity=primary_event.severity if primary_event else None,
            event_turns_remaining=primary_event.turns_remaining if primary_event else 0,
            crew_injured=self._crew_injured,
            active_events=active_names,
            power_routing=self._current_routing,
            cumulative_radiation=round(self._cumulative_radiation, 4),
        )

    def _compute_reward(self, power_budget, delta_o2, delta_co2, all_green, event_effects) -> Reward:
        penalty = 0.0

        if O2_SAFE_MIN <= self._o2_percent <= O2_SAFE_MAX:
            o2_score = 1.0
        elif self._o2_percent < O2_SAFE_MIN:
            o2_score = max(-1.0, (self._o2_percent - O2_SAFE_MIN) / O2_SAFE_MIN)
            penalty += 0.5
        else:
            o2_score = max(0.0, 1.0 - (self._o2_percent - O2_SAFE_MAX) / 5.0)
            if self._o2_percent > O2_FIRE_RISK:
                penalty += 0.3

        if self._co2_ppm <= CO2_SAFE_MAX:
            co2_score = 1.0 - (self._co2_ppm / CO2_SAFE_MAX) * 0.2
        else:
            co2_score = max(-1.0, 1.0 - (self._co2_ppm - CO2_SAFE_MAX) / 2000.0)
            penalty += 0.5 * min(1.0, self._co2_ppm / CO2_CRITICAL)

        health_component    = 0.5 * self._crew_health + 0.25 * o2_score + 0.25 * co2_score
        resource_component  = 0.5 * min(1.0, self._water / 50.0) + 0.5 * min(1.0, self._food / 10.0)
        efficiency_component = power_budget * 0.3 if self._crew_health > 0.8 else 0.0

        if self._water <= 0: penalty += 0.4
        if self._food  <= 0: penalty += 0.2

        survival_reward = 0.1

        # Trend bonus
        trend_bonus = 0.0
        if self._o2_percent < O2_SAFE_MIN and delta_o2 > 0:
            trend_bonus += min(0.1, 0.05 * delta_o2)
        if self._o2_percent > O2_SAFE_MAX and delta_o2 < 0:
            trend_bonus += min(0.1, 0.05 * abs(delta_o2))
        if self._co2_ppm > CO2_SAFE_MAX and delta_co2 < 0:
            trend_bonus += min(0.1, 0.03 * abs(delta_co2) / 500.0)

        streak_bonus = min(0.05, self._green_streak * 0.001)

        # Event survival bonus — reward staying alive during events
        event_bonus = 0.0
        if event_effects["event_names"] and self._crew_health > 0.5:
            event_bonus = 0.05 * len(event_effects["event_names"])

        raw = (0.4 * health_component + 0.2 * resource_component +
               0.1 * efficiency_component + survival_reward +
               trend_bonus + streak_bonus + event_bonus - penalty)

        total = max(-2.0, min(2.0, raw))
        return Reward(
            total=round(total, 4),
            health_component=round(health_component, 4),
            resource_component=round(resource_component, 4),
            efficiency_component=round(efficiency_component, 4),
            penalty=round(penalty, 4),
        )

    def _alarm_tiers(self) -> Dict[str, str]:
        def o2_tier():
            if O2_SAFE_MIN <= self._o2_percent <= O2_SAFE_MAX: return "GREEN"
            if self._o2_percent > O2_FIRE_RISK or self._o2_percent < 17.0: return "CRITICAL"
            return "RED"
        def co2_tier():
            if self._co2_ppm <= CO2_SAFE_MAX: return "GREEN"
            if self._co2_ppm <= 2000: return "YELLOW"
            if self._co2_ppm <= CO2_CRITICAL: return "RED"
            return "CRITICAL"
        def water_tier():
            if self._water > 50: return "GREEN"
            if self._water > 20: return "YELLOW"
            if self._water > 5:  return "RED"
            return "CRITICAL"
        def food_tier():
            if self._food > 20: return "GREEN"
            if self._food > 5:  return "YELLOW"
            if self._food > 0:  return "RED"
            return "CRITICAL"
        return {"o2": o2_tier(), "co2": co2_tier(), "water": water_tier(), "food": food_tier()}
