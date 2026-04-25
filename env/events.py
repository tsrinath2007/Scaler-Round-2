"""
EventSystem — Tiered randomised crisis events for Artemis survival.

Tier 1 (Medium): Dust storms, minor equipment faults, supply pods, lunar night
Tier 2 (Hard):   + Solar flares, meteor impacts, crew emergencies, catastrophic failures
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ActiveEvent:
    name: str
    severity: str          # minor / major / catastrophic
    turns_remaining: int
    tier: int              # 1 or 2
    data: Dict = field(default_factory=dict)


# ── Event definitions ────────────────────────────────────────────────────────

TIER1_EVENTS = [
    {"name": "dust_storm",        "prob": 0.03, "dur": (8, 20),  "severity": "minor"},
    {"name": "equipment_fault",   "prob": 0.02, "dur": (5, 10),  "severity": "minor"},
    {"name": "supply_pod",        "prob": 0.03, "dur": (1, 1),   "severity": "minor"},
    {"name": "lunar_night",       "prob": 0.015,"dur": (15, 25), "severity": "major"},
]

TIER2_EVENTS = [
    {"name": "solar_flare",       "prob": 0.03, "dur": (5, 15),  "severity": "catastrophic"},
    {"name": "meteor_impact",     "prob": 0.015,"dur": (1, 1),   "severity": "catastrophic"},
    {"name": "crew_emergency",    "prob": 0.01, "dur": (5, 10),  "severity": "major"},
    {"name": "equipment_catastrophic", "prob": 0.01, "dur": (10, 20), "severity": "catastrophic"},
]


class EventSystem:
    def __init__(self, tier: str, rng: random.Random, max_active: int = 1):
        self.tier = tier
        self.rng = rng
        self.max_active = max_active
        self.active: List[ActiveEvent] = []
        self.history: List[str] = []
        self.events_survived = 0
        self.total_events = 0
        self.power_routing_changes = 0
        self.last_routing = "balanced"

        self.event_table = list(TIER1_EVENTS)
        if tier == "hard":
            self.event_table += TIER2_EVENTS

    def tick(self) -> List[ActiveEvent]:
        """Advance timers, roll for new events. Returns newly triggered events."""
        # Decrement active event timers
        expired = [e for e in self.active if e.turns_remaining <= 1]
        for e in expired:
            self.events_survived += 1
        self.active = [e for e in self.active if e.turns_remaining > 1]
        for e in self.active:
            e.turns_remaining -= 1

        # Roll for new events if below cap
        new_events = []
        if len(self.active) < self.max_active:
            for edef in self.event_table:
                if len(self.active) + len(new_events) >= self.max_active:
                    break
                # Don't duplicate same event type
                active_names = {e.name for e in self.active}
                if edef["name"] in active_names:
                    continue
                if self.rng.random() < edef["prob"]:
                    dur = self.rng.randint(edef["dur"][0], edef["dur"][1])
                    ev = ActiveEvent(
                        name=edef["name"],
                        severity=edef["severity"],
                        turns_remaining=dur,
                        tier=2 if edef in TIER2_EVENTS else 1,
                        data=self._gen_event_data(edef["name"]),
                    )
                    self.active.append(ev)
                    new_events.append(ev)
                    self.total_events += 1
                    self.history.append(edef["name"])

        return new_events

    def _gen_event_data(self, name: str) -> Dict:
        if name == "dust_storm":
            return {"panel_reduction": self.rng.uniform(0.2, 0.4)}
        if name == "lunar_night":
            return {"panel_reduction": self.rng.uniform(0.3, 0.5)}
        if name == "equipment_fault":
            target = self.rng.choice(["co2_scrubber", "water_recycler"])
            return {"target": target, "efficiency_mult": 0.7}
        if name == "supply_pod":
            crew_bonus = 0
            if self.tier == "hard" and self.rng.random() < 0.35:
                crew_bonus = self.rng.randint(1, 2)
            return {
                "water": self.rng.uniform(30, 80),
                "food": self.rng.uniform(5, 20),
                "crew_bonus": crew_bonus,
            }
        if name == "solar_flare":
            return {
                "panel_reduction": self.rng.uniform(0.4, 0.65),
                "radiation": self.rng.uniform(0.4, 0.8),
            }
        if name == "meteor_impact":
            crew_injured = 1 if self.rng.random() < 0.3 else 0
            return {
                "o2_leak_rate": self.rng.uniform(0.2, 0.5),
                "crew_injured": crew_injured,
            }
        if name == "crew_emergency":
            return {"crew_down": 1}
        if name == "equipment_catastrophic":
            target = self.rng.choice(["co2_scrubber", "water_recycler", "o2_generator"])
            return {"target": target, "efficiency_mult": 0.2}
        return {}

    def apply_effects(self, env) -> Dict:
        """Apply all active event effects to the environment. Returns info dict."""
        info = {
            "solar_panel_mult": 1.0,
            "radiation": 0.0,
            "co2_scrubber_mult": 1.0,
            "water_recycler_mult": 1.0,
            "o2_gen_mult": 1.0,
            "o2_leak": 0.0,
            "crew_injured": 0,
            "crew_added": 0,
            "water_bonus": 0.0,
            "food_bonus": 0.0,
            "event_names": [],
        }

        for ev in self.active:
            info["event_names"].append(ev.name)

            if ev.name in ("dust_storm", "lunar_night"):
                info["solar_panel_mult"] *= (1.0 - ev.data["panel_reduction"])

            elif ev.name == "equipment_fault":
                if ev.data["target"] == "co2_scrubber":
                    info["co2_scrubber_mult"] *= ev.data["efficiency_mult"]
                elif ev.data["target"] == "water_recycler":
                    info["water_recycler_mult"] *= ev.data["efficiency_mult"]

            elif ev.name == "supply_pod":
                info["water_bonus"] += ev.data["water"]
                info["food_bonus"] += ev.data["food"]
                info["crew_added"] += ev.data.get("crew_bonus", 0)

            elif ev.name == "solar_flare":
                info["solar_panel_mult"] *= (1.0 - ev.data["panel_reduction"])
                info["radiation"] = max(info["radiation"], ev.data["radiation"])

            elif ev.name == "meteor_impact":
                info["o2_leak"] += ev.data["o2_leak_rate"]
                info["crew_injured"] += ev.data.get("crew_injured", 0)

            elif ev.name == "crew_emergency":
                info["crew_injured"] += ev.data.get("crew_down", 1)

            elif ev.name == "equipment_catastrophic":
                t = ev.data["target"]
                mult = ev.data["efficiency_mult"]
                if t == "co2_scrubber":
                    info["co2_scrubber_mult"] *= mult
                elif t == "water_recycler":
                    info["water_recycler_mult"] *= mult
                elif t == "o2_generator":
                    info["o2_gen_mult"] *= mult

        return info

    def track_routing(self, new_routing: str):
        if new_routing != self.last_routing:
            self.power_routing_changes += 1
            self.last_routing = new_routing

    def get_primary_event(self) -> Optional[ActiveEvent]:
        """Return the most severe active event for observation display."""
        if not self.active:
            return None
        severity_order = {"catastrophic": 3, "major": 2, "minor": 1}
        return max(self.active, key=lambda e: severity_order.get(e.severity, 0))
