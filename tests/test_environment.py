"""
Tests for the Closed-Loop Life Support OpenEnv environment.
Run with: python -m pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from env.environment import LifeSupportEnv
from env.models import Action, Observation, Reward, EnvironmentState
from tasks.graders import grade_episode, grade_easy, grade_medium, grade_hard


class TestEnvironmentBasics:
    def test_reset_returns_observation(self):
        env = LifeSupportEnv("task_easy", seed=0)
        obs = env.reset()
        assert isinstance(obs, Observation)
        assert 0 <= obs.o2_percent <= 30
        assert 0 <= obs.co2_ppm <= 5000
        assert obs.crew_size == 3

    def test_step_returns_correct_types(self):
        env = LifeSupportEnv("task_easy", seed=0)
        env.reset()
        action = Action()
        obs, reward, done, info = env.step(action)
        assert isinstance(obs, Observation)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert -1.0 <= reward <= 1.0

    def test_state_returns_full_state(self):
        env = LifeSupportEnv("task_easy", seed=0)
        env.reset()
        state = env.state()
        assert isinstance(state, EnvironmentState)
        assert state.task_id == "task_easy"
        assert state.step_count == 0

    def test_step_after_done_raises(self):
        env = LifeSupportEnv("task_easy", seed=0)
        env.reset()
        # Run to completion
        for _ in range(24):
            _, _, done, _ = env.step(Action())
            if done:
                break
        if done:
            with pytest.raises(RuntimeError):
                env.step(Action())

    def test_reset_clears_state(self):
        env = LifeSupportEnv("task_easy", seed=0)
        env.reset()
        env.step(Action())
        env.step(Action())
        env.reset()
        state = env.state()
        assert state.step_count == 0
        assert state.total_reward == 0.0

    def test_all_task_ids(self):
        for task_id in ["task_easy", "task_medium", "task_hard"]:
            env = LifeSupportEnv(task_id, seed=42)
            obs = env.reset()
            assert obs is not None

    def test_invalid_task_id(self):
        with pytest.raises(ValueError):
            LifeSupportEnv("task_impossible")

    def test_reproducible_with_seed(self):
        env1 = LifeSupportEnv("task_easy", seed=7)
        env2 = LifeSupportEnv("task_easy", seed=7)
        obs1 = env1.reset()
        obs2 = env2.reset()
        assert obs1.co2_ppm == obs2.co2_ppm
        assert obs1.o2_percent == obs2.o2_percent

    def test_reward_range(self):
        env = LifeSupportEnv("task_easy", seed=0)
        env.reset()
        for _ in range(24):
            _, reward, done, _ = env.step(Action())
            assert -1.0 <= reward <= 1.0
            if done:
                break


class TestPhysicsPlausibility:
    def test_plant_growth_increases_o2(self):
        env = LifeSupportEnv("task_easy", seed=0)
        env.reset()
        # Run with max plant growth for a few steps
        action_grow = Action(increase_plant_growth=1.0, adjust_oxygen=0.0, crew_activity=0.1)
        o2_readings = []
        for _ in range(5):
            obs, _, done, _ = env.step(action_grow)
            o2_readings.append(obs.o2_percent)
            if done:
                break
        # O2 should generally increase (or at least not crash) with heavy plant growth
        assert o2_readings[-1] > 15.0

    def test_high_crew_activity_consumes_more_o2(self):
        env1 = LifeSupportEnv("task_easy", seed=99)
        env2 = LifeSupportEnv("task_easy", seed=99)
        obs1_start = env1.reset()
        obs2_start = env2.reset()

        for _ in range(5):
            o1, _, d1, _ = env1.step(Action(crew_activity=1.0, increase_plant_growth=0.0))
            o2, _, d2, _ = env2.step(Action(crew_activity=0.0, increase_plant_growth=0.0))
            if d1 or d2:
                break

        # High activity should consume more O2 (lower final O2)
        assert o1.o2_percent <= o2.o2_percent + 0.1  # Allow small tolerance

    def test_water_recycling_recovers_water(self):
        env = LifeSupportEnv("task_medium", seed=0)
        env.reset()
        # Run several steps with max recycling
        for _ in range(10):
            obs, _, done, _ = env.step(Action(recycle_water=1.0))
            if done:
                break
        # Water should not have crashed (recycling helps)
        assert obs.water_liters > 0


class TestGraders:
    def _build_trajectory(self, task_id: str, steps: int, health: float = 0.9,
                           o2: float = 21.0, co2: float = 500.0, water: float = 200.0):
        traj = []
        for i in range(steps):
            traj.append({
                "observation": {
                    "o2_percent": o2,
                    "co2_ppm": co2,
                    "water_liters": water,
                    "food_kg": 20.0,
                    "crew_health": health,
                    "crew_size": 3,
                    "plant_growth_rate": 0.5,
                    "water_recycling_rate": 0.5,
                    "day": (i // 24) + 1,
                    "power_budget": 0.6,
                },
                "reward": 0.5,
                "done": i == steps - 1,
                "done_reason": None,
            })
        return traj

    def test_easy_grader_perfect_episode(self):
        traj = self._build_trajectory("task_easy", 24, water=200.0)
        result = grade_easy(traj)
        assert result.score >= 0.85
        assert result.passed

    def test_easy_grader_failed_episode(self):
        traj = self._build_trajectory("task_easy", 5, health=0.1, o2=14.0, co2=3500.0, water=2.0)
        result = grade_easy(traj)
        assert result.score < 0.5

    def test_medium_grader_returns_score(self):
        traj = self._build_trajectory("task_medium", 168, health=0.85, water=180.0)
        result = grade_medium(traj)
        assert 0.0 <= result.score <= 1.0

    def test_hard_grader_returns_score(self):
        traj = self._build_trajectory("task_hard", 720, health=0.9)
        result = grade_hard(traj)
        assert 0.0 <= result.score <= 1.0

    def test_graders_deterministic(self):
        traj = self._build_trajectory("task_easy", 24)
        r1 = grade_easy(traj)
        r2 = grade_easy(traj)
        assert r1.score == r2.score

    def test_grade_episode_dispatcher(self):
        for task_id in ["task_easy", "task_medium", "task_hard"]:
            steps = {"task_easy": 24, "task_medium": 168, "task_hard": 720}[task_id]
            traj = self._build_trajectory(task_id, steps)
            result = grade_episode(task_id, traj)
            assert 0.0 <= result.score <= 1.0

    def test_empty_trajectory_handled(self):
        result = grade_easy([])
        assert result.score == 0.0
        assert not result.passed

    def test_score_ordering(self):
        """Better episodes should score higher."""
        good_traj = self._build_trajectory("task_easy", 24, health=1.0, o2=21.0, co2=400.0)
        bad_traj = self._build_trajectory("task_easy", 10, health=0.3, o2=17.0, co2=2000.0)
        good_score = grade_easy(good_traj).score
        bad_score = grade_easy(bad_traj).score
        assert good_score > bad_score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
