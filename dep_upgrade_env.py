"""
DepUpgradeEnv — Dependency Upgrade Environment
================================================
An OpenEnv-compliant RL environment where an AI agent upgrades
outdated, vulnerable, or conflicting Python package dependencies.

step(action)  → observation, reward, done, info
reset()       → observation
state()       → dict
"""

from pydantic import BaseModel
from typing import Optional, Any
import copy


# ── Pydantic Models ──────────────────────────────────────────────────────────

class Package(BaseModel):
    name: str
    current_version: str
    latest_version: str
    has_cve: bool = False
    cve_severity: Optional[str] = None   # low / medium / high / critical
    is_outdated: bool = False
    is_conflicting: bool = False
    conflict_reason: Optional[str] = None
    locked: bool = False                  # pinned intentionally, do not touch


class Observation(BaseModel):
    task_id: str
    step: int
    packages: list[Package]
    test_results: dict[str, bool]         # test_name → passed
    issues_remaining: list[str]
    score_so_far: float
    message: str = ""


class Action(BaseModel):
    action_type: str          # upgrade | pin | remove | run_tests | validate | skip
    package: Optional[str] = None
    version: Optional[str] = None        # target version for upgrade/pin


class Reward(BaseModel):
    score: float
    breakdown: dict[str, float]
    done: bool
    message: str


# ── DepUpgradeEnv ────────────────────────────────────────────────────────────

class DepUpgradeEnv:
    """
    OpenEnv-compliant environment for dependency upgrade tasks.
    The agent receives a requirements.txt with issues and must
    resolve them without breaking the test suite.
    """

    TASKS = ["easy", "medium", "hard"]

    def __init__(self, task_id: str = "easy"):
        assert task_id in self.TASKS, f"task_id must be one of {self.TASKS}"
        self.task_id = task_id
        self._step = 0
        self._max_steps = 25
        self._packages: list[Package] = []
        self._tests: dict[str, bool] = {}
        self._aux: dict = {}

        from tasks.easy import EasyTask
        from tasks.medium import MediumTask
        from tasks.hard import HardTask
        self._task_cls = {"easy": EasyTask, "medium": MediumTask, "hard": HardTask}[task_id]

    # ── OpenEnv API ──────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        self._step = 0
        task = self._task_cls()
        self._packages, self._tests, self._aux = task.generate()
        self._original_packages = copy.deepcopy(self._packages)
        return self._observe("Environment reset. Upgrade dependencies to resolve all issues.")

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        self._step += 1
        msg = ""

        try:
            msg = self._apply(action)
        except Exception as e:
            msg = f"Action failed: {e}"

        score, breakdown = self._task_cls().grade(self._packages, self._tests, self._aux)
        done = score >= 0.95 or self._step >= self._max_steps

        reward = Reward(
            score=round(score, 4),
            breakdown=breakdown,
            done=done,
            message=f"Step {self._step}: {msg}",
        )
        return self._observe(msg), reward, done, {"step": self._step}

    def state(self) -> dict:
        return {
            "task_id": self.task_id,
            "step": self._step,
            "max_steps": self._max_steps,
            "packages": [p.model_dump() for p in self._packages],
            "tests": self._tests,
            "aux": self._aux,
        }

    # ── Internals ────────────────────────────────────────────────────────────

    def _observe(self, message: str = "") -> Observation:
        issues = self._get_issues()
        score, breakdown = self._task_cls().grade(self._packages, self._tests, self._aux)
        return Observation(
            task_id=self.task_id,
            step=self._step,
            packages=copy.deepcopy(self._packages),
            test_results=copy.copy(self._tests),
            issues_remaining=issues,
            score_so_far=round(score, 4),
            message=message,
        )

    def _get_issues(self) -> list[str]:
        issues = []
        for p in self._packages:
            if p.has_cve:
                issues.append(f"{p.name}: CVE ({p.cve_severity})")
            if p.is_outdated:
                issues.append(f"{p.name}: outdated ({p.current_version} → {p.latest_version})")
            if p.is_conflicting:
                issues.append(f"{p.name}: conflict — {p.conflict_reason}")
        failing = [k for k, v in self._tests.items() if not v]
        for t in failing:
            issues.append(f"test failing: {t}")
        return issues

    def _get_pkg(self, name: str) -> Package:
        for p in self._packages:
            if p.name == name:
                return p
        raise ValueError(f"Package '{name}' not found")

    def _apply(self, action: Action) -> str:
        at = action.action_type

        if at == "upgrade":
            pkg = self._get_pkg(action.package)
            if pkg.locked:
                return f"'{pkg.name}' is locked — cannot upgrade"
            target = action.version or pkg.latest_version
            pkg.current_version = target
            # resolve CVE / outdated if target matches latest
            if target == pkg.latest_version:
                pkg.has_cve = False
                pkg.cve_severity = None
                pkg.is_outdated = False
            # task-specific side effects
            self._task_cls().on_upgrade(pkg, target, self._packages, self._tests, self._aux)
            return f"Upgraded {pkg.name} to {target}"

        elif at == "pin":
            pkg = self._get_pkg(action.package)
            pkg.locked = True
            pkg.current_version = action.version or pkg.current_version
            return f"Pinned {pkg.name} at {pkg.current_version}"

        elif at == "remove":
            name = action.package
            before = len(self._packages)
            self._packages = [p for p in self._packages if p.name != name]
            if len(self._packages) < before:
                self._task_cls().on_remove(name, self._packages, self._tests, self._aux)
                return f"Removed {name}"
            return f"Package {name} not found"

        elif at == "run_tests":
            self._task_cls().run_tests(self._packages, self._tests, self._aux)
            passing = sum(v for v in self._tests.values())
            total = len(self._tests)
            return f"Tests run: {passing}/{total} passing"

        elif at == "validate":
            issues = self._get_issues()
            if not issues:
                return "All issues resolved!"
            return f"{len(issues)} issues remaining: {issues[0]}..."

        elif at == "skip":
            return "Skipped"

        else:
            raise ValueError(f"Unknown action_type: {at!r}")
