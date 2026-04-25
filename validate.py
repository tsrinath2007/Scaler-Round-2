"""
validate.py — Pre-submission validator for DepUpgradeEnv
Run: python validate.py
     python validate.py --url http://localhost:7860
"""

import sys
import argparse

try:
    import requests as req_lib
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from dep_upgrade_env import DepUpgradeEnv, Action, Observation, Reward

TASKS = ["easy", "medium", "hard"]
results = []


def check(name, passed, detail=""):
    status = "  ✅ PASS" if passed else "  ❌ FAIL"
    print(f"{status} — {name}" + (f"\n         {detail}" if detail else ""))
    results.append((name, passed))
    return passed


def validate_files():
    print("\n[1] Required Files")
    import os
    for f in ["env.py", "server.py", "inference.py", "openenv.yaml",
              "Dockerfile", "requirements.txt", "README.md", "validate.py",
              "tasks/easy.py", "tasks/medium.py", "tasks/hard.py"]:
        check(f"{f} exists", os.path.isfile(f))


def validate_spec():
    print("\n[2] OpenEnv Spec Compliance")

    try:
        import yaml
        with open("openenv.yaml") as f:
            meta = yaml.safe_load(f)
        required = {"name", "version", "tasks", "observation_space", "action_space", "reward"}
        missing = required - set(meta.keys())
        check("openenv.yaml valid", not missing, f"missing: {missing}" if missing else "")
    except Exception as e:
        check("openenv.yaml valid", False, str(e))

    for task_id in TASKS:
        try:
            env = DepUpgradeEnv(task_id=task_id)
            obs = env.reset()
            check(f"reset() returns Observation [{task_id}]", isinstance(obs, Observation))
        except Exception as e:
            check(f"reset() [{task_id}]", False, str(e))

        try:
            env = DepUpgradeEnv(task_id=task_id)
            env.reset()
            obs, reward, done, info = env.step(Action(action_type="validate"))
            ok = all([isinstance(obs, Observation), isinstance(reward, Reward),
                      isinstance(done, bool), isinstance(info, dict)])
            check(f"step() returns correct tuple [{task_id}]", ok)
        except Exception as e:
            check(f"step() [{task_id}]", False, str(e))

        try:
            env = DepUpgradeEnv(task_id=task_id)
            env.reset()
            s = env.state()
            check(f"state() returns dict [{task_id}]", isinstance(s, dict))
        except Exception as e:
            check(f"state() [{task_id}]", False, str(e))


def validate_graders():
    print("\n[3] Grader Quality")

    check("3+ tasks defined", len(TASKS) >= 3)

    for task_id in TASKS:
        try:
            env = DepUpgradeEnv(task_id=task_id)
            env.reset()
            _, reward, _, _ = env.step(Action(action_type="validate"))
            in_range = 0.0 <= reward.score <= 1.0
            check(f"score in [0,1] [{task_id}]", in_range, f"got {reward.score}")
        except Exception as e:
            check(f"score in [0,1] [{task_id}]", False, str(e))

        try:
            env = DepUpgradeEnv(task_id=task_id)
            env.reset()
            _, r1, _, _ = env.step(Action(action_type="validate"))
            env.reset()
            _, r2, _, _ = env.step(Action(action_type="validate"))
            check(f"grader deterministic [{task_id}]", r1.score == r2.score)
        except Exception as e:
            check(f"grader deterministic [{task_id}]", False, str(e))

        # score changes after a good action
        try:
            env = DepUpgradeEnv(task_id=task_id)
            obs = env.reset()
            _, r_before, _, _ = env.step(Action(action_type="validate"))
            # upgrade first CVE package
            cve_pkg = next((p for p in obs.packages if p.has_cve), None)
            if cve_pkg:
                env.step(Action(action_type="upgrade", package=cve_pkg.name,
                                version=cve_pkg.latest_version))
                _, r_after, _, _ = env.step(Action(action_type="validate"))
                check(f"reward increases after fix [{task_id}]",
                      r_after.score >= r_before.score,
                      f"{r_before.score} → {r_after.score}")
            else:
                check(f"reward increases after fix [{task_id}]", True, "no CVE pkg to test")
        except Exception as e:
            check(f"reward increases after fix [{task_id}]", False, str(e))


def validate_env_design():
    print("\n[4] Environment Design")

    for task_id in TASKS:
        try:
            env = DepUpgradeEnv(task_id=task_id)
            obs1 = env.reset()
            env.step(Action(action_type="validate"))
            obs2 = env.reset()
            check(f"reset() cleans step counter [{task_id}]",
                  obs1.step == 0 and obs2.step == 0)
        except Exception as e:
            check(f"reset() clean [{task_id}]", False, str(e))

        try:
            env = DepUpgradeEnv(task_id=task_id)
            obs = env.reset()
            required = {"task_id","step","packages","test_results","issues_remaining"}
            missing = required - set(obs.model_dump().keys())
            check(f"observation has required fields [{task_id}]", not missing,
                  str(missing) if missing else "")
        except Exception as e:
            check(f"observation fields [{task_id}]", False, str(e))


def validate_server(base_url):
    print(f"\n[5] Live Server ({base_url})")
    if not HAS_REQUESTS:
        print("  ⚠️  requests not installed — skipping")
        return

    try:
        r = req_lib.get(f"{base_url}/health", timeout=5)
        check("/health → 200", r.status_code == 200)
    except Exception as e:
        check("/health → 200", False, str(e))

    for task_id in TASKS:
        try:
            r = req_lib.post(f"{base_url}/reset", json={"task_id": task_id}, timeout=5)
            check(f"/reset → 200 [{task_id}]", r.status_code == 200)
        except Exception as e:
            check(f"/reset [{task_id}]", False, str(e))

    try:
        r = req_lib.post(f"{base_url}/step", json={
            "task_id": "easy",
            "action": {"action_type": "validate"}
        }, timeout=5)
        check("/step → 200", r.status_code == 200)
        if r.status_code == 200:
            data = r.json()
            check("/step has reward.score", "score" in data.get("reward", {}))
    except Exception as e:
        check("/step", False, str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None)
    args = parser.parse_args()

    print("=" * 55)
    print("  DepUpgradeEnv — Pre-Submission Validator")
    print("=" * 55)

    validate_files()
    validate_spec()
    validate_graders()
    validate_env_design()
    if args.url:
        validate_server(args.url)

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    failed = [n for n, ok in results if not ok]

    print(f"\n{'='*55}")
    print(f"  RESULT: {passed}/{total} checks passed")
    if failed:
        print("\n  Failed:")
        for n in failed:
            print(f"    ✗ {n}")
        sys.exit(1)
    else:
        print("  🎉 All checks passed — ready to submit!")
        sys.exit(0)


if __name__ == "__main__":
    main()
