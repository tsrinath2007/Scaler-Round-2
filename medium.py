"""
Medium Task
===========
Scenario: A data pipeline app. There are CVEs to fix AND version
conflicts — upgrading one package breaks another.

Specifically:
- pandas 1.3.0 has a CVE → needs upgrade to 2.0.0
- BUT numpy < 1.23 is required by an old scipy (1.7.0)
- scipy 1.7.0 conflicts with pandas 2.0.0 (needs scipy >= 1.9.0)
- celery 4.4.0 has a CVE → safe to upgrade to 5.3.4
- Agent must upgrade scipy FIRST, then pandas, then fix numpy

Correct order: upgrade scipy → upgrade pandas → upgrade numpy → run_tests
"""

from dep_upgrade_env import Package


class MediumTask:

    def generate(self):
        packages = [
            Package(
                name="pandas",
                current_version="1.3.0",
                latest_version="2.0.0",
                has_cve=True,
                cve_severity="high",
                is_outdated=True,
                is_conflicting=True,
                conflict_reason="requires scipy>=1.9.0 but scipy==1.7.0 installed",
            ),
            Package(
                name="scipy",
                current_version="1.7.0",
                latest_version="1.11.4",
                has_cve=False,
                is_outdated=True,
                is_conflicting=True,
                conflict_reason="too old for pandas>=2.0.0",
            ),
            Package(
                name="numpy",
                current_version="1.21.0",
                latest_version="1.26.4",
                has_cve=False,
                is_outdated=True,
            ),
            Package(
                name="celery",
                current_version="4.4.0",
                latest_version="5.3.4",
                has_cve=True,
                cve_severity="medium",
                is_outdated=True,
            ),
            Package(
                name="sqlalchemy",
                current_version="1.4.46",
                latest_version="2.0.23",
                has_cve=False,
                is_outdated=True,
            ),
        ]

        tests = {
            "test_data_pipeline":  False,  # fails: pandas CVE + conflict
            "test_async_tasks":    False,  # fails: celery CVE
            "test_db_queries":     True,   # sqlalchemy version doesn't break tests
            "test_scipy_compute":  False,  # fails: scipy too old
        }

        aux = {
            "scipy_upgraded": False,
            "pandas_upgraded": False,
        }
        return packages, tests, aux

    def on_upgrade(self, pkg, target, packages, tests, aux):
        if pkg.name == "scipy":
            ver = float(".".join(target.split(".")[:2]))
            if ver >= 1.9:
                pkg.is_conflicting = False
                pkg.conflict_reason = None
                aux["scipy_upgraded"] = True
                tests["test_scipy_compute"] = True
                # resolve pandas conflict now that scipy is upgraded
                pandas_pkg = next((p for p in packages if p.name == "pandas"), None)
                if pandas_pkg:
                    pandas_pkg.is_conflicting = False
                    pandas_pkg.conflict_reason = None

        if pkg.name == "pandas":
            if aux.get("scipy_upgraded"):
                pkg.has_cve = False
                pkg.cve_severity = None
                pkg.is_outdated = False
                pkg.is_conflicting = False
                aux["pandas_upgraded"] = True
                tests["test_data_pipeline"] = True
            else:
                # upgraded pandas without fixing scipy first — breaks things
                pkg.is_conflicting = True
                pkg.conflict_reason = "scipy still too old for pandas>=2.0.0"
                tests["test_data_pipeline"] = False
                tests["test_scipy_compute"] = False

        if pkg.name == "celery" and target == "5.3.4":
            tests["test_async_tasks"] = True

        if pkg.name == "numpy":
            pkg.is_outdated = False

    def on_remove(self, name, packages, tests, aux):
        pass

    def run_tests(self, packages, tests, aux):
        scipy = next((p for p in packages if p.name == "scipy"), None)
        pandas = next((p for p in packages if p.name == "pandas"), None)
        celery = next((p for p in packages if p.name == "celery"), None)

        if scipy and not scipy.is_conflicting:
            tests["test_scipy_compute"] = True
        if pandas and not pandas.has_cve and not pandas.is_conflicting:
            tests["test_data_pipeline"] = True
        if celery and not celery.has_cve:
            tests["test_async_tasks"] = True

    def grade(self, packages, tests, aux) -> tuple[float, dict]:
        scores = {}

        # 1. CVEs resolved (35%)
        cve_pkgs = [p for p in packages if p.has_cve]
        scores["cves_resolved"] = round(1.0 - len(cve_pkgs) / 2, 4)

        # 2. Conflicts resolved (30%)
        conflicting = [p for p in packages if p.is_conflicting]
        scores["conflicts_resolved"] = round(1.0 - len(conflicting) / 2, 4)

        # 3. Tests passing (35%)
        passing = sum(v for v in tests.values())
        scores["tests_passing"] = round(passing / len(tests), 4)

        total = (
            0.35 * scores["cves_resolved"] +
            0.30 * scores["conflicts_resolved"] +
            0.35 * scores["tests_passing"]
        )
        return round(total, 4), scores
