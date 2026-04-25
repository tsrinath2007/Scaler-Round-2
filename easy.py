"""
Easy Task
=========
Scenario: A small web app with 2 CVEs and 1 outdated package.
No conflicts. Every upgrade is safe. Tests pass after upgrading.

Agent needs to:
1. Upgrade requests (CVE critical)
2. Upgrade flask (CVE high)
3. Upgrade numpy (outdated)
4. run_tests to confirm
"""

from dep_upgrade_env import Package


class EasyTask:

    def generate(self):
        packages = [
            Package(
                name="requests",
                current_version="2.20.0",
                latest_version="2.31.0",
                has_cve=True,
                cve_severity="critical",
                is_outdated=True,
            ),
            Package(
                name="flask",
                current_version="1.1.2",
                latest_version="3.0.2",
                has_cve=True,
                cve_severity="high",
                is_outdated=True,
            ),
            Package(
                name="numpy",
                current_version="1.21.0",
                latest_version="1.26.4",
                has_cve=False,
                is_outdated=True,
            ),
            Package(
                name="pytest",
                current_version="7.4.0",
                latest_version="7.4.0",
                has_cve=False,
                is_outdated=False,
            ),
        ]

        tests = {
            "test_http_requests": False,   # fails due to requests CVE
            "test_web_routes":    False,   # fails due to flask CVE
            "test_math_ops":      True,    # passes regardless
        }

        aux = {}
        return packages, tests, aux

    def on_upgrade(self, pkg, target, packages, tests, aux):
        if pkg.name == "requests" and target == "2.31.0":
            tests["test_http_requests"] = True
        if pkg.name == "flask" and target == "3.0.2":
            tests["test_web_routes"] = True

    def on_remove(self, name, packages, tests, aux):
        pass

    def run_tests(self, packages, tests, aux):
        # re-evaluate based on current package state
        req = next((p for p in packages if p.name == "requests"), None)
        flask = next((p for p in packages if p.name == "flask"), None)
        if req and not req.has_cve:
            tests["test_http_requests"] = True
        if flask and not flask.has_cve:
            tests["test_web_routes"] = True

    def grade(self, packages, tests, aux) -> tuple[float, dict]:
        scores = {}

        # 1. No CVEs remaining (40%)
        cve_pkgs = [p for p in packages if p.has_cve]
        scores["cves_resolved"] = round(1.0 - len(cve_pkgs) / 2, 4)

        # 2. No outdated packages (30%)
        outdated = [p for p in packages if p.is_outdated]
        total_outdated = 3
        scores["packages_updated"] = round(1.0 - len(outdated) / total_outdated, 4)

        # 3. All tests passing (30%)
        passing = sum(v for v in tests.values())
        scores["tests_passing"] = round(passing / len(tests), 4)

        total = (
            0.4 * scores["cves_resolved"] +
            0.3 * scores["packages_updated"] +
            0.3 * scores["tests_passing"]
        )
        return round(total, 4), scores
