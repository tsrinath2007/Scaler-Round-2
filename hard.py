"""
Hard Task
=========
Scenario: A production ML service. Multiple CVEs, a diamond
dependency conflict, AND a breaking API change in the fix version.

Diamond conflict:
- transformers 4.30.0 requires tokenizers>=0.13,<0.14
- datasets 2.14.0 requires tokenizers>=0.14
- Both depend on tokenizers but need INCOMPATIBLE versions
- Solution: upgrade transformers to 4.36.0 (supports tokenizers>=0.14)

CVEs:
- cryptography 39.0.0 → critical CVE → must upgrade to 41.0.6
- pillow 9.0.0 → high CVE → must upgrade to 10.1.0

Breaking change:
- pillow 10.x removed Image.ANTIALIAS → tests fail until pin workaround applied
- Agent must upgrade pillow AND pin a compat shim OR update usage pattern

Correct strategy:
1. upgrade cryptography (safe, no conflicts)
2. upgrade transformers to 4.36.0 (resolves diamond)
3. upgrade datasets (now compatible)
4. upgrade tokenizers to 0.15.0
5. upgrade pillow to 10.1.0 (CVE fix)
6. pin pillow compat: add pillow-compat==1.0.0 OR pin pillow==9.5.0 (no CVE)
   Actually: correct answer is upgrade + run_tests (tests adapt)
7. run_tests
"""

from dep_upgrade_env import Package


class HardTask:

    def generate(self):
        packages = [
            Package(
                name="cryptography",
                current_version="39.0.0",
                latest_version="41.0.6",
                has_cve=True,
                cve_severity="critical",
                is_outdated=True,
            ),
            Package(
                name="pillow",
                current_version="9.0.0",
                latest_version="10.1.0",
                has_cve=True,
                cve_severity="high",
                is_outdated=True,
            ),
            Package(
                name="transformers",
                current_version="4.30.0",
                latest_version="4.36.0",
                has_cve=False,
                is_outdated=True,
                is_conflicting=True,
                conflict_reason="requires tokenizers<0.14 but datasets needs tokenizers>=0.14",
            ),
            Package(
                name="datasets",
                current_version="2.14.0",
                latest_version="2.16.0",
                has_cve=False,
                is_outdated=True,
                is_conflicting=True,
                conflict_reason="requires tokenizers>=0.14 but transformers needs tokenizers<0.14",
            ),
            Package(
                name="tokenizers",
                current_version="0.13.3",
                latest_version="0.15.0",
                has_cve=False,
                is_outdated=True,
                is_conflicting=True,
                conflict_reason="version pinned between conflicting transformers and datasets requirements",
            ),
            Package(
                name="torch",
                current_version="2.0.0",
                latest_version="2.1.2",
                has_cve=False,
                is_outdated=True,
                locked=True,   # pinned — production model requires exact version
            ),
        ]

        tests = {
            "test_encryption":       False,  # cryptography CVE
            "test_image_processing": False,  # pillow CVE + breaking change
            "test_model_inference":  False,  # transformers/tokenizers conflict
            "test_dataset_loading":  False,  # datasets/tokenizers conflict
            "test_torch_forward":    True,   # torch is fine (locked)
        }

        aux = {
            "transformers_upgraded": False,
            "tokenizers_upgraded": False,
            "datasets_upgraded": False,
            "pillow_upgraded": False,
            "diamond_resolved": False,
        }
        return packages, tests, aux

    def on_upgrade(self, pkg, target, packages, tests, aux):
        if pkg.name == "cryptography" and target == "41.0.6":
            tests["test_encryption"] = True

        if pkg.name == "transformers":
            ver = float(".".join(target.split(".")[:2]))
            if ver >= 4.36:
                pkg.is_conflicting = False
                pkg.conflict_reason = None
                aux["transformers_upgraded"] = True
                # check if diamond resolved
                self._check_diamond(packages, tests, aux)

        if pkg.name == "tokenizers":
            ver = float(".".join(target.split(".")[:2]))
            if ver >= 0.14:
                pkg.is_conflicting = False
                pkg.conflict_reason = None
                aux["tokenizers_upgraded"] = True
                self._check_diamond(packages, tests, aux)

        if pkg.name == "datasets":
            ver = float(".".join(target.split(".")[:2]))
            if ver >= 2.15:
                pkg.is_conflicting = False
                pkg.conflict_reason = None
                aux["datasets_upgraded"] = True
                self._check_diamond(packages, tests, aux)

        if pkg.name == "pillow":
            ver = float(".".join(target.split(".")[:2]))
            if ver >= 10.0:
                pkg.has_cve = False
                pkg.cve_severity = None
                pkg.is_outdated = False
                aux["pillow_upgraded"] = True
                # breaking change: tests fail until run_tests reconciles
                tests["test_image_processing"] = False  # needs run_tests

        if pkg.name == "torch":
            # torch is locked, should not be upgraded
            pkg.locked = True  # re-lock

    def _check_diamond(self, packages, tests, aux):
        if aux.get("transformers_upgraded") and aux.get("tokenizers_upgraded"):
            # diamond resolved
            tok = next((p for p in packages if p.name == "tokenizers"), None)
            tra = next((p for p in packages if p.name == "transformers"), None)
            dat = next((p for p in packages if p.name == "datasets"), None)
            if tok:
                tok.is_conflicting = False
            if tra:
                tra.is_conflicting = False
            if dat and aux.get("datasets_upgraded"):
                dat.is_conflicting = False
            aux["diamond_resolved"] = True
            tests["test_model_inference"] = True
            if aux.get("datasets_upgraded"):
                tests["test_dataset_loading"] = True

    def on_remove(self, name, packages, tests, aux):
        pass

    def run_tests(self, packages, tests, aux):
        crypt = next((p for p in packages if p.name == "cryptography"), None)
        pillow = next((p for p in packages if p.name == "pillow"), None)

        if crypt and not crypt.has_cve:
            tests["test_encryption"] = True

        # pillow breaking change reconciled after run_tests
        if pillow and not pillow.has_cve and aux.get("pillow_upgraded"):
            tests["test_image_processing"] = True

        if aux.get("diamond_resolved"):
            tests["test_model_inference"] = True
            if aux.get("datasets_upgraded"):
                tests["test_dataset_loading"] = True

    def grade(self, packages, tests, aux) -> tuple[float, dict]:
        scores = {}

        # 1. CVEs resolved (30%)
        cve_pkgs = [p for p in packages if p.has_cve]
        scores["cves_resolved"] = round(1.0 - len(cve_pkgs) / 2, 4)

        # 2. Diamond conflict resolved (30%)
        scores["diamond_resolved"] = 1.0 if aux.get("diamond_resolved") else 0.0

        # 3. Locked package respected (10%)
        torch_pkg = next((p for p in packages if p.name == "torch"), None)
        scores["locked_respected"] = 1.0 if (torch_pkg and torch_pkg.locked and
                                               torch_pkg.current_version == "2.0.0") else 0.0

        # 4. Tests passing (30%)
        passing = sum(v for v in tests.values())
        scores["tests_passing"] = round(passing / len(tests), 4)

        total = (
            0.30 * scores["cves_resolved"] +
            0.30 * scores["diamond_resolved"] +
            0.10 * scores["locked_respected"] +
            0.30 * scores["tests_passing"]
        )
        return round(total, 4), scores
