# ABOUTME: Validates the saved pilot/confirmatory prompt registry and its access rules.
# ABOUTME: Prevents exploratory code paths from silently consuming confirmatory prompts.

import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PromptRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        from prompts.registry import (
            ConfirmatoryAccessError,
            load_prompt_registry,
            perturb_prompt_entry,
            resolve_prompt_entries,
        )

        self.ConfirmatoryAccessError = ConfirmatoryAccessError
        self.load_prompt_registry = load_prompt_registry
        self.perturb_prompt_entry = perturb_prompt_entry
        self.resolve_prompt_entries = resolve_prompt_entries

    def test_registry_file_exists(self) -> None:
        self.assertTrue((ROOT / "prompts" / "registry_v1.yaml").is_file())

    def test_phase1_collection_records_required_metadata(self) -> None:
        registry = self.load_prompt_registry()
        collection = registry.collections["oracle_alpha_phase1_v1"]

        self.assertEqual("sequence", collection.aggregation_unit)
        self.assertIn("softmax-constrained", collection.objective_families)
        self.assertEqual(
            (
                "uniform",
                "random_dirichlet",
                "magnitude_proportional",
                "last_layer_only",
            ),
            collection.null_models,
        )
        self.assertIn("paired_t_test", collection.statistical_tests)
        self.assertIn("bootstrap_1000", collection.statistical_tests)

        pilot_entries = self.resolve_prompt_entries(
            collection_id="oracle_alpha_phase1_v1",
            split="pilot",
            exploratory=True,
            registry=registry,
        )
        confirm_entries = self.resolve_prompt_entries(
            collection_id="oracle_alpha_phase1_v1",
            split="confirm",
            exploratory=False,
            registry=registry,
        )
        self.assertEqual(len(pilot_entries), len(confirm_entries))
        self.assertGreater(len(pilot_entries), 0)
        self.assertTrue(
            all(entry.split == "pilot" for entry in pilot_entries),
        )
        self.assertTrue(
            all(entry.split == "confirm" for entry in confirm_entries),
        )

    def test_oracle_alpha_pilot_entries_have_saved_paraphrases(self) -> None:
        registry = self.load_prompt_registry()
        pilot_entries = self.resolve_prompt_entries(
            collection_id="oracle_alpha_phase1_v1",
            split="pilot",
            exploratory=True,
            registry=registry,
        )

        self.assertTrue(
            all("prompt_paraphrase" in entry.perturbations for entry in pilot_entries),
        )
        paraphrased = self.perturb_prompt_entry(
            pilot_entries[0],
            perturbation_name="prompt_paraphrase",
        )
        self.assertNotEqual(pilot_entries[0].text, paraphrased.text)
        self.assertIn("prompt_paraphrase", paraphrased.tags)

    def test_confirmatory_access_rejects_exploratory_mode(self) -> None:
        registry = self.load_prompt_registry()

        with self.assertRaises(self.ConfirmatoryAccessError):
            self.resolve_prompt_entries(
                collection_id="oracle_alpha_phase1_v1",
                split="confirm",
                exploratory=True,
                registry=registry,
            )

    def test_export_script_allows_pilot_and_blocks_confirm_exploratory(self) -> None:
        script = ROOT / "scripts" / "export_prompt_split.py"

        pilot_result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--collection-id",
                "oracle_alpha_phase1_v1",
                "--split",
                "pilot",
                "--exploratory",
            ],
            capture_output=True,
            check=False,
            cwd=ROOT,
            text=True,
        )
        self.assertEqual(0, pilot_result.returncode)
        self.assertGreater(len(json.loads(pilot_result.stdout)), 0)

        confirm_result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--collection-id",
                "oracle_alpha_phase1_v1",
                "--split",
                "confirm",
                "--exploratory",
            ],
            capture_output=True,
            check=False,
            cwd=ROOT,
            text=True,
        )
        self.assertNotEqual(0, confirm_result.returncode)
        self.assertIn(
            "confirmatory prompts are locked",
            confirm_result.stderr,
        )


if __name__ == "__main__":
    unittest.main()
