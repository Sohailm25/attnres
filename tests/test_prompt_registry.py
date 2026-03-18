# ABOUTME: Validates the saved pilot/confirmatory prompt registry and its access rules.
# ABOUTME: Prevents exploratory code paths from silently consuming confirmatory prompts.

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml


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
        self.assertTrue((ROOT / "prompts" / "registry_v5.yaml").is_file())

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
        self.assertEqual(256, len(pilot_entries))
        self.assertEqual(1024, len(confirm_entries))
        self.assertTrue(
            all(entry.split == "pilot" for entry in pilot_entries),
        )
        self.assertTrue(
            all(entry.split == "confirm" for entry in confirm_entries),
        )
        pilot_stratum_counts = {}
        confirm_stratum_counts = {}
        for entry in pilot_entries:
            stratum_tag = next(tag for tag in entry.tags if tag.startswith("stratum_"))
            pilot_stratum_counts[stratum_tag] = (
                pilot_stratum_counts.get(stratum_tag, 0) + 1
            )
        for entry in confirm_entries:
            stratum_tag = next(tag for tag in entry.tags if tag.startswith("stratum_"))
            confirm_stratum_counts[stratum_tag] = (
                confirm_stratum_counts.get(stratum_tag, 0) + 1
            )
        self.assertEqual(
            {
                "stratum_factual_recall": 64,
                "stratum_reasoning_math": 64,
                "stratum_code_procedural": 64,
                "stratum_general_text": 64,
            },
            pilot_stratum_counts,
        )
        self.assertEqual(
            {
                "stratum_factual_recall": 256,
                "stratum_reasoning_math": 256,
                "stratum_code_procedural": 256,
                "stratum_general_text": 256,
            },
            confirm_stratum_counts,
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

    def test_tool_breakage_entries_have_target_text_metadata(self) -> None:
        registry = self.load_prompt_registry()
        pilot_entries = self.resolve_prompt_entries(
            collection_id="tool_breakage_factual_recall_v1",
            split="pilot",
            exploratory=True,
            registry=registry,
        )
        confirm_entries = self.resolve_prompt_entries(
            collection_id="tool_breakage_factual_recall_v1",
            split="confirm",
            exploratory=False,
            registry=registry,
        )

        self.assertEqual(8, len(pilot_entries))
        self.assertEqual(8, len(confirm_entries))
        self.assertTrue(all(entry.target_text for entry in pilot_entries))
        self.assertTrue(all(entry.target_text for entry in confirm_entries))

    def test_tool_breakage_entries_have_saved_subcategory_tags(self) -> None:
        registry = self.load_prompt_registry()
        entries = self.resolve_prompt_entries(
            collection_id="tool_breakage_factual_recall_v1",
            split="pilot",
            exploratory=True,
            registry=registry,
        ) + self.resolve_prompt_entries(
            collection_id="tool_breakage_factual_recall_v1",
            split="confirm",
            exploratory=False,
            registry=registry,
        )

        for entry in entries:
            subcategory_tags = [
                tag for tag in entry.tags if tag.startswith("subcategory_")
            ]
            self.assertEqual(1, len(subcategory_tags), entry.prompt_id)

    def test_tool_breakage_v2_entries_balance_matched_factual_families(self) -> None:
        registry = self.load_prompt_registry()
        pilot_entries = self.resolve_prompt_entries(
            collection_id="tool_breakage_factual_recall_v2",
            split="pilot",
            exploratory=True,
            registry=registry,
        )
        confirm_entries = self.resolve_prompt_entries(
            collection_id="tool_breakage_factual_recall_v2",
            split="confirm",
            exploratory=False,
            registry=registry,
        )

        self.assertEqual(8, len(pilot_entries))
        self.assertEqual(16, len(confirm_entries))

        expected_families = {
            "subcategory_capital_fact",
            "subcategory_element_symbol",
            "subcategory_author_fact",
            "subcategory_moon_fact",
        }
        for entries, expected_count in ((pilot_entries, 2), (confirm_entries, 4)):
            counts = {family: 0 for family in expected_families}
            for entry in entries:
                subcategory_tags = [
                    tag for tag in entry.tags if tag.startswith("subcategory_")
                ]
                self.assertEqual(1, len(subcategory_tags), entry.prompt_id)
                self.assertIn("matched_routing_family", entry.tags)
                counts[subcategory_tags[0]] += 1
            self.assertEqual(
                {family: expected_count for family in expected_families},
                counts,
            )

    def test_tool_breakage_v3_entries_balance_matched_factual_families(self) -> None:
        registry = self.load_prompt_registry()
        pilot_entries = self.resolve_prompt_entries(
            collection_id="tool_breakage_factual_recall_v3",
            split="pilot",
            exploratory=True,
            registry=registry,
        )
        confirm_entries = self.resolve_prompt_entries(
            collection_id="tool_breakage_factual_recall_v3",
            split="confirm",
            exploratory=False,
            registry=registry,
        )

        self.assertEqual(16, len(pilot_entries))
        self.assertEqual(32, len(confirm_entries))

        expected_families = {
            "subcategory_capital_fact",
            "subcategory_element_symbol",
            "subcategory_author_fact",
            "subcategory_moon_fact",
        }
        for entries, expected_count in ((pilot_entries, 4), (confirm_entries, 8)):
            counts = {family: 0 for family in expected_families}
            for entry in entries:
                subcategory_tags = [
                    tag for tag in entry.tags if tag.startswith("subcategory_")
                ]
                self.assertEqual(1, len(subcategory_tags), entry.prompt_id)
                self.assertIn("matched_routing_family", entry.tags)
                counts[subcategory_tags[0]] += 1
            self.assertEqual(
                {family: expected_count for family in expected_families},
                counts,
            )

    def test_safety_alignment_entries_form_matched_refusal_workflow_groups(
        self,
    ) -> None:
        registry = self.load_prompt_registry()
        pilot_entries = self.resolve_prompt_entries(
            collection_id="safety_refusal_discovery_v1",
            split="pilot",
            exploratory=True,
            registry=registry,
        )
        confirm_entries = self.resolve_prompt_entries(
            collection_id="safety_refusal_discovery_v1",
            split="confirm",
            exploratory=False,
            registry=registry,
        )

        self.assertEqual(18, len(pilot_entries))
        self.assertEqual(18, len(confirm_entries))
        self.assertTrue(
            all("safety_alignment" in entry.tags for entry in pilot_entries)
        )
        self.assertTrue(
            all(
                entry.prompt_id.endswith(("-refusal", "-harmful_context", "-benign"))
                for entry in confirm_entries
            )
        )

    def test_safety_alignment_surface_v2_includes_refusal_style_non_refusal_prompts(
        self,
    ) -> None:
        registry = self.load_prompt_registry()
        pilot_entries = self.resolve_prompt_entries(
            collection_id="safety_refusal_surface_v2",
            split="pilot",
            exploratory=True,
            registry=registry,
        )
        confirm_entries = self.resolve_prompt_entries(
            collection_id="safety_refusal_surface_v2",
            split="confirm",
            exploratory=False,
            registry=registry,
        )

        self.assertEqual(18, len(pilot_entries))
        self.assertEqual(18, len(confirm_entries))
        non_refusal_entries = [
            entry
            for entry in pilot_entries + confirm_entries
            if not entry.prompt_id.endswith("-refusal")
        ]
        self.assertTrue(
            any(
                "refusal_style_non_refusal" in entry.tags
                for entry in non_refusal_entries
            )
        )
        self.assertTrue(
            any("policy_style_expected" in entry.tags for entry in non_refusal_entries)
        )

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

    def test_registry_v5_generator_reproduces_committed_registry(self) -> None:
        script = ROOT / "scripts" / "generate_registry_v5.py"
        committed_path = ROOT / "prompts" / "registry_v5.yaml"

        with tempfile.TemporaryDirectory() as temp_dir:
            generated_path = Path(temp_dir) / "registry_v5.yaml"
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--output",
                    str(generated_path),
                ],
                capture_output=True,
                check=False,
                cwd=ROOT,
                text=True,
            )
            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertTrue(generated_path.is_file())
            self.assertEqual(
                yaml.safe_load(committed_path.read_text()),
                yaml.safe_load(generated_path.read_text()),
            )

    def test_registry_v5_general_text_prompts_avoid_known_article_regressions(
        self,
    ) -> None:
        registry = self.load_prompt_registry()
        prompts = registry.collections["oracle_alpha_phase1_v1"].prompt_entries
        general_text_entries = [
            entry for entry in prompts if "stratum_general_text" in entry.tags
        ]

        self.assertTrue(general_text_entries)
        self.assertTrue(
            all("a lemons" not in entry.text for entry in general_text_entries)
        )
        self.assertTrue(
            all("a optician" not in entry.text for entry in general_text_entries)
        )


if __name__ == "__main__":
    unittest.main()
