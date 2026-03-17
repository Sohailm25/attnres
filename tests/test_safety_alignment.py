# ABOUTME: Validates the refusal-feature discovery workflow helpers for the safety lane.
# ABOUTME: Keeps grouping, localization, and direction-separation logic testable before Gemma runs.

from __future__ import annotations

import unittest

import torch

from prompts import PromptEntry


class SafetyAlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        from validation.safety_alignment import (
            DirectionSeparationMetrics,
            SafetyLayerLocalizationSummary,
            group_safety_prompt_entries,
            build_layer_localization_summary,
            discover_normalized_direction,
            matches_refusal_marker,
            paired_projection_summary,
        )

        self.DirectionSeparationMetrics = DirectionSeparationMetrics
        self.SafetyLayerLocalizationSummary = SafetyLayerLocalizationSummary
        self.group_safety_prompt_entries = staticmethod(group_safety_prompt_entries)
        self.build_layer_localization_summary = staticmethod(
            build_layer_localization_summary
        )
        self.discover_normalized_direction = staticmethod(discover_normalized_direction)
        self.matches_refusal_marker = staticmethod(matches_refusal_marker)
        self.paired_projection_summary = staticmethod(paired_projection_summary)

    def test_group_safety_prompt_entries_requires_refusal_harmful_and_benign_roles(
        self,
    ) -> None:
        entries = (
            PromptEntry(
                prompt_id="sa-pilot-001-refusal",
                text="Refuse this harmful request.",
                split="pilot",
                tags=("safety_alignment", "refusal_expected"),
                perturbations={},
            ),
            PromptEntry(
                prompt_id="sa-pilot-001-harmful_context",
                text="Discuss why this harmful request should be refused.",
                split="pilot",
                tags=("safety_alignment", "harmful_context"),
                perturbations={},
            ),
            PromptEntry(
                prompt_id="sa-pilot-001-benign",
                text="Discuss a benign safety topic.",
                split="pilot",
                tags=("safety_alignment", "benign"),
                perturbations={},
            ),
        )

        groups = self.group_safety_prompt_entries(entries)

        self.assertEqual(1, len(groups))
        self.assertEqual("sa-pilot-001", groups[0].group_id)
        self.assertEqual("sa-pilot-001-refusal", groups[0].refusal_entry.prompt_id)
        self.assertEqual(
            "sa-pilot-001-harmful_context",
            groups[0].harmful_context_entry.prompt_id,
        )
        self.assertEqual("sa-pilot-001-benign", groups[0].benign_entry.prompt_id)

    def test_build_layer_localization_summary_selects_largest_mean_divergence_layer(
        self,
    ) -> None:
        positive = torch.tensor(
            [
                [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            ],
            dtype=torch.float32,
        )
        negative = torch.tensor(
            [
                [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]],
            ],
            dtype=torch.float32,
        )

        summary = self.build_layer_localization_summary(
            target_name="refusal",
            position_name="assistant_prefill",
            positive_residuals=positive,
            negative_residuals=negative,
        )

        self.assertEqual("refusal", summary.target_name)
        self.assertEqual("assistant_prefill", summary.position_name)
        self.assertEqual(1, summary.selected_layer)
        self.assertAlmostEqual(2.0, summary.mean_divergence_by_layer[1], places=6)

    def test_discover_normalized_direction_and_projection_summary_are_consistent(
        self,
    ) -> None:
        positive = torch.tensor(
            [
                [3.0, 0.0],
                [2.0, 0.0],
                [4.0, 0.0],
            ],
            dtype=torch.float32,
        )
        negative = torch.tensor(
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        direction = self.discover_normalized_direction(
            positive_residuals=positive,
            negative_residuals=negative,
        )
        metrics = self.paired_projection_summary(
            positive_residuals=positive,
            negative_residuals=negative,
            direction=direction,
        )

        self.assertAlmostEqual(1.0, float(direction.norm().item()), places=6)
        self.assertAlmostEqual(1.0, metrics.pair_accuracy, places=6)
        self.assertGreater(metrics.mean_margin, 0.0)

    def test_matches_refusal_marker_only_triggers_on_first_person_refusal(self) -> None:
        self.assertTrue(
            self.matches_refusal_marker(
                "I cannot help with that request because it would be unsafe."
            )
        )
        self.assertFalse(
            self.matches_refusal_marker(
                "Requests like that should be refused because they are unsafe."
            )
        )


if __name__ == "__main__":
    unittest.main()
