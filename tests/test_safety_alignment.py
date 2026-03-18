# ABOUTME: Validates the refusal-feature discovery workflow helpers for the safety lane.
# ABOUTME: Keeps grouping, localization, and direction-separation logic testable before Gemma runs.

from __future__ import annotations

import unittest

import torch

from prompts import PromptEntry


class SafetyAlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        from validation.safety_alignment import (
            ContinuationPreferenceSummary,
            DirectionSeparationMetrics,
            DirectionProjectionSummary,
            InterventionBehaviorSummary,
            MediatorPartitionSummary,
            ProjectionTrajectorySummary,
            _expected_behavior_mode,
            SafetyLayerLocalizationSummary,
            TrajectoryInterventionComparisonSummary,
            behavior_matches_expected,
            classify_behavior_mode,
            _projection_intervention_hook,
            build_continuation_preference_summary,
            build_direction_projection_summary,
            build_intervention_behavior_summary,
            build_mediator_partition_summary,
            build_projection_trajectory_summary,
            build_trajectory_intervention_comparison_summary,
            group_safety_prompt_entries,
            build_layer_localization_summary,
            discover_normalized_direction,
            matches_policy_style_marker,
            matches_refusal_marker,
            paired_projection_summary,
            replace_direction_projection,
        )

        self.ContinuationPreferenceSummary = ContinuationPreferenceSummary
        self.DirectionSeparationMetrics = DirectionSeparationMetrics
        self.DirectionProjectionSummary = DirectionProjectionSummary
        self.InterventionBehaviorSummary = InterventionBehaviorSummary
        self.MediatorPartitionSummary = MediatorPartitionSummary
        self.ProjectionTrajectorySummary = ProjectionTrajectorySummary
        self.expected_behavior_mode = staticmethod(_expected_behavior_mode)
        self.SafetyLayerLocalizationSummary = SafetyLayerLocalizationSummary
        self.TrajectoryInterventionComparisonSummary = (
            TrajectoryInterventionComparisonSummary
        )
        self.behavior_matches_expected = staticmethod(behavior_matches_expected)
        self.classify_behavior_mode = staticmethod(classify_behavior_mode)
        self.projection_intervention_hook = staticmethod(_projection_intervention_hook)
        self.build_continuation_preference_summary = staticmethod(
            build_continuation_preference_summary
        )
        self.build_direction_projection_summary = staticmethod(
            build_direction_projection_summary
        )
        self.build_intervention_behavior_summary = staticmethod(
            build_intervention_behavior_summary
        )
        self.build_mediator_partition_summary = staticmethod(
            build_mediator_partition_summary
        )
        self.build_projection_trajectory_summary = staticmethod(
            build_projection_trajectory_summary
        )
        self.build_trajectory_intervention_comparison_summary = staticmethod(
            build_trajectory_intervention_comparison_summary
        )
        self.group_safety_prompt_entries = staticmethod(group_safety_prompt_entries)
        self.build_layer_localization_summary = staticmethod(
            build_layer_localization_summary
        )
        self.discover_normalized_direction = staticmethod(discover_normalized_direction)
        self.matches_policy_style_marker = staticmethod(matches_policy_style_marker)
        self.matches_refusal_marker = staticmethod(matches_refusal_marker)
        self.paired_projection_summary = staticmethod(paired_projection_summary)
        self.replace_direction_projection = staticmethod(replace_direction_projection)

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

    def test_matches_policy_style_marker_captures_institutional_policy_language(
        self,
    ) -> None:
        self.assertTrue(
            self.matches_policy_style_marker(
                "## Policy Note: Counterfeiting Currency\n\n"
                "We cannot provide assistance or information related to counterfeiting."
            )
        )
        self.assertFalse(
            self.matches_policy_style_marker(
                "At a high level, requests like that are dangerous and should be refused."
            )
        )
        self.assertTrue(
            self.matches_policy_style_marker(
                "## Policy Note: Counterfeiting Currency\n\n"
                "It is strictly prohibited to assist in any activity related to counterfeiting currency."
            )
        )

    def test_expected_behavior_mode_uses_tags_for_refusal_style_non_refusal(
        self,
    ) -> None:
        refusal_entry = PromptEntry(
            prompt_id="sa-pilot-001-refusal",
            text="Refuse this harmful request.",
            split="pilot",
            tags=("safety_alignment", "refusal_expected"),
            perturbations={},
        )
        standard_non_refusal_entry = PromptEntry(
            prompt_id="sa-pilot-001-benign",
            text="Discuss a benign safety topic.",
            split="pilot",
            tags=("safety_alignment", "non_refusal_expected"),
            perturbations={},
        )
        refusal_style_non_refusal_entry = PromptEntry(
            prompt_id="sa2-pilot-001-harmful_context",
            text="Write a refusal message and redirect.",
            split="pilot",
            tags=(
                "safety_alignment",
                "non_refusal_expected",
                "refusal_style_non_refusal",
            ),
            perturbations={},
        )

        self.assertEqual(
            "refusal_like_expected",
            self.expected_behavior_mode(refusal_entry),
        )
        self.assertEqual(
            "non_refusal_expected",
            self.expected_behavior_mode(standard_non_refusal_entry),
        )
        self.assertEqual(
            "refusal_like_expected",
            self.expected_behavior_mode(refusal_style_non_refusal_entry),
        )

    def test_expected_behavior_mode_prefers_policy_style_tag(self) -> None:
        policy_style_entry = PromptEntry(
            prompt_id="sa2-pilot-002-benign",
            text="Draft a short safety policy response.",
            split="pilot",
            tags=(
                "safety_alignment",
                "benign",
                "non_refusal_expected",
                "policy_style_expected",
                "refusal_style_non_refusal",
            ),
            perturbations={},
        )

        self.assertEqual(
            "policy_style_expected",
            self.expected_behavior_mode(policy_style_entry),
        )

    def test_classify_behavior_mode_distinguishes_refusal_policy_and_explanation(
        self,
    ) -> None:
        self.assertEqual(
            "refusal_like_expected",
            self.classify_behavior_mode("I cannot help with that request."),
        )
        self.assertEqual(
            "policy_style_expected",
            self.classify_behavior_mode(
                "Subject: Regarding Your Request\n\n"
                "We cannot provide assistance or information on that topic."
            ),
        )
        self.assertEqual(
            "policy_style_expected",
            self.classify_behavior_mode(
                "## Policy Note: Counterfeiting Currency\n\n"
                "It is strictly prohibited to assist in any activity related to counterfeiting currency."
            ),
        )

    def test_policy_style_expected_accepts_refusal_like_but_not_plain_explanation(
        self,
    ) -> None:
        self.assertTrue(
            self.behavior_matches_expected(
                expected_behavior_mode="policy_style_expected",
                observed_behavior_mode="policy_style_expected",
            )
        )
        self.assertTrue(
            self.behavior_matches_expected(
                expected_behavior_mode="policy_style_expected",
                observed_behavior_mode="refusal_like_expected",
            )
        )
        self.assertFalse(
            self.behavior_matches_expected(
                expected_behavior_mode="policy_style_expected",
                observed_behavior_mode="non_refusal_expected",
            )
        )
        self.assertEqual(
            "non_refusal_expected",
            self.classify_behavior_mode(
                "At a high level, the request is dangerous and should be refused."
            ),
        )

    def test_build_direction_projection_summary_uses_direction_coefficients(
        self,
    ) -> None:
        positive = torch.tensor(
            [
                [4.0, 1.0],
                [6.0, 2.0],
            ],
            dtype=torch.float32,
        )
        negative = torch.tensor(
            [
                [1.0, 5.0],
                [2.0, 6.0],
            ],
            dtype=torch.float32,
        )
        direction = torch.tensor([1.0, 0.0], dtype=torch.float32)

        summary = self.build_direction_projection_summary(
            positive_residuals=positive,
            negative_residuals=negative,
            direction=direction,
        )

        self.assertAlmostEqual(5.0, summary.positive_mean, places=6)
        self.assertAlmostEqual(1.5, summary.negative_mean, places=6)
        self.assertAlmostEqual(3.5, summary.mean_gap, places=6)

    def test_replace_direction_projection_preserves_orthogonal_component(self) -> None:
        residual = torch.tensor([3.0, 4.0], dtype=torch.float32)
        direction = torch.tensor([2.0, 0.0], dtype=torch.float32)

        replaced = self.replace_direction_projection(
            residual=residual,
            direction=direction,
            target_projection=1.0,
        )

        self.assertAlmostEqual(1.0, float(replaced[0].item()), places=6)
        self.assertAlmostEqual(4.0, float(replaced[1].item()), places=6)

    def test_build_intervention_behavior_summary_reports_delta_and_flip_rate(
        self,
    ) -> None:
        summary = self.build_intervention_behavior_summary(
            baseline_refusal_like=(True, True, False, False),
            intervened_refusal_like=(False, True, False, True),
        )

        self.assertEqual(4, summary.num_prompts)
        self.assertAlmostEqual(0.5, summary.baseline_refusal_rate, places=6)
        self.assertAlmostEqual(0.5, summary.intervened_refusal_rate, places=6)
        self.assertAlmostEqual(0.0, summary.refusal_rate_delta, places=6)
        self.assertAlmostEqual(0.5, summary.changed_prompt_fraction, places=6)

    def test_projection_intervention_hook_accepts_transformerlens_keyword_signature(
        self,
    ) -> None:
        hook = self.projection_intervention_hook(
            direction=torch.tensor([1.0, 0.0], dtype=torch.float32),
            position_index=0,
            target_projection=1.0,
        )
        residual = torch.tensor([[[3.0, 4.0]]], dtype=torch.float32)

        updated = hook(residual, hook=None)

        self.assertAlmostEqual(1.0, float(updated[0, 0, 0].item()), places=6)
        self.assertAlmostEqual(4.0, float(updated[0, 0, 1].item()), places=6)

    def test_build_continuation_preference_summary_reports_margin_shift(self) -> None:
        summary = self.build_continuation_preference_summary(
            baseline_positive_logprobs=(-1.0, -1.5),
            baseline_negative_logprobs=(-2.0, -2.5),
            intervened_positive_logprobs=(-0.8, -1.2),
            intervened_negative_logprobs=(-2.1, -2.2),
        )

        self.assertAlmostEqual(-1.25, summary.baseline_positive_mean_logprob, places=6)
        self.assertAlmostEqual(-2.25, summary.baseline_negative_mean_logprob, places=6)
        self.assertAlmostEqual(1.0, summary.baseline_margin, places=6)
        self.assertAlmostEqual(-1.0, summary.intervened_positive_mean_logprob, places=6)
        self.assertAlmostEqual(
            -2.15, summary.intervened_negative_mean_logprob, places=6
        )
        self.assertAlmostEqual(1.15, summary.intervened_margin, places=6)
        self.assertAlmostEqual(0.15, summary.margin_delta, places=6)

    def test_build_projection_trajectory_summary_projects_mean_depth_profile(
        self,
    ) -> None:
        residuals = torch.tensor(
            [
                [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
                [[2.0, 0.0], [4.0, 0.0], [6.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        direction = torch.tensor([1.0, 0.0], dtype=torch.float32)

        summary = self.build_projection_trajectory_summary(
            label="active",
            residuals_by_layer=residuals,
            direction=direction,
            selected_layer=1,
        )

        self.assertEqual("active", summary.label)
        self.assertEqual((1.5, 3.0, 4.5), summary.mean_projection_by_layer)
        self.assertAlmostEqual(3.0, summary.selected_layer_mean, places=6)
        self.assertAlmostEqual(4.5, summary.final_layer_mean, places=6)

    def test_build_mediator_partition_summary_splits_prompt_roles_by_threshold(
        self,
    ) -> None:
        summary = self.build_mediator_partition_summary(
            prompt_ids=(
                "sa-confirm-001-refusal",
                "sa-confirm-001-harmful_context",
                "sa-confirm-001-benign",
                "sa-confirm-002-refusal",
            ),
            roles=("refusal", "harmful_context", "benign", "refusal"),
            projection_scores=(2.0, 0.5, -0.5, 1.5),
            threshold=1.0,
        )

        self.assertEqual(1.0, summary.threshold)
        self.assertEqual(
            ("sa-confirm-001-refusal", "sa-confirm-002-refusal"),
            summary.active_prompt_ids,
        )
        self.assertEqual(
            ("sa-confirm-001-harmful_context", "sa-confirm-001-benign"),
            summary.inactive_prompt_ids,
        )
        self.assertEqual({"refusal": 2}, summary.active_role_counts)
        self.assertEqual(
            {"benign": 1, "harmful_context": 1},
            summary.inactive_role_counts,
        )
        self.assertAlmostEqual(1.75, summary.active_mean_projection, places=6)
        self.assertAlmostEqual(0.0, summary.inactive_mean_projection, places=6)

    def test_build_trajectory_intervention_comparison_summary_reports_layer_deltas(
        self,
    ) -> None:
        baseline = torch.tensor(
            [
                [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
                [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        intervened = torch.tensor(
            [
                [[0.5, 0.0], [1.0, 0.0], [1.5, 0.0]],
                [[0.5, 0.0], [1.0, 0.0], [1.5, 0.0]],
            ],
            dtype=torch.float32,
        )
        direction = torch.tensor([1.0, 0.0], dtype=torch.float32)

        summary = self.build_trajectory_intervention_comparison_summary(
            arm_name="refusal_suppression_on_refusal_prompts",
            direction_name="refusal",
            position_name="assistant_prefill",
            target_role="refusal",
            selected_layer=1,
            baseline_residuals_by_layer=baseline,
            intervened_residuals_by_layer=intervened,
            direction=direction,
        )

        self.assertEqual(
            "refusal_suppression_on_refusal_prompts",
            summary.arm_name,
        )
        self.assertAlmostEqual(-1.0, summary.selected_layer_delta, places=6)
        self.assertAlmostEqual(-1.5, summary.final_layer_delta, places=6)


if __name__ == "__main__":
    unittest.main()
