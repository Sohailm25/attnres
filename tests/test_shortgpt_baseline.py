# ABOUTME: Validates static block-influence baseline helpers used by the OIH anchor lane.
# ABOUTME: Prevents pruning/normalization bugs from distorting dynamic-vs-static comparisons.

from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


class ShortGPTBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        from validation.shortgpt_baseline import (
            StaticBlockInfluenceEvalSummary,
            build_static_alpha_from_group_influences,
            mean_alpha_policy_from_sequences,
            select_best_static_evaluation,
            select_kept_groups_from_influences,
        )

        self.StaticBlockInfluenceEvalSummary = StaticBlockInfluenceEvalSummary
        self.build_static_alpha_from_group_influences = (
            build_static_alpha_from_group_influences
        )
        self.mean_alpha_policy_from_sequences = mean_alpha_policy_from_sequences
        self.select_best_static_evaluation = select_best_static_evaluation
        self.select_kept_groups_from_influences = select_kept_groups_from_influences

    def test_select_kept_groups_prunes_lowest_influence_groups(self) -> None:
        kept = self.select_kept_groups_from_influences(
            mean_group_influences=(-0.4, 0.3, 0.1, -0.2),
            prune_fraction=0.25,
        )
        self.assertEqual((1, 2, 3), kept)

    def test_build_static_alpha_from_group_influences_is_group_uniform(self) -> None:
        alpha = self.build_static_alpha_from_group_influences(
            group_indices=torch.tensor([0, 0, 1, 1, 2]),
            mean_group_influences=(0.10, -0.50, 0.30),
            prune_fraction=1 / 3,
        )
        self.assertEqual((5,), tuple(alpha.shape))
        self.assertAlmostEqual(1.0, float(alpha.sum().item()), places=6)
        self.assertAlmostEqual(0.25, float(alpha[0].item()), places=6)
        self.assertAlmostEqual(0.25, float(alpha[1].item()), places=6)
        self.assertAlmostEqual(0.0, float(alpha[2].item()), places=6)
        self.assertAlmostEqual(0.0, float(alpha[3].item()), places=6)
        self.assertAlmostEqual(0.5, float(alpha[4].item()), places=6)

    def test_build_static_alpha_rejects_pruning_all_groups(self) -> None:
        with self.assertRaises(ValueError):
            self.build_static_alpha_from_group_influences(
                group_indices=torch.tensor([0, 1, 2]),
                mean_group_influences=(0.1, 0.2, 0.3),
                prune_fraction=1.0,
            )

    def test_mean_alpha_policy_from_sequences_normalizes_mean(self) -> None:
        mean_policy = self.mean_alpha_policy_from_sequences(
            (
                (0.6, 0.4, 0.0),
                (0.2, 0.7, 0.1),
            )
        )
        self.assertAlmostEqual(1.0, sum(mean_policy), places=6)
        self.assertGreater(mean_policy[1], mean_policy[0])

    def test_select_best_static_evaluation_prefers_higher_improvement(self) -> None:
        evaluations = {
            "pruned": self.StaticBlockInfluenceEvalSummary(
                num_prompts=10,
                mean_uniform_loss=2.0,
                mean_static_loss=1.9,
                mean_improvement_over_uniform=0.1,
                positive_improvement_count=7,
            ),
            "pilot_mean_alpha": self.StaticBlockInfluenceEvalSummary(
                num_prompts=10,
                mean_uniform_loss=2.0,
                mean_static_loss=1.8,
                mean_improvement_over_uniform=0.2,
                positive_improvement_count=8,
            ),
        }
        label, _ = self.select_best_static_evaluation(evaluations)
        self.assertEqual("pilot_mean_alpha", label)


if __name__ == "__main__":
    unittest.main()
