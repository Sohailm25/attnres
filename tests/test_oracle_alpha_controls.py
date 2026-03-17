# ABOUTME: Validates the saved Phase 1 oracle-alpha control plan and its helper math.
# ABOUTME: Prevents claim-bearing work from skipping stability, predictiveness, or explicit MIB handling.

import tempfile
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class OracleAlphaControlTests(unittest.TestCase):
    def setUp(self) -> None:
        from validation.oracle_alpha_controls import (
            MIBPlan,
            bootstrap_mean_confidence_interval,
            compare_predictiveness_metric_values,
            linear_alpha_predictiveness_summary,
            load_oracle_alpha_control_registry,
            mean_pairwise_js_divergence,
            mean_top1_source_agreement,
            mean_topk_jaccard_similarity,
            ridge_alpha_predictiveness_summary,
        )

        self.MIBPlan = MIBPlan
        self.bootstrap_mean_confidence_interval = bootstrap_mean_confidence_interval
        self.compare_predictiveness_metric_values = compare_predictiveness_metric_values
        self.linear_alpha_predictiveness_summary = linear_alpha_predictiveness_summary
        self.load_oracle_alpha_control_registry = load_oracle_alpha_control_registry
        self.mean_pairwise_js_divergence = mean_pairwise_js_divergence
        self.mean_top1_source_agreement = mean_top1_source_agreement
        self.mean_topk_jaccard_similarity = mean_topk_jaccard_similarity
        self.ridge_alpha_predictiveness_summary = ridge_alpha_predictiveness_summary

    def test_control_plan_file_exists(self) -> None:
        self.assertTrue((ROOT / "configs" / "oracle_alpha_controls_v1.yaml").is_file())

    def test_control_plan_loads_and_matches_prompt_registry(self) -> None:
        registry = self.load_oracle_alpha_control_registry()
        plan = registry.plans["oracle_alpha_phase1_v1"]

        self.assertEqual("sequence", plan.aggregation_unit)
        self.assertEqual(1000, plan.bootstrap_resamples)
        self.assertEqual("pilot", plan.predictiveness.train_split)
        self.assertEqual("confirm", plan.predictiveness.eval_split)
        self.assertEqual("planned", plan.mib_anchor.mib_status)
        self.assertEqual("MIB", plan.mib_anchor.benchmark)
        self.assertIn("runner", plan.mib_anchor.rationale.lower())
        self.assertIn("random_dirichlet", plan.null_models)
        self.assertIn("prompt_paraphrase", plan.stability_suite.prompt_perturbations)
        self.assertEqual("r_squared", plan.predictiveness.primary_metric)
        self.assertEqual("mean_js_divergence", plan.predictiveness.secondary_metric)

    def test_compare_predictiveness_metric_values_respects_direction(self) -> None:
        self.assertGreater(
            self.compare_predictiveness_metric_values(
                metric_name="r_squared",
                left=0.4,
                right=0.2,
            ),
            0.0,
        )
        self.assertGreater(
            self.compare_predictiveness_metric_values(
                metric_name="mean_js_divergence",
                left=0.1,
                right=0.3,
            ),
            0.0,
        )

    def test_bootstrap_mean_confidence_interval_contains_observed_mean(self) -> None:
        interval = self.bootstrap_mean_confidence_interval(
            [0.1, 0.2, 0.4, 0.5],
            num_resamples=1000,
            confidence_level=0.95,
            seed=11,
        )

        self.assertLessEqual(interval.lower, interval.mean)
        self.assertGreaterEqual(interval.upper, interval.mean)
        self.assertEqual(1000, interval.num_resamples)

    def test_mean_pairwise_js_divergence_is_zero_for_identical_distributions(
        self,
    ) -> None:
        distributions = [
            [0.2, 0.3, 0.5],
            [0.2, 0.3, 0.5],
            [0.2, 0.3, 0.5],
        ]

        self.assertAlmostEqual(
            0.0,
            self.mean_pairwise_js_divergence(distributions),
        )

    def test_mean_topk_jaccard_similarity_scores_expected_overlap(self) -> None:
        distributions = [
            [0.60, 0.30, 0.10],
            [0.50, 0.40, 0.10],
            [0.10, 0.30, 0.60],
        ]

        self.assertAlmostEqual(
            1.0 / 3.0,
            self.mean_topk_jaccard_similarity(distributions, k=1),
        )

    def test_mean_top1_source_agreement_scores_expected_overlap(self) -> None:
        distributions = [
            [0.60, 0.30, 0.10],
            [0.50, 0.40, 0.10],
            [0.10, 0.30, 0.60],
        ]

        self.assertAlmostEqual(
            1.0 / 3.0,
            self.mean_top1_source_agreement(distributions),
        )

    def test_linear_alpha_predictiveness_summary_is_high_on_linear_signal(self) -> None:
        summary = self.linear_alpha_predictiveness_summary(
            train_features=[
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            train_targets=[
                [0.70, 0.30],
                [0.80, 0.20],
                [0.20, 0.80],
                [0.30, 0.70],
            ],
            eval_features=[
                [0.25, 0.75],
                [0.75, 0.25],
            ],
            eval_targets=[
                [0.35, 0.65],
                [0.65, 0.35],
            ],
        )

        self.assertGreater(summary.r_squared, 0.95)
        self.assertLess(summary.mean_js_divergence, 0.02)
        self.assertEqual(4, summary.num_train_examples)
        self.assertEqual(2, summary.num_eval_examples)

    def test_ridge_alpha_predictiveness_summary_is_high_on_linear_signal(self) -> None:
        summary = self.ridge_alpha_predictiveness_summary(
            train_features=[
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            train_targets=[
                [0.70, 0.30],
                [0.80, 0.20],
                [0.20, 0.80],
                [0.30, 0.70],
            ],
            eval_features=[
                [0.25, 0.75],
                [0.75, 0.25],
            ],
            eval_targets=[
                [0.35, 0.65],
                [0.65, 0.35],
            ],
            regularization_strength=1e-3,
        )

        self.assertGreater(summary.r_squared, 0.90)
        self.assertLess(summary.mean_js_divergence, 0.05)
        self.assertEqual(4, summary.num_train_examples)
        self.assertEqual(2, summary.num_eval_examples)

    def test_loader_rejects_omitted_mib_plan_without_rationale(self) -> None:
        bad_config = """
version: 1
registry_id: bad
plans:
  oracle_alpha_phase1_v1:
    collection_id: oracle_alpha_phase1_v1
    aggregation_unit: sequence
    bootstrap_resamples: 1000
    null_models:
      - uniform
    stability_suite:
      restart_seeds: [11, 17]
      prompt_perturbations: [prompt_resample, prompt_paraphrase]
      report_metrics: [mean_pairwise_js_divergence]
    predictiveness:
      train_collection_id: oracle_alpha_phase1_v1
      train_split: pilot
      eval_collection_id: oracle_alpha_phase1_v1
      eval_split: confirm
      model_family: ridge_regression
      target: oracle_alpha_vector
      features: cached_early_hidden_state_summary
      primary_metric: r_squared
      secondary_metric: mean_js_divergence
      failure_action: weaken_claim
    mib_anchor:
      mib_status: omitted
      revisit_trigger: after_runner_exists
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(bad_config)
            path = Path(handle.name)

        try:
            with self.assertRaises(ValueError):
                self.load_oracle_alpha_control_registry(path=path)
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
