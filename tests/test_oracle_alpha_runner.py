# ABOUTME: Exercises the first development-model oracle-alpha execution path.
# ABOUTME: Ensures the runner consumes the saved registries and reports sequence-level controls.

import contextlib
import io
from pathlib import Path
import unittest
import warnings

from transformer_lens import HookedTransformer
from transformers.utils import logging as transformers_logging


ROOT = Path(__file__).resolve().parents[1]
warnings.filterwarnings(
    "ignore",
    message=r"`torch_dtype` is deprecated! Use `dtype` instead!",
)
transformers_logging.set_verbosity_error()


class OracleAlphaRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from validation.oracle_alpha_runner import (
            run_oracle_alpha_collection,
            run_oracle_alpha_predictiveness_check,
            run_oracle_alpha_stability_suite,
        )

        cls.run_oracle_alpha_collection = staticmethod(run_oracle_alpha_collection)
        cls.run_oracle_alpha_predictiveness_check = staticmethod(
            run_oracle_alpha_predictiveness_check
        )
        cls.run_oracle_alpha_stability_suite = staticmethod(
            run_oracle_alpha_stability_suite
        )
        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(io.StringIO()):
                cls.model = HookedTransformer.from_pretrained(
                    "tiny-stories-1M",
                    device="cpu",
                    dtype="float32",
                )

    def test_runner_consumes_saved_registries_on_tiny_model(self) -> None:
        summary = self.run_oracle_alpha_collection(
            model=self.model,
            collection_id="oracle_alpha_phase1_v1",
            split="pilot",
            exploratory=True,
            max_sequences=2,
            optimization_steps=6,
            learning_rate=0.1,
            seed=11,
        )

        self.assertEqual("oracle_alpha_phase1_v1", summary.collection_id)
        self.assertEqual("pilot", summary.split)
        self.assertTrue(summary.exploratory)
        self.assertEqual(2, summary.num_sequences)
        self.assertEqual(1000, summary.bootstrap_interval.num_resamples)
        self.assertIn("random_dirichlet", summary.null_model_mean_losses)
        self.assertIn("magnitude_proportional", summary.null_model_mean_losses)
        self.assertIn("last_layer_only", summary.null_model_mean_losses)

        for result in summary.sequence_results:
            self.assertLessEqual(result.optimized_loss, result.uniform_loss + 1e-6)
            self.assertGreater(result.num_sources, self.model.cfg.n_layers)
            self.assertEqual(result.num_sources, len(result.best_alpha))
            self.assertEqual(result.num_sources, len(result.final_alpha))

    def test_stability_suite_reports_restart_and_perturbation_metrics(self) -> None:
        summary = self.run_oracle_alpha_stability_suite(
            model=self.model,
            collection_id="oracle_alpha_phase1_v1",
            split="pilot",
            exploratory=True,
            max_sequences=3,
            optimization_steps=4,
            learning_rate=0.1,
            restart_seeds=(11, 17),
            resample_count=2,
            resample_size=2,
        )

        self.assertEqual(3, summary.base_run.num_sequences)
        self.assertEqual(2, len(summary.restart_runs))
        self.assertEqual(2, len(summary.resample_runs))
        self.assertIsNotNone(summary.paraphrase_run)
        self.assertGreaterEqual(
            summary.restart_metrics.mean_pairwise_js_divergence, 0.0
        )
        self.assertGreaterEqual(
            summary.paraphrase_metrics.mean_pairwise_js_divergence, 0.0
        )
        self.assertGreaterEqual(
            summary.resample_metrics.mean_pairwise_js_divergence, 0.0
        )
        self.assertLessEqual(summary.restart_metrics.mean_top1_source_agreement, 1.0)
        self.assertLessEqual(summary.paraphrase_metrics.mean_top1_source_agreement, 1.0)
        self.assertLessEqual(summary.resample_metrics.mean_top1_source_agreement, 1.0)
        self.assertGreater(summary.restart_per_sequence_metrics.matched_prompt_count, 0)
        self.assertGreaterEqual(
            summary.restart_per_sequence_metrics.mean_pairwise_js_divergence, 0.0
        )
        self.assertLessEqual(
            summary.restart_per_sequence_metrics.mean_top1_source_agreement, 1.0
        )
        self.assertIsNotNone(summary.paraphrase_per_sequence_metrics)
        self.assertGreater(
            summary.paraphrase_per_sequence_metrics.matched_prompt_count,
            0,
        )
        self.assertIsNotNone(summary.resample_per_sequence_metrics)
        self.assertGreater(
            summary.resample_per_sequence_metrics.matched_prompt_count,
            0,
        )

    def test_different_restart_seeds_change_alpha_outputs(self) -> None:
        first = self.run_oracle_alpha_collection(
            model=self.model,
            collection_id="oracle_alpha_phase1_v1",
            split="pilot",
            exploratory=True,
            max_sequences=1,
            optimization_steps=3,
            learning_rate=0.1,
            seed=11,
        )
        second = self.run_oracle_alpha_collection(
            model=self.model,
            collection_id="oracle_alpha_phase1_v1",
            split="pilot",
            exploratory=True,
            max_sequences=1,
            optimization_steps=3,
            learning_rate=0.1,
            seed=17,
        )

        self.assertNotEqual(
            first.sequence_results[0].final_alpha,
            second.sequence_results[0].final_alpha,
        )

    def test_predictiveness_check_uses_pilot_to_confirm_splits(self) -> None:
        summary = self.run_oracle_alpha_predictiveness_check(
            model=self.model,
            collection_id="oracle_alpha_phase1_v1",
            max_train_sequences=3,
            max_eval_sequences=2,
            optimization_steps=4,
            learning_rate=0.1,
            seed=11,
            regularization_grid=(1e-3, 1e-1, 1.0),
            candidate_feature_sources=(
                "mean_pooled_h_1[t]_resid_post_layer_0",
                "mean_pooled_h_1[t]_plus_h_4[t]_concat",
            ),
        )

        self.assertEqual("pilot", summary.train_run.split)
        self.assertEqual("confirm", summary.eval_run.split)
        self.assertFalse(summary.eval_run.exploratory)
        self.assertEqual("oracle_alpha_logit_vector", summary.target)
        self.assertEqual(3, summary.predictiveness_summary.num_train_examples)
        self.assertEqual(2, summary.predictiveness_summary.num_eval_examples)
        self.assertEqual("r_squared", summary.tuning_primary_metric)
        self.assertEqual("mean_js_divergence", summary.tuning_secondary_metric)
        self.assertEqual(2, len(summary.candidate_feature_summaries))
        self.assertIn(
            summary.feature_source,
            (
                "mean_pooled_h_1[t]_resid_post_layer_0",
                "mean_pooled_h_1[t]_plus_h_4[t]_concat",
            ),
        )
        self.assertIn(summary.selected_regularization_strength, (1e-3, 1e-1, 1.0))
        self.assertEqual(2, len(summary.eval_predictions))
        self.assertTrue(
            all(
                candidate.feature_source
                in (
                    "mean_pooled_h_1[t]_resid_post_layer_0",
                    "mean_pooled_h_1[t]_plus_h_4[t]_concat",
                )
                for candidate in summary.candidate_feature_summaries
            )
        )
        self.assertTrue(
            all(
                candidate.tuning_primary_metric == "r_squared"
                and candidate.tuning_secondary_metric == "mean_js_divergence"
                for candidate in summary.candidate_feature_summaries
            )
        )
        for prediction in summary.eval_predictions:
            self.assertEqual(prediction.num_sources, len(prediction.predicted_alpha))
            self.assertAlmostEqual(1.0, sum(prediction.predicted_alpha), places=6)
            self.assertGreaterEqual(prediction.predicted_loss, 0.0)
            self.assertGreaterEqual(prediction.js_divergence_to_oracle, 0.0)

    def test_predictiveness_check_accepts_token_aware_feature_sources(self) -> None:
        summary = self.run_oracle_alpha_predictiveness_check(
            model=self.model,
            collection_id="oracle_alpha_phase1_v1",
            max_train_sequences=3,
            max_eval_sequences=2,
            optimization_steps=4,
            learning_rate=0.1,
            seed=11,
            regularization_grid=(1e-3, 1e-1, 1.0),
            candidate_feature_sources=(
                "position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat",
                "start_mid_end_h_4[t]_resid_post_layer_3_concat",
            ),
        )

        self.assertIn(
            summary.feature_source,
            (
                "position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat",
                "start_mid_end_h_4[t]_resid_post_layer_3_concat",
            ),
        )
        self.assertEqual(2, len(summary.candidate_feature_summaries))
        self.assertTrue(
            all(
                candidate.feature_source
                in (
                    "position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat",
                    "start_mid_end_h_4[t]_resid_post_layer_3_concat",
                )
                for candidate in summary.candidate_feature_summaries
            )
        )

    def test_predictiveness_check_accepts_prompt_level_and_hybrid_sources(self) -> None:
        summary = self.run_oracle_alpha_predictiveness_check(
            model=self.model,
            collection_id="oracle_alpha_phase1_v1",
            max_train_sequences=3,
            max_eval_sequences=2,
            optimization_steps=4,
            learning_rate=0.1,
            seed=11,
            regularization_grid=(1e-3, 1e-1, 1.0),
            candidate_feature_sources=(
                "prompt_shape_scalar_features_v1",
                "mean_pooled_token_embedding",
                "position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_plus_prompt_shape_scalar_features_v1_concat",
                "position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_plus_mean_pooled_token_embedding_concat",
            ),
        )

        self.assertIn(
            summary.feature_source,
            (
                "prompt_shape_scalar_features_v1",
                "mean_pooled_token_embedding",
                "position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_plus_prompt_shape_scalar_features_v1_concat",
                "position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_plus_mean_pooled_token_embedding_concat",
            ),
        )
        self.assertEqual(4, len(summary.candidate_feature_summaries))


if __name__ == "__main__":
    unittest.main()
