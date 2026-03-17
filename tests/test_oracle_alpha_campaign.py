# ABOUTME: Exercises the checkpointed oracle-alpha campaign runner for prereg-scale runs.
# ABOUTME: Ensures prompt-level checkpoints and feature caches make larger runs resumable.

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import warnings

from huggingface_hub import logging as huggingface_logging
import logging
from transformer_lens import HookedTransformer
from transformers.utils import logging as transformers_logging


ROOT = Path(__file__).resolve().parents[1]
warnings.filterwarnings(
    "ignore",
    message=r"`torch_dtype` is deprecated! Use `dtype` instead!",
)
huggingface_logging.set_verbosity_error()
logging.getLogger("huggingface_hub.file_download").setLevel(logging.CRITICAL)
transformers_logging.set_verbosity_error()


class OracleAlphaCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from validation.oracle_alpha_campaign import (
            run_oracle_alpha_predictiveness_campaign,
        )

        cls.run_oracle_alpha_predictiveness_campaign = staticmethod(
            run_oracle_alpha_predictiveness_campaign
        )
        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(io.StringIO()):
                cls.model = HookedTransformer.from_pretrained(
                    "tiny-stories-1M",
                    device="cpu",
                    dtype="float32",
                )

    def test_campaign_writes_manifest_summary_and_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            summary = self.run_oracle_alpha_predictiveness_campaign(
                model=self.model,
                collection_id="oracle_alpha_phase1_v1",
                output_dir=output_dir,
                max_train_sequences=2,
                max_eval_sequences=1,
                optimization_steps=4,
                learning_rate=0.1,
                seed=11,
                regularization_grid=(1e-3, 1e-1),
                candidate_feature_sources=(
                    "position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat",
                ),
                candidate_target_names=("oracle_alpha_logit_vector",),
            )

            self.assertEqual(2, summary.train_run.num_sequences)
            self.assertEqual(1, summary.eval_run.num_sequences)
            self.assertTrue((output_dir / "campaign_manifest.json").is_file())
            self.assertTrue((output_dir / "predictiveness_summary.json").is_file())
            self.assertTrue(
                any((output_dir / "checkpoints" / "oracle_runs" / "pilot").iterdir())
            )
            self.assertTrue(
                any((output_dir / "checkpoints" / "oracle_runs" / "confirm").iterdir())
            )
            self.assertTrue(
                any(
                    (output_dir / "checkpoints" / "feature_vectors" / "pilot").iterdir()
                )
            )
            manifest = json.loads((output_dir / "campaign_manifest.json").read_text())
            self.assertEqual("oracle_alpha_phase1_v1", manifest["collection_id"])
            self.assertEqual(
                ["position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat"],
                manifest["candidate_feature_sources"],
            )

    def test_campaign_resume_uses_saved_oracle_and_feature_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            self.run_oracle_alpha_predictiveness_campaign(
                model=self.model,
                collection_id="oracle_alpha_phase1_v1",
                output_dir=output_dir,
                max_train_sequences=2,
                max_eval_sequences=1,
                optimization_steps=4,
                learning_rate=0.1,
                seed=11,
                regularization_grid=(1e-3, 1e-1),
                candidate_feature_sources=(
                    "position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat",
                ),
                candidate_target_names=("oracle_alpha_logit_vector",),
            )

            with (
                mock.patch(
                    "validation.oracle_alpha_runner._optimize_sequence",
                    side_effect=AssertionError("oracle checkpoints should be reused"),
                ),
                mock.patch(
                    "validation.oracle_alpha_runner._feature_vector_for_source",
                    side_effect=AssertionError("feature checkpoints should be reused"),
                ),
            ):
                resumed_summary = self.run_oracle_alpha_predictiveness_campaign(
                    model=self.model,
                    collection_id="oracle_alpha_phase1_v1",
                    output_dir=output_dir,
                    max_train_sequences=2,
                    max_eval_sequences=1,
                    optimization_steps=4,
                    learning_rate=0.1,
                    seed=11,
                    regularization_grid=(1e-3, 1e-1),
                    candidate_feature_sources=(
                        "position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat",
                    ),
                    candidate_target_names=("oracle_alpha_logit_vector",),
                )

            self.assertEqual(2, resumed_summary.train_run.num_sequences)
            self.assertEqual(1, resumed_summary.eval_run.num_sequences)


if __name__ == "__main__":
    unittest.main()
