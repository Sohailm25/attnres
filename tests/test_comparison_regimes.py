# ABOUTME: Exercises the preregistered oracle routing-regime comparison helpers.
# ABOUTME: Locks the regime schedule, sparsity summaries, and checkpointed summary writing before Gemma runs.

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from huggingface_hub import logging as huggingface_logging
import logging
from transformer_lens import HookedTransformer
from transformers.utils import logging as transformers_logging


ROOT = Path(__file__).resolve().parents[1]
huggingface_logging.set_verbosity_error()
logging.getLogger("huggingface_hub.file_download").setLevel(logging.CRITICAL)
transformers_logging.set_verbosity_error()


class ComparisonRegimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from validation.comparison_regimes import (
            build_regime_specs,
            summarize_regime_run,
            write_regime_comparison_summary,
        )

        cls.build_regime_specs = staticmethod(build_regime_specs)
        cls.summarize_regime_run = staticmethod(summarize_regime_run)
        cls.write_regime_comparison_summary = staticmethod(
            write_regime_comparison_summary
        )
        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(io.StringIO()):
                cls.model = HookedTransformer.from_pretrained(
                    "tiny-stories-1M",
                    device="cpu",
                    dtype="float32",
                )

    def test_build_regime_specs_uses_preregistered_topk_schedule(self) -> None:
        specs = self.build_regime_specs(53)

        self.assertEqual("softmax-constrained", specs[0].regime)
        self.assertEqual("unconstrained", specs[1].regime)
        self.assertEqual(
            (2, 4, 8, 13, 26),
            tuple(spec.top_k for spec in specs if spec.regime == "top-k"),
        )

    def test_summarize_regime_run_reports_positive_count_and_sparsity(self) -> None:
        payload = {
            "num_sequences": 2,
            "sequence_mean_improvement": 0.3,
            "bootstrap_interval": {"mean": 0.3, "lower": 0.2, "upper": 0.4},
            "null_model_mean_losses": {"uniform": 4.0},
            "sequence_results": [
                {
                    "uniform_loss": 4.0,
                    "optimized_loss": 3.5,
                    "best_alpha": [0.7, 0.2, 0.1],
                },
                {
                    "uniform_loss": 4.0,
                    "optimized_loss": 4.2,
                    "best_alpha": [0.3, 0.3, 0.4],
                },
            ],
        }

        summary = self.summarize_regime_run(
            payload,
            regime="softmax-constrained",
        )

        self.assertEqual(1, summary.positive_prompt_count)
        self.assertGreater(summary.mean_effective_sources, 1.0)
        self.assertGreaterEqual(summary.mean_gini, 0.0)
        self.assertLessEqual(summary.mean_gini, 1.0)
        self.assertGreater(summary.mean_top1_mass, 0.3)

    def test_write_regime_comparison_summary_writes_checkpoints_and_summary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "comparison"

            summary_path = self.write_regime_comparison_summary(
                model=self.model,
                collection_id="oracle_alpha_phase1_v1",
                split="confirm",
                exploratory=False,
                output_dir=output_dir,
                optimization_steps=1,
                learning_rate=0.1,
                seed=11,
            )

            payload = json.loads(summary_path.read_text())
            self.assertEqual(self.model.cfg.model_name, payload["model_name"])
            self.assertEqual("oracle_alpha_phase1_v1", payload["collection_id"])
            self.assertEqual("confirm", payload["split"])
            self.assertGreaterEqual(len(payload["regime_results"]), 3)
            self.assertTrue(
                (output_dir / "runs" / "softmax-constrained.json").is_file()
            )
            self.assertTrue((output_dir / "runs" / "unconstrained.json").is_file())


if __name__ == "__main__":
    unittest.main()
