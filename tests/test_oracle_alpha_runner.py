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
        from validation.oracle_alpha_runner import run_oracle_alpha_collection

        cls.run_oracle_alpha_collection = staticmethod(run_oracle_alpha_collection)
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


if __name__ == "__main__":
    unittest.main()
