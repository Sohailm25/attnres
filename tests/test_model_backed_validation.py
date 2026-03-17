# ABOUTME: Exercises the TransformerLens-backed Phase 1 reconstruction smoke checks.
# ABOUTME: Guards real cache extraction, residual accounting, and final-norm-aware logit reconstruction.

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


class ModelBackedValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from validation.model_backed import model_backed_reconstruction_metrics

        cls.compute_metrics = staticmethod(model_backed_reconstruction_metrics)
        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(io.StringIO()):
                cls.model = HookedTransformer.from_pretrained(
                    "tiny-stories-1M",
                    device="cpu",
                    dtype="float32",
                )

    def test_model_backed_reconstruction_matches_original_logits(self) -> None:
        metrics = self.compute_metrics(
            model=self.model,
            prompt="The cat sat on the",
        )

        self.assertIn("tinystories", metrics.model_name.lower())
        self.assertIn("1m", metrics.model_name.lower())
        self.assertGreater(metrics.num_sources, self.model.cfg.n_layers)
        self.assertIn("embed", metrics.source_labels)
        self.assertLess(metrics.final_residual_max_abs_error, 1e-5)
        self.assertLess(metrics.uniform_logits_max_abs_error, 1e-4)

    def test_model_backed_residual_identities_hold_per_layer(self) -> None:
        metrics = self.compute_metrics(
            model=self.model,
            prompt="Once upon a time there was",
        )

        self.assertEqual(self.model.cfg.n_layers, len(metrics.layer_metrics))
        for layer_metrics in metrics.layer_metrics:
            self.assertLess(layer_metrics.resid_post_max_abs_error, 1e-5)
            if layer_metrics.resid_mid_max_abs_error is not None:
                self.assertLess(layer_metrics.resid_mid_max_abs_error, 1e-5)


if __name__ == "__main__":
    unittest.main()
