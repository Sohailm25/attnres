# ABOUTME: Exercises the minimal custom tuned-lens training and evaluation path.
# ABOUTME: Keeps the Gemma-2 tool-breakage baseline grounded in held-out raw-vs-tuned comparisons before routing is introduced.

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from transformer_lens import HookedTransformer

from prompts import PromptEntry


ROOT = Path(__file__).resolve().parents[1]


class TunedLensTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from validation.tuned_lens import (
            LowRankAffineResidualLens,
            ResidualTranslationExamples,
            TunedLensLayerMetric,
            TunedLensViabilitySummary,
            collect_residual_translation_examples,
            evaluate_low_rank_residual_lens,
            fit_low_rank_residual_lens,
            save_tuned_lens_artifacts,
        )

        cls.LowRankAffineResidualLens = LowRankAffineResidualLens
        cls.ResidualTranslationExamples = ResidualTranslationExamples
        cls.TunedLensLayerMetric = TunedLensLayerMetric
        cls.TunedLensViabilitySummary = TunedLensViabilitySummary
        cls.collect_residual_translation_examples = staticmethod(
            collect_residual_translation_examples
        )
        cls.evaluate_low_rank_residual_lens = staticmethod(
            evaluate_low_rank_residual_lens
        )
        cls.fit_low_rank_residual_lens = staticmethod(fit_low_rank_residual_lens)
        cls.save_tuned_lens_artifacts = staticmethod(save_tuned_lens_artifacts)

        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(io.StringIO()):
                cls.model = HookedTransformer.from_pretrained(
                    "tiny-stories-1M",
                    device="cpu",
                    dtype="float32",
                )

    def test_low_rank_residual_lens_beats_raw_readout_on_synthetic_affine_data(
        self,
    ) -> None:
        torch.manual_seed(11)
        num_layers = 2
        num_positions = 12
        d_model = 4
        vocab_size = 5
        rank = 2

        final_residuals = torch.randn(num_positions, d_model)
        left = torch.randn(num_layers, d_model, rank) * 0.15
        right = torch.randn(num_layers, rank, d_model) * 0.15
        bias = torch.randn(num_layers, d_model) * 0.05
        transformed = []
        for layer in range(num_layers):
            delta = final_residuals @ left[layer] @ right[layer]
            transformed.append(final_residuals - delta - bias[layer])
        layer_residuals = torch.stack(transformed)
        final_position_mask = torch.tensor(
            [False] * (num_positions - 3) + [True, True, True]
        )

        examples = self.ResidualTranslationExamples(
            model_name="synthetic",
            prompt_ids=tuple(f"p{i}" for i in range(3)),
            num_prompts=3,
            num_layers=num_layers,
            num_positions=num_positions,
            d_model=d_model,
            layer_residuals=layer_residuals,
            final_residuals=final_residuals,
            final_position_mask=final_position_mask,
        )
        projection = torch.randn(d_model, vocab_size) * 0.3

        lens = self.fit_low_rank_residual_lens(
            examples=examples,
            readout_fn=lambda residuals: residuals @ projection,
            rank=rank,
            num_steps=300,
            learning_rate=0.05,
            weight_decay=0.0,
            seed=11,
            device="cpu",
        )
        summary = self.evaluate_low_rank_residual_lens(
            lens=lens,
            examples=examples,
            readout_fn=lambda residuals: residuals @ projection,
            train_collection_id="synthetic_train",
            train_split="pilot",
            eval_collection_id="synthetic_eval",
            eval_split="pilot",
        )

        self.assertLess(
            summary.mean_tuned_kl_to_final,
            summary.mean_raw_kl_to_final,
        )
        self.assertLess(
            summary.final_position_mean_tuned_kl_to_final,
            summary.final_position_mean_raw_kl_to_final,
        )
        self.assertGreaterEqual(
            summary.mean_tuned_top1_agreement,
            summary.mean_raw_top1_agreement,
        )

    def test_collect_residual_translation_examples_from_tiny_model(self) -> None:
        prompt_entries = (
            PromptEntry(
                prompt_id="tb-a",
                text="The cat sat on the",
                split="pilot",
                tags=("tiny",),
                perturbations={},
            ),
            PromptEntry(
                prompt_id="tb-b",
                text="Once upon a time there was",
                split="pilot",
                tags=("tiny",),
                perturbations={},
            ),
        )

        examples = self.collect_residual_translation_examples(
            model=self.model,
            prompt_entries=prompt_entries,
        )

        self.assertEqual(2, examples.num_prompts)
        self.assertEqual(self.model.cfg.n_layers, examples.num_layers)
        self.assertEqual(self.model.cfg.d_model, examples.d_model)
        self.assertEqual(self.model.cfg.n_layers, examples.layer_residuals.shape[0])
        self.assertEqual(examples.num_positions, examples.layer_residuals.shape[1])
        self.assertEqual(examples.num_positions, examples.final_residuals.shape[0])
        self.assertEqual(2, int(examples.final_position_mask.sum().item()))

    def test_collect_residual_translation_examples_reuses_prompt_cache(self) -> None:
        prompt_entries = (
            PromptEntry(
                prompt_id="tb-a",
                text="The cat sat on the",
                split="pilot",
                tags=("tiny",),
                perturbations={},
            ),
            PromptEntry(
                prompt_id="tb-b",
                text="Once upon a time there was",
                split="pilot",
                tags=("tiny",),
                perturbations={},
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "prompt_cache"
            first = self.collect_residual_translation_examples(
                model=self.model,
                prompt_entries=prompt_entries,
                checkpoint_dir=checkpoint_dir,
            )
            self.assertEqual(2, len(list(checkpoint_dir.glob("*.pt"))))

            with mock.patch.object(
                self.model,
                "run_with_cache",
                side_effect=AssertionError("cache reuse failed"),
            ):
                second = self.collect_residual_translation_examples(
                    model=self.model,
                    prompt_entries=prompt_entries,
                    checkpoint_dir=checkpoint_dir,
                )

        self.assertEqual(first.prompt_ids, second.prompt_ids)
        self.assertTrue(torch.equal(first.layer_residuals, second.layer_residuals))
        self.assertTrue(torch.equal(first.final_residuals, second.final_residuals))
        self.assertTrue(
            torch.equal(first.final_position_mask, second.final_position_mask)
        )

    def test_fit_low_rank_residual_lens_resume_matches_continuous_training(
        self,
    ) -> None:
        torch.manual_seed(11)
        num_layers = 2
        num_positions = 12
        d_model = 4
        rank = 2

        final_residuals = torch.randn(num_positions, d_model)
        left = torch.randn(num_layers, d_model, rank) * 0.15
        right = torch.randn(num_layers, rank, d_model) * 0.15
        bias = torch.randn(num_layers, d_model) * 0.05
        transformed = []
        for layer in range(num_layers):
            delta = final_residuals @ left[layer] @ right[layer]
            transformed.append(final_residuals - delta - bias[layer])
        examples = self.ResidualTranslationExamples(
            model_name="synthetic",
            prompt_ids=("p0", "p1", "p2"),
            num_prompts=3,
            num_layers=num_layers,
            num_positions=num_positions,
            d_model=d_model,
            layer_residuals=torch.stack(transformed),
            final_residuals=final_residuals,
            final_position_mask=torch.tensor(
                [False] * (num_positions - 3) + [True, True, True]
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "training_checkpoint.pt"
            resumed_lens = self.fit_low_rank_residual_lens(
                examples=examples,
                readout_fn=None,
                rank=rank,
                num_steps=100,
                learning_rate=0.05,
                weight_decay=0.0,
                seed=11,
                device="cpu",
                checkpoint_path=checkpoint_path,
                checkpoint_every_steps=25,
            )
            self.assertTrue(checkpoint_path.is_file())
            resumed_lens = self.fit_low_rank_residual_lens(
                examples=examples,
                readout_fn=None,
                rank=rank,
                num_steps=200,
                learning_rate=0.05,
                weight_decay=0.0,
                seed=11,
                device="cpu",
                checkpoint_path=checkpoint_path,
                checkpoint_every_steps=25,
            )

            direct_lens = self.fit_low_rank_residual_lens(
                examples=examples,
                readout_fn=None,
                rank=rank,
                num_steps=200,
                learning_rate=0.05,
                weight_decay=0.0,
                seed=11,
                device="cpu",
            )

        for key, value in direct_lens.state_dict().items():
            self.assertTrue(torch.allclose(value, resumed_lens.state_dict()[key]))

    def test_save_tuned_lens_artifacts_writes_checkpoint_and_summary(self) -> None:
        lens = self.LowRankAffineResidualLens(num_layers=2, d_model=4, rank=2)
        summary = self.TunedLensViabilitySummary(
            model_name="synthetic",
            train_collection_id="oracle_alpha_phase1_v1",
            train_split="pilot",
            eval_collection_id="tool_breakage_factual_recall_v1",
            eval_split="pilot",
            translator_rank=2,
            training_objective="residual_mse_then_logit_eval",
            num_train_prompts=8,
            num_eval_prompts=4,
            num_train_positions=24,
            num_eval_positions=12,
            mean_raw_kl_to_final=0.4,
            mean_tuned_kl_to_final=0.2,
            mean_raw_top1_agreement=0.3,
            mean_tuned_top1_agreement=0.6,
            final_position_mean_raw_kl_to_final=0.5,
            final_position_mean_tuned_kl_to_final=0.25,
            final_position_mean_raw_top1_agreement=0.25,
            final_position_mean_tuned_top1_agreement=0.75,
            layer_metrics=(
                self.TunedLensLayerMetric(
                    layer=0,
                    raw_kl_to_final=0.4,
                    tuned_kl_to_final=0.2,
                    raw_top1_agreement=0.25,
                    tuned_top1_agreement=0.5,
                    final_position_raw_kl_to_final=0.5,
                    final_position_tuned_kl_to_final=0.25,
                    final_position_raw_top1_agreement=0.0,
                    final_position_tuned_top1_agreement=1.0,
                ),
            ),
            checkpoint_path=None,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            saved_summary = self.save_tuned_lens_artifacts(
                output_dir=output_dir,
                lens=lens,
                summary=summary,
            )

            checkpoint_path = output_dir / "checkpoint.pt"
            summary_path = output_dir / "summary.json"
            self.assertTrue(checkpoint_path.is_file())
            self.assertTrue(summary_path.is_file())
            self.assertEqual(str(checkpoint_path), saved_summary.checkpoint_path)
            payload = json.loads(summary_path.read_text())
            self.assertEqual("synthetic", payload["model_name"])
            self.assertIn("layer_metrics", payload)


if __name__ == "__main__":
    unittest.main()
