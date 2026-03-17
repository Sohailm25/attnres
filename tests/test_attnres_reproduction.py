# ABOUTME: Validates the first local AttnRes proxy scaffolding for the Figure 8 lane.
# ABOUTME: Keeps the reproduction slice honest about routing export shapes and matched baseline behavior.

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch


class AttnResReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from validation.attnres_reproduction import (
            AttnResProxyConfig,
            RoutingSnapshot,
            BlockAttnResTinyLM,
            StandardResidualTinyLM,
            build_next_token_examples,
            run_attnres_proxy_viability_from_texts,
            summarize_figure8_proxy_metrics,
            train_language_model,
        )

        cls.AttnResProxyConfig = AttnResProxyConfig
        cls.RoutingSnapshot = RoutingSnapshot
        cls.BlockAttnResTinyLM = BlockAttnResTinyLM
        cls.StandardResidualTinyLM = StandardResidualTinyLM
        cls.build_next_token_examples = staticmethod(build_next_token_examples)
        cls.run_attnres_proxy_viability_from_texts = staticmethod(
            run_attnres_proxy_viability_from_texts
        )
        cls.summarize_figure8_proxy_metrics = staticmethod(
            summarize_figure8_proxy_metrics
        )
        cls.train_language_model = staticmethod(train_language_model)

    def test_block_attnres_forward_exports_normalized_routing(self) -> None:
        config = self.AttnResProxyConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=3,
            d_ff=32,
            max_seq_len=8,
            dropout=0.0,
            num_blocks=3,
        )
        model = self.BlockAttnResTinyLM(config)
        tokens = torch.randint(0, config.vocab_size, (2, config.max_seq_len))

        logits, routing_steps = model(tokens, return_routing=True)

        self.assertEqual(
            (2, config.max_seq_len, config.vocab_size),
            tuple(logits.shape),
        )
        self.assertEqual(2 * config.n_layers + 1, len(routing_steps))
        self.assertEqual("0_pre_attn", routing_steps[0].target_label)
        self.assertEqual(("embed",), routing_steps[0].source_labels)
        self.assertEqual("0_pre_mlp", routing_steps[1].target_label)
        self.assertEqual(
            ("embed", "block_0_attn"),
            routing_steps[1].source_labels,
        )
        self.assertEqual("final_output", routing_steps[-1].target_label)
        self.assertEqual(
            tuple(f"block_{index}" for index in range(config.num_blocks)) + ("embed",),
            tuple(sorted(routing_steps[-1].source_labels)),
        )
        for step in routing_steps:
            self.assertEqual(tokens.shape[0], step.alpha.shape[1])
            self.assertEqual(tokens.shape[1], step.alpha.shape[2])
            self.assertTrue(
                torch.allclose(
                    step.alpha.sum(dim=0),
                    torch.ones_like(step.alpha.sum(dim=0)),
                    atol=1e-5,
                )
            )

    def test_standard_baseline_forward_omits_routing_export(self) -> None:
        config = self.AttnResProxyConfig(
            vocab_size=16,
            d_model=16,
            n_heads=4,
            n_layers=2,
            d_ff=32,
            max_seq_len=6,
            dropout=0.0,
            num_blocks=2,
        )
        model = self.StandardResidualTinyLM(config)
        tokens = torch.randint(0, config.vocab_size, (2, config.max_seq_len))

        logits, routing_steps = model(tokens, return_routing=True)

        self.assertEqual(
            (2, config.max_seq_len, config.vocab_size),
            tuple(logits.shape),
        )
        self.assertEqual((), routing_steps)

    def test_build_next_token_examples_chunks_non_overlapping_windows(self) -> None:
        examples = self.build_next_token_examples(
            token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            sequence_length=3,
        )

        self.assertTrue(
            torch.equal(
                examples,
                torch.tensor(
                    [
                        [1, 2, 3, 4],
                        [4, 5, 6, 7],
                    ],
                    dtype=torch.long,
                ),
            )
        )

    def test_train_language_model_writes_checkpoint(self) -> None:
        config = self.AttnResProxyConfig(
            vocab_size=16,
            d_model=16,
            n_heads=4,
            n_layers=2,
            d_ff=32,
            max_seq_len=4,
            dropout=0.0,
            num_blocks=2,
        )
        model = self.StandardResidualTinyLM(config)
        examples = torch.tensor(
            [
                [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [3, 4, 5, 6, 7],
                [4, 5, 6, 7, 8],
            ],
            dtype=torch.long,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "training_state.pt"
            summary = self.train_language_model(
                model=model,
                model_label="baseline",
                train_examples=examples,
                eval_examples=examples,
                batch_size=2,
                num_steps=2,
                learning_rate=1e-3,
                weight_decay=0.0,
                seed=11,
                device="cpu",
                checkpoint_path=checkpoint_path,
                checkpoint_interval=1,
            )

            self.assertTrue(checkpoint_path.is_file())
            self.assertTrue(torch.isfinite(torch.tensor(summary.final_train_loss)))
            self.assertTrue(torch.isfinite(torch.tensor(summary.final_eval_loss)))

    def test_train_language_model_writes_best_checkpoint_and_eval_history(self) -> None:
        config = self.AttnResProxyConfig(
            vocab_size=16,
            d_model=16,
            n_heads=4,
            n_layers=2,
            d_ff=32,
            max_seq_len=4,
            dropout=0.0,
            num_blocks=2,
        )
        model = self.StandardResidualTinyLM(config)
        examples = torch.tensor(
            [
                [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [3, 4, 5, 6, 7],
                [4, 5, 6, 7, 8],
            ],
            dtype=torch.long,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "training_state.pt"
            best_checkpoint_path = Path(tmp_dir) / "best_state.pt"
            eval_history_path = Path(tmp_dir) / "eval_history.json"
            summary = self.train_language_model(
                model=model,
                model_label="baseline",
                train_examples=examples,
                eval_examples=examples,
                batch_size=2,
                num_steps=3,
                learning_rate=1e-3,
                weight_decay=0.0,
                seed=11,
                device="cpu",
                checkpoint_path=checkpoint_path,
                best_checkpoint_path=best_checkpoint_path,
                eval_history_path=eval_history_path,
                checkpoint_interval=1,
            )

            self.assertTrue(checkpoint_path.is_file())
            self.assertTrue(best_checkpoint_path.is_file())
            self.assertTrue(eval_history_path.is_file())
            history = json.loads(eval_history_path.read_text())
            self.assertGreaterEqual(len(history), 3)
            self.assertEqual(summary.best_eval_step, history[-1]["best_eval_step"])
            self.assertEqual(str(best_checkpoint_path), summary.best_checkpoint_path)
            self.assertEqual(str(eval_history_path), summary.eval_history_path)

    def test_summarize_figure8_proxy_metrics_tracks_operationalized_patterns(
        self,
    ) -> None:
        routing_steps = (
            self.RoutingSnapshot(
                target_label="0_pre_attn",
                source_labels=("embed",),
                alpha=torch.ones((1, 2, 2), dtype=torch.float32),
            ),
            self.RoutingSnapshot(
                target_label="0_pre_mlp",
                source_labels=("embed", "block_0_attn"),
                alpha=torch.tensor(
                    [
                        [[0.2, 0.1], [0.2, 0.1]],
                        [[0.8, 0.9], [0.8, 0.9]],
                    ],
                    dtype=torch.float32,
                ),
            ),
            self.RoutingSnapshot(
                target_label="1_pre_attn",
                source_labels=("embed", "block_0"),
                alpha=torch.tensor(
                    [
                        [[0.7, 0.6], [0.7, 0.6]],
                        [[0.3, 0.4], [0.3, 0.4]],
                    ],
                    dtype=torch.float32,
                ),
            ),
            self.RoutingSnapshot(
                target_label="1_pre_mlp",
                source_labels=("embed", "block_0", "block_1_attn"),
                alpha=torch.tensor(
                    [
                        [[0.05, 0.05], [0.05, 0.05]],
                        [[0.15, 0.15], [0.15, 0.15]],
                        [[0.8, 0.8], [0.8, 0.8]],
                    ],
                    dtype=torch.float32,
                ),
            ),
            self.RoutingSnapshot(
                target_label="final_output",
                source_labels=("embed", "block_0", "block_1"),
                alpha=torch.tensor(
                    [
                        [[0.8, 0.75], [0.8, 0.75]],
                        [[0.1, 0.15], [0.1, 0.15]],
                        [[0.1, 0.1], [0.1, 0.1]],
                    ],
                    dtype=torch.float32,
                ),
            ),
        )

        metrics = self.summarize_figure8_proxy_metrics(routing_steps=routing_steps)

        target_metrics = {
            metric.target_label: metric for metric in metrics.target_summaries
        }
        self.assertAlmostEqual(
            1.0,
            target_metrics["0_pre_attn"].locality_score,
            places=5,
        )
        self.assertAlmostEqual(
            0.85,
            target_metrics["0_pre_mlp"].locality_score,
            places=5,
        )
        self.assertAlmostEqual(
            0.35,
            target_metrics["1_pre_attn"].locality_score,
            places=5,
        )
        self.assertAlmostEqual(
            0.49166667,
            metrics.deep_embedding_persistence,
            places=5,
        )
        self.assertGreater(
            metrics.mean_pre_attn_entropy,
            metrics.mean_pre_mlp_entropy,
        )
        final_skip_peaks = dict(
            target_metrics["final_output"].skip_connection_peak_fractions
        )
        self.assertAlmostEqual(1.0, final_skip_peaks["embed"], places=5)

    def test_run_attnres_proxy_viability_from_texts_writes_summary_and_checkpoints(
        self,
    ) -> None:
        config = self.AttnResProxyConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=2,
            d_ff=32,
            max_seq_len=8,
            dropout=0.0,
            num_blocks=2,
        )
        train_texts = (
            "routing reveals depth structure in tiny models",
            "attention residuals route across prior blocks",
            "figure eight needs exportable routing traces",
        )
        eval_texts = ("tiny evaluation text for routing export",)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "attnres_proxy"
            summary = self.run_attnres_proxy_viability_from_texts(
                config=config,
                dataset_name="synthetic_text",
                train_texts=train_texts,
                eval_texts=eval_texts,
                output_dir=output_dir,
                batch_size=2,
                num_steps=2,
                learning_rate=1e-3,
                weight_decay=0.0,
                seed=11,
                device="cpu",
                checkpoint_interval=1,
            )

            self.assertTrue((output_dir / "summary.json").is_file())
            self.assertTrue(
                (output_dir / "checkpoints" / "baseline_training_state.pt").is_file()
            )
            self.assertTrue(
                (output_dir / "checkpoints" / "attnres_training_state.pt").is_file()
            )
            self.assertTrue(
                (output_dir / "checkpoints" / "baseline_best_state.pt").is_file()
            )
            self.assertTrue(
                (output_dir / "checkpoints" / "attnres_best_state.pt").is_file()
            )
            self.assertTrue(
                (output_dir / "checkpoints" / "baseline_eval_history.json").is_file()
            )
            self.assertTrue(
                (output_dir / "checkpoints" / "attnres_eval_history.json").is_file()
            )
            self.assertEqual("synthetic_text", summary.dataset_name)
            self.assertGreater(len(summary.figure8_proxy_metrics.target_summaries), 0)
            self.assertIsNotNone(summary.best_checkpoint_figure8_proxy_metrics)
            self.assertTrue(
                torch.isfinite(torch.tensor(summary.baseline_summary.best_eval_loss))
            )
            self.assertTrue(
                torch.isfinite(torch.tensor(summary.attnres_summary.best_eval_loss))
            )

    def test_run_attnres_proxy_viability_from_texts_supports_compact_subword_mode(
        self,
    ) -> None:
        class StubTokenizer:
            def __init__(self) -> None:
                self._mapping = {
                    "alpha beta": [101, 202, 303],
                    "beta gamma": [202, 404],
                    "eval alpha beta gamma delta": [505, 101, 202, 404, 606],
                    "\n\n": [9000],
                }

            def __call__(
                self,
                text: str,
                *,
                add_special_tokens: bool = False,
            ) -> dict[str, list[int]]:
                if add_special_tokens:
                    raise AssertionError(
                        "stub tokenizer does not expect special tokens"
                    )
                return {"input_ids": list(self._mapping[text])}

            def decode(self, token_ids: list[int]) -> str:
                return "|".join(str(token_id) for token_id in token_ids)

        config = self.AttnResProxyConfig(
            vocab_size=16,
            d_model=16,
            n_heads=4,
            n_layers=2,
            d_ff=32,
            max_seq_len=4,
            dropout=0.0,
            num_blocks=2,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "attnres_proxy_subword"
            summary = self.run_attnres_proxy_viability_from_texts(
                config=config,
                dataset_name="synthetic_text",
                train_texts=("alpha beta", "beta gamma"),
                eval_texts=("eval alpha beta gamma delta",),
                output_dir=output_dir,
                batch_size=1,
                num_steps=2,
                learning_rate=1e-3,
                weight_decay=0.0,
                seed=11,
                device="cpu",
                checkpoint_interval=1,
                tokenizer_mode="compact_subword",
                tokenizer=StubTokenizer(),
                tokenizer_name="stub-tokenizer",
            )

            manifest_path = output_dir / "tokenizer_manifest.json"
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual("compact_subword", manifest["tokenizer_mode"])
            self.assertEqual("stub-tokenizer", manifest["tokenizer_name"])
            self.assertEqual(
                [101, 202, 303, 404, 505, 606, 9000],
                manifest["observed_original_token_ids"],
            )
            self.assertEqual("compact_subword:stub-tokenizer", summary.tokenizer_mode)
            self.assertTrue(
                torch.isfinite(torch.tensor(summary.baseline_summary.best_eval_loss))
            )


if __name__ == "__main__":
    unittest.main()
