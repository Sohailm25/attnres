# ABOUTME: Exercises the Phase 6 pilot router-distillation comparison on saved per-token exports.
# ABOUTME: Pins dataset loading, target handling, sequence aggregation, stratified pilot splitting, and the h_1[t] versus h_4[t] comparison surface.

import json
from pathlib import Path
import tempfile
import unittest

import torch


class RouterDistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from validation.router_distillation import (
            RouterDistillationExample,
            aggregate_token_logits_to_sequence_alpha,
            compare_router_aggregations,
            compare_router_capacities,
            compare_router_families,
            compare_router_input_sources,
            compare_router_target_parameterizations,
            load_router_distillation_pilot_dataset,
            stratified_router_train_eval_split,
            target_matrix_for_router_distillation,
        )

        cls.RouterDistillationExample = RouterDistillationExample
        cls.aggregate_token_logits_to_sequence_alpha = staticmethod(
            aggregate_token_logits_to_sequence_alpha
        )
        cls.compare_router_aggregations = staticmethod(compare_router_aggregations)
        cls.compare_router_capacities = staticmethod(compare_router_capacities)
        cls.compare_router_families = staticmethod(compare_router_families)
        cls.compare_router_input_sources = staticmethod(compare_router_input_sources)
        cls.compare_router_target_parameterizations = staticmethod(
            compare_router_target_parameterizations
        )
        cls.load_router_distillation_pilot_dataset = staticmethod(
            load_router_distillation_pilot_dataset
        )
        cls.stratified_router_train_eval_split = staticmethod(
            stratified_router_train_eval_split
        )
        cls.target_matrix_for_router_distillation = staticmethod(
            target_matrix_for_router_distillation
        )

    def test_load_router_distillation_pilot_dataset_reads_manifest_and_checkpoints(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            export_dir = Path(temp_dir)
            checkpoint_dir = export_dir / "checkpoints" / "prompt_exports"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = checkpoint_dir / "prompt-1.pt"
            torch.save(
                {
                    "prompt_id": "prompt-1",
                    "prompt": "The capital of France is",
                    "split": "pilot",
                    "target_text": "Paris",
                    "tags": (
                        "oracle_alpha",
                        "stratum_factual_recall",
                        "subcategory_capital_fact",
                    ),
                    "perturbations": {"prompt_paraphrase": "France has its capital in"},
                    "token_ids": torch.tensor([1, 2, 3], dtype=torch.long),
                    "h_1[t]": torch.tensor(
                        [[1.0, 0.0], [0.5, 0.0], [0.25, 0.0]],
                        dtype=torch.float32,
                    ),
                    "h_4[t]": torch.tensor(
                        [[0.0, 1.0], [0.0, 0.5], [0.0, 0.25]],
                        dtype=torch.float32,
                    ),
                    "source_labels": ("embed", "0_attn_out"),
                    "final_alpha": torch.tensor([0.75, 0.25], dtype=torch.float32),
                },
                checkpoint_path,
            )
            (export_dir / "dataset_manifest.json").write_text(
                json.dumps(
                    {
                        "collection_id": "oracle_alpha_phase1_v1",
                        "model_name": "test-model",
                        "split": "pilot",
                        "output_dir": str(export_dir),
                        "checkpoint_dir": str(checkpoint_dir),
                        "dataset_fields": [
                            "prompt_id",
                            "prompt",
                            "split",
                            "target_text",
                            "tags",
                            "perturbations",
                            "token_ids",
                            "h_1[t]",
                            "h_4[t]",
                            "source_labels",
                            "final_alpha",
                        ],
                        "entries": [
                            {
                                "prompt_id": "prompt-1",
                                "split": "pilot",
                                "prompt": "The capital of France is",
                                "target_text": "Paris",
                                "tags": [
                                    "oracle_alpha",
                                    "stratum_factual_recall",
                                    "subcategory_capital_fact",
                                ],
                                "perturbation_names": ["prompt_paraphrase"],
                                "checkpoint_path": str(checkpoint_path),
                                "num_tokens": 3,
                                "d_model": 2,
                                "num_sources": 2,
                            }
                        ],
                    },
                    indent=2,
                )
                + "\n"
            )

            dataset = self.load_router_distillation_pilot_dataset(export_dir)

            self.assertEqual("oracle_alpha_phase1_v1", dataset.collection_id)
            self.assertEqual("test-model", dataset.model_name)
            self.assertEqual(1, len(dataset.examples))
            example = dataset.examples[0]
            self.assertEqual("prompt-1", example.prompt_id)
            self.assertEqual(
                ("oracle_alpha", "stratum_factual_recall", "subcategory_capital_fact"),
                example.tags,
            )
            self.assertEqual((3, 2), tuple(example.h_1.shape))
            self.assertEqual((3, 2), tuple(example.h_4.shape))
            self.assertEqual((2,), tuple(example.final_alpha.shape))

    def test_aggregate_token_logits_to_sequence_alpha_uses_mean_logits_then_softmax(
        self,
    ) -> None:
        token_logits = torch.tensor([[2.0, 0.0], [0.0, 0.0]], dtype=torch.float32)

        alpha = self.aggregate_token_logits_to_sequence_alpha(
            token_logits=token_logits,
            aggregation="mean_token_logits_then_softmax",
        )

        expected = torch.softmax(torch.tensor([1.0, 0.0]), dim=0)
        self.assertTrue(torch.allclose(expected, alpha, atol=1e-6))

    def test_aggregate_token_logits_to_sequence_alpha_uses_last_token_logits_then_softmax(
        self,
    ) -> None:
        token_logits = torch.tensor(
            [
                [2.0, 0.0],
                [0.0, 2.0],
                [1.0, 3.0],
            ],
            dtype=torch.float32,
        )

        alpha = self.aggregate_token_logits_to_sequence_alpha(
            token_logits=token_logits,
            aggregation="last_token_logits_then_softmax",
        )

        expected = torch.softmax(torch.tensor([1.0, 3.0]), dim=0)
        self.assertTrue(torch.allclose(expected, alpha, atol=1e-6))

    def test_stratified_router_train_eval_split_preserves_subcategory_balance(
        self,
    ) -> None:
        examples = [
            self._synthetic_example(
                prompt_id=f"prompt-{index}",
                subcategory="subcategory_capital_fact"
                if index < 4
                else "subcategory_author_fact",
                signal=float(index + 1),
            )
            for index in range(8)
        ]

        train_examples, eval_examples = self.stratified_router_train_eval_split(
            examples,
            eval_fraction=0.25,
            seed=11,
        )

        self.assertEqual(6, len(train_examples))
        self.assertEqual(2, len(eval_examples))
        eval_subcategories = sorted(
            self._subcategory_tag(example.tags) for example in eval_examples
        )
        self.assertEqual(
            ["subcategory_author_fact", "subcategory_capital_fact"],
            eval_subcategories,
        )

    def test_compare_router_input_sources_prefers_h1_when_h1_carries_signal(
        self,
    ) -> None:
        examples = [
            self._synthetic_example(
                prompt_id=f"prompt-{index}",
                subcategory=(
                    "subcategory_capital_fact"
                    if index < 4
                    else "subcategory_author_fact"
                    if index < 8
                    else "subcategory_element_fact"
                    if index < 12
                    else "subcategory_city_fact"
                ),
                signal=float(index - 7.5),
            )
            for index in range(16)
        ]

        comparison = self.compare_router_input_sources(
            examples=examples,
            candidate_input_fields=("h_1[t]", "h_4[t]"),
            aggregation="mean_token_logits_then_softmax",
            eval_fraction=0.25,
            hidden_dim=8,
            learning_rate=0.05,
            max_epochs=250,
            patience=40,
            seed=11,
            device="cpu",
        )

        self.assertEqual("h_1[t]", comparison.selected_input_field)
        summaries = {
            summary.input_field: summary for summary in comparison.input_summaries
        }
        self.assertGreater(summaries["h_1[t]"].eval_summary.r_squared, 0.8)
        self.assertLess(summaries["h_4[t]"].eval_summary.r_squared, 0.2)
        self.assertEqual(
            "mean_token_logits_then_softmax",
            summaries["h_1[t]"].aggregation,
        )

    def test_target_matrix_for_router_distillation_roundtrips_raw_and_logit_targets(
        self,
    ) -> None:
        examples = (
            self._synthetic_example(
                prompt_id="prompt-a",
                subcategory="subcategory_capital_fact",
                signal=1.5,
            ),
            self._synthetic_example(
                prompt_id="prompt-b",
                subcategory="subcategory_author_fact",
                signal=-0.75,
            ),
        )

        raw_targets = self.target_matrix_for_router_distillation(
            examples=examples,
            target_name="oracle_alpha_vector",
        )
        logit_targets = self.target_matrix_for_router_distillation(
            examples=examples,
            target_name="oracle_alpha_logit_vector",
        )

        expected = torch.stack([example.final_alpha for example in examples], dim=0)
        self.assertTrue(torch.allclose(expected, raw_targets, atol=1e-6))
        self.assertTrue(
            torch.allclose(
                torch.zeros(logit_targets.shape[0]),
                logit_targets.mean(dim=1),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(expected, torch.softmax(logit_targets, dim=1), atol=1e-6)
        )

    def test_compare_router_target_parameterizations_reuses_one_fixed_split(
        self,
    ) -> None:
        examples = [
            self._synthetic_example(
                prompt_id=f"prompt-{index}",
                subcategory=(
                    "subcategory_capital_fact"
                    if index < 4
                    else "subcategory_author_fact"
                    if index < 8
                    else "subcategory_element_fact"
                    if index < 12
                    else "subcategory_city_fact"
                ),
                signal=float(index - 7.5),
            )
            for index in range(16)
        ]

        comparison = self.compare_router_target_parameterizations(
            examples=examples,
            candidate_input_fields=("h_1[t]", "h_4[t]"),
            candidate_target_names=(
                "oracle_alpha_vector",
                "oracle_alpha_logit_vector",
            ),
            aggregation="mean_token_logits_then_softmax",
            eval_fraction=0.25,
            hidden_dim=8,
            learning_rate=0.05,
            max_epochs=250,
            patience=40,
            seed=11,
            device="cpu",
        )

        self.assertEqual(4, len(comparison.candidate_summaries))
        self.assertEqual(
            {"oracle_alpha_vector", "oracle_alpha_logit_vector"},
            {summary.target_name for summary in comparison.candidate_summaries},
        )
        for summary in comparison.candidate_summaries:
            self.assertEqual(comparison.train_prompt_ids, summary.train_prompt_ids)
            self.assertEqual(comparison.eval_prompt_ids, summary.eval_prompt_ids)
        self.assertIn(
            comparison.selected_target_name,
            {"oracle_alpha_vector", "oracle_alpha_logit_vector"},
        )
        self.assertIn(
            comparison.selected_input_field,
            {"h_1[t]", "h_4[t]"},
        )

    def test_compare_router_aggregations_prefers_last_token_when_order_carries_signal(
        self,
    ) -> None:
        examples = [
            self._order_sensitive_example(
                prompt_id=f"prompt-{index}",
                subcategory=(
                    "subcategory_capital_fact"
                    if index < 4
                    else "subcategory_author_fact"
                    if index < 8
                    else "subcategory_element_fact"
                    if index < 12
                    else "subcategory_city_fact"
                ),
                positive=(index % 2 == 0),
            )
            for index in range(16)
        ]

        comparison = self.compare_router_aggregations(
            examples=examples,
            candidate_input_fields=("h_1[t]",),
            fixed_target_name="oracle_alpha_logit_vector",
            candidate_aggregations=(
                "mean_token_logits_then_softmax",
                "last_token_logits_then_softmax",
            ),
            eval_fraction=0.25,
            hidden_dim=8,
            learning_rate=0.05,
            max_epochs=250,
            patience=40,
            seed=11,
            device="cpu",
        )

        self.assertEqual(
            "last_token_logits_then_softmax",
            comparison.selected_aggregation,
        )
        summaries = {
            summary.aggregation: summary for summary in comparison.input_summaries
        }
        self.assertLess(
            summaries["mean_token_logits_then_softmax"].eval_summary.r_squared,
            0.2,
        )
        self.assertGreater(
            summaries["last_token_logits_then_softmax"].eval_summary.r_squared,
            0.8,
        )

    def test_compare_router_capacities_prefers_larger_hidden_dim_when_width_is_bottleneck(
        self,
    ) -> None:
        examples = [
            self._quadrant_parity_example(
                prompt_id=f"prompt-{index}",
                subcategory=(
                    "subcategory_capital_fact"
                    if index < 8
                    else "subcategory_author_fact"
                    if index < 16
                    else "subcategory_element_fact"
                    if index < 24
                    else "subcategory_city_fact"
                ),
                point_index=index,
            )
            for index in range(32)
        ]

        comparison = self.compare_router_capacities(
            examples=examples,
            candidate_input_fields=("h_1[t]",),
            fixed_target_name="oracle_alpha_logit_vector",
            fixed_aggregation="mean_token_logits_then_softmax",
            candidate_hidden_dims=(1, 8),
            eval_fraction=0.25,
            learning_rate=0.05,
            max_epochs=400,
            patience=60,
            seed=11,
            device="cpu",
        )

        self.assertEqual(8, comparison.selected_hidden_dim)
        summaries = {
            summary.hidden_dim: summary for summary in comparison.input_summaries
        }
        self.assertLess(summaries[1].eval_summary.r_squared, 0.3)
        self.assertGreater(summaries[8].eval_summary.r_squared, 0.8)

    def test_compare_router_families_prefers_mlp_when_linear_underfits_xor_structure(
        self,
    ) -> None:
        examples = [
            self._quadrant_parity_example(
                prompt_id=f"prompt-{index}",
                subcategory=(
                    "subcategory_capital_fact"
                    if index < 8
                    else "subcategory_author_fact"
                    if index < 16
                    else "subcategory_element_fact"
                    if index < 24
                    else "subcategory_city_fact"
                ),
                point_index=index,
                signal_field="h_4[t]",
            )
            for index in range(32)
        ]

        comparison = self.compare_router_families(
            examples=examples,
            fixed_input_field="h_4[t]",
            fixed_target_name="oracle_alpha_logit_vector",
            fixed_aggregation="mean_token_logits_then_softmax",
            candidate_router_families=("linear", "mlp"),
            hidden_dim=8,
            eval_fraction=0.25,
            learning_rate=0.05,
            max_epochs=400,
            patience=60,
            seed=11,
            device="cpu",
        )

        self.assertEqual("mlp", comparison.selected_router_family)
        summaries = {
            summary.router_family: summary for summary in comparison.input_summaries
        }
        self.assertLess(summaries["linear"].eval_summary.r_squared, 0.3)
        self.assertGreater(summaries["mlp"].eval_summary.r_squared, 0.8)

    def _synthetic_example(
        self,
        *,
        prompt_id: str,
        subcategory: str,
        signal: float,
    ):
        logits = torch.tensor([signal, -signal], dtype=torch.float32)
        final_alpha = torch.softmax(logits, dim=0)
        return self.RouterDistillationExample(
            prompt_id=prompt_id,
            prompt=f"Prompt {prompt_id}",
            split="pilot",
            target_text=None,
            tags=("oracle_alpha", "stratum_factual_recall", subcategory),
            perturbation_names=("prompt_paraphrase",),
            token_ids=torch.tensor([1, 2, 3], dtype=torch.long),
            h_1=torch.tensor(
                [
                    [signal, 0.0],
                    [signal, 0.1],
                    [signal, -0.1],
                ],
                dtype=torch.float32,
            ),
            h_4=torch.tensor(
                [
                    [0.0, 1.0],
                    [0.0, -1.0],
                    [0.0, 0.5],
                ],
                dtype=torch.float32,
            ),
            source_labels=("embed", "0_attn_out"),
            final_alpha=final_alpha,
        )

    def _quadrant_parity_example(
        self,
        *,
        prompt_id: str,
        subcategory: str,
        point_index: int,
        signal_field: str = "h_1[t]",
    ):
        angle = ((point_index + 0.5) / 32.0) * (2.0 * torch.pi)
        x_value = float(torch.cos(torch.tensor(angle)))
        y_value = float(torch.sin(torch.tensor(angle)))
        positive = (x_value > 0) ^ (y_value > 0)
        final_alpha = torch.tensor(
            [0.85, 0.15] if positive else [0.15, 0.85],
            dtype=torch.float32,
        )
        return self.RouterDistillationExample(
            prompt_id=prompt_id,
            prompt=f"Prompt {prompt_id}",
            split="pilot",
            target_text=None,
            tags=("oracle_alpha", "stratum_factual_recall", subcategory),
            perturbation_names=("prompt_paraphrase",),
            token_ids=torch.tensor([1, 2], dtype=torch.long),
            h_1=(
                torch.tensor(
                    [
                        [x_value, y_value],
                        [x_value, y_value],
                    ],
                    dtype=torch.float32,
                )
                if signal_field == "h_1[t]"
                else torch.tensor(
                    [
                        [0.5, 0.5],
                        [0.5, 0.5],
                    ],
                    dtype=torch.float32,
                )
            ),
            h_4=(
                torch.tensor(
                    [
                        [x_value, y_value],
                        [x_value, y_value],
                    ],
                    dtype=torch.float32,
                )
                if signal_field == "h_4[t]"
                else torch.tensor(
                    [
                        [0.5, 0.5],
                        [0.5, 0.5],
                    ],
                    dtype=torch.float32,
                )
            ),
            source_labels=("embed", "0_attn_out"),
            final_alpha=final_alpha,
        )

    def _order_sensitive_example(
        self,
        *,
        prompt_id: str,
        subcategory: str,
        positive: bool,
    ):
        final_alpha = torch.tensor(
            [0.85, 0.15] if positive else [0.15, 0.85],
            dtype=torch.float32,
        )
        first_token = [1.0, 0.0] if positive else [0.0, 1.0]
        second_token = [0.0, 1.0] if positive else [1.0, 0.0]
        return self.RouterDistillationExample(
            prompt_id=prompt_id,
            prompt=f"Prompt {prompt_id}",
            split="pilot",
            target_text=None,
            tags=("oracle_alpha", "stratum_factual_recall", subcategory),
            perturbation_names=("prompt_paraphrase",),
            token_ids=torch.tensor([1, 2], dtype=torch.long),
            h_1=torch.tensor(
                [first_token, second_token],
                dtype=torch.float32,
            ),
            h_4=torch.tensor(
                [
                    [0.5, 0.5],
                    [0.5, 0.5],
                ],
                dtype=torch.float32,
            ),
            source_labels=("embed", "0_attn_out"),
            final_alpha=final_alpha,
        )

    def _subcategory_tag(self, tags: tuple[str, ...]) -> str:
        for tag in tags:
            if tag.startswith("subcategory_"):
                return tag
        raise AssertionError("missing subcategory tag")


if __name__ == "__main__":
    unittest.main()
