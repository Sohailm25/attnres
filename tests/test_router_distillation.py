# ABOUTME: Exercises the Phase 6 pilot router-distillation comparison on saved per-token exports.
# ABOUTME: Pins dataset loading, target handling, sequence aggregation, stratified pilot splitting, and the h_1[t] versus h_4[t] comparison surface.

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


class RouterDistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from validation.router_distillation import (
            RouterDistillationExample,
            RouterDistillationPromptDiagnostic,
            RouterDistillationTeacherMismatchPromptDiagnostic,
            aggregate_token_logits_to_sequence_alpha,
            audit_exact_tokenwise_teacher_mismatch,
            audit_router_supervision_granularity,
            compare_router_aggregations,
            compare_router_capacities,
            compare_router_families,
            compare_router_input_sources,
            compare_router_supervision_objectives,
            compare_router_target_parameterizations,
            build_oracle_alpha_tokenwise_teacher_lookup,
            compact_router_distillation_teacher_mismatch_payload,
            exact_tokenwise_oracle_alpha_logit_targets,
            load_router_distillation_pilot_dataset,
            router_supervision_loss,
            summarize_exact_teacher_mismatch,
            summarize_router_supervision_granularity,
            stratified_router_train_eval_split,
            target_matrix_for_router_distillation,
            tokenwise_oracle_alpha_target_logit_contribution_targets,
        )

        cls.RouterDistillationExample = RouterDistillationExample
        cls.RouterDistillationPromptDiagnostic = RouterDistillationPromptDiagnostic
        cls.RouterDistillationTeacherMismatchPromptDiagnostic = (
            RouterDistillationTeacherMismatchPromptDiagnostic
        )
        cls.aggregate_token_logits_to_sequence_alpha = staticmethod(
            aggregate_token_logits_to_sequence_alpha
        )
        cls.audit_exact_tokenwise_teacher_mismatch = staticmethod(
            audit_exact_tokenwise_teacher_mismatch
        )
        cls.audit_router_supervision_granularity = staticmethod(
            audit_router_supervision_granularity
        )
        cls.compare_router_aggregations = staticmethod(compare_router_aggregations)
        cls.compare_router_capacities = staticmethod(compare_router_capacities)
        cls.compare_router_families = staticmethod(compare_router_families)
        cls.compare_router_input_sources = staticmethod(compare_router_input_sources)
        cls.compare_router_supervision_objectives = staticmethod(
            compare_router_supervision_objectives
        )
        cls.compare_router_target_parameterizations = staticmethod(
            compare_router_target_parameterizations
        )
        cls.build_oracle_alpha_tokenwise_teacher_lookup = staticmethod(
            build_oracle_alpha_tokenwise_teacher_lookup
        )
        cls.compact_router_distillation_teacher_mismatch_payload = staticmethod(
            compact_router_distillation_teacher_mismatch_payload
        )
        cls.exact_tokenwise_oracle_alpha_logit_targets = staticmethod(
            exact_tokenwise_oracle_alpha_logit_targets
        )
        cls.load_router_distillation_pilot_dataset = staticmethod(
            load_router_distillation_pilot_dataset
        )
        cls.router_supervision_loss = staticmethod(router_supervision_loss)
        cls.summarize_exact_teacher_mismatch = staticmethod(
            summarize_exact_teacher_mismatch
        )
        cls.summarize_router_supervision_granularity = staticmethod(
            summarize_router_supervision_granularity
        )
        cls.stratified_router_train_eval_split = staticmethod(
            stratified_router_train_eval_split
        )
        cls.target_matrix_for_router_distillation = staticmethod(
            target_matrix_for_router_distillation
        )
        cls.tokenwise_oracle_alpha_target_logit_contribution_targets = staticmethod(
            tokenwise_oracle_alpha_target_logit_contribution_targets
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

    def test_compare_router_families_respects_fixed_split_and_all_token_supervision(
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
        train_prompt_ids = tuple(
            f"prompt-{index}"
            for index in (
                0,
                1,
                2,
                3,
                8,
                9,
                10,
                11,
                16,
                17,
                18,
                19,
                24,
                25,
                26,
                27,
            )
        )
        eval_prompt_ids = tuple(
            f"prompt-{index}"
            for index in (
                4,
                5,
                6,
                7,
                12,
                13,
                14,
                15,
                20,
                21,
                22,
                23,
                28,
                29,
                30,
                31,
            )
        )

        comparison = self.compare_router_families(
            examples=examples,
            fixed_input_field="h_4[t]",
            fixed_target_name="oracle_alpha_logit_vector",
            fixed_aggregation="mean_token_logits_then_softmax",
            candidate_router_families=("linear", "mlp"),
            hidden_dim=8,
            train_prompt_ids=train_prompt_ids,
            eval_prompt_ids=eval_prompt_ids,
            supervision_objective="all_tokens_target_mse",
            eval_fraction=0.25,
            learning_rate=0.05,
            max_epochs=400,
            patience=60,
            seed=11,
            device="cpu",
        )

        self.assertEqual(
            "all_tokens_target_mse", comparison.fixed_supervision_objective
        )
        self.assertEqual(train_prompt_ids, comparison.train_prompt_ids)
        self.assertEqual(eval_prompt_ids, comparison.eval_prompt_ids)
        summaries = {
            summary.router_family: summary for summary in comparison.input_summaries
        }
        self.assertEqual(
            "all_tokens_target_mse",
            summaries["linear"].supervision_objective,
        )
        self.assertEqual(
            "all_tokens_target_mse",
            summaries["mlp"].supervision_objective,
        )
        self.assertGreater(
            summaries["mlp"].eval_summary.r_squared,
            summaries["linear"].eval_summary.r_squared,
        )

    def test_summarize_router_supervision_granularity_highlights_stratum_gap(
        self,
    ) -> None:
        diagnostics = (
            self.RouterDistillationPromptDiagnostic(
                prompt_id="fact-1",
                prompt="The capital of France is",
                tags=(
                    "oracle_alpha",
                    "stratum_factual_recall",
                    "subcategory_capital_fact",
                ),
                num_tokens=7,
                oracle_entropy=3.4,
                oracle_top1_mass=0.42,
                mean_js_divergence=0.05,
                target_mse=0.10,
            ),
            self.RouterDistillationPromptDiagnostic(
                prompt_id="fact-2",
                prompt="The symbol for sodium is",
                tags=(
                    "oracle_alpha",
                    "stratum_factual_recall",
                    "subcategory_element_fact",
                ),
                num_tokens=8,
                oracle_entropy=3.35,
                oracle_top1_mass=0.44,
                mean_js_divergence=0.06,
                target_mse=0.12,
            ),
            self.RouterDistillationPromptDiagnostic(
                prompt_id="code-1",
                prompt="def clean_names(names):",
                tags=(
                    "oracle_alpha",
                    "stratum_code_procedural",
                    "subcategory_python_snippet",
                ),
                num_tokens=20,
                oracle_entropy=3.38,
                oracle_top1_mass=0.41,
                mean_js_divergence=0.13,
                target_mse=0.31,
            ),
            self.RouterDistillationPromptDiagnostic(
                prompt_id="code-2",
                prompt="def lowercase_tags(tags):",
                tags=(
                    "oracle_alpha",
                    "stratum_code_procedural",
                    "subcategory_python_snippet",
                ),
                num_tokens=22,
                oracle_entropy=3.36,
                oracle_top1_mass=0.43,
                mean_js_divergence=0.14,
                target_mse=0.33,
            ),
        )

        summary = self.summarize_router_supervision_granularity(
            prompt_diagnostics=diagnostics
        )

        self.assertEqual("stratum_code_procedural", summary.worst_stratum_tag)
        self.assertGreater(summary.stratum_mean_js_range, 0.05)
        stratum_summaries = {
            group.group_key: group for group in summary.stratum_summaries
        }
        self.assertGreater(
            stratum_summaries["stratum_code_procedural"].mean_js_divergence,
            stratum_summaries["stratum_factual_recall"].mean_js_divergence,
        )
        correlations = {
            item.attribute_name: item.pearson_r
            for item in summary.attribute_correlations
        }
        self.assertGreater(correlations["num_tokens"], 0.9)
        self.assertLess(abs(correlations["oracle_entropy"]), 0.3)

    def test_audit_router_supervision_granularity_flags_unmodeled_stratum(
        self,
    ) -> None:
        source_labels = ("embed", "0_attn_out")
        examples = []
        for index in range(4):
            positive = index % 2 == 0
            factual_logits = torch.tensor(
                [2.0, -2.0] if positive else [-2.0, 2.0],
                dtype=torch.float32,
            )
            examples.append(
                self.RouterDistillationExample(
                    prompt_id=f"fact-{index}",
                    prompt=f"Fact {index}",
                    split="pilot",
                    target_text=None,
                    tags=(
                        "oracle_alpha",
                        "stratum_factual_recall",
                        "subcategory_capital_fact",
                    ),
                    perturbation_names=(),
                    token_ids=torch.tensor([1, 2, 3], dtype=torch.long),
                    h_1=torch.zeros((3, 2), dtype=torch.float32),
                    h_4=torch.tensor(
                        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
                        if positive
                        else [[-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]],
                        dtype=torch.float32,
                    ),
                    source_labels=source_labels,
                    final_alpha=torch.softmax(factual_logits, dim=0),
                )
            )
        for index in range(4):
            positive = index % 2 == 0
            code_logits = torch.tensor(
                [2.0, -2.0] if positive else [-2.0, 2.0],
                dtype=torch.float32,
            )
            examples.append(
                self.RouterDistillationExample(
                    prompt_id=f"code-{index}",
                    prompt=f"Code {index}",
                    split="pilot",
                    target_text=None,
                    tags=(
                        "oracle_alpha",
                        "stratum_code_procedural",
                        "subcategory_python_snippet",
                    ),
                    perturbation_names=(),
                    token_ids=torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
                    h_1=torch.zeros((5, 2), dtype=torch.float32),
                    h_4=torch.tensor(
                        [[0.25, 0.25]] * 5,
                        dtype=torch.float32,
                    ),
                    source_labels=source_labels,
                    final_alpha=torch.softmax(code_logits, dim=0),
                )
            )

        audit = self.audit_router_supervision_granularity(
            examples=examples,
            fixed_input_field="h_4[t]",
            fixed_target_name="oracle_alpha_logit_vector",
            fixed_aggregation="mean_token_logits_then_softmax",
            fixed_router_family="linear",
            hidden_dim=8,
            train_prompt_ids=(
                "fact-0",
                "fact-1",
                "code-0",
                "code-1",
            ),
            eval_prompt_ids=(
                "fact-2",
                "fact-3",
                "code-2",
                "code-3",
            ),
            learning_rate=0.05,
            max_epochs=300,
            patience=60,
            seed=11,
            device="cpu",
        )

        self.assertEqual(4, audit.eval_prompt_count)
        self.assertEqual(
            "stratum_code_procedural", audit.supervision_summary.worst_stratum_tag
        )
        self.assertEqual(
            "richer_token_or_span_supervision",
            audit.supervision_summary.recommended_next_step,
        )
        self.assertGreater(
            audit.supervision_summary.stratum_mean_js_range,
            0.05,
        )

    def test_router_supervision_loss_penalizes_token_disagreement_beyond_sequence_match(
        self,
    ) -> None:
        token_logits = torch.tensor(
            [[[2.0, -2.0], [-2.0, 2.0], [2.0, -2.0]]],
            dtype=torch.float32,
        )
        token_mask = torch.tensor([[True, True, True]])
        targets = torch.tensor([[2.0 / 3.0, -2.0 / 3.0]], dtype=torch.float32)

        sequence_loss = self.router_supervision_loss(
            token_logits=token_logits,
            token_mask=token_mask,
            targets=targets,
            aggregation="mean_token_logits_then_softmax",
            target_name="oracle_alpha_logit_vector",
            supervision_objective="sequence_target_mse",
        )
        all_token_loss = self.router_supervision_loss(
            token_logits=token_logits,
            token_mask=token_mask,
            targets=targets,
            aggregation="mean_token_logits_then_softmax",
            target_name="oracle_alpha_logit_vector",
            supervision_objective="all_tokens_target_mse",
        )

        self.assertAlmostEqual(0.0, float(sequence_loss.item()), places=6)
        self.assertGreater(float(all_token_loss.item()), 0.0)

    def test_summarize_exact_teacher_mismatch_prefers_within_prompt_variance(
        self,
    ) -> None:
        diagnostics = (
            self.RouterDistillationTeacherMismatchPromptDiagnostic(
                prompt_id="prompt-1",
                subset_role="train",
                tags=("stratum_factual_recall",),
                num_teacher_positions=8,
                oracle_entropy=3.2,
                repeated_target_entropy=3.2,
                mean_teacher_entropy=3.6,
                last_teacher_entropy=3.5,
                within_prompt_js_to_mean_teacher=0.16,
                within_prompt_js_to_oracle=0.22,
                mean_teacher_js_to_oracle=0.08,
                last_teacher_js_to_oracle=0.18,
                mean_teacher_js_to_repeated_target=0.08,
                top1_token_agreement_with_oracle=0.05,
                top1_token_agreement_with_mean_teacher=0.08,
            ),
            self.RouterDistillationTeacherMismatchPromptDiagnostic(
                prompt_id="prompt-2",
                subset_role="eval",
                tags=("stratum_reasoning_math",),
                num_teacher_positions=9,
                oracle_entropy=3.3,
                repeated_target_entropy=3.3,
                mean_teacher_entropy=3.7,
                last_teacher_entropy=3.6,
                within_prompt_js_to_mean_teacher=0.15,
                within_prompt_js_to_oracle=0.20,
                mean_teacher_js_to_oracle=0.09,
                last_teacher_js_to_oracle=0.19,
                mean_teacher_js_to_repeated_target=0.09,
                top1_token_agreement_with_oracle=0.04,
                top1_token_agreement_with_mean_teacher=0.09,
            ),
        )

        summary = self.summarize_exact_teacher_mismatch(prompt_diagnostics=diagnostics)

        self.assertEqual(
            "within_prompt_teacher_variance",
            summary.dominant_failure_mechanism,
        )
        self.assertEqual(
            "do_not_expand_exact_tokenwise_teacher_work",
            summary.recommended_next_step,
        )
        self.assertGreater(
            summary.mean_within_prompt_js_to_mean_teacher,
            summary.mean_mean_teacher_js_to_oracle,
        )

    def test_summarize_exact_teacher_mismatch_flags_aggregation_mismatch_when_last_beats_mean(
        self,
    ) -> None:
        diagnostics = (
            self.RouterDistillationTeacherMismatchPromptDiagnostic(
                prompt_id="prompt-1",
                subset_role="train",
                tags=("stratum_factual_recall",),
                num_teacher_positions=6,
                oracle_entropy=3.0,
                repeated_target_entropy=3.0,
                mean_teacher_entropy=3.2,
                last_teacher_entropy=3.1,
                within_prompt_js_to_mean_teacher=0.03,
                within_prompt_js_to_oracle=0.10,
                mean_teacher_js_to_oracle=0.14,
                last_teacher_js_to_oracle=0.05,
                mean_teacher_js_to_repeated_target=0.14,
                top1_token_agreement_with_oracle=0.60,
                top1_token_agreement_with_mean_teacher=0.70,
            ),
        )

        summary = self.summarize_exact_teacher_mismatch(prompt_diagnostics=diagnostics)

        self.assertEqual(
            "sequence_aggregation_mismatch",
            summary.dominant_failure_mechanism,
        )
        self.assertLess(
            summary.mean_last_teacher_js_to_oracle,
            summary.mean_mean_teacher_js_to_oracle,
        )

    def test_audit_exact_tokenwise_teacher_mismatch_reports_variance_when_mean_teacher_matches_oracle(
        self,
    ) -> None:
        source_labels = ("embed", "0_attn_out")
        examples = (
            self.RouterDistillationExample(
                prompt_id="prompt-train",
                prompt="prompt-train",
                split="pilot",
                target_text=None,
                tags=("oracle_alpha", "stratum_factual_recall"),
                perturbation_names=(),
                token_ids=torch.tensor([1, 2, 3], dtype=torch.long),
                h_1=torch.zeros((3, 2), dtype=torch.float32),
                h_4=torch.zeros((3, 2), dtype=torch.float32),
                source_labels=source_labels,
                final_alpha=torch.tensor([0.5, 0.5], dtype=torch.float32),
            ),
            self.RouterDistillationExample(
                prompt_id="prompt-eval",
                prompt="prompt-eval",
                split="pilot",
                target_text=None,
                tags=("oracle_alpha", "stratum_general_text"),
                perturbation_names=(),
                token_ids=torch.tensor([4, 5, 6], dtype=torch.long),
                h_1=torch.zeros((3, 2), dtype=torch.float32),
                h_4=torch.zeros((3, 2), dtype=torch.float32),
                source_labels=source_labels,
                final_alpha=torch.tensor([0.5, 0.5], dtype=torch.float32),
            ),
        )
        tokenwise_teacher_lookup = {
            "prompt-train": (
                torch.tensor(
                    [
                        [2.0, -2.0],
                        [-2.0, 2.0],
                        [0.0, 0.0],
                    ],
                    dtype=torch.float32,
                ),
                torch.tensor([True, True, False]),
            ),
            "prompt-eval": (
                torch.tensor(
                    [
                        [2.0, -2.0],
                        [-2.0, 2.0],
                        [0.0, 0.0],
                    ],
                    dtype=torch.float32,
                ),
                torch.tensor([True, True, False]),
            ),
        }

        audit = self.audit_exact_tokenwise_teacher_mismatch(
            examples=examples,
            train_prompt_ids=("prompt-train",),
            eval_prompt_ids=("prompt-eval",),
            target_name="oracle_alpha_logit_vector",
            tokenwise_teacher_lookup=tokenwise_teacher_lookup,
            collection_id="oracle_alpha_phase1_v1",
            model_name="test-model",
        )

        self.assertEqual(2, audit.mismatch_summary.prompt_count)
        self.assertEqual(
            "within_prompt_teacher_variance",
            audit.mismatch_summary.dominant_failure_mechanism,
        )
        self.assertAlmostEqual(
            0.0,
            audit.prompt_diagnostics[0].mean_teacher_js_to_oracle,
            places=6,
        )
        self.assertGreater(
            audit.prompt_diagnostics[0].within_prompt_js_to_mean_teacher,
            0.1,
        )
        payload = self.compact_router_distillation_teacher_mismatch_payload(audit)
        self.assertIn("mismatch_summary", payload)
        self.assertEqual(2, len(payload["prompt_diagnostics"]))

    def test_tokenwise_oracle_alpha_target_logit_contribution_targets_uses_shared_scale(
        self,
    ) -> None:
        residual_stack = torch.tensor(
            [
                [[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]],
                [[0.0, 2.0], [2.0, 0.0], [1.0, -1.0]],
            ],
            dtype=torch.float32,
        )
        final_alpha = torch.tensor([0.75, 0.25], dtype=torch.float32)
        token_ids = torch.tensor([0, 1, 0], dtype=torch.long)
        final_norm_weight = torch.tensor([1.0, 1.0], dtype=torch.float32)
        unembed = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        teacher_targets, teacher_mask = (
            self.tokenwise_oracle_alpha_target_logit_contribution_targets(
                residual_stack=residual_stack,
                final_alpha=final_alpha,
                token_ids=token_ids,
                final_norm_weight=final_norm_weight,
                unembed=unembed,
                eps=0.0,
            )
        )

        self.assertEqual((3, 2), tuple(teacher_targets.shape))
        self.assertEqual([True, True, False], teacher_mask.tolist())
        self.assertTrue(
            torch.allclose(
                teacher_targets[0],
                torch.tensor(
                    [-0.2236068, 0.2236068],
                    dtype=torch.float32,
                ),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                teacher_targets[1],
                torch.tensor(
                    [-0.2236068, 0.2236068],
                    dtype=torch.float32,
                ),
                atol=1e-6,
            )
        )

    def test_router_supervision_loss_uses_tokenwise_teacher_and_ignores_last_token(
        self,
    ) -> None:
        token_logits = torch.tensor(
            [
                [
                    [0.5, -0.5],
                    [-0.5, 0.5],
                    [9.0, -9.0],
                ]
            ],
            dtype=torch.float32,
        )
        token_mask = torch.tensor([[True, True, True]])
        targets = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        tokenwise_teacher_targets = torch.tensor(
            [
                [
                    [0.5, -0.5],
                    [-0.5, 0.5],
                    [0.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        )
        tokenwise_teacher_mask = torch.tensor([[True, True, False]])

        for supervision_objective in (
            "next_token_positions_oracle_alpha_target_logit_contribution_mse",
            "next_token_positions_exact_oracle_alpha_logit_mse",
        ):
            teacher_loss = self.router_supervision_loss(
                token_logits=token_logits,
                token_mask=token_mask,
                targets=targets,
                aggregation="mean_token_logits_then_softmax",
                target_name="oracle_alpha_logit_vector",
                supervision_objective=supervision_objective,
                tokenwise_teacher_targets=tokenwise_teacher_targets,
                tokenwise_teacher_mask=tokenwise_teacher_mask,
            )
            all_token_loss = self.router_supervision_loss(
                token_logits=token_logits,
                token_mask=token_mask,
                targets=targets,
                aggregation="mean_token_logits_then_softmax",
                target_name="oracle_alpha_logit_vector",
                supervision_objective="all_tokens_target_mse",
            )

            self.assertAlmostEqual(0.0, float(teacher_loss.item()), places=6)
            self.assertGreater(float(all_token_loss.item()), 10.0)

    def test_exact_tokenwise_oracle_alpha_logit_targets_follow_per_position_best_source(
        self,
    ) -> None:
        residual_stack = torch.tensor(
            [
                [[3.0, 0.0], [3.0, 0.0], [0.0, 0.0]],
                [[0.0, 3.0], [0.0, 3.0], [0.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        token_ids = torch.tensor([9, 0, 1], dtype=torch.long)
        final_norm_weight = torch.tensor([1.0, 1.0], dtype=torch.float32)
        unembed = torch.eye(2, dtype=torch.float32)

        teacher_targets, teacher_mask = self.exact_tokenwise_oracle_alpha_logit_targets(
            residual_stack=residual_stack,
            token_ids=token_ids,
            final_norm_weight=final_norm_weight,
            unembed=unembed,
            eps=0.0,
            output_logits_soft_cap=0.0,
            optimization_steps=80,
            learning_rate=0.2,
        )

        self.assertEqual([True, True, False], teacher_mask.tolist())
        self.assertGreater(float(teacher_targets[0, 0]), 0.5)
        self.assertLess(float(teacher_targets[0, 1]), -0.5)
        self.assertLess(float(teacher_targets[1, 0]), -0.5)
        self.assertGreater(float(teacher_targets[1, 1]), 0.5)

    def test_build_oracle_alpha_tokenwise_teacher_lookup_accepts_rmspre(self) -> None:
        example = self.RouterDistillationExample(
            prompt_id="prompt-1",
            prompt="Prompt 1",
            split="pilot",
            target_text=None,
            tags=("oracle_alpha", "stratum_factual_recall", "subcategory_capital_fact"),
            perturbation_names=(),
            token_ids=torch.tensor([0, 1, 0], dtype=torch.long),
            h_1=torch.zeros((3, 2), dtype=torch.float32),
            h_4=torch.zeros((3, 2), dtype=torch.float32),
            source_labels=("embed", "0_attn_out"),
            final_alpha=torch.tensor([0.75, 0.25], dtype=torch.float32),
        )
        residual_stack = torch.tensor(
            [
                [[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]],
                [[0.0, 2.0], [2.0, 0.0], [1.0, -1.0]],
            ],
            dtype=torch.float32,
        )
        fake_model = SimpleNamespace(
            cfg=SimpleNamespace(normalization_type="RMSPre", eps=0.0),
            ln_final=SimpleNamespace(b=None),
            unembed=SimpleNamespace(W_U=torch.eye(2, dtype=torch.float32)),
        )

        with patch(
            "validation.router_distillation._fixed_residual_sources",
            return_value=(
                residual_stack,
                example.source_labels,
                example.token_ids,
            ),
        ):
            lookup = self.build_oracle_alpha_tokenwise_teacher_lookup(
                model=fake_model,
                examples=(example,),
                prompt_ids=(example.prompt_id,),
            )

        teacher_targets, teacher_mask = lookup[example.prompt_id]
        self.assertEqual([True, True, False], teacher_mask.tolist())
        self.assertTrue(
            torch.allclose(
                teacher_targets[0],
                torch.tensor([-0.2236068, 0.2236068], dtype=torch.float32),
                atol=1e-6,
            )
        )

    def test_compare_router_supervision_objectives_prefers_last_third_under_prefix_shift(
        self,
    ) -> None:
        source_labels = ("embed", "0_attn_out")
        subcategories = (
            "subcategory_capital_fact",
            "subcategory_author_fact",
            "subcategory_element_fact",
            "subcategory_city_fact",
        )
        examples = []
        train_prompt_ids = []
        eval_prompt_ids = []
        for subcategory_index, subcategory in enumerate(subcategories):
            for label_name, positive in (("pos", True), ("neg", False)):
                target_logits = torch.tensor(
                    [2.0, -2.0] if positive else [-2.0, 2.0],
                    dtype=torch.float32,
                )
                prefix_value = 1.0 if positive else -1.0
                last_value = 1.0 if positive else -1.0
                prompt_id = f"train-{subcategory_index}-{label_name}"
                train_prompt_ids.append(prompt_id)
                examples.append(
                    self.RouterDistillationExample(
                        prompt_id=prompt_id,
                        prompt=prompt_id,
                        split="pilot",
                        target_text=None,
                        tags=("oracle_alpha", "stratum_factual_recall", subcategory),
                        perturbation_names=(),
                        token_ids=torch.tensor([1, 2, 3], dtype=torch.long),
                        h_1=torch.zeros((3, 2), dtype=torch.float32),
                        h_4=torch.tensor(
                            [
                                [3.0 * prefix_value, 0.0],
                                [3.0 * prefix_value, 0.0],
                                [0.0, last_value],
                            ],
                            dtype=torch.float32,
                        ),
                        source_labels=source_labels,
                        final_alpha=torch.softmax(target_logits, dim=0),
                    )
                )

                eval_prompt_id = f"eval-{subcategory_index}-{label_name}"
                eval_prompt_ids.append(eval_prompt_id)
                flipped_prefix = -prefix_value
                examples.append(
                    self.RouterDistillationExample(
                        prompt_id=eval_prompt_id,
                        prompt=eval_prompt_id,
                        split="pilot",
                        target_text=None,
                        tags=("oracle_alpha", "stratum_factual_recall", subcategory),
                        perturbation_names=(),
                        token_ids=torch.tensor([1, 2, 3], dtype=torch.long),
                        h_1=torch.zeros((3, 2), dtype=torch.float32),
                        h_4=torch.tensor(
                            [
                                [3.0 * flipped_prefix, 0.0],
                                [3.0 * flipped_prefix, 0.0],
                                [0.0, last_value],
                            ],
                            dtype=torch.float32,
                        ),
                        source_labels=source_labels,
                        final_alpha=torch.softmax(target_logits, dim=0),
                    )
                )

        comparison = self.compare_router_supervision_objectives(
            examples=examples,
            fixed_input_field="h_4[t]",
            fixed_target_name="oracle_alpha_logit_vector",
            fixed_aggregation="mean_token_logits_then_softmax",
            fixed_router_family="linear",
            hidden_dim=8,
            train_prompt_ids=tuple(train_prompt_ids),
            eval_prompt_ids=tuple(eval_prompt_ids),
            candidate_supervision_objectives=(
                "sequence_target_mse",
                "last_third_tokens_target_mse",
            ),
            learning_rate=0.05,
            max_epochs=300,
            patience=60,
            seed=11,
            device="cpu",
        )

        self.assertEqual(
            "last_third_tokens_target_mse",
            comparison.selected_supervision_objective,
        )
        summaries = {
            summary.supervision_objective: summary
            for summary in comparison.input_summaries
        }
        self.assertGreater(
            summaries["last_third_tokens_target_mse"].eval_summary.r_squared,
            0.8,
        )
        self.assertGreater(
            summaries["last_third_tokens_target_mse"].eval_summary.r_squared,
            summaries["sequence_target_mse"].eval_summary.r_squared + 0.05,
        )
        self.assertLess(
            summaries["last_third_tokens_target_mse"].eval_summary.mean_js_divergence,
            summaries["sequence_target_mse"].eval_summary.mean_js_divergence,
        )

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
