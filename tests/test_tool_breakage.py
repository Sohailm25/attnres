# ABOUTME: Validates the first routed-versus-original factual-recall tool-breakage helpers.
# ABOUTME: Keeps the Gemma baseline honest about target-token traces and routed-prefix construction.

from __future__ import annotations

import contextlib
import io
import unittest
import warnings

import torch
from transformer_lens import HookedTransformer

from prompts import PromptEntry

warnings.filterwarnings(
    "ignore",
    message=r"`torch_dtype` is deprecated! Use `dtype` instead!",
)


class ToolBreakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from validation.tool_breakage import (
            ToolBreakageLayerTrace,
            ToolBreakagePromptResult,
            ToolBreakageRunSummary,
            ToolBreakageCounterfactualArmResult,
            ToolBreakageCounterfactualLayerTrace,
            ToolBreakageCounterfactualPromptResult,
            build_routed_residual_traces,
            build_counterfactual_alpha_controls,
            summarize_tool_breakage_prompt,
            summarize_tool_breakage_counterfactual_run,
            summarize_tool_breakage_run,
            target_token_for_entry,
        )

        cls.ToolBreakageLayerTrace = ToolBreakageLayerTrace
        cls.ToolBreakagePromptResult = ToolBreakagePromptResult
        cls.ToolBreakageRunSummary = ToolBreakageRunSummary
        cls.ToolBreakageCounterfactualArmResult = ToolBreakageCounterfactualArmResult
        cls.ToolBreakageCounterfactualLayerTrace = ToolBreakageCounterfactualLayerTrace
        cls.ToolBreakageCounterfactualPromptResult = (
            ToolBreakageCounterfactualPromptResult
        )
        cls.build_routed_residual_traces = staticmethod(build_routed_residual_traces)
        cls.build_counterfactual_alpha_controls = staticmethod(
            build_counterfactual_alpha_controls
        )
        cls.summarize_tool_breakage_prompt = staticmethod(
            summarize_tool_breakage_prompt
        )
        cls.summarize_tool_breakage_counterfactual_run = staticmethod(
            summarize_tool_breakage_counterfactual_run
        )
        cls.summarize_tool_breakage_run = staticmethod(summarize_tool_breakage_run)
        cls.target_token_for_entry = staticmethod(target_token_for_entry)

        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(io.StringIO()):
                cls.model = HookedTransformer.from_pretrained(
                    "tiny-stories-1M",
                    device="cpu",
                    dtype=torch.float32,
                )

    def test_build_routed_residual_traces_respects_prefix_alpha(self) -> None:
        residual_stack = torch.tensor(
            [[1.0], [2.0], [4.0], [8.0], [16.0], [32.0]],
            dtype=torch.float32,
        ).reshape(6, 1, 1)
        source_labels = (
            "embed",
            "pos_embed",
            "0_attn_out",
            "0_mlp_out",
            "1_attn_out",
            "1_mlp_out",
        )
        alpha = torch.tensor([0.1, 0.1, 0.2, 0.2, 0.1, 0.3], dtype=torch.float32)

        traces = self.build_routed_residual_traces(
            residual_stack=residual_stack,
            source_labels=source_labels,
            alpha=alpha,
            num_layers=2,
        )

        self.assertEqual((2, 1, 1), tuple(traces.shape))
        self.assertAlmostEqual(4.5, float(traces[0, 0, 0]), places=5)
        self.assertAlmostEqual(13.9, float(traces[1, 0, 0]), places=5)

    def test_summarize_tool_breakage_prompt_tracks_non_monotonicity(self) -> None:
        final_logits = torch.log(
            torch.tensor(
                [
                    [0.2, 0.7, 0.1],
                    [0.1, 0.8, 0.1],
                ],
                dtype=torch.float32,
            )
        )
        raw_original_logits = torch.log(
            torch.tensor(
                [
                    [[0.6, 0.2, 0.2], [0.6, 0.2, 0.2]],
                    [[0.3, 0.4, 0.3], [0.3, 0.4, 0.3]],
                    [[0.2, 0.7, 0.1], [0.1, 0.8, 0.1]],
                ],
                dtype=torch.float32,
            )
        )
        raw_routed_logits = torch.log(
            torch.tensor(
                [
                    [[0.6, 0.2, 0.2], [0.6, 0.2, 0.2]],
                    [[0.2, 0.6, 0.2], [0.2, 0.6, 0.2]],
                    [[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]],
                ],
                dtype=torch.float32,
            )
        )
        tuned_original_logits = torch.log(
            torch.tensor(
                [
                    [[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]],
                    [[0.25, 0.55, 0.2], [0.2, 0.55, 0.25]],
                    [[0.2, 0.7, 0.1], [0.1, 0.8, 0.1]],
                ],
                dtype=torch.float32,
            )
        )
        tuned_routed_logits = torch.log(
            torch.tensor(
                [
                    [[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]],
                    [[0.15, 0.7, 0.15], [0.15, 0.7, 0.15]],
                    [[0.4, 0.45, 0.15], [0.4, 0.45, 0.15]],
                ],
                dtype=torch.float32,
            )
        )

        prompt_result = self.summarize_tool_breakage_prompt(
            model_name="synthetic",
            prompt_id="tb-pilot-001",
            prompt="The capital of France is",
            split="pilot",
            target_text="Paris",
            target_token_id=1,
            target_token_text=" Paris",
            oracle_alpha=(0.2, 0.3, 0.5),
            source_labels=("embed", "pos_embed", "0_attn_out"),
            final_logits=final_logits,
            raw_original_logits=raw_original_logits,
            raw_routed_logits=raw_routed_logits,
            tuned_original_logits=tuned_original_logits,
            tuned_routed_logits=tuned_routed_logits,
        )

        self.assertFalse(prompt_result.raw_original_non_monotonic)
        self.assertTrue(prompt_result.raw_routed_non_monotonic)
        self.assertFalse(prompt_result.tuned_original_non_monotonic)
        self.assertTrue(prompt_result.tuned_routed_non_monotonic)
        self.assertAlmostEqual(
            0.0,
            prompt_result.layer_traces[-1].raw_original_final_position_kl_to_final,
            places=6,
        )
        self.assertEqual(
            1,
            prompt_result.layer_traces[-1].raw_original_final_position_target_rank,
        )

        run_summary = self.summarize_tool_breakage_run(
            model_name="synthetic",
            collection_id="tool_breakage_factual_recall_v1",
            split="pilot",
            tuned_lens_checkpoint_path="checkpoint.pt",
            prompt_results=(prompt_result,),
        )
        self.assertEqual(1, run_summary.num_prompts)
        self.assertEqual(0.0, run_summary.fraction_raw_original_non_monotonic_prompts)
        self.assertEqual(1.0, run_summary.fraction_raw_routed_non_monotonic_prompts)
        self.assertEqual(
            1.0, run_summary.fraction_raw_routing_increases_non_monotonicity_prompts
        )
        self.assertEqual(
            1.0, run_summary.fraction_tuned_routing_increases_non_monotonicity_prompts
        )
        self.assertEqual(
            1.0, run_summary.fraction_raw_routing_worsens_final_target_rank_prompts
        )
        self.assertEqual(
            0.0, run_summary.fraction_tuned_routing_worsens_final_target_rank_prompts
        )
        self.assertEqual(
            0.0, run_summary.fraction_raw_routing_worsens_best_target_rank_prompts
        )
        self.assertEqual(
            0.0, run_summary.fraction_tuned_routing_worsens_best_target_rank_prompts
        )
        self.assertEqual(
            0.0, run_summary.fraction_raw_routing_increases_target_rank_range_prompts
        )
        self.assertEqual(
            0.0, run_summary.fraction_tuned_routing_increases_target_rank_range_prompts
        )

    def test_summarize_tool_breakage_run_tracks_relative_rank_instability(self) -> None:
        def make_trace(
            *,
            layer: int,
            raw_original_rank: int,
            raw_routed_rank: int,
            tuned_original_rank: int,
            tuned_routed_rank: int,
        ) -> object:
            return self.ToolBreakageLayerTrace(
                layer=layer,
                raw_original_mean_kl_to_final=0.0,
                raw_routed_mean_kl_to_final=0.0,
                tuned_original_mean_kl_to_final=0.0,
                tuned_routed_mean_kl_to_final=0.0,
                raw_original_mean_top1_agreement=0.0,
                raw_routed_mean_top1_agreement=0.0,
                tuned_original_mean_top1_agreement=0.0,
                tuned_routed_mean_top1_agreement=0.0,
                raw_original_final_position_kl_to_final=0.0,
                raw_routed_final_position_kl_to_final=0.0,
                tuned_original_final_position_kl_to_final=0.0,
                tuned_routed_final_position_kl_to_final=0.0,
                raw_original_final_position_top1_agreement=0.0,
                raw_routed_final_position_top1_agreement=0.0,
                tuned_original_final_position_top1_agreement=0.0,
                tuned_routed_final_position_top1_agreement=0.0,
                raw_original_final_position_target_probability=0.2,
                raw_routed_final_position_target_probability=0.1,
                tuned_original_final_position_target_probability=0.3,
                tuned_routed_final_position_target_probability=0.1,
                raw_original_final_position_target_rank=raw_original_rank,
                raw_routed_final_position_target_rank=raw_routed_rank,
                tuned_original_final_position_target_rank=tuned_original_rank,
                tuned_routed_final_position_target_rank=tuned_routed_rank,
            )

        prompt_result = self.ToolBreakagePromptResult(
            prompt_id="tb-pilot-xyz",
            prompt="Synthetic prompt",
            split="pilot",
            target_text="answer",
            target_token_id=1,
            target_token_text=" answer",
            source_labels=("embed", "0_attn_out"),
            oracle_alpha=(0.4, 0.6),
            raw_original_non_monotonic=True,
            raw_routed_non_monotonic=True,
            tuned_original_non_monotonic=True,
            tuned_routed_non_monotonic=True,
            layer_traces=(
                make_trace(
                    layer=0,
                    raw_original_rank=4,
                    raw_routed_rank=6,
                    tuned_original_rank=3,
                    tuned_routed_rank=9,
                ),
                make_trace(
                    layer=1,
                    raw_original_rank=2,
                    raw_routed_rank=10,
                    tuned_original_rank=1,
                    tuned_routed_rank=4,
                ),
            ),
        )

        run_summary = self.summarize_tool_breakage_run(
            model_name="synthetic",
            collection_id="tool_breakage_factual_recall_v1",
            split="pilot",
            tuned_lens_checkpoint_path="checkpoint.pt",
            prompt_results=(prompt_result,),
        )

        self.assertEqual(
            1.0, run_summary.fraction_raw_routing_worsens_final_target_rank_prompts
        )
        self.assertEqual(
            1.0, run_summary.fraction_tuned_routing_worsens_final_target_rank_prompts
        )
        self.assertEqual(
            1.0, run_summary.fraction_raw_routing_worsens_best_target_rank_prompts
        )
        self.assertEqual(
            1.0, run_summary.fraction_tuned_routing_worsens_best_target_rank_prompts
        )
        self.assertEqual(
            1.0, run_summary.fraction_raw_routing_increases_target_rank_range_prompts
        )
        self.assertEqual(
            1.0, run_summary.fraction_tuned_routing_increases_target_rank_range_prompts
        )

    def test_target_token_for_entry_uses_prompt_plus_target_tokenization(self) -> None:
        entry = PromptEntry(
            prompt_id="tb-a",
            text="The cat sat on the",
            split="pilot",
            tags=("tiny",),
            perturbations={},
            target_text="mat",
        )

        token_id, token_text = self.target_token_for_entry(
            model=self.model,
            entry=entry,
            prepend_bos=False,
        )
        prompt_tokens = self.model.to_tokens(entry.text, prepend_bos=False)
        full_tokens = self.model.to_tokens(
            f"{entry.text} {entry.target_text}", prepend_bos=False
        )
        expected_token_id = int(full_tokens[0, prompt_tokens.shape[-1]].item())

        self.assertEqual(expected_token_id, token_id)
        self.assertTrue(token_text)

    def test_build_counterfactual_alpha_controls_uses_cyclic_shift_and_fixed_mean(
        self,
    ) -> None:
        def make_prompt_result(prompt_id: str, alpha: tuple[float, ...]) -> object:
            return self.ToolBreakagePromptResult(
                prompt_id=prompt_id,
                prompt=f"Prompt {prompt_id}",
                split="confirm",
                target_text="answer",
                target_token_id=1,
                target_token_text=" answer",
                source_labels=("embed", "0_attn_out"),
                oracle_alpha=alpha,
                raw_original_non_monotonic=False,
                raw_routed_non_monotonic=True,
                tuned_original_non_monotonic=False,
                tuned_routed_non_monotonic=True,
                layer_traces=(),
            )

        confirm_prompt_results = (
            make_prompt_result("tb-confirm-001", (0.9, 0.1)),
            make_prompt_result("tb-confirm-002", (0.2, 0.8)),
            make_prompt_result("tb-confirm-003", (0.6, 0.4)),
        )
        pilot_prompt_results = (
            make_prompt_result("tb-pilot-001", (0.8, 0.2)),
            make_prompt_result("tb-pilot-002", (0.4, 0.6)),
        )

        controls = self.build_counterfactual_alpha_controls(
            confirm_prompt_results=confirm_prompt_results,
            fixed_alpha_prompt_results=pilot_prompt_results,
        )

        self.assertEqual(
            ("pilot_mean_alpha", "prompt_permuted_alpha"),
            tuple(sorted(controls["tb-confirm-001"].keys())),
        )
        self.assertEqual(
            "tb-confirm-002",
            controls["tb-confirm-001"]["prompt_permuted_alpha"].alpha_source_prompt_id,
        )
        self.assertEqual(
            (0.2, 0.8),
            controls["tb-confirm-001"]["prompt_permuted_alpha"].alpha,
        )
        self.assertEqual(
            (0.6, 0.4),
            controls["tb-confirm-002"]["prompt_permuted_alpha"].alpha,
        )
        self.assertEqual(
            (0.9, 0.1),
            controls["tb-confirm-003"]["prompt_permuted_alpha"].alpha,
        )
        self.assertAlmostEqual(
            0.6,
            controls["tb-confirm-001"]["pilot_mean_alpha"].alpha[0],
            places=6,
        )
        self.assertAlmostEqual(
            0.4,
            controls["tb-confirm-001"]["pilot_mean_alpha"].alpha[1],
            places=6,
        )
        self.assertIsNone(
            controls["tb-confirm-001"]["pilot_mean_alpha"].alpha_source_prompt_id
        )

    def test_summarize_tool_breakage_counterfactual_run_tracks_routed_vs_control_metrics(
        self,
    ) -> None:
        def make_baseline_trace(
            *,
            layer: int,
            raw_original_rank: int,
            raw_routed_rank: int,
            tuned_original_rank: int,
            tuned_routed_rank: int,
            tuned_original_kl: float,
            tuned_routed_kl: float,
        ) -> object:
            return self.ToolBreakageLayerTrace(
                layer=layer,
                raw_original_mean_kl_to_final=0.0,
                raw_routed_mean_kl_to_final=0.0,
                tuned_original_mean_kl_to_final=tuned_original_kl,
                tuned_routed_mean_kl_to_final=tuned_routed_kl,
                raw_original_mean_top1_agreement=0.0,
                raw_routed_mean_top1_agreement=0.0,
                tuned_original_mean_top1_agreement=0.5,
                tuned_routed_mean_top1_agreement=0.2,
                raw_original_final_position_kl_to_final=0.0,
                raw_routed_final_position_kl_to_final=0.0,
                tuned_original_final_position_kl_to_final=tuned_original_kl + 0.5,
                tuned_routed_final_position_kl_to_final=tuned_routed_kl + 0.5,
                raw_original_final_position_top1_agreement=0.0,
                raw_routed_final_position_top1_agreement=0.0,
                tuned_original_final_position_top1_agreement=0.0,
                tuned_routed_final_position_top1_agreement=0.0,
                raw_original_final_position_target_probability=0.2,
                raw_routed_final_position_target_probability=0.1,
                tuned_original_final_position_target_probability=0.2,
                tuned_routed_final_position_target_probability=0.1,
                raw_original_final_position_target_rank=raw_original_rank,
                raw_routed_final_position_target_rank=raw_routed_rank,
                tuned_original_final_position_target_rank=tuned_original_rank,
                tuned_routed_final_position_target_rank=tuned_routed_rank,
            )

        def make_control_trace(
            *,
            layer: int,
            raw_rank: int,
            tuned_rank: int,
            tuned_kl: float,
        ) -> object:
            return self.ToolBreakageCounterfactualLayerTrace(
                layer=layer,
                raw_mean_kl_to_final=0.0,
                tuned_mean_kl_to_final=tuned_kl,
                raw_mean_top1_agreement=0.0,
                tuned_mean_top1_agreement=0.4,
                raw_final_position_kl_to_final=0.0,
                tuned_final_position_kl_to_final=tuned_kl + 0.5,
                raw_final_position_top1_agreement=0.0,
                tuned_final_position_top1_agreement=0.0,
                raw_final_position_target_probability=0.15,
                tuned_final_position_target_probability=0.12,
                raw_final_position_target_rank=raw_rank,
                tuned_final_position_target_rank=tuned_rank,
            )

        baseline_prompt_result = self.ToolBreakagePromptResult(
            prompt_id="tb-confirm-xyz",
            prompt="Synthetic prompt",
            split="confirm",
            target_text="answer",
            target_token_id=1,
            target_token_text=" answer",
            source_labels=("embed", "0_attn_out"),
            oracle_alpha=(0.4, 0.6),
            raw_original_non_monotonic=True,
            raw_routed_non_monotonic=True,
            tuned_original_non_monotonic=False,
            tuned_routed_non_monotonic=True,
            layer_traces=(
                make_baseline_trace(
                    layer=0,
                    raw_original_rank=4,
                    raw_routed_rank=7,
                    tuned_original_rank=3,
                    tuned_routed_rank=8,
                    tuned_original_kl=1.0,
                    tuned_routed_kl=3.0,
                ),
                make_baseline_trace(
                    layer=1,
                    raw_original_rank=2,
                    raw_routed_rank=9,
                    tuned_original_rank=2,
                    tuned_routed_rank=10,
                    tuned_original_kl=1.2,
                    tuned_routed_kl=3.2,
                ),
            ),
        )
        prompt_result = self.ToolBreakageCounterfactualPromptResult(
            baseline_prompt_result=baseline_prompt_result,
            counterfactual_results=(
                self.ToolBreakageCounterfactualArmResult(
                    arm_name="prompt_permuted_alpha",
                    alpha=(0.6, 0.4),
                    alpha_source_prompt_id="tb-confirm-abc",
                    raw_non_monotonic=True,
                    tuned_non_monotonic=False,
                    layer_traces=(
                        make_control_trace(
                            layer=0,
                            raw_rank=5,
                            tuned_rank=5,
                            tuned_kl=2.0,
                        ),
                        make_control_trace(
                            layer=1,
                            raw_rank=4,
                            tuned_rank=6,
                            tuned_kl=2.1,
                        ),
                    ),
                ),
            ),
        )

        summary = self.summarize_tool_breakage_counterfactual_run(
            model_name="synthetic",
            collection_id="tool_breakage_factual_recall_v1",
            split="confirm",
            tuned_lens_checkpoint_path="checkpoint.pt",
            baseline_summary_path="baseline-summary.json",
            fixed_alpha_summary_path="pilot-summary.json",
            prompt_results=(prompt_result,),
        )

        self.assertEqual(1, summary.num_prompts)
        self.assertEqual(1, len(summary.control_summaries))
        control_summary = summary.control_summaries[0]
        self.assertEqual("prompt_permuted_alpha", control_summary.arm_name)
        self.assertAlmostEqual(
            0.95,
            control_summary.mean_tuned_kl_delta_arm_minus_original,
            places=6,
        )
        self.assertAlmostEqual(
            1.05,
            control_summary.mean_tuned_kl_delta_routed_minus_arm,
            places=6,
        )
        self.assertEqual(
            1.0,
            control_summary.fraction_tuned_routed_worsens_final_target_rank_vs_arm_prompts,
        )
        self.assertEqual(
            1.0,
            control_summary.fraction_tuned_routed_increases_target_rank_range_vs_arm_prompts,
        )


if __name__ == "__main__":
    unittest.main()
