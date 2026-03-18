# ABOUTME: Tests the saved-artifact family-conditioned donor-arm profile for Gemma tool-breakage.
# ABOUTME: Keeps the post-run heterogeneity analysis deterministic and prompt-level.

from __future__ import annotations

import unittest


class ToolBreakageFamilyProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from validation.tool_breakage_family_profile import (
            build_tool_breakage_counterfactual_tag_profile_summary,
            build_tool_breakage_family_profile_summary,
        )

        cls.build_tool_breakage_counterfactual_tag_profile_summary = staticmethod(
            build_tool_breakage_counterfactual_tag_profile_summary
        )
        cls.build_tool_breakage_family_profile_summary = staticmethod(
            build_tool_breakage_family_profile_summary
        )

    @staticmethod
    def _sample_counterfactual_prompt_results() -> tuple[dict[str, object], ...]:
        return (
            {
                "baseline_prompt_result": {
                    "prompt_id": "tb-confirm-001",
                    "prompt": "The author of Pride and Prejudice was",
                    "split": "confirm",
                    "target_text": "Jane Austen",
                    "source_labels": ("embed", "0_attn_out"),
                    "oracle_alpha": (0.85, 0.15),
                    "layer_traces": (
                        {
                            "tuned_routed_mean_kl_to_final": 1.2,
                            "tuned_routed_final_position_kl_to_final": 0.6,
                            "tuned_routed_final_position_target_rank": 3,
                        },
                        {
                            "tuned_routed_mean_kl_to_final": 1.6,
                            "tuned_routed_final_position_kl_to_final": 1.1,
                            "tuned_routed_final_position_target_rank": 6,
                        },
                    ),
                },
                "counterfactual_results": (
                    {
                        "arm_name": "within_family_permuted_alpha",
                        "alpha": (0.70, 0.30),
                        "alpha_source_prompt_id": "tb-confirm-002",
                        "layer_traces": (
                            {
                                "tuned_mean_kl_to_final": 1.0,
                                "tuned_final_position_kl_to_final": 0.5,
                                "tuned_final_position_target_rank": 2,
                            },
                            {
                                "tuned_mean_kl_to_final": 1.1,
                                "tuned_final_position_kl_to_final": 0.8,
                                "tuned_final_position_target_rank": 4,
                            },
                        ),
                    },
                    {
                        "arm_name": "cross_family_permuted_alpha",
                        "alpha": (0.35, 0.65),
                        "alpha_source_prompt_id": "tb-confirm-003",
                        "layer_traces": (
                            {
                                "tuned_mean_kl_to_final": 0.7,
                                "tuned_final_position_kl_to_final": 0.3,
                                "tuned_final_position_target_rank": 1,
                            },
                            {
                                "tuned_mean_kl_to_final": 0.9,
                                "tuned_final_position_kl_to_final": 0.4,
                                "tuned_final_position_target_rank": 2,
                            },
                        ),
                    },
                ),
            },
            {
                "baseline_prompt_result": {
                    "prompt_id": "tb-confirm-002",
                    "prompt": "The author of Emma was",
                    "split": "confirm",
                    "target_text": "Jane Austen",
                    "source_labels": ("embed", "0_attn_out"),
                    "oracle_alpha": (0.75, 0.25),
                    "layer_traces": (
                        {
                            "tuned_routed_mean_kl_to_final": 0.9,
                            "tuned_routed_final_position_kl_to_final": 0.4,
                            "tuned_routed_final_position_target_rank": 1,
                        },
                        {
                            "tuned_routed_mean_kl_to_final": 1.0,
                            "tuned_routed_final_position_kl_to_final": 0.5,
                            "tuned_routed_final_position_target_rank": 2,
                        },
                    ),
                },
                "counterfactual_results": (
                    {
                        "arm_name": "within_family_permuted_alpha",
                        "alpha": (0.85, 0.15),
                        "alpha_source_prompt_id": "tb-confirm-001",
                        "layer_traces": (
                            {
                                "tuned_mean_kl_to_final": 0.8,
                                "tuned_final_position_kl_to_final": 0.3,
                                "tuned_final_position_target_rank": 1,
                            },
                            {
                                "tuned_mean_kl_to_final": 0.9,
                                "tuned_final_position_kl_to_final": 0.4,
                                "tuned_final_position_target_rank": 1,
                            },
                        ),
                    },
                    {
                        "arm_name": "cross_family_permuted_alpha",
                        "alpha": (0.40, 0.60),
                        "alpha_source_prompt_id": "tb-confirm-003",
                        "layer_traces": (
                            {
                                "tuned_mean_kl_to_final": 0.7,
                                "tuned_final_position_kl_to_final": 0.2,
                                "tuned_final_position_target_rank": 1,
                            },
                            {
                                "tuned_mean_kl_to_final": 0.8,
                                "tuned_final_position_kl_to_final": 0.3,
                                "tuned_final_position_target_rank": 1,
                            },
                        ),
                    },
                ),
            },
            {
                "baseline_prompt_result": {
                    "prompt_id": "tb-confirm-003",
                    "prompt": "The capital of Italy is",
                    "split": "confirm",
                    "target_text": "Rome",
                    "source_labels": ("embed", "0_attn_out"),
                    "oracle_alpha": (0.10, 0.90),
                    "layer_traces": (
                        {
                            "tuned_routed_mean_kl_to_final": 1.3,
                            "tuned_routed_final_position_kl_to_final": 0.7,
                            "tuned_routed_final_position_target_rank": 5,
                        },
                        {
                            "tuned_routed_mean_kl_to_final": 1.1,
                            "tuned_routed_final_position_kl_to_final": 0.6,
                            "tuned_routed_final_position_target_rank": 4,
                        },
                    ),
                },
                "counterfactual_results": (
                    {
                        "arm_name": "within_family_permuted_alpha",
                        "alpha": (0.20, 0.80),
                        "alpha_source_prompt_id": "tb-confirm-004",
                        "layer_traces": (
                            {
                                "tuned_mean_kl_to_final": 1.0,
                                "tuned_final_position_kl_to_final": 0.5,
                                "tuned_final_position_target_rank": 2,
                            },
                            {
                                "tuned_mean_kl_to_final": 0.8,
                                "tuned_final_position_kl_to_final": 0.4,
                                "tuned_final_position_target_rank": 2,
                            },
                        ),
                    },
                    {
                        "arm_name": "cross_family_permuted_alpha",
                        "alpha": (0.85, 0.15),
                        "alpha_source_prompt_id": "tb-confirm-001",
                        "layer_traces": (
                            {
                                "tuned_mean_kl_to_final": 0.9,
                                "tuned_final_position_kl_to_final": 0.4,
                                "tuned_final_position_target_rank": 2,
                            },
                            {
                                "tuned_mean_kl_to_final": 0.7,
                                "tuned_final_position_kl_to_final": 0.3,
                                "tuned_final_position_target_rank": 2,
                            },
                        ),
                    },
                ),
            },
            {
                "baseline_prompt_result": {
                    "prompt_id": "tb-confirm-004",
                    "prompt": "The capital of Spain is",
                    "split": "confirm",
                    "target_text": "Madrid",
                    "source_labels": ("embed", "0_attn_out"),
                    "oracle_alpha": (0.15, 0.85),
                    "layer_traces": (
                        {
                            "tuned_routed_mean_kl_to_final": 0.7,
                            "tuned_routed_final_position_kl_to_final": 0.4,
                            "tuned_routed_final_position_target_rank": 2,
                        },
                        {
                            "tuned_routed_mean_kl_to_final": 0.8,
                            "tuned_routed_final_position_kl_to_final": 0.5,
                            "tuned_routed_final_position_target_rank": 3,
                        },
                    ),
                },
                "counterfactual_results": (
                    {
                        "arm_name": "within_family_permuted_alpha",
                        "alpha": (0.10, 0.90),
                        "alpha_source_prompt_id": "tb-confirm-003",
                        "layer_traces": (
                            {
                                "tuned_mean_kl_to_final": 0.8,
                                "tuned_final_position_kl_to_final": 0.6,
                                "tuned_final_position_target_rank": 3,
                            },
                            {
                                "tuned_mean_kl_to_final": 0.9,
                                "tuned_final_position_kl_to_final": 0.7,
                                "tuned_final_position_target_rank": 4,
                            },
                        ),
                    },
                    {
                        "arm_name": "cross_family_permuted_alpha",
                        "alpha": (0.75, 0.25),
                        "alpha_source_prompt_id": "tb-confirm-002",
                        "layer_traces": (
                            {
                                "tuned_mean_kl_to_final": 1.1,
                                "tuned_final_position_kl_to_final": 0.8,
                                "tuned_final_position_target_rank": 5,
                            },
                            {
                                "tuned_mean_kl_to_final": 1.0,
                                "tuned_final_position_kl_to_final": 0.6,
                                "tuned_final_position_target_rank": 4,
                            },
                        ),
                    },
                ),
            },
        )

    @staticmethod
    def _sample_prompt_tags_by_id() -> dict[str, tuple[str, ...]]:
        return {
            "tb-confirm-001": (
                "subcategory_author_fact",
                "route_mode_author_cluster_6",
            ),
            "tb-confirm-002": (
                "subcategory_author_fact",
                "route_mode_author_cluster_6",
            ),
            "tb-confirm-003": (
                "subcategory_capital_fact",
                "route_mode_capital_cluster_2",
            ),
            "tb-confirm-004": (
                "subcategory_capital_fact",
                "route_mode_capital_cluster_2",
            ),
        }

    def test_family_profile_summary_tracks_prompt_level_donor_assignments(self) -> None:
        counterfactual_prompt_results = self._sample_counterfactual_prompt_results()
        prompt_tags_by_id = {
            prompt_id: (tags[0],)
            for prompt_id, tags in self._sample_prompt_tags_by_id().items()
        }

        summary = self.build_tool_breakage_family_profile_summary(
            counterfactual_prompt_results=counterfactual_prompt_results,
            prompt_tags_by_id=prompt_tags_by_id,
        )

        self.assertEqual(4, summary["num_prompts"])
        self.assertEqual(
            ["cross_family_permuted_alpha", "within_family_permuted_alpha"],
            summary["arm_names"],
        )
        self.assertEqual(
            2, summary["by_subcategory"]["subcategory_author_fact"]["prompt_count"]
        )
        self.assertEqual(
            1.0,
            summary["by_subcategory"]["subcategory_author_fact"][
                "multiword_target_fraction"
            ],
        )
        self.assertEqual(
            0.0,
            summary["by_subcategory"]["subcategory_capital_fact"][
                "multiword_target_fraction"
            ],
        )
        self.assertAlmostEqual(
            0.225,
            summary["by_subcategory"]["subcategory_author_fact"]["arm_summaries"][
                "within_family_permuted_alpha"
            ]["mean_tuned_kl_delta_routed_minus_arm"],
        )
        self.assertEqual(
            1.0,
            summary["by_subcategory"]["subcategory_author_fact"]["arm_summaries"][
                "within_family_permuted_alpha"
            ]["fraction_routed_worse_final_target_rank_vs_arm"],
        )
        self.assertAlmostEqual(
            0.1,
            summary["by_subcategory"]["subcategory_capital_fact"]["arm_summaries"][
                "within_family_permuted_alpha"
            ]["mean_tuned_kl_delta_routed_minus_arm"],
        )
        self.assertEqual(
            "tb-confirm-002",
            summary["prompt_profiles"][0]["arm_profiles"][
                "within_family_permuted_alpha"
            ]["alpha_source_prompt_id"],
        )
        self.assertEqual(
            "Jane Austen",
            summary["prompt_profiles"][0]["arm_profiles"][
                "within_family_permuted_alpha"
            ]["alpha_source_target_text"],
        )
        self.assertEqual(
            "subcategory_author_fact",
            summary["prompt_profiles"][0]["arm_profiles"][
                "within_family_permuted_alpha"
            ]["alpha_source_subcategory"],
        )

    def test_counterfactual_tag_profile_summary_groups_by_route_mode(self) -> None:
        summary = self.build_tool_breakage_counterfactual_tag_profile_summary(
            counterfactual_prompt_results=self._sample_counterfactual_prompt_results(),
            prompt_tags_by_id=self._sample_prompt_tags_by_id(),
            group_tag_prefixes=("subcategory_", "route_mode_"),
        )

        self.assertEqual(["subcategory_", "route_mode_"], summary["group_tag_prefixes"])
        self.assertEqual(
            "route_mode_author_cluster_6",
            summary["prompt_profiles"][0]["tags_by_prefix"]["route_mode_"],
        )
        self.assertEqual(
            2,
            summary["group_summaries"]["route_mode_"]["by_tag"][
                "route_mode_author_cluster_6"
            ]["prompt_count"],
        )
        self.assertAlmostEqual(
            0.225,
            summary["group_summaries"]["route_mode_"]["by_tag"][
                "route_mode_author_cluster_6"
            ]["arm_summaries"]["within_family_permuted_alpha"][
                "mean_tuned_kl_delta_routed_minus_arm"
            ],
        )

    def test_family_profile_summary_requires_single_subcategory_tag(self) -> None:
        with self.assertRaises(ValueError):
            self.build_tool_breakage_family_profile_summary(
                counterfactual_prompt_results=(
                    {
                        "baseline_prompt_result": {
                            "prompt_id": "tb-confirm-001",
                            "prompt": "The capital of Italy is",
                            "split": "confirm",
                            "target_text": "Rome",
                            "source_labels": ("embed", "0_attn_out"),
                            "oracle_alpha": (0.5, 0.5),
                            "layer_traces": (
                                {
                                    "tuned_routed_mean_kl_to_final": 1.0,
                                    "tuned_routed_final_position_kl_to_final": 0.5,
                                    "tuned_routed_final_position_target_rank": 2,
                                },
                            ),
                        },
                        "counterfactual_results": (
                            {
                                "arm_name": "within_family_permuted_alpha",
                                "alpha": (0.5, 0.5),
                                "alpha_source_prompt_id": None,
                                "layer_traces": (
                                    {
                                        "tuned_mean_kl_to_final": 0.8,
                                        "tuned_final_position_kl_to_final": 0.4,
                                        "tuned_final_position_target_rank": 1,
                                    },
                                ),
                            },
                        ),
                    },
                ),
                prompt_tags_by_id={"tb-confirm-001": ("tool_breakage",)},
            )
