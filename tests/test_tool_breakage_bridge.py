# ABOUTME: Tests the saved-artifact bridge between factual-recall oracle clusters and Gemma tool-breakage prompts.
# ABOUTME: Keeps the cluster-family bridge analysis deterministic and derived from existing artifacts.

from __future__ import annotations

import unittest


class ToolBreakageBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from validation.tool_breakage_bridge import (
            build_tool_breakage_factual_bridge_summary,
            build_tool_breakage_route_mode_coverage_summary,
        )

        cls.build_tool_breakage_factual_bridge_summary = staticmethod(
            build_tool_breakage_factual_bridge_summary
        )
        cls.build_tool_breakage_route_mode_coverage_summary = staticmethod(
            build_tool_breakage_route_mode_coverage_summary
        )

    def test_bridge_summary_assigns_prompts_to_nearest_factual_clusters(self) -> None:
        factual_prompt_results = (
            {
                "prompt_id": "oa-confirm-001",
                "final_alpha": (0.90, 0.08, 0.02),
                "source_labels": ("embed", "0_attn_out", "0_mlp_out"),
            },
            {
                "prompt_id": "oa-confirm-002",
                "final_alpha": (0.85, 0.10, 0.05),
                "source_labels": ("embed", "0_attn_out", "0_mlp_out"),
            },
            {
                "prompt_id": "oa-confirm-003",
                "final_alpha": (0.05, 0.10, 0.85),
                "source_labels": ("embed", "0_attn_out", "0_mlp_out"),
            },
            {
                "prompt_id": "oa-confirm-004",
                "final_alpha": (0.02, 0.08, 0.90),
                "source_labels": ("embed", "0_attn_out", "0_mlp_out"),
            },
        )
        factual_tags_by_prompt_id = {
            "oa-confirm-001": ("stratum_factual_recall", "subcategory_capital_fact"),
            "oa-confirm-002": ("stratum_factual_recall", "subcategory_capital_fact"),
            "oa-confirm-003": ("stratum_factual_recall", "subcategory_author_fact"),
            "oa-confirm-004": ("stratum_factual_recall", "subcategory_author_fact"),
        }
        tool_prompt_results = (
            {
                "prompt_id": "tb-confirm-001",
                "prompt": "The capital of Italy is",
                "split": "confirm",
                "target_text": "Rome",
                "source_labels": ("embed", "0_attn_out", "0_mlp_out"),
                "oracle_alpha": (0.88, 0.09, 0.03),
                "layer_traces": (
                    {
                        "tuned_original_final_position_target_rank": 1,
                        "tuned_routed_final_position_target_rank": 2,
                        "tuned_original_final_position_kl_to_final": 0.6,
                        "tuned_routed_final_position_kl_to_final": 1.1,
                    },
                    {
                        "tuned_original_final_position_target_rank": 1,
                        "tuned_routed_final_position_target_rank": 3,
                        "tuned_original_final_position_kl_to_final": 0.5,
                        "tuned_routed_final_position_kl_to_final": 1.4,
                    },
                ),
            },
            {
                "prompt_id": "tb-confirm-002",
                "prompt": "The author of Pride and Prejudice was",
                "split": "confirm",
                "target_text": "Jane Austen",
                "source_labels": ("embed", "0_attn_out", "0_mlp_out"),
                "oracle_alpha": (0.04, 0.12, 0.84),
                "layer_traces": (
                    {
                        "tuned_original_final_position_target_rank": 2,
                        "tuned_routed_final_position_target_rank": 2,
                        "tuned_original_final_position_kl_to_final": 0.7,
                        "tuned_routed_final_position_kl_to_final": 0.8,
                    },
                    {
                        "tuned_original_final_position_target_rank": 3,
                        "tuned_routed_final_position_target_rank": 5,
                        "tuned_original_final_position_kl_to_final": 0.9,
                        "tuned_routed_final_position_kl_to_final": 1.5,
                    },
                ),
            },
        )
        tool_tags_by_prompt_id = {
            "tb-confirm-001": ("factual_recall", "subcategory_capital_fact"),
            "tb-confirm-002": ("factual_recall", "subcategory_author_fact"),
        }

        summary = self.build_tool_breakage_factual_bridge_summary(
            factual_prompt_results=factual_prompt_results,
            factual_tags_by_prompt_id=factual_tags_by_prompt_id,
            tool_prompt_results=tool_prompt_results,
            tool_tags_by_prompt_id=tool_tags_by_prompt_id,
            cluster_count=2,
        )

        self.assertEqual(2, summary["num_tool_prompts"])
        self.assertEqual(2, summary["num_factual_clusters"])
        self.assertEqual(1.0, summary["matching_subcategory_fraction"])
        self.assertEqual(1.0, summary["overlap_prompt_fraction"])
        self.assertEqual(
            "subcategory_capital_fact",
            summary["prompt_assignments"][0]["nearest_cluster_dominant_subcategory"],
        )
        self.assertEqual(
            "subcategory_author_fact",
            summary["prompt_assignments"][1]["nearest_cluster_dominant_subcategory"],
        )
        self.assertEqual(
            2,
            summary["prompt_assignments"][0][
                "tuned_final_target_rank_delta_under_routing"
            ],
        )
        self.assertEqual(
            2,
            summary["prompt_assignments"][1][
                "tuned_target_rank_range_delta_under_routing"
            ],
        )
        self.assertAlmostEqual(
            0.9,
            summary["prompt_assignments"][0][
                "tuned_final_position_kl_delta_under_routing"
            ],
        )
        self.assertEqual(
            1,
            summary["by_tool_subcategory"]["subcategory_capital_fact"]["prompt_count"],
        )
        self.assertEqual(
            1,
            summary["by_nearest_cluster_subcategory"]["subcategory_author_fact"][
                "prompt_count"
            ],
        )

    def test_route_mode_coverage_summary_tracks_covered_and_uncovered_modes(
        self,
    ) -> None:
        bridge_summary = {
            "num_factual_clusters": 3,
            "prompt_assignments": (
                {
                    "prompt_id": "tb-001",
                    "tool_subcategory": "subcategory_capital_fact",
                    "nearest_cluster_label": 2,
                    "nearest_cluster_dominant_subcategory": (
                        "subcategory_capital_fact"
                    ),
                    "nearest_cluster_js_distance": 0.12,
                    "tuned_final_position_kl_delta_under_routing": 1.0,
                },
                {
                    "prompt_id": "tb-002",
                    "tool_subcategory": "subcategory_author_fact",
                    "nearest_cluster_label": 2,
                    "nearest_cluster_dominant_subcategory": (
                        "subcategory_capital_fact"
                    ),
                    "nearest_cluster_js_distance": 0.31,
                    "tuned_final_position_kl_delta_under_routing": 3.0,
                },
                {
                    "prompt_id": "tb-003",
                    "tool_subcategory": "subcategory_author_fact",
                    "nearest_cluster_label": 5,
                    "nearest_cluster_dominant_subcategory": "subcategory_author_fact",
                    "nearest_cluster_js_distance": 0.22,
                    "tuned_final_position_kl_delta_under_routing": 2.0,
                },
            ),
        }
        route_mode_summary = {
            "cluster_count": 3,
            "family_summaries": (
                {
                    "family_tag": "subcategory_capital_fact",
                    "modes": (
                        {"cluster_label": 2, "size": 2},
                        {"cluster_label": 3, "size": 2},
                    ),
                },
                {
                    "family_tag": "subcategory_author_fact",
                    "modes": ({"cluster_label": 5, "size": 2},),
                },
            ),
        }

        summary = self.build_tool_breakage_route_mode_coverage_summary(
            bridge_summary=bridge_summary,
            factual_route_mode_summary=route_mode_summary,
        )

        self.assertEqual(3, summary["num_factual_modes"])
        self.assertEqual(2, summary["modes_with_any_assignment"])
        self.assertEqual(2, summary["modes_with_matching_family_assignment"])

        family_summaries = {
            item["family_tag"]: item for item in summary["family_coverage"]
        }
        self.assertEqual(
            [3],
            family_summaries["subcategory_capital_fact"][
                "uncovered_mode_cluster_labels"
            ],
        )
        self.assertEqual(
            1,
            family_summaries["subcategory_capital_fact"][
                "modes_with_matching_family_assignment"
            ],
        )
        self.assertEqual(
            1,
            family_summaries["subcategory_author_fact"][
                "modes_with_matching_family_assignment"
            ],
        )

        mode_summaries = {
            item["cluster_label"]: item for item in summary["mode_coverage"]
        }
        self.assertEqual(2, mode_summaries[2]["assigned_tool_prompt_count"])
        self.assertEqual(1, mode_summaries[2]["matching_family_tool_prompt_count"])
        self.assertEqual(
            {"subcategory_capital_fact": 1, "subcategory_author_fact": 1},
            mode_summaries[2]["assigned_tool_subcategory_counts"],
        )
        self.assertFalse(mode_summaries[3]["covered_by_any_tool_prompt"])
