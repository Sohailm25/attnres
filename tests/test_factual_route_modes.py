# ABOUTME: Tests the saved-artifact factual route-mode characterization helpers.
# ABOUTME: Keeps the next oracle analysis focused on within-family modes, not just cluster purity.

import unittest


class FactualRouteModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from validation.factual_route_modes import build_factual_route_mode_summary

        cls.build_factual_route_mode_summary = staticmethod(
            build_factual_route_mode_summary
        )

    def test_build_factual_route_mode_summary_recovers_within_family_modes(
        self,
    ) -> None:
        source_labels = (
            "embed",
            "0_attn_out",
            "0_mlp_out",
            "1_attn_out",
            "1_mlp_out",
        )
        prompt_results = (
            {
                "prompt_id": "p1",
                "prompt": "The capital of Canada is",
                "source_labels": source_labels,
                "final_alpha": (0.05, 0.62, 0.10, 0.13, 0.10),
            },
            {
                "prompt_id": "p2",
                "prompt": "The capital of Japan is",
                "source_labels": source_labels,
                "final_alpha": (0.05, 0.59, 0.11, 0.15, 0.10),
            },
            {
                "prompt_id": "p3",
                "prompt": "The capital of Spain is",
                "source_labels": source_labels,
                "final_alpha": (0.05, 0.12, 0.10, 0.13, 0.60),
            },
            {
                "prompt_id": "p4",
                "prompt": "The author of 1984 was",
                "source_labels": source_labels,
                "final_alpha": (0.05, 0.10, 0.11, 0.62, 0.12),
            },
            {
                "prompt_id": "p5",
                "prompt": "The author of Beloved was",
                "source_labels": source_labels,
                "final_alpha": (0.05, 0.11, 0.12, 0.59, 0.13),
            },
            {
                "prompt_id": "p6",
                "prompt": "The author of Frankenstein was",
                "source_labels": source_labels,
                "final_alpha": (0.05, 0.12, 0.58, 0.14, 0.11),
            },
        )
        tags_by_prompt_id = {
            "p1": ("stratum_factual_recall", "subcategory_capital_fact"),
            "p2": ("stratum_factual_recall", "subcategory_capital_fact"),
            "p3": ("stratum_factual_recall", "subcategory_capital_fact"),
            "p4": ("stratum_factual_recall", "subcategory_author_fact"),
            "p5": ("stratum_factual_recall", "subcategory_author_fact"),
            "p6": ("stratum_factual_recall", "subcategory_author_fact"),
        }

        summary = self.build_factual_route_mode_summary(
            prompt_results=prompt_results,
            tags_by_prompt_id=tags_by_prompt_id,
            cluster_count=4,
            top_sources=2,
            example_prompts=1,
        )

        self.assertEqual(6, summary["num_sequences"])
        self.assertEqual(2, summary["num_families"])

        family_summaries = {
            item["family_tag"]: item for item in summary["family_summaries"]
        }
        capital = family_summaries["subcategory_capital_fact"]
        author = family_summaries["subcategory_author_fact"]

        self.assertEqual(3, capital["prompt_count"])
        self.assertEqual(2, capital["mode_count"])
        self.assertEqual([1, 2], capital["mode_sizes"])
        self.assertGreater(capital["mode_centroid_js"]["mean"], 0.0)
        self.assertEqual(
            "0_attn_out",
            capital["family_top_mean_sources"][0]["source_label"],
        )
        self.assertCountEqual(
            ["0_attn_out", "1_mlp_out"],
            [
                mode["top_enriched_sources_vs_family"][0]["source_label"]
                for mode in capital["modes"]
            ],
        )

        self.assertEqual(3, author["prompt_count"])
        self.assertEqual(2, author["mode_count"])
        self.assertEqual([1, 2], author["mode_sizes"])
        self.assertEqual(
            "1_attn_out",
            author["family_top_mean_sources"][0]["source_label"],
        )
        self.assertCountEqual(
            ["1_attn_out", "0_mlp_out"],
            [
                mode["top_enriched_sources_vs_family"][0]["source_label"]
                for mode in author["modes"]
            ],
        )

    def test_build_factual_route_mode_summary_requires_one_subcategory_tag(
        self,
    ) -> None:
        prompt_results = (
            {
                "prompt_id": "p1",
                "prompt": "The capital of Canada is",
                "source_labels": ("embed", "0_attn_out", "0_mlp_out"),
                "final_alpha": (0.1, 0.7, 0.2),
            },
            {
                "prompt_id": "p2",
                "prompt": "The capital of Japan is",
                "source_labels": ("embed", "0_attn_out", "0_mlp_out"),
                "final_alpha": (0.1, 0.68, 0.22),
            },
        )
        tags_by_prompt_id = {
            "p1": ("stratum_factual_recall",),
            "p2": ("stratum_factual_recall", "subcategory_capital_fact"),
        }

        with self.assertRaises(ValueError):
            self.build_factual_route_mode_summary(
                prompt_results=prompt_results,
                tags_by_prompt_id=tags_by_prompt_id,
                cluster_count=2,
            )


if __name__ == "__main__":
    unittest.main()
