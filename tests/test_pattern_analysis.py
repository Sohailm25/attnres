# ABOUTME: Exercises the prereg-scale sequence-level pattern-analysis helpers.
# ABOUTME: Keeps the first Phase 2 slice scoped to JSD structure, source-type mass, and compact artifact writing.

import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PatternAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from validation.pattern_analysis import (
            build_sequence_level_pattern_summary,
            scan_average_linkage_clusters,
            summarize_source_type_mass,
            write_pattern_analysis_summary,
        )

        cls.build_sequence_level_pattern_summary = staticmethod(
            build_sequence_level_pattern_summary
        )
        cls.scan_average_linkage_clusters = staticmethod(scan_average_linkage_clusters)
        cls.summarize_source_type_mass = staticmethod(summarize_source_type_mass)
        cls.write_pattern_analysis_summary = staticmethod(
            write_pattern_analysis_summary
        )

    def test_summarize_source_type_mass_groups_embed_attention_and_mlp(self) -> None:
        source_labels = (
            "embed",
            "pos_embed",
            "0_attn_out",
            "0_mlp_out",
            "1_attn_out",
            "1_mlp_out",
        )
        alpha_vectors = (
            (0.10, 0.20, 0.30, 0.10, 0.20, 0.10),
            (0.05, 0.15, 0.35, 0.10, 0.25, 0.10),
        )

        summary = self.summarize_source_type_mass(alpha_vectors, source_labels)

        self.assertAlmostEqual(0.25, summary.mean_embedding_mass)
        self.assertAlmostEqual(0.55, summary.mean_attention_mass)
        self.assertAlmostEqual(0.20, summary.mean_mlp_mass)

    def test_scan_average_linkage_clusters_finds_two_cluster_structure(self) -> None:
        distributions = (
            (0.88, 0.08, 0.02, 0.02),
            (0.83, 0.11, 0.03, 0.03),
            (0.02, 0.02, 0.08, 0.88),
            (0.03, 0.03, 0.12, 0.82),
        )

        result = self.scan_average_linkage_clusters(
            distributions,
            max_clusters=4,
        )

        self.assertEqual("average", result.linkage_method)
        self.assertEqual(2, result.best_k)
        self.assertGreater(result.best_silhouette, 0.5)
        self.assertEqual({2, 3, 4}, set(result.silhouette_by_k))

    def test_build_sequence_level_pattern_summary_reports_random_control(self) -> None:
        source_labels = (
            "embed",
            "pos_embed",
            "0_attn_out",
            "0_mlp_out",
        )
        sequence_results = (
            {
                "prompt_id": "p1",
                "split": "confirm",
                "source_labels": source_labels,
                "final_alpha": (0.60, 0.20, 0.15, 0.05),
            },
            {
                "prompt_id": "p2",
                "split": "confirm",
                "source_labels": source_labels,
                "final_alpha": (0.58, 0.22, 0.14, 0.06),
            },
            {
                "prompt_id": "p3",
                "split": "confirm",
                "source_labels": source_labels,
                "final_alpha": (0.05, 0.10, 0.20, 0.65),
            },
            {
                "prompt_id": "p4",
                "split": "confirm",
                "source_labels": source_labels,
                "final_alpha": (0.07, 0.12, 0.18, 0.63),
            },
        )

        summary = self.build_sequence_level_pattern_summary(
            sequence_results,
            random_seed=11,
            max_clusters=4,
        )

        self.assertEqual(4, summary.num_sequences)
        self.assertGreater(summary.mean_entropy, 0.0)
        self.assertGreater(summary.mean_effective_sources, 1.0)
        self.assertEqual("average", summary.oracle_cluster_scan.linkage_method)
        self.assertEqual("average", summary.random_control_cluster_scan.linkage_method)
        self.assertGreater(summary.oracle_cluster_scan.best_silhouette, 0.0)
        self.assertGreaterEqual(len(summary.top1_source_counts), 1)

    def test_write_pattern_analysis_summary_writes_compact_json(self) -> None:
        source_labels = (
            "embed",
            "pos_embed",
            "0_attn_out",
            "0_mlp_out",
        )
        raw_run = {
            "model_name": "tiny-stories-1M",
            "collection_id": "oracle_alpha_phase1_v1",
            "split": "confirm",
            "num_sequences": 2,
            "sequence_mean_improvement": 0.1,
            "bootstrap_interval": {"mean": 0.1, "lower": 0.05, "upper": 0.15},
            "null_model_mean_losses": {"random_dirichlet": 1.2},
            "sequence_results": [
                {
                    "prompt_id": "p1",
                    "prompt": "first",
                    "split": "confirm",
                    "num_sources": 4,
                    "source_labels": source_labels,
                    "uniform_loss": 4.0,
                    "optimized_loss": 3.8,
                    "null_losses": {"random_dirichlet": 4.1},
                    "best_alpha_entropy": 1.0,
                    "best_alpha": [0.5, 0.2, 0.2, 0.1],
                    "final_alpha": [0.5, 0.2, 0.2, 0.1],
                },
                {
                    "prompt_id": "p2",
                    "prompt": "second",
                    "split": "confirm",
                    "num_sources": 4,
                    "source_labels": source_labels,
                    "uniform_loss": 4.2,
                    "optimized_loss": 4.0,
                    "null_losses": {"random_dirichlet": 4.3},
                    "best_alpha_entropy": 1.0,
                    "best_alpha": [0.1, 0.1, 0.2, 0.6],
                    "final_alpha": [0.1, 0.1, 0.2, 0.6],
                },
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            raw_path = Path(temp_dir) / "oracle_eval_run.json"
            raw_path.write_text(json.dumps(raw_run))
            output_path = Path(temp_dir) / "pattern_summary.json"

            self.write_pattern_analysis_summary(
                run_path=raw_path,
                output_path=output_path,
                random_seed=11,
                max_clusters=4,
            )

            payload = json.loads(output_path.read_text())
            self.assertEqual("tiny-stories-1M", payload["model_name"])
            self.assertEqual("confirm", payload["split"])
            self.assertEqual(2, payload["num_sequences"])
            self.assertIn("oracle_cluster_scan", payload)
            self.assertIn("random_control_cluster_scan", payload)


if __name__ == "__main__":
    unittest.main()
