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
            assign_average_linkage_clusters,
            build_grouped_view_pattern_summary,
            build_prompt_resampling_stability_summary,
            build_sequence_level_pattern_summary,
            scan_average_linkage_clusters,
            summarize_source_type_mass,
            write_pattern_analysis_summary,
        )

        cls.build_grouped_view_pattern_summary = staticmethod(
            build_grouped_view_pattern_summary
        )
        cls.build_prompt_resampling_stability_summary = staticmethod(
            build_prompt_resampling_stability_summary
        )
        cls.build_sequence_level_pattern_summary = staticmethod(
            build_sequence_level_pattern_summary
        )
        cls.assign_average_linkage_clusters = staticmethod(
            assign_average_linkage_clusters
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

    def test_summarize_source_type_mass_supports_single_distribution(self) -> None:
        source_labels = (
            "embed",
            "pos_embed",
            "0_attn_out",
            "0_mlp_out",
            "1_attn_out",
            "1_mlp_out",
        )
        alpha_vectors = ((0.10, 0.20, 0.30, 0.10, 0.20, 0.10),)

        summary = self.summarize_source_type_mass(alpha_vectors, source_labels)

        self.assertAlmostEqual(0.30, summary.mean_embedding_mass)
        self.assertAlmostEqual(0.50, summary.mean_attention_mass)
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
        self.assertEqual((2, 2), tuple(result.cluster_sizes_by_k[2]))
        self.assertAlmostEqual(0.5, result.largest_cluster_fraction_by_k[2])

    def test_assign_average_linkage_clusters_recovers_expected_partition(self) -> None:
        distributions = (
            (0.88, 0.08, 0.02, 0.02),
            (0.83, 0.11, 0.03, 0.03),
            (0.02, 0.02, 0.08, 0.88),
            (0.03, 0.03, 0.12, 0.82),
        )

        assignments = self.assign_average_linkage_clusters(
            distributions,
            cluster_count=2,
        )

        self.assertEqual(4, len(assignments))
        self.assertEqual(assignments[0], assignments[1])
        self.assertEqual(assignments[2], assignments[3])
        self.assertNotEqual(assignments[0], assignments[2])

    def test_build_grouped_view_pattern_summary_aggregates_depth_and_type(self) -> None:
        source_labels = (
            "embed",
            "pos_embed",
            "0_attn_out",
            "0_mlp_out",
            "1_attn_out",
            "1_mlp_out",
            "2_attn_out",
            "2_mlp_out",
        )
        sequence_results = (
            {
                "prompt_id": "p1",
                "split": "confirm",
                "source_labels": source_labels,
                "final_alpha": (0.10, 0.10, 0.25, 0.15, 0.20, 0.10, 0.05, 0.05),
            },
            {
                "prompt_id": "p2",
                "split": "confirm",
                "source_labels": source_labels,
                "final_alpha": (0.08, 0.12, 0.22, 0.18, 0.18, 0.12, 0.05, 0.05),
            },
        )

        grouped = self.build_grouped_view_pattern_summary(
            sequence_results,
            view_name="depth_thirds_by_type",
            random_seed=11,
            max_clusters=2,
        )

        self.assertEqual("depth_thirds_by_type", grouped.view_name)
        self.assertEqual(
            (
                "embedding",
                "early_attention",
                "early_mlp",
                "middle_attention",
                "middle_mlp",
                "late_attention",
                "late_mlp",
            ),
            grouped.grouped_source_labels,
        )
        self.assertEqual(7, grouped.summary.num_sources)
        self.assertAlmostEqual(
            0.20, grouped.summary.source_type_mass.mean_embedding_mass
        )
        self.assertAlmostEqual(
            0.475, grouped.summary.source_type_mass.mean_attention_mass
        )
        self.assertAlmostEqual(0.325, grouped.summary.source_type_mass.mean_mlp_mass)

    def test_build_prompt_resampling_stability_summary_reports_view_metrics(
        self,
    ) -> None:
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
                "final_alpha": (0.62, 0.18, 0.15, 0.05),
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
                "final_alpha": (0.05, 0.08, 0.17, 0.70),
            },
            {
                "prompt_id": "p4",
                "split": "confirm",
                "source_labels": source_labels,
                "final_alpha": (0.06, 0.09, 0.18, 0.67),
            },
            {
                "prompt_id": "p5",
                "split": "confirm",
                "source_labels": source_labels,
                "final_alpha": (0.60, 0.20, 0.14, 0.06),
            },
            {
                "prompt_id": "p6",
                "split": "confirm",
                "source_labels": source_labels,
                "final_alpha": (0.04, 0.10, 0.19, 0.67),
            },
        )

        stability = self.build_prompt_resampling_stability_summary(
            sequence_results,
            view_name="raw_source",
            random_seed=11,
            max_clusters=4,
            num_resamples=5,
            sample_size=4,
        )

        self.assertEqual("raw_source", stability.view_name)
        self.assertEqual(5, stability.num_resamples)
        self.assertEqual(4, stability.sample_size)
        self.assertGreaterEqual(stability.oracle_beats_random_fraction, 0.0)
        self.assertLessEqual(stability.oracle_beats_random_fraction, 1.0)
        self.assertIn(2, stability.oracle_best_k_counts)
        self.assertIn(2, stability.oracle_largest_cluster_fraction_mean_by_k)

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
        self.assertIn(2, summary.oracle_cluster_scan.cluster_sizes_by_k)
        self.assertIn(2, summary.oracle_cluster_scan.largest_cluster_fraction_by_k)

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
            self.assertIn("cluster_views", payload)
            self.assertIn("resampling_stability", payload)

    def test_write_pattern_analysis_summary_includes_subset_summaries(self) -> None:
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
            "num_sequences": 4,
            "sequence_mean_improvement": 0.2,
            "bootstrap_interval": {"mean": 0.2, "lower": 0.1, "upper": 0.3},
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
                    "uniform_loss": 4.1,
                    "optimized_loss": 3.9,
                    "null_losses": {"random_dirichlet": 4.2},
                    "best_alpha_entropy": 1.0,
                    "best_alpha": [0.45, 0.25, 0.2, 0.1],
                    "final_alpha": [0.45, 0.25, 0.2, 0.1],
                },
                {
                    "prompt_id": "p3",
                    "prompt": "third",
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
                {
                    "prompt_id": "p4",
                    "prompt": "fourth",
                    "split": "confirm",
                    "num_sources": 4,
                    "source_labels": source_labels,
                    "uniform_loss": 4.3,
                    "optimized_loss": 4.1,
                    "null_losses": {"random_dirichlet": 4.4},
                    "best_alpha_entropy": 1.0,
                    "best_alpha": [0.15, 0.1, 0.15, 0.6],
                    "final_alpha": [0.15, 0.1, 0.15, 0.6],
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
                subset_prompt_ids_by_name={
                    "stratum_a": ("p1", "p2"),
                    "stratum_b": ("p3", "p4"),
                },
            )

            payload = json.loads(output_path.read_text())
            self.assertIn("subset_summaries", payload)
            self.assertEqual(
                {"stratum_a", "stratum_b"}, set(payload["subset_summaries"])
            )
            self.assertEqual(
                2,
                payload["subset_summaries"]["stratum_a"]["summary"]["num_sequences"],
            )
            self.assertIn(
                "cluster_views",
                payload["subset_summaries"]["stratum_b"],
            )


if __name__ == "__main__":
    unittest.main()
