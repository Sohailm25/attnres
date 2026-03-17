# ABOUTME: Exercises the first Phase 1 implementation slice for dependency freezing.
# ABOUTME: Guards normalization-aware reconstruction and cache-validity logic before oracle-alpha runs.

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DependencyFreezeTests(unittest.TestCase):
    def test_requirements_are_directly_pinned(self) -> None:
        requirements = (ROOT / "requirements.txt").read_text().splitlines()
        expected = {
            "torch==2.10.0",
            "torchvision==0.25.0",
            "transformer-lens==2.17.0",
            "sae-lens==6.38.0",
            "nnsight==0.6.1",
            "pyvene==0.1.8",
            "circuitsvis==1.43.3",
            "datasets==4.8.2",
            "wandb==0.25.1",
            "plotly==6.6.0",
            "jupyter==1.1.1",
            "numpy==2.4.3",
            "pandas==3.0.1",
            "scipy==1.17.1",
            "scikit-learn==1.8.0",
            "matplotlib==3.10.8",
            "seaborn==0.13.2",
            "pyyaml==6.0.3",
            "pre-commit==4.5.1",
            "ruff==0.15.6",
        }
        self.assertTrue(expected.issubset(set(requirements)))
        for line in requirements:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            self.assertRegex(stripped, r"^[A-Za-z0-9._-]+==.+$")

    def test_lockfile_exists_for_full_phase1_resolution(self) -> None:
        lockfile = ROOT / "requirements.lock.txt"
        self.assertTrue(lockfile.is_file())
        content = lockfile.read_text()
        for snippet in [
            "torch==2.10.0",
            "transformer-lens==2.17.0",
            "sae-lens==6.38.0",
            "nnsight==0.6.1",
            "pyvene==0.1.8",
            "ruff==0.15.6",
        ]:
            self.assertIn(snippet, content)


class ReconstructionValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        from validation.reconstruction import (
            CacheReconstructionMetrics,
            cache_reconstruction_metrics,
            max_abs_difference,
            routed_logits,
            shared_final_norm_logit_contributions,
        )

        self.CacheReconstructionMetrics = CacheReconstructionMetrics
        self.cache_reconstruction_metrics = cache_reconstruction_metrics
        self.max_abs_difference = max_abs_difference
        self.routed_logits = routed_logits
        self.shared_final_norm_logit_contributions = (
            shared_final_norm_logit_contributions
        )

        self.sources = [
            [1.0, -1.0],
            [0.5, 1.5],
            [-0.5, 2.0],
        ]
        self.final_hidden = [1.0, 2.5]
        self.unembed = [
            [2.0, 0.0],
            [0.0, -1.0],
        ]
        self.eps = 1e-6

    def test_cache_reconstruction_metrics_match_exact_sum(self) -> None:
        metrics = self.cache_reconstruction_metrics(
            embedding=self.sources[0],
            sublayer_outputs=self.sources[1:],
            final_hidden=self.final_hidden,
        )
        self.assertIsInstance(metrics, self.CacheReconstructionMetrics)
        self.assertEqual(3, metrics.num_sources)
        self.assertAlmostEqual(0.0, metrics.max_abs_error)
        self.assertAlmostEqual(0.0, metrics.l2_error)

    def test_uniform_routing_matches_original_logits_after_final_norm(self) -> None:
        original_logits = self.routed_logits(
            sources=self.sources,
            alpha=[1.0, 1.0, 1.0],
            unembed=self.unembed,
            eps=self.eps,
        )
        uniform_logits = self.routed_logits(
            sources=self.sources,
            alpha=[1.0 / len(self.sources)] * len(self.sources),
            unembed=self.unembed,
            eps=self.eps,
        )
        self.assertLess(
            self.max_abs_difference(original_logits, uniform_logits),
            1e-5,
        )

        logits_divided_by_layer_count = [
            value / len(self.sources) for value in original_logits
        ]
        self.assertGreater(
            self.max_abs_difference(
                uniform_logits,
                logits_divided_by_layer_count,
            ),
            0.1,
        )

    def test_shared_final_norm_contributions_sum_back_to_full_logits(self) -> None:
        full_logits = self.routed_logits(
            sources=self.sources,
            alpha=[0.2, 0.3, 0.5],
            unembed=self.unembed,
            eps=self.eps,
        )
        contributions = self.shared_final_norm_logit_contributions(
            sources=self.sources,
            alpha=[0.2, 0.3, 0.5],
            unembed=self.unembed,
            eps=self.eps,
        )
        summed = [
            sum(component[index] for component in contributions)
            for index in range(len(full_logits))
        ]
        self.assertLess(self.max_abs_difference(full_logits, summed), 1e-6)

    def test_independent_per_source_normalization_is_not_exact(self) -> None:
        from validation.reconstruction import rms_norm

        alpha = [0.2, 0.3, 0.5]
        full_logits = self.routed_logits(
            sources=self.sources,
            alpha=alpha,
            unembed=self.unembed,
            eps=self.eps,
        )

        naive_logits = [0.0 for _ in full_logits]
        for source, weight in zip(self.sources, alpha, strict=True):
            normalized = rms_norm(source, eps=self.eps)
            naive_logits = [
                current
                + weight
                * sum(
                    row[column] * normalized[column]
                    for column in range(len(normalized))
                )
                for current, row in zip(naive_logits, self.unembed, strict=True)
            ]

        self.assertGreater(
            self.max_abs_difference(full_logits, naive_logits),
            0.01,
        )


if __name__ == "__main__":
    unittest.main()
