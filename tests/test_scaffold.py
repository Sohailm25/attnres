# ABOUTME: Validates that the local experiment scaffold exists and encodes the right thesis.
# ABOUTME: Prevents drift back to persona-circuits or Modal-specific assumptions.

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ScaffoldTests(unittest.TestCase):
    def test_expected_top_level_directories_exist(self) -> None:
        expected = {
            "background-work",
            "configs",
            "history",
            "journal",
            "knowledge",
            "notebooks",
            "background-work/papers",
            "prompts",
            "results",
            "scratch",
            "scripts",
            "sessions",
            "tests",
        }
        missing = sorted(path for path in expected if not (ROOT / path).is_dir())
        self.assertEqual([], missing)

    def test_expected_control_docs_exist(self) -> None:
        expected = {
            "AGENTS.md",
            "CURRENT_STATE.md",
            "DECISIONS.md",
            "README.md",
            "SCRATCHPAD.md",
            "THOUGHT_LOG.md",
            ".gitignore",
            ".pre-commit-config.yaml",
            "requirements.txt",
            "configs/experiment.yaml",
            "background-work/PROPOSAL_REVIEW.md",
            "background-work/RESEARCH_POSITIONING.md",
            "background-work/SAFETY_PUBLICATION_POLICY.md",
            "background-work/papers/DOWNLOAD_MANIFEST.md",
            "history/PREREG.md",
            "history/20260316-deepresearch-review-and-actions.md",
            "history/20260316-thesis-alignment-and-gap-closure.md",
            "history/20260316-second-review-readiness.md",
            "history/20260316-methodology-gap-audit.md",
            "history/20260316-secondary-red-team-review.md",
            "journal/current_state.md",
            "sessions/SESSION_TEMPLATE.md",
            "results/RESULTS_INDEX.md",
            "scripts/download_reference_papers.py",
        }
        missing = sorted(path for path in expected if not (ROOT / path).is_file())
        self.assertEqual([], missing)

    def test_agents_is_local_and_gap_aware(self) -> None:
        content = (ROOT / "AGENTS.md").read_text()
        required_snippets = [
            "research/master-research-document.docx",
            "research/decision-matrix.md",
            "research/artifact2.md",
            "research/artifact3.md",
            "MacBook Pro",
            "no Modal",
            "oracle-alpha",
            "Figure 8",
            "tool-breakage",
            "softmax",
            "top-k",
            "safety",
            "known",
            "observed",
            "inferred",
            "unknown",
            "Epistemic Standards",
            "Adversarial self-questioning",
            "Implementation skepticism",
            "Research Navigation Guide",
            "Document Discipline",
            "Session Check-In Protocol",
            "background-work/papers",
            "smallest experiment",
            "tight feedback loops",
            "mock-up of the main plot",
            "Motivation / Methods / Results / Limitations / Next Steps",
            "bd issue",
            "tmux",
            "checkpoint",
            "resume command",
            "effective depth mixture",
            "MIB",
            "stability",
            "predictiveness",
            "controlled dynamic-routing counterfactual",
            "harmfulness",
            "reproducible proxy",
        ]
        for snippet in required_snippets:
            self.assertIn(snippet, content)
        self.assertNotIn("LessWrong", content)

    def test_current_state_and_prereg_encode_thesis_corrections(self) -> None:
        current_state = (ROOT / "CURRENT_STATE.md").read_text()
        prereg = (ROOT / "history/PREREG.md").read_text()
        combined = current_state + "\n" + prereg
        required_snippets = [
            "upper bound on the routing signal available in standard architectures",
            "lower bound on the benefit of depth routing",
            "co-adaptation",
            "tool-breakage",
            "w_l analog",
            "8 clusters",
            "softmax-constrained",
            "unconstrained",
            "top-k",
            "Figure 8",
            "refusal",
            "locality score",
            "alpha_0",
            "Entropy",
            "2/L",
            "non-monotonic",
            "R^2 > 0.5",
            "silhouette > 0.2",
            "d > 0.2",
            "MIB",
            "stability",
            "out-of-sample",
            "controlled dynamic-routing counterfactual",
            "harmfulness",
            "reproducible proxy",
        ]
        for snippet in required_snippets:
            self.assertIn(snippet, combined)
        self.assertNotIn("LessWrong", combined)

    def test_prereg_has_operationalized_metrics_and_gates(self) -> None:
        prereg = (ROOT / "history/PREREG.md").read_text()
        required_snippets = [
            "Jensen-Shannon",
            "Cohen's d",
            "bootstrap",
            "paired t-test",
            "pilot",
            "confirmatory",
            "locality score > 1/L",
            "above-uniform weight",
            "Entropy(pre-attn) > Entropy(pre-MLP)",
            "α* > 2/L",
            "k ∈ {2,4,8",
            "non-monotonic curves on >50% of prompts",
            "tuned lens",
            "2-layer MLP on h_1[t]",
            "pre-register",
            "MIB",
            "out-of-sample",
            "controlled dynamic-routing counterfactual",
            "harmfulness",
            "reproducible proxy",
        ]
        for snippet in required_snippets:
            self.assertIn(snippet, prereg)
        self.assertNotIn("LessWrong", prereg)

    def test_methodology_audit_captures_known_subtle_hazards(self) -> None:
        audit = (ROOT / "history/20260316-methodology-gap-audit.md").read_text()
        required_snippets = [
            "Ward linkage",
            'only "euclidean" is accepted',
            "Jensen-Shannon",
            "RMSNorm",
            "match the original logits",
            "not logits / L",
            "shared final normalization factor",
            "per-source LayerNorm",
            "sequence-level",
            "pseudoreplication",
            "resid_post",
            "sublayer outputs",
            "dependency freeze",
        ]
        for snippet in required_snippets:
            self.assertIn(snippet, audit)
        self.assertNotIn("public preregistration on LessWrong", audit)

    def test_paper_manifest_has_core_reference_set(self) -> None:
        manifest = (ROOT / "background-work/papers/DOWNLOAD_MANIFEST.md").read_text()
        required_snippets = [
            "Attention Residuals",
            "DeepCrossAttention",
            "DenseFormer",
            "Hyper-Connections",
            "MUDDFormer",
            "The Curse of Depth in Large Language Models",
            "ShortGPT",
            "Pythia",
            "Gemma Scope",
            "Route Sparse Autoencoder",
            "Backward Lens",
            "Tuned Lens",
            "LayerSkip",
            "MIB",
            "Safety Layers in Aligned LLMs",
            "LLMs Encode Harmfulness and Refusal Separately",
            "Weight-sparse transformers have interpretable circuits",
            "OLMo 2",
            "EvoLM",
            "Refusal in Language Models Is Mediated by a Single Direction",
            "Interpretability in the Wild",
            "In-context Learning and Induction Heads",
            "Measuring Faithfulness in Chain-of-Thought Reasoning",
        ]
        for snippet in required_snippets:
            self.assertIn(snippet, manifest)

    def test_secondary_red_team_review_captures_remaining_publishability_risks(
        self,
    ) -> None:
        review = (ROOT / "history/20260316-secondary-red-team-review.md").read_text()
        required_snippets = [
            "raw logit lens",
            "often brittle",
            "tuned lens",
            "pilot/confirmatory split",
            "h_1[t]",
            "refusal-feature discovery",
        ]
        for snippet in required_snippets:
            self.assertIn(snippet, review)

    def test_safety_publication_policy_exists_and_forbids_dual_use_sloppiness(
        self,
    ) -> None:
        policy = (ROOT / "background-work/SAFETY_PUBLICATION_POLICY.md").read_text()
        required_snippets = [
            "jailbreak",
            "confirmatory",
            "dual-use",
            "do not publish",
        ]
        for snippet in required_snippets:
            self.assertIn(snippet, policy)

    def test_thought_log_is_first_class_and_reflective(self) -> None:
        agents = (ROOT / "AGENTS.md").read_text()
        thought_log = (ROOT / "THOUGHT_LOG.md").read_text()

        for snippet in [
            "THOUGHT_LOG.md",
            "research reflections",
            "hunches",
            "predictions",
            "surprises",
            "confidence",
        ]:
            self.assertIn(snippet, agents)

        for snippet in [
            "Working Hypotheses",
            "Hunches and Guesses",
            "Feel of the Experiment",
            "Surprises and Tensions",
            "Predictions",
            "interesting facts",
            "not claim-bearing evidence",
        ]:
            self.assertIn(snippet, thought_log)

    def test_results_scaffold_covers_all_major_lanes(self) -> None:
        expected = {
            "results/infrastructure",
            "results/oracle_alpha",
            "results/pattern_analysis",
            "results/figure8_validation",
            "results/block_structure",
            "results/comparison_regimes",
            "results/tool_breakage",
            "results/training_dynamics",
            "results/router_training",
            "results/safety_alignment",
            "results/figures",
        }
        missing = sorted(path for path in expected if not (ROOT / path).is_dir())
        self.assertEqual([], missing)

    def test_journal_and_knowledge_scaffold_exists(self) -> None:
        expected = {
            "journal/logs",
            "knowledge/general/insights",
            "knowledge/general/learnings",
            "knowledge/general/accomplishments",
            "knowledge/general/references",
        }
        missing = sorted(path for path in expected if not (ROOT / path).is_dir())
        self.assertEqual([], missing)


if __name__ == "__main__":
    unittest.main()
