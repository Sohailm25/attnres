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
            "history/PREREG.md",
            "history/20260316-thesis-alignment-and-gap-closure.md",
            "journal/current_state.md",
            "sessions/SESSION_TEMPLATE.md",
            "results/RESULTS_INDEX.md",
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
        ]
        for snippet in required_snippets:
            self.assertIn(snippet, content)

    def test_current_state_and_prereg_encode_thesis_corrections(self) -> None:
        current_state = (ROOT / "CURRENT_STATE.md").read_text()
        prereg = (ROOT / "history/PREREG.md").read_text()
        combined = current_state + "\n" + prereg
        required_snippets = [
            "upper bound on the routing signal available in standard architectures",
            "co-adaptation",
            "tool-breakage",
            "w_l analog",
            "8 clusters",
            "softmax-constrained",
            "unconstrained",
            "top-k",
            "Figure 8",
            "refusal",
        ]
        for snippet in required_snippets:
            self.assertIn(snippet, combined)

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
