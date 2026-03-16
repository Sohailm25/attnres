# ABOUTME: Downloads and indexes the local paper cache for the depth-routing experiment.
# ABOUTME: Keeps the core references available as local files under background-work/papers/files.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import shutil
import sys
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = ROOT / "background-work" / "papers"
FILES_DIR = PAPERS_DIR / "files"
MANIFEST_PATH = PAPERS_DIR / "DOWNLOAD_MANIFEST.md"


@dataclass(frozen=True)
class PaperSpec:
    title: str
    slug: str
    url: str
    note: str
    suffix: str = ".pdf"
    local_source: str | None = None

    @property
    def filename(self) -> str:
        return f"{self.slug}{self.suffix}"

    @property
    def output_path(self) -> Path:
        return FILES_DIR / self.filename

    @property
    def local_source_path(self) -> Path | None:
        if self.local_source is None:
            return None
        return ROOT / self.local_source


PAPERS: list[PaperSpec] = [
    PaperSpec(
        title="Attention Residuals",
        slug="attention_residuals",
        url="https://github.com/MoonshotAI/Attention-Residuals",
        local_source="research/Attention_Residuals.pdf",
        note="Primary motivating architecture; Figure 8 predictions originate here.",
    ),
    PaperSpec(
        title="DeepCrossAttention",
        slug="deepcrossattention_supercharging_transformer_residual_connections",
        url="https://arxiv.org/pdf/2502.06785.pdf",
        note="Dynamic residual-connection baseline for comparison with AttnRes.",
    ),
    PaperSpec(
        title="DenseFormer",
        slug="denseformer",
        url="https://proceedings.neurips.cc/paper_files/paper/2024/file/f67449c7ab72f441d3a713b046c6818c-Paper-Conference.pdf",
        note="Static depth-weighting reference and pattern-stability comparison.",
    ),
    PaperSpec(
        title="Hyper-Connections",
        slug="hyper_connections",
        url="https://proceedings.iclr.cc/paper_files/paper/2025/file/f1e8ff97057e0c3f6bcb7018c78d4df1-Paper-Conference.pdf",
        note="Parallel-stream residual mixing baseline.",
    ),
    PaperSpec(
        title="MUDDFormer",
        slug="muddformer",
        url="https://arxiv.org/pdf/2502.12170.pdf",
        note="Dynamic multiway dense connectivity reference.",
    ),
    PaperSpec(
        title="The Curse of Depth in Large Language Models",
        slug="the_curse_of_depth_in_large_language_models",
        url="https://arxiv.org/pdf/2502.05795.pdf",
        note="Main theoretical redundancy and depth-decay reference.",
    ),
    PaperSpec(
        title="ShortGPT",
        slug="shortgpt_layers_in_large_language_models_are_more_redundant_than_you_expect",
        url="https://aclanthology.org/2025.findings-acl.1035.pdf",
        note="Layer redundancy baseline and Block Influence comparison.",
    ),
    PaperSpec(
        title="Pythia",
        slug="pythia_a_suite_for_analyzing_large_language_models_across_training_and_scaling",
        url="https://arxiv.org/pdf/2304.01373.pdf",
        note="Training-dynamics model family and checkpoint suite.",
    ),
    PaperSpec(
        title="Gemma Scope",
        slug="gemma_scope_open_sparse_autoencoders_everywhere_all_at_once_on_gemma_2",
        url="https://arxiv.org/pdf/2408.05147.pdf",
        note="Main SAE reference for Gemma-2-2B analysis.",
    ),
    PaperSpec(
        title="Route Sparse Autoencoder",
        slug="route_sparse_autoencoder_to_interpret_large_language_models",
        url="https://aclanthology.org/2025.emnlp-main.346.pdf",
        note="Routing-aware SAE reference for depth-routing feature analysis.",
    ),
    PaperSpec(
        title="Backward Lens",
        slug="backward_lens_projecting_language_model_gradients_into_the_vocabulary_space",
        url="https://aclanthology.org/2024.emnlp-main.142.pdf",
        note="Vocabulary-space attribution method relevant to routing-aware lens work.",
    ),
    PaperSpec(
        title="Tuned Lens",
        slug="eliciting_latent_predictions_from_transformers_with_the_tuned_lens",
        url="https://arxiv.org/pdf/2303.08112.pdf",
        note="Baseline lens methodology for intermediate prediction analysis.",
    ),
    PaperSpec(
        title="LayerSkip",
        slug="layerskip_enabling_early_exit_inference_and_self_speculative_decoding",
        url="https://arxiv.org/pdf/2404.16710.pdf",
        note="Dynamic depth-use comparison and early-exit baseline.",
    ),
    PaperSpec(
        title="Root Mean Square Layer Normalization",
        slug="root_mean_square_layer_normalization",
        url="https://arxiv.org/pdf/1910.07467.pdf",
        note="Normalization reference for reconstruction and exact routed-logit decomposition.",
    ),
    PaperSpec(
        title="Circuit Tracing",
        slug="circuit_tracing_revealing_computational_graphs_in_language_models",
        url="https://transformer-circuits.pub/2025/attribution-graphs/biology.html",
        suffix=".html",
        note="Tool-breakage and attribution-graph baseline for the dynamic-routing claim.",
    ),
]


def fetch(spec: PaperSpec, refresh: bool) -> tuple[str, int]:
    output_path = spec.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not refresh:
        return "cached", output_path.stat().st_size

    source_path = spec.local_source_path
    if source_path is not None and source_path.exists():
        shutil.copyfile(source_path, output_path)
        return "copied-local", output_path.stat().st_size

    request = urllib.request.Request(
        spec.url,
        headers={"User-Agent": "resattn-paper-cache/1.0"},
    )
    with urllib.request.urlopen(request) as response:
        data = response.read()
    output_path.write_bytes(data)
    return "downloaded", len(data)


def render_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def write_manifest(rows: list[tuple[PaperSpec, str, int]]) -> None:
    lines = [
        "# Download Manifest",
        "",
        "Local paper cache for the depth-routing experiment.",
        "",
        f"- Paper count: `{len(rows)}`",
        f"- Files directory: `{FILES_DIR.relative_to(ROOT)}`",
        "- Generated by: `scripts/download_reference_papers.py`",
        "",
        "| Title | Local file | Source URL | Status | Size | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for spec, status, num_bytes in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    spec.title,
                    f"`{spec.output_path.relative_to(ROOT)}`",
                    spec.url,
                    status,
                    render_size(num_bytes),
                    spec.note,
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "Use `python scripts/download_reference_papers.py --refresh` to re-download all entries."
    )
    MANIFEST_PATH.write_text("\n".join(lines))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-download all papers even if a local cached file already exists",
    )
    args = parser.parse_args(argv)

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    FILES_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[PaperSpec, str, int]] = []
    for spec in PAPERS:
        status, num_bytes = fetch(spec, refresh=args.refresh)
        rows.append((spec, status, num_bytes))

    write_manifest(rows)

    for spec, status, num_bytes in rows:
        print(f"{status:12} {render_size(num_bytes):>8}  {spec.title}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
