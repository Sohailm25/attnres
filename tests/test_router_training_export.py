# ABOUTME: Exercises the pilot-only per-token export path for Phase 6 router distillation.
# ABOUTME: Ensures the export joins saved oracle targets to per-token hidden states with resumable checkpoints.

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import warnings

from huggingface_hub import logging as huggingface_logging
import logging
import torch
from transformer_lens import HookedTransformer
from transformers.utils import logging as transformers_logging


ROOT = Path(__file__).resolve().parents[1]
warnings.filterwarnings(
    "ignore",
    message=r"`torch_dtype` is deprecated! Use `dtype` instead!",
)
huggingface_logging.set_verbosity_error()
logging.getLogger("huggingface_hub.file_download").setLevel(logging.CRITICAL)
transformers_logging.set_verbosity_error()


class RouterTrainingExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from validation.router_training_export import (
            run_router_distillation_pilot_export,
        )

        cls.run_router_distillation_pilot_export = staticmethod(
            run_router_distillation_pilot_export
        )
        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(io.StringIO()):
                cls.model = HookedTransformer.from_pretrained(
                    "tiny-stories-1M",
                    device="cpu",
                    dtype="float32",
                )

    def test_export_writes_manifest_summary_and_prompt_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            oracle_checkpoint_dir = temp_root / "oracle_checkpoints"
            oracle_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            output_dir = temp_root / "export"

            self._write_oracle_checkpoint(
                oracle_checkpoint_dir=oracle_checkpoint_dir,
                prompt_id="oa5-pilot-factual_recall-001",
                prompt="The capital city of France is",
                final_alpha=(0.25, 0.75),
                source_labels=("embed", "blocks.0.hook_resid_post"),
            )
            self._write_oracle_checkpoint(
                oracle_checkpoint_dir=oracle_checkpoint_dir,
                prompt_id="oa5-pilot-reasoning_math-001",
                prompt="Mira had 7 pencils, gave away 2, and bought 3 more. Mira now has",
                final_alpha=(0.6, 0.4),
                source_labels=("embed", "blocks.0.hook_resid_post"),
            )

            summary = self.run_router_distillation_pilot_export(
                model=self.model,
                collection_id="oracle_alpha_phase1_v1",
                oracle_checkpoint_dir=oracle_checkpoint_dir,
                output_dir=output_dir,
                max_sequences=2,
            )

            self.assertEqual(2, summary.num_prompts)
            self.assertTrue((output_dir / "dataset_manifest.json").is_file())
            self.assertTrue((output_dir / "summary.json").is_file())
            checkpoint_files = sorted(
                (output_dir / "checkpoints" / "prompt_exports").glob("*.pt")
            )
            self.assertEqual(2, len(checkpoint_files))

            manifest = json.loads((output_dir / "dataset_manifest.json").read_text())
            self.assertEqual("oracle_alpha_phase1_v1", manifest["collection_id"])
            self.assertEqual("pilot", manifest["split"])
            self.assertEqual(
                [
                    "prompt_id",
                    "prompt",
                    "split",
                    "target_text",
                    "tags",
                    "perturbations",
                    "token_ids",
                    "h_1[t]",
                    "h_4[t]",
                    "source_labels",
                    "final_alpha",
                ],
                manifest["dataset_fields"],
            )
            self.assertEqual(2, len(manifest["entries"]))
            checkpoint_payload = torch.load(checkpoint_files[0], map_location="cpu")
            self.assertEqual(
                {
                    "prompt_id",
                    "prompt",
                    "split",
                    "target_text",
                    "tags",
                    "perturbations",
                    "token_ids",
                    "h_1[t]",
                    "h_4[t]",
                    "source_labels",
                    "final_alpha",
                },
                set(checkpoint_payload.keys()),
            )
            self.assertEqual(
                checkpoint_payload["token_ids"].shape[0],
                checkpoint_payload["h_1[t]"].shape[0],
            )
            self.assertEqual(
                checkpoint_payload["h_1[t]"].shape,
                checkpoint_payload["h_4[t]"].shape,
            )

    def test_export_resume_reuses_saved_prompt_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            oracle_checkpoint_dir = temp_root / "oracle_checkpoints"
            oracle_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            output_dir = temp_root / "export"

            self._write_oracle_checkpoint(
                oracle_checkpoint_dir=oracle_checkpoint_dir,
                prompt_id="oa5-pilot-factual_recall-001",
                prompt="The capital city of France is",
                final_alpha=(0.25, 0.75),
                source_labels=("embed", "blocks.0.hook_resid_post"),
            )

            self.run_router_distillation_pilot_export(
                model=self.model,
                collection_id="oracle_alpha_phase1_v1",
                oracle_checkpoint_dir=oracle_checkpoint_dir,
                output_dir=output_dir,
                max_sequences=1,
            )

            with mock.patch(
                "validation.router_training_export._compute_prompt_export_payload",
                side_effect=AssertionError(
                    "prompt export checkpoints should be reused"
                ),
            ):
                resumed = self.run_router_distillation_pilot_export(
                    model=self.model,
                    collection_id="oracle_alpha_phase1_v1",
                    oracle_checkpoint_dir=oracle_checkpoint_dir,
                    output_dir=output_dir,
                    max_sequences=1,
                )

            self.assertEqual(1, resumed.num_prompts)

    def _write_oracle_checkpoint(
        self,
        *,
        oracle_checkpoint_dir: Path,
        prompt_id: str,
        prompt: str,
        final_alpha: tuple[float, ...],
        source_labels: tuple[str, ...],
    ) -> None:
        path = oracle_checkpoint_dir / f"{prompt_id}.json"
        path.write_text(
            json.dumps(
                {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "split": "pilot",
                    "num_sources": len(source_labels),
                    "source_labels": list(source_labels),
                    "uniform_loss": 1.0,
                    "optimized_loss": 0.5,
                    "null_losses": {
                        "uniform": 1.0,
                        "random_dirichlet": 1.5,
                        "magnitude_proportional": 1.2,
                        "last_layer_only": 2.0,
                    },
                    "best_alpha_entropy": 0.5,
                    "best_alpha": list(final_alpha),
                    "final_alpha": list(final_alpha),
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    unittest.main()
