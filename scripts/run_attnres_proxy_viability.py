# ABOUTME: Runs the first local Block AttnRes viability slice for the Figure 8 proxy lane.
# ABOUTME: Trains a matched tiny baseline and Block AttnRes proxy on a small text corpus with resumable checkpoints.

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from datasets import load_dataset
import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-train-texts", type=int, default=512)
    parser.add_argument("--max-eval-texts", type=int, default=128)
    parser.add_argument(
        "--tokenizer-mode",
        default="character",
        choices=("character", "compact_subword"),
    )
    parser.add_argument("--tokenizer-name", default="gpt2")
    parser.add_argument("--separator-text", default="\n\n")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument("--checkpoint-every-steps", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _resolve_device(requested_device: str | None) -> str:
    if requested_device:
        if requested_device == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        return requested_device
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_texts(
    *,
    dataset_name: str,
    dataset_config: str,
    split: str,
    text_field: str,
    max_texts: int,
) -> list[str]:
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    texts = []
    for record in dataset:
        text = record[text_field].strip()
        if not text:
            continue
        texts.append(text)
        if len(texts) >= max_texts:
            break
    if not texts:
        raise ValueError(
            f"no non-empty texts found for {dataset_name}/{dataset_config} split={split}"
        )
    return texts


def main() -> int:
    from validation.attnres_reproduction import (
        AttnResProxyConfig,
        run_attnres_proxy_viability_from_texts,
    )

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    tokenizer = None
    tokenizer_name = None
    if args.tokenizer_mode == "compact_subword":
        tokenizer_name = args.tokenizer_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    train_texts = _load_texts(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.train_split,
        text_field=args.text_field,
        max_texts=args.max_train_texts,
    )
    eval_texts = _load_texts(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.eval_split,
        text_field=args.text_field,
        max_texts=args.max_eval_texts,
    )
    config = AttnResProxyConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        num_blocks=args.n_layers,
    )
    run_attnres_proxy_viability_from_texts(
        config=config,
        dataset_name=f"{args.dataset_name}/{args.dataset_config}",
        train_texts=train_texts,
        eval_texts=eval_texts,
        output_dir=output_dir,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
        checkpoint_interval=args.checkpoint_every_steps,
        tokenizer_mode=args.tokenizer_mode,
        tokenizer=tokenizer,
        tokenizer_name=tokenizer_name,
        separator_text=args.separator_text,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
