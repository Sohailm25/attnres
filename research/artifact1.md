# Attention Residuals weights have not been released

**MoonshotAI has not released any pre-trained model weights for their Attention Residuals (AttnRes) architecture.** The GitHub repository published on March 15–16, 2026 is a paper-only release containing the research PDF and illustrative pseudocode — no checkpoints, no training code, and no download links. None of the five categories of weights you asked about are publicly available as of March 16, 2026.

## The GitHub repository contains only a paper and images

The repository at `github.com/MoonshotAI/Attention-Residuals` holds exactly **three items**: a `README.md` with PyTorch-style pseudocode and benchmark tables, the paper PDF (`Attention_Residuals.pdf`, ~930 KB), and an `assets/` folder with figure images. There are no `.pt`, `.bin`, `.safetensors`, or any other model files. The repository has a single commit, **zero releases**, zero packages, and no links to external model hosting. The README makes no mention of "weights," "checkpoints," "download," or "pre-trained" anywhere. No license file is included. Notably, the paper is not even on arXiv — the citation URL simply points back to the GitHub repo.

## Kimi Linear on HuggingFace is the pre-AttnRes baseline

MoonshotAI maintains an active HuggingFace organization page (`huggingface.co/moonshotai`) with **15+ published models**, including `Kimi-Linear-48B-A3B-Base` and `Kimi-Linear-48B-A3B-Instruct`. These are the Kimi Linear models released with the earlier KDA (Kimi Delta Attention) paper from October 2025 — they are the **baseline architecture without AttnRes**. The AttnRes paper uses this exact architecture as its testbed and reports improvements (e.g., **+7.5 on GPQA-Diamond**, +3.6 on Math, +3.1 on HumanEval), but those enhanced checkpoints trained on 1.4T tokens were not published. Third-party derivatives like Cerebras's pruned variant and community AWQ quantizations are all based on the original, pre-AttnRes Kimi Linear weights.

## No scaling law checkpoints or third-party reproductions exist

The smaller models from the paper's scaling law experiments (**194M to 528M activated parameters**, five model sizes comparing PreNorm, Full AttnRes, and Block AttnRes) have not been released. No HuggingFace models tagged with "AttnRes" or "attention residuals" from any source exist. Given the paper dropped barely a day ago, no third-party reproductions with trained weights have appeared either — all GitHub search results for "attention residuals" point to the unrelated 2017 CVPR Residual Attention Network, not this work.

## Community reaction confirms paper-only status

Early community discussion on LINUX DO (Chinese tech forum), MarkTechPost, and Nerds Chalk all summarize the paper's findings and link to the repo, with no mention of downloadable weights. One user on LINUX DO asked when Kimi 3 would arrive, suggesting the community views AttnRes as a research contribution that may eventually ship inside a future production model rather than as a standalone weight release. MoonshotAI's official announcement via Kimi's X/Twitter account simply introduced the research — no open-weight release was announced.

## What you can actually use today

In practical terms, the release gives you:

- **Architecture pseudocode** (Block AttnRes in PyTorch style) sufficient to reimplement the method in your own codebase
- **Training recipes and hyperparameters** documented in the paper, including the scaling law experimental setup
- **Benchmark numbers** for validation if you train your own models with AttnRes

What it does not give you is any runnable training code, configuration files, or model artifacts. MoonshotAI has a strong track record of open-sourcing major model weights (Kimi K2, K2.5, Kimi Linear, Moonlight, Kimi Audio), so AttnRes-enhanced weights could plausibly appear in a future model release — possibly as part of a Kimi K3 or an updated Kimi Linear — but **no timeline or commitment has been stated**.
