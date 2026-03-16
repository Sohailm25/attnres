# Depth-routing interpretability on a MacBook Pro: a complete execution plan

An M4 MacBook Pro with **128GB unified RAM** is more than capable of running full oracle-α analysis on models up to 9B parameters—with every layer's activations cached simultaneously—and can train small routing modules in days rather than weeks. The key insight driving this plan is that **memory is abundant but compute is the bottleneck**, which shapes every decision below. Gemma-2-2B emerges as the optimal primary model: it fits in under 15GB total (weights plus full activation cache), has the richest SAE ecosystem of any open model, and is one of only two models supported by Anthropic's circuit-tracer library. This plan maps a realistic 8-week path to a publishable contribution, using only local hardware.

---

## What fits in 128GB and what that means for activation caching

The fundamental question for oracle-α analysis is not whether a model's weights fit—every model through 50B+ fits at some quantization level—but whether **weights plus full layer-output caches** fit simultaneously. The answer is overwhelmingly yes for the target model range.

**Model weights alone** occupy far less than 128GB for any model in the 1B–9B range. Llama-3.1-8B at fp16 requires ~16GB; Gemma-2-2B needs ~5GB; Pythia-2.8B needs ~5.6GB. Even Qwen-2.5-72B fits at Q4_K_M quantization (~39GB), and the 128GB machine has been confirmed running **104B-parameter models** at Q6_K quantization on comparable Apple Silicon configurations.

**Activation caching** is the critical calculation. For a model with L layers, hidden dimension H, and sequence length S, caching all residual-stream outputs requires L × S × H × 4 bytes at fp32:

| Model | Layers × Hidden | Cache at S=512 | Cache at S=1024 | Cache at S=2048 | Weights (fp16) | **Total at S=2048** |
|---|---|---|---|---|---|---|
| Pythia-1B | 16 × 2048 | 64 MB | 128 MB | 256 MB | 2.0 GB | **~4.3 GB** |
| Gemma-2-2B | 26 × 2304 | 117 MB | 234 MB | 468 MB | 5.2 GB | **~7.7 GB** |
| Llama-3.2-3B | 28 × 3072 | 168 MB | 336 MB | 672 MB | 6.0 GB | **~8.7 GB** |
| Pythia-2.8B | 32 × 2560 | 160 MB | 320 MB | 640 MB | 5.6 GB | **~8.2 GB** |
| Llama-3.1-8B | 32 × 4096 | 256 MB | 512 MB | 1.0 GB | 16.0 GB | **~20 GB** |
| Gemma-2-9B | 42 × 3584 | 294 MB | 588 MB | 1.15 GB | 18.4 GB | **~22.6 GB** |

Even the largest configuration—Gemma-2-9B at sequence length 2048—uses under **23GB**, leaving over 100GB free. The specific calculation the researcher asked about (8B model, 32 layers, hidden 4096, S=2048) yields exactly **1.0GB** for the residual stream cache. TransformerLens's `run_with_cache()` stores additional tensors (attention patterns, MLP outputs, Q/K/V projections), which can expand total cache to **5–10GB** for an 8B model, but filtering to only residual stream outputs via `names_filter=lambda name: "resid_post" in name` keeps memory minimal.

One critical caveat: **attention pattern caching scales quadratically**. For Llama-3.1-8B at S=2048, storing all attention patterns (32 layers × 32 heads × 2048 × 2048 × 4 bytes) consumes ~17GB. The oracle-α experiment only needs residual stream outputs, so this is avoidable, but researchers should be deliberate about which activations they cache.

**Batch processing over many inputs** is where disk caching becomes essential. Processing 10,000 sequences at S=1024 through Gemma-2-2B generates ~2.3TB of residual activations. The strategy is straightforward: process inputs in batches of 16–64, write per-layer activations to the NVMe SSD (which delivers **7+ GB/s** read speeds on M4 hardware), then memory-map files back via `numpy.memmap` or PyTorch's memory-mapped storage during the α-optimization phase.

**Quantization and interpretability are largely incompatible.** GGUF/llama.cpp models run in a C++ runtime with no Python hook mechanism—they cannot expose intermediate activations to TransformerLens. PyTorch-native quantization (BitsAndBytes NF4/INT8) preserves hook access, but since every target model fits comfortably in fp16 within 128GB, **quantization is unnecessary and should be avoided** for this experiment. Kimi-Linear-48B-A3B fits at Q4_K_M (~28GB), but its MoE architecture and lack of interpretability tooling make it impractical for oracle-α analysis.

---

## The interpretability toolchain works on Apple Silicon, with known workarounds

The entire PyTorch-based interpretability stack is functional on MPS, though the ecosystem requires specific configuration. **MLX should be avoided entirely for interpretability work**—it lacks a hook system equivalent to PyTorch's `register_forward_hook()`, and no interpretability tools exist in the MLX ecosystem.

**TransformerLens** is the backbone tool and works on MPS. It supports **50+ model families** including all target architectures: GPT-2 (all sizes), Pythia (all sizes), Llama-3.x, Gemma-2, Gemma-3, Qwen/Qwen3, Mistral, and OLMo. The `run_with_cache()` method is purpose-built for activation extraction experiments. Three environment variables are essential: `PYTORCH_ENABLE_MPS_FALLBACK=1` (graceful CPU fallback for unsupported ops), `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` (prevents premature OOM errors), and using **float32 throughout** rather than float16, since Apple Silicon lacks Tensor Core-style fp16 acceleration and fp32 is often actually faster.

**NNsight** (v0.6, February 2026) is confirmed working on MPS—its documentation explicitly shows `device='mps:0'` outputs. It works with any PyTorch model through a hook-based deferred execution system and is more flexible than TransformerLens for custom intervention experiments. This is the recommended fallback if TransformerLens lacks support for a specific model or operation.

**SAELens** supports loading and running pretrained SAEs on MPS. Training SAEs on MPS is possible but slower than CUDA and requires PyTorch ≥2.4 (which fixed a critical Adam optimizer bug on MPS where `addcmul_`/`addcdiv_` silently failed on non-contiguous tensors). The pretrained SAE catalog is extensive:

- **Gemma-2-2B**: GemmaScope SAEs (Google DeepMind) cover **all layers and sublayers** with multiple dictionary widths (16K, 32K, 65K, 131K features)—over **30 million learned features** total
- **GPT-2 Small**: Joseph Bloom's RES-JB SAEs across all 12 layers, the most studied SAE set in the field
- **Pythia models**: EleutherAI's 32K-feature residual stream SAEs
- **Gemma-3-4B**: GemmaScope 2 adds transcoders and cross-layer transcoders

**Anthropic's circuit-tracer** builds on TransformerLens and supports **Gemma-2-2B**, **Llama-3.2-1B**, and **Qwen3-4B**. It generates attribution graphs showing direct effects between transcoder features. GemmaScope transcoders with 426K and 2.5M features are available for Gemma-2-2B. Since it uses TransformerLens under the hood, the same MPS caveats apply.

**baukit** and **pyvene** are both device-agnostic PyTorch hook libraries that work on MPS without modification. baukit's `Trace`/`TraceDict` provides lightweight activation capture; pyvene (Stanford NLP) offers structured intervention abstractions including activation patching, interchange interventions, and trainable interventions.

| Tool | MPS Status | Best For | Model Coverage |
|---|---|---|---|
| TransformerLens | ✅ Works (fp32, fallback enabled) | Activation caching, hooks | 50+ families |
| NNsight | ✅ Confirmed | Flexible interventions | Any PyTorch model |
| SAELens | ✅ Works | SAE loading/training | Gemma-2, GPT-2, Pythia, Llama |
| circuit-tracer | ⚠️ Partial | Attribution graphs | Gemma-2-2B, Llama-3.2-1B, Qwen3-4B |
| baukit | ✅ Works | Lightweight activation capture | Any PyTorch model |
| pyvene | ✅ Works | Structured interventions | Any PyTorch model |
| MLX | ❌ No interp tools | Inference/training only | N/A for interpretability |

PyTorch MPS as of early 2026 (PyTorch 2.9.x stable) remains in beta but is functional for the operations needed here. Known remaining limitations include no FP8 support, no distributed training, and some missing operations that fall back to CPU. The upcoming Metal4 framework (announced at WWDC25) promises native tensor support that should substantially improve MPS performance in future PyTorch releases.

---

## Gemma-2-2B is the clear first choice, with Pythia-2.8B as the scientific complement

The model selection balances five factors: layer count (more layers → richer routing structure), SAE dictionary availability (for comparing α-routing with feature-level analysis), tooling support, memory feasibility, and research novelty.

**Gemma-2-2B earns the top slot** by dominating on ecosystem support. GemmaScope provides the most comprehensive SAE coverage of any open model—every layer, every sublayer, multiple dictionary widths. Anthropic's circuit-tracer was built for it. Its 26 layers provide good depth, and its alternating sliding-window/global attention architecture creates genuinely non-trivial routing questions: do depth-routing patterns differ between local-attention and global-attention layers? This architectural feature is a natural source of publishable insight. At ~5GB weights plus ~500MB activation cache (S=2048), it occupies under **8GB**—leaving 120GB free for batch processing.

**Pythia-2.8B is the ideal second model** for a different reason: its **154 training checkpoints** enable studying how depth-routing patterns evolve during training. No other model offers this capability. Its 32 layers provide excellent depth, and it was designed specifically for interpretability research by EleutherAI. The Tuned Lens paper (Belrose et al., 2023)—the closest existing work to oracle-α analysis—was developed primarily on Pythia. Using the same model family enables direct comparison. SAE coverage exists but is less comprehensive than Gemma's.

**GPT-2 XL (1.5B, 48 layers) serves as a validation baseline.** Its 48 layers make it the deepest option for routing analysis, and its gold-standard TransformerLens support means rapid prototyping. The risk is that reviewers may consider GPT-2-based findings less novel, so it should be positioned as a validation of generalization rather than the primary result.

Using **two or three models strengthens the paper significantly**—it demonstrates that depth-routing patterns are general phenomena rather than architecture-specific artifacts. The recommended approach: develop the method on GPT-2 XL (fastest iteration), validate on Gemma-2-2B (richest ecosystem), and extend to Pythia-2.8B (training dynamics angle).

The existing literature establishes clear context. The **Tuned Lens** showed that easy tokens are predicted at early layers while hard tokens require deeper processing. **ShortGPT** found that middle layers are most redundant while head and tail layers are "core." **LayerSkip** and **CALM** demonstrated that models can skip layers dynamically with minimal performance loss. The oracle-α analysis extends this line of work by optimizing **per-input, per-layer contribution weights**, revealing fine-grained routing structure that aggregate metrics miss. Combining this with SAE feature analysis to explain *why* certain layers matter for certain inputs is the novel contribution.

---

## Training strategy: freeze the base, train only the router

Training a full 200M-parameter transformer from scratch on M4 Max hardware would take **2–4 weeks for 5B tokens** (estimated throughput: 2,000–8,000 tokens/second depending on optimization level). This is feasible but slow, and Chinchilla scaling suggests 200M parameters is optimally trained on ~4B tokens.

The far better strategy is **freezing a pretrained model and training only the α-routing component**. This reduces trainable parameters from 200M to perhaps **1–10M** (the routing network alone), which transforms the training timeline from weeks to days:

- Forward pass through frozen layers is inference-only (no gradient computation)
- Optimizer states (Adam m and v) are needed only for the router (~40–400MB)
- Effective throughput for the routing component could reach **10,000–50,000 tokens/second**
- Training on 1B tokens would take **hours, not weeks**

In MLX, this is elegant: `model.freeze()` locks all parameters, then `model.router.unfreeze()` enables gradients only for the routing component. PyTorch achieves the same via `param.requires_grad = False` on base parameters.

**Knowledge distillation** is a compelling complementary strategy. With 128GB, the researcher can hold a quantized 7B–14B teacher model (~4–8GB at 4-bit) alongside the student model simultaneously—something impossible on a 24GB GPU. The teacher generates soft targets that the routing module learns to approximate, potentially providing richer supervision than hard labels alone.

For the interpretability phase, **MLX is not recommended** despite its speed advantage, because the entire interpretability toolchain is PyTorch-based. The practical choice is **PyTorch with MPS backend** for the oracle-α analysis (which needs TransformerLens hooks), with MLX reserved for any standalone training experiments where interpretability tool integration isn't needed.

Cloud compute is worth considering for the final training run only. A single A100 hour costs $0.42–$2.50, and an H100 could train the routing module in hours for under $50. But for iterative development—testing architectures, debugging loss functions, exploring hyperparameters—the always-available, zero-cost M4 is the right environment.

---

## Environment setup and data pipeline

The Python environment should use **conda** (specifically miniforge for ARM-native packages) with PyTorch nightly or ≥2.4 stable:

```bash
# Core environment
conda create -n interp python=3.11
conda activate interp
pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
pip install transformer-lens sae-lens nnsight pyvene circuitsvis
pip install wandb plotly jupyter

# Essential environment variables (add to .zshrc)
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

For data, **OpenWebText** (`Skylion007/openwebtext` on HuggingFace, ~38GB) is the standard corpus for interpretability work. The oracle-α analysis does not require massive data—**10K–100K diverse sequences** are sufficient to identify robust routing patterns. Standard practice for SAE activation studies uses as few as 100–150 samples at sequence lengths of 150–300 tokens. The researcher should start with 1,000 sequences and scale up as patterns emerge.

Data acquisition and preprocessing takes about a day. Download via HuggingFace `datasets`, tokenize with the model's tokenizer, chunk into fixed-length sequences, and save as memory-mapped arrays. For the training phase, The Pile (800GB) is the canonical dataset for Pythia-family comparisons, but a curated subset of 1–5GB is sufficient for router training.

**Weights & Biases** integrates natively with both PyTorch and TransformerLens for experiment logging. Log α-weight distributions, loss curves, per-layer importance heatmaps, and routing pattern clusters. **CircuitsVis** (maintained by the TransformerLens org) provides attention pattern and token-level visualizations in Jupyter notebooks. **Plotly** handles the custom heatmaps and trajectory plots needed for depth-routing visualization.

---

## Week-by-week execution plan

**Weeks 1–2: Infrastructure and oracle-α analysis on GPT-2 XL.** Set up the conda environment, verify TransformerLens runs on MPS with `run_with_cache()`, and implement the oracle-α optimization loop. For each input sequence, cache all 48 layers' residual stream outputs, then optimize α weights that linearly combine layer outputs to minimize cross-entropy loss on the next token. Process 1,000 sequences from OpenWebText. Deliverable: working code and initial α-weight heatmaps showing per-input, per-layer importance patterns. Estimated wall-clock: **3–5 days** for setup, **2–3 days** for processing 1,000 sequences (GPT-2 XL inference at ~1,000 sequences/hour on MPS).

**Weeks 3–4: Scale to Gemma-2-2B and pattern analysis.** Port the oracle-α code to Gemma-2-2B, process 10,000+ sequences, and begin pattern discovery. Cluster inputs by their α profiles (k-means or spectral clustering on the L-dimensional α vectors). Load GemmaScope SAEs via SAELens and characterize what SAE features activate differently across routing clusters. Compare oracle-α results with Tuned Lens predictions on the same model. Deliverable: routing pattern taxonomy showing distinct clusters (e.g., factual recall uses layers 8–14, syntactic processing uses layers 2–6). Estimated wall-clock: **5–7 days** for 10K sequences on Gemma-2-2B.

**Week 5: Pythia training dynamics extension.** Load Pythia-2.8B at multiple training checkpoints (e.g., steps 0, 1K, 10K, 50K, 143K) and run oracle-α analysis at each. Track how routing patterns evolve during training: do early checkpoints route uniformly while late checkpoints specialize? This is a unique contribution enabled by Pythia's checkpoint suite. Estimated wall-clock: **3–4 days** (5 checkpoints × 5K sequences each).

**Weeks 6–7: Attention Residual router training.** Design a lightweight routing network (e.g., a 2-layer MLP taking the first layer's output and predicting α weights for all subsequent layers). Freeze Gemma-2-2B's base weights and train only the router to match oracle-α outputs. Use MLX if pure training speed matters, or PyTorch MPS if integration with the analysis pipeline is more important. Training budget: 100M–500M tokens with the routing loss. Estimated wall-clock: **2–5 days** for router training, **2–3 days** for evaluation and ablation studies.

**Week 8: Paper writing and visualization.** Write up results as a LessWrong post with interactive CircuitsVis visualizations and an arXiv preprint. Structure: (1) oracle-α reveals input-dependent depth routing, (2) routing patterns correlate with SAE feature clusters, (3) routing evolves during training (Pythia evidence), (4) a learned router approximates oracle routing with minimal parameters. Release code on GitHub. Target venues: **LessWrong/Alignment Forum** for immediate community impact, **ICML or NeurIPS mechanistic interpretability workshops** for academic credit.

**Total estimated compute time**: ~15–20 days of active computation spread across 8 calendar weeks, with the remaining time spent on analysis, writing, and iteration. The 128GB MacBook Pro is never the bottleneck—every model fits with massive headroom, and the primary constraint is the **~18 TFLOPS** of compute available for inference passes. Processing 10K sequences through Gemma-2-2B with full activation caching should take roughly 6–10 hours, making multiple experimental iterations feasible within each week.

## Conclusion

The 128GB M4 MacBook Pro is not merely adequate for this research—it is **genuinely well-suited** to it. Activation caching for the largest target model (Gemma-2-9B at S=2048) uses under 23GB total, leaving over 100GB free. The entire interpretability toolchain (TransformerLens, SAELens, NNsight, circuit-tracer) functions on MPS with known, manageable workarounds. The frozen-base-plus-trainable-router approach eliminates the training bottleneck by reducing the problem from weeks-long pretraining to days-long router optimization.

The strongest strategic choice is to lead with **Gemma-2-2B** (best SAE ecosystem and circuit-tracer support), validate generality on **Pythia-2.8B** (unique training dynamics angle via 154 checkpoints), and use **GPT-2 XL** for rapid prototyping (48 layers, gold-standard tooling). This three-model approach produces a paper that demonstrates general phenomena rather than architecture-specific artifacts. The combination of oracle-α routing analysis, SAE feature characterization of routing clusters, and training-dynamics evolution through Pythia checkpoints constitutes a novel and publishable contribution to mechanistic interpretability—achievable entirely on local hardware within 8 weeks.
