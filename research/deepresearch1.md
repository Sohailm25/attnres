# Latent depth-routing in transformers: a full research audit

**The project is well-positioned but faces a rapidly shifting landscape.** The core claim — that standard transformers contain latent depth-routing structure invisible to current interpretability tools — is strongly supported by converging 2024–2026 literature (Curse of Depth, ShortGPT, Gromov et al., MUDDFormer's formal proof). The oracle-alpha framing remains genuinely novel: no prior work extracts input-dependent routing weights from frozen standard transformers post-hoc. However, **three developments demand immediate response**: the Attention Residuals paper dropped publicly on March 15–16, 2026 (literally today), the refusal-direction literature has grown significantly more nuanced than a single direction, and MUDDFormer's Appendix A already proves the formal equivalence between dynamic dense connections and depth-wise self-attention — meaning part of the theoretical contribution may need reframing as empirical validation rather than novel insight. The tooling ecosystem is stable but requires careful version pinning (TransformerLens v3 beta is not ready), and the safety lane needs substantial methodological upgrading given 2025 results showing harmfulness and refusal are encoded separately.

---

## 1. What still looks strong

**Oracle-alpha as post-hoc routing extraction is genuinely novel.** Extensive search across 2024–2026 literature found zero prior work framing per-token, per-layer contribution weights as "latent routing" in frozen standard transformers. Adjacent work (ShortGPT's Block Influence, Gromov et al.'s layer pruning, Razzhigaev et al.'s linearity analysis) all compute **static, global** importance scores. The project's input-dependent formulation — optimizing α_l(t) per token per layer — goes strictly further. This gap is real and publishable.

**The architecture progression validates the framing.** The field has produced a clean ladder: DenseFormer (static weights, NeurIPS 2024) → Hyper-Connections (learnable matrices, ICLR 2025, used in DeepSeek-V3) → MUDDFormer (dynamic input-dependent multiway routing, ICML 2025) → DeepCrossAttention (depth-wise cross-attention, ICML 2025) → Attention Residuals (full softmax attention over depth, March 2026). Each step makes depth-routing more explicit and achieves better compute efficiency. The project's claim that standard transformers contain *latent* versions of this structure slots perfectly into this narrative as the interpretability complement.

**MUDDFormer's formal proof is an asset, not a threat.** MUDDFormer's Appendix A (arXiv:2502.12170) explicitly proves that dynamic dense connections are equivalent to depth-wise self-attention. Rather than scooping the project, this provides the formal backbone. The project should cite it as establishing the mathematical framework, then contribute the empirical demonstration that frozen standard transformers already implement a soft version of this.

**The Curse of Depth provides theoretical grounding.** Sun et al. (arXiv:2502.05795, Feb 2025) prove that Pre-LN transformers cause deep layers to degenerate into near-identity functions via exponential variance growth in the residual stream. This is the *mechanism* explaining why standard transformers have non-uniform depth utilization. Oracle-alpha should recover this pattern — shallow layers contributing more — and the project can frame explicit depth-routing (AttnRes et al.) as architecturally solving what the Curse of Depth creates.

**The Pythia training-dynamics lane is strengthened by PolyPythias.** PolyPythias (ICLR 2025, arXiv:2503.09543) adds **45 new training runs** (9 seeds × 5 sizes, 14M–410M) to the original Pythia suite. This enables separating robust depth-allocation patterns from seed-specific artifacts. Combined with SAE-Track (arXiv:2412.17626) for feature tracking across checkpoints and the "Embryology of a Language Model" susceptibility analysis (arXiv:2508.00331), the training-dynamics lane has excellent methodological support.

**TransformerLens provides full sublayer access for Figure 8 validation.** Version 2.17.0 (July 2025, stable) supports `cache["attn_out", layer]`, `cache["mlp_out", layer]`, and per-head results via `use_attn_result=True`. Gemma-2 is explicitly supported with its architectural quirks (interleaved attention, soft-capping). NNsight v0.6.1 with the nnterp wrapper (NeurIPS 2025 MI Workshop) provides an alternative with exact HuggingFace numerical fidelity.

---

## 2. What looks weak, risky, or likely to be criticized

### The AttnRes paper released today changes the landscape — **must-fix**

The Attention Residuals paper (Moonshot AI / Kimi team, GitHub: MoonshotAI/Attention-Residuals) was published March 15–16, 2026, with code and Kimi Linear weights (48B MoE, 3B active). It implements the exact architecture the project references — softmax attention over depth with learned per-layer pseudo-query vectors. **The project must now engage with this as a published, publicly available work, not a hypothetical.** Specifically:

- The Figure 8 reference needs verification. I could not confirm that AttnRes contains a "Figure 8" showing the patterns claimed. MUDDFormer (arXiv:2502.12170) has a Figure 8 showing "attention head activation ratio by layers." The project must clarify which paper's Figure 8 is being validated and ensure the claims match the actual figure.
- Public AttnRes weights exist on HuggingFace (`moonshotai/Kimi-Linear-48B-A3B-Instruct`). The project should use these for direct comparison if computationally feasible.

### Layer importance is interdependent, not decomposable — **must-fix**

"Pruning as a Cooperative Game" (arXiv:2602.07804, 2025) demonstrates that layer importance is **context-dependent on other layers** — removing one layer changes the optimal importance of all others. Optimizing oracle-alpha independently per layer (treating each α_l as separable) will miss these interactions. **Reviewers will flag this.** The project must either: (a) jointly optimize all α simultaneously with explicit interaction terms, (b) use Shapley-value-inspired methods to quantify interaction effects, or (c) explicitly acknowledge the limitation and bound the approximation error.

### Refusal is not a single direction — **must-fix for safety lane**

Three 2025 papers fundamentally complicate the planned safety lane:

- **"The Geometry of Refusal" (ICML 2025)**: Finds *multiple independent* refusal directions forming multi-dimensional "concept cones," not a single direction. The project cannot assume Arditi et al.'s single-direction finding generalizes.
- **"LLMs Encode Harmfulness and Refusal Separately" (NeurIPS 2025, arXiv:2507.11878)**: Harmfulness is encoded at the instruction token position; refusal at the post-instruction position. These are separate computational stages with distinct causal roles. Jailbreaks suppress refusal signals without suppressing harmfulness recognition.
- **"Safety Layers in Aligned LLMs" (ICLR 2025, arXiv:2408.17003)**: A small set of **contiguous middle layers** are most crucial for distinguishing malicious from normal queries. Refusal execution concentrates in deeper layers.

The safety lane's requirement for "explicit refusal-feature discovery/validation" is necessary but insufficient. The project needs to distinguish harmfulness features from refusal features, test for multi-dimensionality, and localize safety-critical layers before making routing claims. The GemmaScope SAE-based refusal feature work (Yeo et al., EMNLP Findings 2025) provides a direct template using attribution patching + activation steering on GemmaScope features.

### Tuned lens has no Gemma-2 support and is in maintenance mode — **should-fix**

The tuned-lens repo (AlignmentResearch/tuned-lens, v0.2.0) has **no pre-trained Gemma-2 lenses**, no documented Gemma support, and appears to be in maintenance mode (last issue Feb 2025). The tool-breakage lane requires "tuned-lens-aware comparison," but this may require training a custom tuned lens for whatever model is used. Alternatives: (1) train a lens using the existing repo's training code, (2) use nnterp's built-in logit lens + Patchscopes, (3) use the "Logit Prisms" approach (Nguyen 2024) for per-component decomposition, (4) cite the "Transformers Don't Need LayerNorm at Inference Time" result (arXiv:2507.02559) to justify working with linearized models where raw logit lens is exact.

### Circuit instability undermines significance claims — **should-fix**

"Mechanistic Interpretability as Statistical Estimation" (arXiv:2510.00845, Oct 2025) shows that **all circuit extraction methods produce highly unstable circuits across nearly all sources of variation** — small perturbations in data or analysis pipeline yield substantially different structures. "Everything, Everywhere, All at Once: Is MI Identifiable?" (ICLR 2025) proves multiple incompatible explanations can satisfy MI criteria. SAEs were shown to extract plausible explanations from random weights (Heap et al., 2025). The project's planned pilot/confirmatory split is necessary but may not be sufficient. Bootstrap resampling, stability metrics, and explicit uncertainty quantification are needed throughout.

### The "existing tools cannot see this structure" claim needs precise scoping — **should-fix**

Anthropic's circuit-tracer (May 2025) with cross-layer transcoders and attribution graphs *does* reveal per-layer information flow patterns — these are effectively a fine-grained version of routing structure. Logit Prisms decompose outputs into per-component contributions. The claim should be more precise: existing tools decompose contributions but don't extract *input-dependent routing weights* comparable to those in explicit routing architectures. The distinction between "contribution analysis" and "routing extraction" needs to be razor-sharp, or reviewers will argue the project is re-describing known phenomena.

### TransformerLens v3 is in beta with breaking changes — **should-fix**

TransformerLens v3.0.0 (beta since December 2025) introduces TransformerBridge, fundamentally changing the API. **Pin to v2.17.0** for all experiments. The v3 beta has known issues (PRs reverting weight processing decoupling, boolean mask handling fixes). The BridgeCompatibilityScripts repo exists specifically to test migration. Do not migrate until v3 reaches stable release.

---

## 3. New papers, tools, and results the project should engage with

### Architecture papers (2024–2026)

| Paper | ArXiv/Link | Date | Key relevance |
|---|---|---|---|
| **Attention Residuals** | github.com/MoonshotAI/Attention-Residuals | Mar 2026 | Core architecture; public weights; softmax over depth |
| **MUDDFormer** | 2502.12170 | Feb 2025 | Formal proof: dynamic dense ≡ depth-wise attention |
| **DeepCrossAttention** | 2502.06785 | Feb 2025 | Depth-wise cross-attention; ICML 2025 |
| **Hyper-Connections** | 2409.19606 | Sep 2024 | Multi-stream residuals; used in DeepSeek-V3; ICLR 2025 |
| **Manifold-Constrained HC (mHC)** | 2512.24880 | Dec 2024 | Stability fix for HC; DeepSeek team |
| **Curse of Depth** | 2502.05795 | Feb 2025 | Theoretical: Pre-LN causes deep-layer identity collapse |
| **Mixture of Depths** | 2404.02258 | Apr 2024 | ICML 2024; token-level depth routing |
| **Mixture of Recursions** | 2507.10524 | Jul 2025 | NeurIPS 2025; recursive + adaptive depth |
| **DenseFormer** | 2402.02622 | Feb 2024 | NeurIPS 2024; static depth-weighted averaging |
| **Transformer Layers as Painters** | 2407.09298 | Jul 2024 | AAAI 2025; three layer classes; middle-layer interchangeability |
| **Your Transformer is Secretly Linear** | 2405.12250 | May 2024 | ACL 2024; near-linearity justifies scalar layer weights |
| **The Unreasonable Ineffectiveness of Deeper Layers** | 2403.17887 | Mar 2024 | NeurIPS 2024 Workshop; half of layers prunable |
| **Pruning as a Cooperative Game** | 2602.07804 | 2025 | Layer importance is interdependent |
| **FlexiDepth** | (2025) | 2025 | Plug-in router for adaptive layer-skipping in pre-trained LLMs |
| **Value Residual Learning** | 2410.17897 | Oct 2024 | Value-specific cross-layer residuals |
| **Depth-Recurrent Attention Mixtures** | 2601.21582 | Jan 2026 | Depth as first-class sequential dimension |

### Interpretability tools and methods

| Tool/Paper | ArXiv/Link | Date | Key relevance |
|---|---|---|---|
| **circuit-tracer** | github.com/safety-research/circuit-tracer | May 2025 | Cross-layer transcoders + attribution graphs; Gemma-2-2b support |
| **GemmaScope 2** | deepmind.google/blog/gemma-scope-2 | Sep 2025 | SAEs + CLTs for all Gemma 3 sizes; Matryoshka training |
| **nnterp** | github.com/Butanium/nnterp | Nov 2025 | Standardized sublayer access across 50+ models via NNsight |
| **NNsight v0.6.1** | github.com/ndif-team/nnsight | Feb 2026 | ICLR 2025; wraps any PyTorch model; remote execution |
| **SAELens v6** | github.com/decoderesearch/SAELens | 2025 | Major refactor; JumpReLU/TopK SAE support |
| **Logit Prisms** | (Nguyen 2024) | 2024 | Per-component logit decomposition |
| **RouteSAEs** | aclanthology.org/2025.emnlp-main.346 | 2025 | Routed cross-layer SAE features |
| **Transformers Don't Need LayerNorm at Inference** | 2507.02559 | Jul 2025 | Makes DLA exact; justifies linear residual decomposition |
| **Language-Model-SAEs** | github.com/OpenMOSS/Language-Model-SAEs | 2025 | CrossCoder for training snapshot comparison |

### Safety and statistical methodology

| Paper | ArXiv/Link | Date | Key relevance |
|---|---|---|---|
| **Geometry of Refusal: Concept Cones** | (ICML 2025) | 2025 | Multiple refusal directions; concept cones |
| **LLMs Encode Harmfulness and Refusal Separately** | 2507.11878 | Jul 2025 | Distinct computational stages; different token positions |
| **Safety Layers in Aligned LLMs** | 2408.17003 | Aug 2024 | Middle layers for harm; deeper layers for refusal; ICLR 2025 |
| **Understanding Refusal with SAEs** | (EMNLP Findings 2025) | 2025 | GemmaScope SAE-based refusal features |
| **CAST: Conditional Activation Steering** | (ICLR 2025) | 2025 | Selective refusal based on context |
| **SafeConstellations** | 2508.11290 | Aug 2025 | Layer-wise trajectory patterns for refusal/non-refusal |
| **MI as Statistical Estimation** | 2510.00845 | Oct 2025 | Circuit extraction instability; bootstrap needed |
| **Hypothesis Testing the Circuit Hypothesis** | 2410.13032 | Oct 2024 | NeurIPS 2024; formalized circuit testing |
| **Is MI Identifiable?** | (ICLR 2025) | 2025 | Multiple valid explanations; non-uniqueness |
| **SAEs Don't Find Canonical Units** | 2502.04878 | Feb 2025 | Features depend on training choices |

### Training dynamics

| Paper | ArXiv/Link | Date | Key relevance |
|---|---|---|---|
| **PolyPythias** | 2503.09543 | Mar 2025 | 45 new Pythia runs; ICLR 2025 |
| **SAE-Track** | 2412.17626 | Dec 2024 | Track SAE features across checkpoints |
| **Birth of Knowledge** | 2505.19440 | May 2025 | Feature emergence across time/layers/scale; semantic reactivation |
| **Embryology of a Language Model** | 2508.00331 | Aug 2025 | SLT susceptibility analysis for phase detection |
| **OLMo 2 Early Training Checkpoints** | allenai/OLMo-2-0425-1B-early-training | Apr 2025 | Every 1K steps for 37K steps from step 0 |
| **Hidden Dynamics of Massive Activations** | 2508.03616 | Aug 2025 | MA trajectories across all Pythia checkpoints |
| **LLM Circuit Analyses Are Consistent** | 2407.10827 | Jul 2024 | Circuits stable across training/scale in Pythia |

---

## 4. Best updated methodology recommendations

### Oracle-alpha optimization — **must-fix: joint optimization**

Oracle-alpha should be optimized **jointly across all layers simultaneously** using gradient-based methods, not layer-by-layer. The objective should be minimizing KL divergence between the oracle-routed output and the original model's output distribution, subject to the constraint that the reconstruction passes through the model's final layer normalization. Specifically: compute the routed residual stream as r(t) = Σ_l α_l(t) · f_l(t) where f_l(t) is layer l's sublayer output for token t, apply the model's final RMSNorm to r(t), then project through the unembedding matrix. Optimize α to minimize KL(p_original ‖ p_routed). The normalization must use the **full mixture's norm**, not per-source norms. Add a comparison against three baselines: (1) ShortGPT's Block Influence scores, (2) angular distance between layer inputs/outputs, and (3) tuned lens prediction-change magnitude.

### Figure 8 validation — **must-fix: identify the actual figure**

The team must confirm which paper's Figure 8 is being validated. If it is the AttnRes paper, verify the figure exists and characterize its content from the PDF at github.com/MoonshotAI/Attention-Residuals/blob/master/Attention_Residuals.pdf. If it is MUDDFormer's Figure 8 (attention head activation ratio by layers), the validation requires matching oracle-alpha patterns to MUDDFormer's learned activation ratios. Sublayer-level extraction (attention output + MLP output separately) is correctly identified as necessary. Use TransformerLens v2.17.0's `cache["attn_out", l]` and `cache["mlp_out", l]` or NNsight + nnterp's `model.attentions_output[l]` and `model.mlps_output[l]`.

### Statistical methodology — **must-fix: bootstrap + FDR**

Implement **bootstrap resampling** (≥1000 resamples) for all oracle-alpha estimates, reporting 95% confidence intervals. Use Benjamini-Hochberg FDR control when testing multiple layers or features for significance. Apply the pilot/confirmatory split as planned: use 50% of prompts for exploratory analysis (discover patterns, tune methods), hold out 50% for confirmatory testing (pre-registered hypotheses only). Report stability metrics: resample prompts, rerun analysis, compute the Jaccard similarity of discovered patterns. Cite Shi et al. (NeurIPS 2024, arXiv:2410.13032) as the methodological template for formalized circuit hypothesis testing.

### JSD clustering — **must-fix: use √JSD with non-Ward linkage**

The constraint against Ward linkage on raw JSD is correct but incomplete. √JSD is a proper metric (Fuglede & Topsoe 2004), but it is **not** a squared Euclidean distance, so Ward's criterion is not formally valid even on √JSD. Use **complete linkage** or **average linkage** with √JSD as the distance metric. If Ward is preferred for its compactness properties, first embed JSD distances into Euclidean space via classical MDS, then apply Ward to the embedding. Report cophenetic correlation and compare at least two linkage methods to demonstrate robustness.

### Safety lane — **must-fix: three-stage protocol**

Replace the current single-step approach with:

1. **Layer localization**: Use the safety-layers methodology (Li et al., ICLR 2025) to identify which layers in the target model carry safety-critical representations. Compute layer-wise cosine similarity between residual streams for matched harmful/benign prompt pairs. Identify the divergence layers.
2. **Feature discovery**: At the identified safety layers, use GemmaScope SAEs (if using Gemma-2) or train task-specific SAEs. Apply hybrid attribution patching + activation steering (per Yeo et al., EMNLP Findings 2025) to find causally relevant features. Distinguish harmfulness-encoding features from refusal-execution features per Zhao et al. (NeurIPS 2025).
3. **Routing analysis**: Only after features are validated, analyze how oracle-alpha weights differ between harmful and benign prompts, focusing on the identified safety layers and features. Test whether routing differences predict safety behavior above chance using held-out prompts.

### Tool-breakage demonstration — **should-fix: clarify the claim**

The tool-breakage lane must be precise about *what* breaks. Anthropic's circuit-tracer with CLTs already reveals cross-layer information flow. The project should demonstrate that **input-dependent routing weights** (oracle-alpha patterns) are invisible to: (1) standard logit lens (raw projection of residual stream), (2) tuned lens (learned affine probes), (3) residual-stream SAEs (which don't model cross-layer mixing weights), and (4) standard attribution patching (which attributes to components but doesn't extract routing structure). Train a tuned lens on the target model (since no pre-trained Gemma-2 lens exists) and show that tuned lens convergence patterns differ from oracle-alpha patterns. The comparison should be against the tuned lens's per-layer prediction-change trajectory, not just the raw logit lens.

---

## 5. Best updated framing for a serious blog post or paper

**Recommended title**: "The Residual Stream Is Already a Router: Extracting Latent Depth-Routing Structure from Standard Transformers"

**Core narrative**: The transformer architecture community has independently converged on the insight that depth-routing improves efficiency (DenseFormer → HC → MUDDFormer → DCA → AttnRes), while the interpretability community has independently discovered that layers contribute non-uniformly (ShortGPT, Curse of Depth, Gromov et al.). This paper bridges both communities by showing that standard frozen transformers already implement soft depth-routing — extractable via oracle-alpha analysis — that matches patterns from architectures explicitly trained with depth-routing. Current interpretability tools decompose residual stream contributions but cannot extract these routing weights, creating a blind spot for safety-critical analysis.

**Strongest claims (in order of defensibility)**:

1. Oracle-alpha analysis extracts input-dependent, per-token routing weights from frozen standard transformers that are significantly non-uniform (against preregistered null of uniform routing).
2. These routing patterns exhibit block structure matching independently-discovered patterns in trained depth-routing architectures (MUDDFormer, AttnRes).
3. Current interpretability tools (logit lens, tuned lens, SAEs, attribution patching) cannot directly extract these routing weights, creating a specific blind spot.
4. Safety-relevant prompts produce systematically different routing patterns, localized to layers independently identified as safety-critical.
5. Explicit depth-routing architectures make this hidden structure interpretable and actionable for safety.

**What to avoid claiming**: Do not claim that oracle-alpha on frozen models is equivalent to co-adapted AttnRes training (already noted). Do not claim that the routing structure is "the same" as learned routing — it is a latent, unrealized routing pattern that explicit architectures can actualize. Do not claim that existing tools "fail" — claim they have a specific blind spot regarding routing-weight extraction.

**Positioning against competitors**: The project is not proposing a new architecture (that's AttnRes/MUDDFormer/DCA). It is not proposing a new interpretability tool (that's circuit-tracer). It occupies the unique intersection: using interpretability methods to reveal that architectural innovations in depth-routing are addressing a real, measurable structural property of standard transformers, with implications for safety interpretability. Frame it as "the interpretability case for depth-routing architectures."

---

## 6. Concrete repo changes before implementation

**Must-fix (before any experiments):**

- **Pin TransformerLens==2.17.0** in requirements.txt. Do not use v3 beta. Add a comment explaining the TransformerBridge migration is pending v3 stable release.
- **Pin NNsight==0.6.1** as an alternative backend. Install nnterp for standardized sublayer access.
- **Add SAELens>=6.37.0** for GemmaScope integration. Migrate any pre-v6 SAE code.
- **Create a `configs/` directory** with separate YAML configs for exploratory vs. confirmatory runs, specifying the prompt split and pre-registered hypotheses.
- **Add a `baselines/` module** implementing ShortGPT Block Influence, angular distance, and tuned-lens prediction-change as comparison methods for oracle-alpha.
- **Fix the reconstruction normalization**: ensure the oracle-alpha objective applies final RMSNorm to the full routed mixture, not to individual layer contributions. Add an assertion that `reconstruct(alpha, sublayer_outputs).logits ≈ model.forward(input).logits` when alpha is uniform.
- **Add sequence-level aggregation as the default** for all statistical tests. Token-level results should be reported as supplementary.
- **Create `scripts/download_models.py`** to download all required model weights and checkpoints in advance.

**Should-fix (before claim-bearing runs):**

- **Add √JSD clustering** with complete linkage as the default in `clustering.py`. Implement cophenetic correlation and silhouette score validation. Add comparison against average linkage and Ward-on-MDS-embedding.
- **Implement bootstrap confidence intervals** for oracle-alpha estimates (≥1000 resamples per prompt set).
- **Add Benjamini-Hochberg FDR correction** in the statistical testing module.
- **Create safety-lane preprocessing**: implement the three-stage protocol (layer localization → feature discovery → routing analysis) as a pipeline.
- **Add router input variable naming check**: ensure all code/comments refer to h_l[t] (per-token), not h_l (sequence-level).
- **Add a `validation/` module** with sanity checks: uniform-alpha reconstruction matches original logits, alpha gradients are non-zero, alpha optimization converges.

**Optional (quality-of-life):**

- Add Weights & Biases or similar logging for oracle-alpha optimization runs.
- Create visualization scripts for alpha heatmaps (tokens × layers) and block-structure dendrograms.
- Add NNsight backend as a fallback for models not supported by TransformerLens.

---

## 7. Experiments to add, remove, or reorder

### Add — **must-add**:

- **Lane 0 (blocking, run first): Sanity-check reconstruction.** Before any oracle-alpha experiments, verify that uniform α = 1/L reconstruction matches original model logits after final normalization. If this fails, the entire decomposition is wrong. Budget: 1 hour.
- **Lane 1b: Cooperative-interaction test.** After oracle-alpha optimization, test for inter-layer interactions by comparing independently-optimized α with jointly-optimized α. Compute Shapley-value-inspired interaction terms for the top-5 most important layers. This preempts the cooperative-game criticism.
- **Lane 7 revision: Three-stage safety protocol.** Replace the current safety lane with the three-stage protocol (layer localization → feature discovery → routing analysis). Add harmfulness vs. refusal feature separation per Zhao et al. Test for multi-dimensional concept cones per ICML 2025.

### Add — **should-add**:

- **Baseline comparison lane**: Run ShortGPT BI scores, angular distance, and tuned-lens prediction-change on the same model and prompts. Show oracle-alpha captures input-dependent variation these static methods miss. This is the most convincing evidence for the "latent routing" claim.
- **Cross-architecture validation**: If compute allows, run oracle-alpha on both a Pre-LN model (GPT-2 or Pythia, affected by Curse of Depth) and a model using Peri-LN or other normalization (Gemma-2 uses RMSNorm with soft-capping). If routing patterns differ systematically, this demonstrates the Curse of Depth connection.
- **PolyPythias seed comparison (Lane 8 enhancement)**: Use multiple Pythia seeds at the same size to separate robust depth-allocation dynamics from seed-specific noise. Run oracle-alpha on 3+ seeds of Pythia-160M across 10+ checkpoints each.

### Reorder:

- **Run Lane 1 (oracle-alpha) before Lane 2 (Figure 8)** — you need the oracle-alpha methodology working before you can validate it against AttnRes/MUDDFormer patterns.
- **Run Lane 5 (tool-breakage) after Lanes 1–3** — the breakage demonstration depends on having working oracle-alpha results to show what tools miss.
- **Run Lane 7 (safety) last among the core lanes** — it depends on validated oracle-alpha, validated features, and validated statistical methodology.

### Remove or downgrade:

- **Lane 3 (softmax vs. unconstrained vs. top-k)**: Consider downgrading to a supplementary comparison within Lane 1 rather than a full separate lane. The softmax constraint is an implementation choice, not a standalone finding. Unless the comparison reveals surprising qualitative differences, this dilutes the narrative.

---

## 8. Recommended experiment ordering

1. **Lane 0**: Sanity-check reconstruction (1 hour, blocking)
2. **Lane 1**: Oracle-alpha against preregistered nulls (exploratory split)
3. **Lane 1b**: Cooperative-interaction test
4. **Baseline comparison**: BI / angular distance / tuned-lens vs. oracle-alpha
5. **Lane 3**: Softmax vs. unconstrained vs. top-k (abbreviated, within Lane 1)
6. **Lane 4**: Block-structure / ~8-cluster test
7. **Lane 2**: Figure 8 validation (after confirming which Figure 8)
8. **Lane 5**: Tool-breakage demonstration
9. **Lane 6**: Router training + w_l-analog geometry
10. **Lane 7**: Safety lane (three-stage protocol)
11. **Lane 1 confirmatory**: Rerun oracle-alpha on held-out prompts
12. **Lane 8**: Training-dynamics extension via Pythia/PolyPythias

---

## 9. Download list

**Papers to save locally (with arxiv IDs):**

- Attention Residuals: github.com/MoonshotAI/Attention-Residuals/blob/master/Attention_Residuals.pdf
- MUDDFormer: 2502.12170
- DeepCrossAttention: 2502.06785
- Hyper-Connections: 2409.19606
- mHC: 2512.24880
- DenseFormer: 2402.02622
- Curse of Depth: 2502.05795
- Mixture of Depths: 2404.02258
- Mixture of Recursions: 2507.10524
- ShortGPT: 2403.03853
- Unreasonable Ineffectiveness of Deeper Layers: 2403.17887
- Your Transformer is Secretly Linear: 2405.12250
- Transformer Layers as Painters: 2407.09298
- Pruning as a Cooperative Game: 2602.07804
- Refusal in LLMs (Arditi et al.): 2406.11717
- LLMs Encode Harmfulness and Refusal Separately: 2507.11878
- Safety Layers in Aligned LLMs: 2408.17003
- SafeConstellations: 2508.11290
- GemmaScope: 2408.05147
- Circuit Tracing (Anthropic): transformer-circuits.pub/2025/attribution-graphs/methods.html
- Biology of an LLM (Anthropic): transformer-circuits.pub/2025/attribution-graphs/biology.html
- Sparse Feature Circuits: 2403.19647
- Hypothesis Testing the Circuit Hypothesis: 2410.13032
- MI as Statistical Estimation: 2510.00845
- Is MI Identifiable?: (ICLR 2025, Méloux et al.)
- SAEs Don't Find Canonical Units: 2502.04878
- Tuned Lens: 2303.08112
- Backward Lens: (EMNLP 2024, Katz et al.)
- Logit Prisms: (Nguyen 2024)
- Transformers Don't Need LayerNorm at Inference: 2507.02559
- PolyPythias: 2503.09543
- SAE-Track: 2412.17626
- Birth of Knowledge: 2505.19440
- Embryology of a Language Model: 2508.00331
- Hidden Dynamics of Massive Activations: 2508.03616
- LLM Circuit Analyses Are Consistent: 2407.10827
- Understanding Refusal with SAEs (Yeo et al.): EMNLP Findings 2025
- Depth-Recurrent Attention Mixtures: 2601.21582
- RouteSAEs: aclanthology.org/2025.emnlp-main.346
- FlexiDepth (2025)
- Value Residual Learning: 2410.17897
- Scaling Monosemanticity: transformer-circuits.pub/2024/scaling-monosemanticity
- RASA: 2602.04448

**Repos to clone:**

- `git clone https://github.com/MoonshotAI/Attention-Residuals`
- `git clone https://github.com/Caiyun-AI/MUDDFormer`
- `git clone https://github.com/lucidrains/deep-cross-attention`
- `git clone https://github.com/epfml/DenseFormer`
- `git clone https://github.com/safety-research/circuit-tracer`
- `git clone https://github.com/TransformerLensOrg/TransformerLens` (pin v2.17.0)
- `git clone https://github.com/ndif-team/nnsight`
- `git clone https://github.com/Butanium/nnterp`
- `git clone https://github.com/decoderesearch/SAELens`
- `git clone https://github.com/AlignmentResearch/tuned-lens`
- `git clone https://github.com/andyrdt/refusal_direction`
- `git clone https://github.com/OpenMOSS/Language-Model-SAEs`
- `git clone https://github.com/shacharKZ/BackwardLens`

**Models/weights to download:**

- Pythia suite: EleutherAI/pythia-{70m,160m,410m,1b,1.4b,2.8b,6.9b} (select checkpoints)
- PolyPythias: EleutherAI/pythia-{70m,160m}-seed{1,2,3} (select checkpoints)
- GemmaScope SAEs: google/gemma-scope-2b-pt-{res,mlp,att}
- OLMo 2 early checkpoints: allenai/OLMo-2-0425-1B-early-training (optional, for larger-scale validation)

---

## 10. Open questions requiring human judgment

**1. Which Figure 8?** The "Figure 8 validation" lane references a specific figure, but it is unclear whether this is from the AttnRes paper, MUDDFormer, or another source. Someone needs to read the actual AttnRes PDF (released today) and MUDDFormer paper to identify which figure contains the patterns to be matched. This determines the entire experimental protocol for Lane 2. **Classification: blocking.**

**2. Model choice for primary experiments.** The project must decide between GPT-2 (well-understood, TransformerLens native), Pythia (checkpoints for training dynamics, TransformerLens native), and Gemma-2 (GemmaScope SAEs available, safety features better studied, but larger compute requirements). The choice cascades through tuned lens availability, SAE availability, and safety-lane feasibility. A reasonable default: Pythia-410M or Pythia-1B for Lanes 1–6 and 8, Gemma-2-2B for Lane 7 (safety, leveraging GemmaScope). **Classification: important, decide before coding.**

**3. Compute budget for oracle-alpha optimization.** Joint optimization of α across all layers and tokens is significantly more expensive than per-layer optimization. For a 24-layer model on 1000 prompts of 512 tokens, this is ~12M α parameters per optimization run. The team needs to decide whether to optimize per-prompt (expensive, maximally informative) or amortize across prompts (cheaper, loses input-dependence signal). **Classification: important.**

**4. How to handle the Curse of Depth confound.** The Curse of Depth predicts that oracle-alpha will assign low weight to deep layers in Pre-LN models purely due to architectural pathology, not meaningful routing. The project needs to decide how to separate "routing structure" from "deep-layer collapse." Options: (a) study a non-Pre-LN model where the curse doesn't apply, (b) compare Pre-LN vs. Peri-LN models and show differential routing, (c) explicitly model the curse and show oracle-alpha captures structure *beyond* the predicted decay. **Classification: important, affects the central claim.**

**5. Threshold for "matches patterns" in Figure 8 validation.** What quantitative criterion defines success? Rank correlation > 0.7? Visual similarity? The team needs to pre-register a specific statistical test and effect size threshold before running the validation. **Classification: important for credibility.**

**6. Whether to use circuit-tracer attribution graphs as a "ground truth" comparison.** Anthropic's circuit-tracer produces per-prompt computation graphs showing cross-layer information flow. These could serve as a fine-grained ground truth against which oracle-alpha's coarser routing weights are compared. However, circuit-tracer currently supports only Gemma-2-2b and Llama-3.2-1b, and requires pre-trained transcoders. The team should decide whether this comparison strengthens or overcomplicates the narrative. **Classification: optional but high-value if feasible.**

**7. Publication venue.** The work bridges architecture, interpretability, and safety — an unusual combination. Possible venues: ICML/NeurIPS main (if the oracle-alpha results are strong), the MI workshop track (if framed primarily as interpretability methodology), or a high-quality blog post on the Alignment Forum (if speed matters more than venue prestige). The framing should be chosen to match the target venue. **Classification: strategic, affects writing scope.**
