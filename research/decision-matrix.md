# Decision Matrix: Critical Variables for the Depth-Routing Interpretability Experiment

## How to Use This Document

Every section below represents a phase of the experimental plan. Within each phase, decisions are ranked by **blast radius** — how much downstream work gets invalidated if you get this wrong. Decisions marked 🔴 can void the entire experiment. Decisions marked 🟡 degrade quality but are recoverable. Decisions marked 🟢 are preferences that affect presentation but not validity.

---

## PHASE 0: Pre-Experiment Foundations

These decisions constrain everything downstream. Get them wrong and no amount of good execution saves the results.

### 🔴 DECISION 0.1: What exactly is the hypothesis?

The hypothesis must be falsifiable and specific. Vague framing like "depth routing reveals structure" is not testable.

**Proposed hypothesis (primary):** "Standard transformer residual streams contain latent, input-dependent depth-routing structure: for a given input, there exist non-uniform softmax-constrained layer weights α* that reduce next-token prediction loss compared to the uniform weights (α = 1/L) implied by standard residual connections."

**Proposed hypothesis (secondary):** "The optimal routing patterns α* are (a) systematically structured rather than random, (b) correlated with input characteristics (task type, difficulty, token position), and (c) qualitatively consistent with the patterns independently reported in trained AttnRes models (Figure 8: diagonal dominance, embedding persistence, layer-type specialization)."

**Why this matters:** If the primary hypothesis is false (uniform weights are already optimal), the entire project collapses. If the secondary hypothesis is false (optimal weights exist but show no interpretable structure), you still have a result but the interpretability angle weakens significantly.

**Validation gate:** Before proceeding past Phase 1, confirm on at least 100 sequences that oracle-α achieves measurably lower loss than uniform weights. Define "measurably" as: mean cross-entropy reduction > 0.01 nats with p < 0.01 via paired t-test.

### 🔴 DECISION 0.2: What is the null model?

Every experimental claim needs a comparison. You must define null models *before* running experiments, not after seeing results.

**Required null models:**
1. **Uniform weights** (α = 1/L for all layers) — the standard residual stream
2. **Random weights** (α sampled from Dirichlet distribution, averaged over K=100 draws) — controls for whether *any* reweighting helps vs. *structured* reweighting
3. **Magnitude-proportional weights** (α_i ∝ ‖v_i‖) — controls for whether structure comes from routing or just from letting loud layers be louder
4. **Last-layer-only** (α_L = 1, all others 0) — baseline: does the model even benefit from earlier layers?

**Why this matters:** If you only compare oracle-α to uniform, a reviewer will ask: "Maybe any random perturbation of weights improves loss due to regularization effects." The random baseline kills this objection. The magnitude-proportional baseline tests whether RMSNorm (which the AttnRes paper uses specifically to prevent magnitude-based dominance) is doing meaningful work.

### 🔴 DECISION 0.3: Primary model selection

The model you choose determines what tooling is available, what prior results exist for comparison, and what the community will trust.

**Recommendation: Gemma-2-2B as primary, GPT-2 XL as development/validation**

| Criterion | Gemma-2-2B | GPT-2 XL | Pythia-2.8B | Llama-3.1-8B |
|---|---|---|---|---|
| Layer count | 26 | 48 | 32 | 32 |
| SAE dictionaries | ✅ GemmaScope (all layers, all sublayers) | ✅ RES-JB (all layers) | ⚠️ Partial | ❌ None public |
| Circuit tracer support | ✅ | ❌ | ❌ | ❌ |
| TransformerLens support | ✅ | ✅ | ✅ | ✅ |
| Training checkpoints | ❌ | ❌ | ✅ 154 checkpoints | ❌ |
| Community trust/novelty | High (modern arch) | Medium (old but canonical) | High (interp community) | High (most used open model) |
| Memory for full cache (S=1024) | ~5.4 GB | ~4.5 GB | ~6 GB | ~16.5 GB |

**Risk if wrong:** Choosing a model without SAE support means you can't connect routing patterns to interpretable features — that's the paper's strongest contribution. Choosing a model the community considers trivial (e.g., GPT-2 Small) invites "but does this scale?" objections.

**Validation gate:** Before committing, run a 10-sequence smoke test confirming: (a) TransformerLens loads and caches activations without error on MPS, (b) full cache fits in memory at target sequence length, (c) the oracle-α optimization converges within 200 steps.

### 🟡 DECISION 0.4: Sequence length

Sequence length affects memory, compute time, and the richness of positional effects in routing patterns.

**Recommendation: S = 512 for development, S = 1024 for final results**

- S = 128: Too short. Many interesting phenomena (multi-hop reasoning, long-range dependencies) don't manifest.
- S = 512: Good balance. Captures most linguistic structure. Cache is small (~117 MB for Gemma-2-2B). Fast iteration.
- S = 1024: Final results. Captures longer dependencies. Cache is ~234 MB. Still very manageable.
- S = 2048+: Marginal benefit for routing analysis. Doubling compute for diminishing returns.

**Risk if wrong:** Too short means routing patterns are trivial (every token uses similar layers). Too long wastes compute on the iterative development cycles that dominate wall-clock time.

### 🟡 DECISION 0.5: Corpus selection and composition

The input data determines *which* routing patterns you'll discover. A corpus biased toward one domain will show domain-specific patterns that don't generalize.

**Recommendation: Stratified sample from OpenWebText with explicit category tags**

**Required strata (minimum 500 sequences each):**
- Factual recall: "The capital of France is ___" style completions
- Multi-hop reasoning: Questions requiring chained inference
- Code: Python/JS snippets
- Mathematics: Arithmetic and symbolic manipulation
- Creative text: Stories, poetry
- Conversational: Dialogue, chat-style text
- Multilingual: If model supports it (Gemma-2 does)

**Why stratification matters:** The paper's biggest benchmark gain is on GPQA-Diamond (+7.5) and Math (+3.6), suggesting depth routing matters most for multi-step reasoning. If your corpus is 90% web text, you'll see the average routing pattern but miss the task-specific variation that's the most publishable finding.

**Validation gate:** After generating oracle-α for each stratum, run a statistical test (e.g., MANOVA or permutation test) confirming that α* distributions differ significantly across strata. If they don't, either the stratification is wrong or routing isn't task-dependent (which is itself a finding).

**What NOT to do:** Do not use model-generated text (it biases toward the model's own distribution). Do not use a single benchmark (it shows one task, not generalization). Do not use deduplicated data without checking that common patterns survive (deduplication can remove exactly the repeated structures that routing patterns track).

---

## PHASE 1: Oracle-α Implementation (Weeks 1-2)

### 🔴 DECISION 1.1: How to compute the "output" from reweighted layer activations

This is the most subtle technical decision. The standard model computes:

```
output = Unembed(LayerNorm(h_L))
```

where h_L is the final residual stream state after all layers. When we reweight, we're replacing h_L with:

```
h_L_reweighted = Σᵢ αᵢ · vᵢ
```

**The critical question:** Do we apply the final LayerNorm to the reweighted sum, or do we also need to account for any transformations that the model applies after the last residual accumulation?

**Correct approach:** Apply the model's own final LayerNorm and unembedding. Specifically:
1. Cache all layer outputs v_i (post-layer, pre-next-layer-norm)
2. Compute h_reweighted = Σ αᵢ · vᵢ
3. Apply the model's final RMSNorm: h_normed = RMSNorm(h_reweighted)
4. Compute logits = W_U · h_normed

**Risk if wrong:** If you apply LayerNorm at the wrong point in the computation, the optimization will find α weights that exploit LayerNorm artifacts rather than genuine routing benefits. This would produce beautiful-looking but entirely spurious results.

**Validation gate:** For α = uniform (1/L for all i), verify that the reweighted logits exactly match the original model's logits divided by L (since the original model uses weight 1, not 1/L). If they don't match, there's a bug in how you're extracting or combining layer outputs.

### 🔴 DECISION 1.2: Which layer outputs to cache — residual stream positions matter

In a standard transformer block, there are multiple points where you could extract the "layer output":

```
x → LayerNorm → Attention → + (residual) → LayerNorm → MLP → + (residual)
          ↑                    ↑                              ↑
      "resid_pre"         "resid_mid"                   "resid_post"
```

**Options:**
- **resid_post only** (one vector per transformer block): Treats each Attn+MLP block as one unit. Simpler, fewer sources (L/2 if treating blocks as layers).
- **resid_mid + resid_post** (two vectors per block): Separates attention and MLP contributions. Matches the AttnRes paper, which treats "each self-attention or MLP as an individual layer."
- **Actual sublayer outputs** (attention output and MLP output separately, before residual addition): The purest decomposition. v_attn = Attn(LayerNorm(x)) and v_mlp = MLP(LayerNorm(x + v_attn)).

**Recommendation: Sublayer outputs (matching the paper's definition)**

The AttnRes paper explicitly says: "In Transformer models, we treat each self-attention or MLP as an individual layer." This means L = 2 × (number of transformer blocks). For Gemma-2-2B with 26 transformer blocks, L = 52 sublayers + 1 embedding = 53 sources.

**Risk if wrong:** If you use resid_post (one per block), you can't observe the layer-type specialization pattern from Figure 8 (pre-attention layers broad, pre-MLP layers local). This is one of your testable predictions — you need the granularity to test it.

**Validation gate:** Confirm that Σ vᵢ (summing all sublayer outputs + embedding) reconstructs the original final hidden state h_L exactly (within floating-point tolerance, < 1e-5 L2 error). If it doesn't, you have the wrong decomposition.

### 🔴 DECISION 1.3: Optimization algorithm and convergence criteria for α*

You're solving: minimize CrossEntropy(logits(softmax(z)), target) where z ∈ ℝ^L.

**Key decisions:**
- **Optimizer:** Adam with lr = 0.1, decaying to 0.01. The problem is small (L ≈ 50 variables) and smooth.
- **Number of steps:** 500 steps maximum, with early stopping if loss change < 1e-6 for 20 consecutive steps.
- **Initialization:** z = 0 (uniform softmax weights). This is the null hypothesis starting point.
- **Per-token vs. per-sequence:** Optimize α separately for each token position in the sequence, or share α across the sequence?

**Per-token vs. per-sequence is critical:**
- Per-token (one α vector per position): Maximum flexibility. Reveals position-dependent routing. But 512 × 50 = 25,600 total optimization variables per sequence. More expensive but richer.
- Per-sequence (one α vector shared across all positions): Simpler, faster. Shows input-level routing preferences. Loses position-dependent structure.

**Recommendation: Per-token optimization, then aggregate statistics**

Per-token gives you the richest data. You can always average across positions to get per-sequence patterns, but you can't go the other direction. The per-token patterns reveal whether routing changes at different positions in a sequence (e.g., early tokens might use shallow layers while the final prediction might recruit deep layers).

**Risk if wrong:** Per-sequence optimization averages over positions and may miss the most interesting signal: how routing shifts within a sequence as the model processes increasingly complex context.

**Validation gate:** Run the optimization on 10 sequences with both per-token and per-sequence settings. Compare: (a) does per-token achieve lower total loss? (b) do per-token α's show meaningful position-dependent variation, or are they essentially constant across positions? If constant, per-sequence is sufficient and saves 10x compute.

### 🟡 DECISION 1.4: How to handle the interaction between LayerNorm and reweighting

This is a known subtlety. In Pre-LN transformers, each sublayer normalizes its input:

```
v_i = Attn(LayerNorm(h_i))  where h_i depends on all previous v's
```

When you change α weights, you change h_i for every subsequent layer, which changes what those layers would have computed. Your cached v_i values are computed under the *original* uniform weights, not under the *reweighted* inputs.

**This means oracle-α is an approximation, not an exact optimum.**

**How to handle this honestly:**
1. State clearly in the paper: "Oracle-α optimizes over a linear combination of layer outputs computed under standard residual connections. The resulting weights represent the best *post-hoc* routing given fixed layer computations, not the routing that would emerge if layers were trained with routing-aware inputs."
2. This is actually a *conservative* estimate: AttnRes layers would co-adapt to the routing mechanism, likely producing even more differentiated outputs. Oracle-α is a lower bound on the benefit of depth routing.

**Risk if wrong:** If you claim oracle-α represents what AttnRes would learn, reviewers familiar with the co-adaptation argument will reject the paper. If you frame it correctly as a lower bound, it strengthens the argument: "Even without co-adaptation, depth routing helps. With co-adaptation (as in AttnRes), the benefit should be larger."

### 🟢 DECISION 1.5: Numerical precision

**Use float32 throughout.** MPS float16 is not faster on Apple Silicon (no Tensor Cores), and the softmax computation in the α optimization is numerically sensitive. The optimization involves exp() and log() operations where float16 overflow/underflow can produce NaN gradients silently.

---

## PHASE 2: Pattern Analysis (Weeks 3-4)

### 🔴 DECISION 2.1: Clustering methodology for α patterns

You have N sequences × T tokens, each with an L-dimensional α* vector. You need to find structure in this space.

**Critical choice: what distance metric to use**

- **Cosine similarity on α vectors:** Treats routing as a directional preference. Two tokens with α = [0.5, 0.5, 0, 0] and α = [0.25, 0.25, 0, 0] would look identical. This is appropriate if you care about *which* layers matter, not *how much* they matter.
- **Jensen-Shannon divergence:** α vectors are probability distributions (they sum to 1). JSD is the natural distance metric for distributions. It's symmetric, bounded [0, 1], and captures differences in both the shape and magnitude of routing preferences.
- **L2 distance on α vectors:** Simple but treats all differences equally. A shift from 0.50→0.48 in one layer counts the same as 0.02→0.00 in another, even though the latter is a qualitative change (turning a layer off).

**Recommendation: JSD as primary metric, cosine as sensitivity check**

**Clustering algorithm:** Hierarchical clustering with Ward linkage on JSD distances. This produces a dendrogram that you can cut at different levels, naturally revealing whether ~8 clusters emerge (matching the AttnRes paper's N≈8 blocks finding).

**Risk if wrong:** K-means with the wrong K will force structure where none exists. Always report the silhouette score across K=2..20 and show the elbow plot. If there's no clear elbow, state that routing patterns form a continuum rather than discrete clusters — that's a legitimate finding.

**Validation gate:** Run clustering on the random-baseline α's (Decision 0.2). Random weights should show NO meaningful clustering. If they do, your clustering methodology is finding noise.

### 🔴 DECISION 2.2: How to connect α patterns to SAE features

This is the bridge from "routing exists" to "routing is interpretable." You need to show that different routing patterns correspond to different *meaningful* model behaviors.

**Approach 1 (simpler, recommended first): Correlation analysis**
- For each SAE feature f, compute its activation a_f across your corpus
- For each source layer i, compute the mean oracle-α weight α_i across the same corpus
- Compute correlation between a_f and α_i vectors: which features activate when which layers get high routing weight?
- Use rank correlation (Spearman) to handle non-linearity

**Approach 2 (stronger, if Approach 1 shows signal): Conditional analysis**
- Split tokens into groups based on α cluster membership (from Decision 2.1)
- For each cluster, compute the average SAE feature activation profile
- Test: do clusters differ in which features are active? (Chi-squared test on binarized feature activations, or Mann-Whitney U on continuous activations)

**Risk if wrong:** If you correlate across the wrong axis (e.g., correlating α weights and SAE activations at different layers rather than the same token position), you'll find spurious patterns. Always compute correlations on *matched* token positions.

**Validation gate:** The top-correlated SAE features for each routing cluster should have interpretable descriptions on Neuronpedia. If the features correlated with "high weight on layer 5" are nonsensical (e.g., "starts with E but isn't elephant"), the connection isn't meaningful.

### 🟡 DECISION 2.3: Statistical testing framework

With 10,000+ sequences × 512+ tokens, you'll have millions of data points. Everything will be "statistically significant." You need effect sizes and practical significance, not just p-values.

**Required reporting:**
- Cohen's d for all comparisons (oracle vs. uniform, oracle vs. random, etc.)
- 95% bootstrap confidence intervals on all key metrics
- Bonferroni correction for multiple comparisons when testing across layers/features
- Report BOTH mean and median (routing patterns may be heavy-tailed)

**Risk if wrong:** Reporting p < 0.001 without effect sizes is a red flag for any competent reviewer. A loss improvement of 0.001 nats that's "statistically significant" at N=500,000 is meaningless if it doesn't correspond to a behavioral difference.

---

## PHASE 3: Figure 8 Predictions (Weeks 3-5)

### 🔴 DECISION 3.1: How to operationalize each Figure 8 prediction

Each prediction needs a specific, quantitative test, not a visual "it looks similar."

**Prediction 1: Diagonal dominance**
- Metric: For each target layer l, compute the fraction of α* weight on source layer l-1 (the immediate predecessor). Call this "locality score."
- Test: Is the locality score significantly > 1/L (the uniform baseline)? Compute across all tokens and report mean ± std.
- Null: If α* randomly distributed, expected locality score = 1/L.

**Prediction 2: Embedding persistence**
- Metric: α*_{0→l} (the weight on the embedding) as a function of target layer l.
- Test: Does the embedding receive above-uniform weight (> 1/L) at deep layers? Specifically at layers > L/2.
- Comparison: Plot α*_{0→l} vs. layer depth. If it stays above 1/L throughout depth, that matches Figure 8.

**Prediction 3: Layer-type specialization**
- Metric: For pre-attention layers (even sublayer index), compute the entropy of the α* distribution (high entropy = broad receptive field). For pre-MLP layers (odd sublayer index), compute the same.
- Test: Is entropy(α*_pre_attn) > entropy(α*_pre_mlp)? This would confirm that attention layers use broader depth routing while MLP layers focus locally.
- Statistical test: Paired t-test across matched layer pairs.

**Prediction 4: Learned skip connections**
- Metric: Identify off-diagonal peaks: for each target layer l, find source layers i where α*_{i→l} > 2/L (twice the uniform baseline) AND i ≠ l-1 (not the immediate predecessor).
- Test: Are these peaks consistent across inputs? Compute the fraction of tokens for which each (source, target) pair exceeds the 2/L threshold. Peaks that appear in >50% of tokens are "structural" (architecture-driven). Peaks that appear in 10-50% are "task-conditional."

**Risk if wrong:** Without precise operationalizations, you'll cherry-pick visual similarities between your heatmaps and Figure 8. Define the tests before looking at results.

---

## PHASE 4: Comparison Regimes (Week 4)

### 🔴 DECISION 4.1: How to implement the three routing regimes

**Unconstrained:** α ∈ ℝ^L, no normalization. Use sigmoid element-wise to keep weights in [0,1] but don't require sum = 1. This corresponds to independent per-layer gating.

**Softmax-constrained:** α = softmax(z), z ∈ ℝ^L. This is the AttnRes regime. Weights sum to 1, enforcing competition.

**Top-k sparse:** α = softmax(top_k(z, k)), where all but the k largest logits are set to -∞ before softmax. Test k ∈ {2, 4, 8, L/4, L/2}.

**What to measure for each regime:**
1. Loss improvement over uniform (primary metric)
2. Sparsity of resulting α: Gini coefficient and effective number of sources (exp(entropy(α)))
3. Pattern structure: does the regime recover the Figure 8 predictions?

**The key comparison:** If softmax > unconstrained in loss, that's the strongest possible evidence for the AttnRes paper's design. It means competition (zero-sum allocation of attention over depth) is better than independence (each layer can be turned up or down freely). This would explain why softmax beats sigmoid in the paper's Table 4 ablation.

**Risk if wrong:** If you use sigmoid for "unconstrained" but initialize differently than softmax, the comparison is confounded by initialization. Always initialize all regimes from the same starting point (z = 0, which gives uniform weights under softmax and 0.5 under sigmoid).

---

## PHASE 5: Tool Breakage Demonstration (Week 6)

### 🟡 DECISION 5.1: Which known circuit to use for the demonstration

You need a well-documented circuit where the ground truth is established. Options:

- **Factual recall ("The capital of France is ___"):** Anthropic documented this circuit in Claude 3.5 Haiku. Gemma-2-2B has similar circuits studied in GemmaScope.
- **Induction heads:** The most canonical circuit in mechanistic interpretability. Well-documented in GPT-2.
- **Greater-than (number comparison):** Studied in Hanna et al. (2023). Clean, well-specified.

**Recommendation: Factual recall on Gemma-2-2B**

It has the richest existing documentation via GemmaScope and circuit-tracer. It's directly relevant to benchmarks where AttnRes shows gains (TriviaQA +1.9).

### 🟡 DECISION 5.2: How to demonstrate logit lens non-monotonicity

**Procedure:**
1. Run standard logit lens on the original model for a factual recall prompt. Show the smooth, monotonic progression of the correct answer's probability across layers.
2. Reweight the residual stream at each layer using oracle-α weights. Recompute logit lens projections.
3. Show that the progression becomes non-monotonic: the correct answer may appear, disappear, and reappear as you scan through layers.

**The key visualization:** A line plot with layer depth on X-axis and P(correct_token) on Y-axis. Two lines: uniform (smooth, monotonically increasing) vs. oracle-α reweighted (non-monotonic, with dips). This is the single most visually compelling figure in the paper.

**Risk if wrong:** If oracle-α reweighting produces a line that's ALSO monotonic (just steeper), the "tools break" claim is weakened. Run this on 50+ prompts before committing to it as a main result.

---

## PHASE 6: Router Training (Week 7)

### 🔴 DECISION 6.1: Router architecture

The router takes some input and predicts α weights for each target layer. Critical choices:

**Input to the router:**
- Option A: First layer's output h_1 (the embedding after position encoding). Simple, but the router can only see the raw token — no contextual processing.
- Option B: A shallow (2-layer) MLP on h_1. Adds parameters but lets the router compute features of the input.
- Option C: The output of a specific early layer (e.g., layer 4). This gives the router access to some processed context.

**Recommendation: Option B (shallow MLP on h_1)**

This matches the AttnRes paper's design where w_l is a static query. The MLP learns to project the embedding into a space where dot products with layer outputs predict routing value. It's also the simplest design that allows input-dependent routing.

**Output structure:**
- One α vector for the final output layer only (simplest, tests whether routing helps at the end)
- One α vector per target layer (full AttnRes analog, much more parameters)

**Start with per-final-layer-only,** then expand if results are positive.

### 🟡 DECISION 6.2: Training objective

**Option A: Distillation from oracle-α** — Train the router to predict oracle-α* weights (MSE loss on the α vectors). Pros: Clean supervised signal, fast convergence. Cons: The router can't discover patterns that oracle-α missed due to the co-adaptation gap.

**Option B: End-to-end language modeling loss** — Freeze base model, train router to minimize cross-entropy on next-token prediction. The router learns routing that actually helps the frozen model. Pros: Learns routing for the right reason. Cons: More expensive, harder to optimize.

**Option C: Both** — Pre-train on oracle-α distillation (warm start), then fine-tune end-to-end.

**Recommendation: Option C.** The distillation phase gets you to a good initialization in hours. The end-to-end phase refines it for the actual objective. Compare the three to see how much end-to-end improves over pure distillation — that gap measures the co-adaptation effect.

### 🟡 DECISION 6.3: Training data and budget

**Use the same corpus as the oracle-α analysis** for the distillation phase. For end-to-end training, use a held-out subset of OpenWebText (at least 100M tokens) to avoid overfitting to the analysis prompts.

**Budget:** Start with 10M tokens for distillation (hours on MPS), 100M tokens for end-to-end (1-2 days on MPS). Monitor validation loss every 1000 steps. Stop if validation loss hasn't improved for 5 checkpoints.

---

## PHASE 7: Analysis and Write-Up (Week 8)

### 🟡 DECISION 7.1: What to claim vs. what to suggest

**Safe claims (directly supported by experiments):**
- Oracle-α achieves lower loss than uniform weights (quantify the improvement)
- Oracle-α patterns show specific, quantifiable structure (list which Figure 8 predictions were confirmed)
- Routing patterns differ across task categories (report JSD between strata)
- SAE features correlate with routing clusters (report top features per cluster)

**Reasonable suggestions (supported but not proven):**
- AttnRes would likely recover and enhance these patterns through co-adaptation
- Existing interpretability tools would need modification for AttnRes models
- The α weights represent a new, lower-dimensional interpretability surface

**Overclaims to avoid:**
- "AttnRes makes models more interpretable" (you haven't tested a trained AttnRes model)
- "This proves depth routing is necessary" (oracle-α is a post-hoc optimization, not a training signal)
- "X layers are redundant" (oracle-α with low weight ≠ redundancy; the layer's computation may enable other layers)

### 🟡 DECISION 7.2: Visualization standards

- All heatmaps must use a colorblind-friendly palette (viridis, not red-green)
- All statistical claims must include error bars or confidence intervals
- All comparisons must include the null baselines (uniform, random)
- All per-layer plots must annotate attention vs. MLP sublayers

### 🟢 DECISION 7.3: Publication target and framing

- **LessWrong/Alignment Forum (primary):** Fastest impact. Framing: "What depth routing reveals about transformer computation." Include interactive visualizations.
- **arXiv preprint:** For academic credit. Same content, formatted as a standard ML paper.
- **NeurIPS 2026 MI Workshop:** 4-page paper, deadline likely July 2026. Focus on the tool-breakage demonstration and the oracle-α methodology.
- **ICML 2027 full paper:** If the router training produces strong results and you later get access to a trained AttnRes model.

---

## CROSS-CUTTING DECISIONS

### 🔴 Reproducibility requirements

- Fix all random seeds (PyTorch, NumPy, Python hash seed)
- Log all hyperparameters to W&B before running experiments
- Version-lock all dependencies in a requirements.txt
- Pre-register the hypothesis and analysis plan (post to LessWrong as a "pre-registration" before running experiments — this dramatically increases community trust)

### 🔴 When to stop and reassess

Define these BEFORE starting:

1. **If oracle-α shows < 0.5% loss improvement over uniform on 1000+ sequences:** The depth routing signal is too weak to support the thesis. Pivot to a different framing (e.g., "standard residual connections are already near-optimal for depth aggregation, which explains why DenseFormer showed no gain").

2. **If α* patterns show no clustering structure (silhouette score < 0.2 across all K):** Routing is input-specific rather than task-categorical. This is still publishable but requires different analysis (per-token rather than per-cluster).

3. **If Figure 8 predictions all fail:** Oracle-α finds structure that is qualitatively different from trained AttnRes. This is a VERY interesting negative result — it means AttnRes's patterns are training artifacts, not inherent task properties. Write it up as such.

4. **If MPS produces numerical instability (NaN gradients in > 5% of optimizations):** Fall back to CPU. Slower but numerically reliable. Do not continue with MPS if it's producing unreliable results.

### 🟡 Order of operations for maximum information gain

Run experiments in this order because each gates the next:

1. Smoke test: 10 sequences, oracle-α on GPT-2 XL (2 hours). **Gate: does it work at all?**
2. Small-scale: 100 sequences, Gemma-2-2B, all null models (4 hours). **Gate: is the effect real?**
3. Scale up: 5,000 sequences with stratification. **Gate: is there structure?**
4. SAE integration: Connect routing clusters to GemmaScope features. **Gate: is it interpretable?**
5. Figure 8 tests: Run quantitative predictions. **Gate: does it match AttnRes?**
6. Tool breakage: Logit lens demonstration. **Gate: does the story cohere?**
7. Router training: Only if all above gates pass.
8. Pythia dynamics: Only if you have time and the main story is solid.

Each gate takes 2-6 hours. You'll know within the first week whether the project has legs, long before committing to the full 8-week plan.
