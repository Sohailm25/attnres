# High‑Rigor Research Audit on Latent Depth Routing and Oracle‑Alpha Interpretability

## Executive summary

The research landscape shifted materially in 2024–2026 in precisely the directions your project is probing: depth-wise information mixing is now an explicit design axis in mainstream architecture work (e.g., depth-attention over layer outputs), and interpretability tooling has moved beyond “logit lens + activation patching” toward feature-based circuit tracing, transcoders, and benchmarked causal localization. citeturn22search1turn2search2turn4search3turn18search0turn17view0turn7search13

Two developments are especially “project-shaping” as of March 16, 2026. First, entity["company","Moonshot AI","beijing, CN"] publicly released **Attention Residuals (AttnRes)** (March 2026), framing residual accumulation itself as a depth-attention mechanism, with a practical “~8 blocks” variant (Block AttnRes). citeturn22search1 Second, entity["company","OpenAI","ai lab"] released **Weight‑sparse transformers have interpretable circuits** (Nov 2025) and open-sourced associated tooling (circuit_sparsity) that explicitly discusses (i) why common causal methods can miss mechanism components, and (ii) why stronger faithfulness criteria (e.g., variants of causal scrubbing) are needed for safety-relevant claims. citeturn26search19turn26search1turn21view1turn21view2turn27search0

Your current plan is directionally aligned with where the field is going, but several lanes (as currently stated) will be criticized unless you (a) adopt the now-standard evaluation mindset (MIB-style causal localization, careful faithfulness), and (b) harden “oracle‑alpha” against identifiability/overfitting critiques. citeturn17view0turn21view1turn10search3turn10search12

The strongest honest reframing, given 2025–2026 context, is not “standard transformers secretly implement AttnRes,” but rather: **standard transformers contain a task- and token-dependent *effective* depth-mixture that is recoverable, structured, and partly invisible to common layerwise tools; making routing explicit (AttnRes / Hyper‑Connections / DCA / MoD) turns this into a control surface that can be benchmarked and instrumented.** This remains publishable if you (i) demonstrate out-of-sample predictiveness and stability of your recovered routing, (ii) show tool failure under controlled dynamic-routing counterfactuals, and (iii) precommit to strong faithfulness validation rather than interpretation-by-visualization. citeturn22search1turn2search2turn4search3turn18search0turn21view2turn17view0

## Landscape update as of March 16, 2026

### Depth routing and residual mixing architectures

**Attention Residuals (AttnRes, March 2026)**: AttnRes replaces uniform residual accumulation with depth-wise attention over prior layer outputs, using a learned per-layer pseudo-query to compute mixing weights; a “Block AttnRes” partitions layers into ~8 blocks to reduce memory from O(Ld) to O(Nd) while purportedly recovering much of full AttnRes’s gains. citeturn22search1 This is the closest “adjacent architecture” to your thesis that separating routing from computation exposes a new interpretability surface (the routing weights).

**Hyper-Connections (ICLR 2025)** and related “skip”/multi-branch residual designs formalize and systematize richer residual pathways; the core point for your project is that “residual wiring” is no longer assumed to be a fixed sum—researchers treat it as an object worth learning/regularizing. citeturn2search2

**DenseFormer (2024)** explicitly studies dynamic weighted averaging (DWA) across layers and reports efficiency/training benefits, strengthening the broader premise that depth-wise mixing weights can be meaningful signals rather than pure optimization hacks. citeturn2search0

**DeepCrossAttention (2025)** generalizes residual learning with input‑dependent weights that dynamically combine layer outputs and adds “depth-wise cross-attention” to support cross-layer interactions. citeturn4search3

**Frac‑Connections (2025)** adds learnable scaling parameters on residual branches (scalars rather than full routers), representing a “minimal routing knob” that can still create structured depth utilization. citeturn2search12

**Layer redundancy / underutilized depth** became a major empirical theme: **ShortGPT (Mar 2024; Findings ACL 2025)** argues many LLM layers are redundant and can be removed with limited loss; **The Curse of Depth (arXiv Feb 2025; NeurIPS 2025)** attributes deep-layer ineffectiveness to Pre‑LayerNorm variance growth and proposes LayerNorm Scaling to restore deep-layer training contribution. citeturn4search1turn4search6turn4search2 These lines can cut both ways for you: they support “depth structure exists,” but also provide a competing explanation (optimization pathology rather than latent routing).

**Adaptive token-level computation** (not your core focus, but highly relevant to “tool breakage under dynamic routing”): **Mixture‑of‑Depths (Apr 2024)** enforces a total compute budget by selecting only k tokens to participate in attention/MLP at a layer, creating inherently dynamic token‑layer participation. citeturn18search0 **Mixture‑of‑Recursions (2025; NeurIPS 2025 poster)** adds routers that assign per-token recursion depths, explicitly tying training to inference-time routing behavior. citeturn18search11turn18search5 These are strong “external validity” motivations for your tool-breakage lane: real architectures already rely on routing/halting dynamics, and interpretability tooling is lagging.

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["Attention Residuals depth attention diagram","Hyper-Connections transformer residual pathways figure","DeepCrossAttention depth-wise cross attention diagram","Mixture-of-Depths transformer routing diagram"],"num_per_query":1}

### Interpretability tools and their 2024–2026 evolution

Tooling bifurcated into (i) intervention/caching frameworks, and (ii) “feature/circuit infrastructure” that tries to operationalize causal mechanisms.

On the framework side, **TransformerLens** remains the de facto exploratory mech‑interp toolkit for open transformer models, emphasizing activation caching and easy interventions. citeturn5search0turn5search4turn5search28 Its recent release track (v3 beta March 2026) suggests an actively evolving API surface—this matters because it increases the odds your project’s infra should abstract away from any single tool’s internal naming/layout. citeturn5search24

**NNsight** has increasingly become the “model-wrapper + intervention language” for larger and more heterogeneous stacks (including remote execution via NDIF), and as of Feb 26, 2026 highlights ecosystem integrations and updated guidance for agents and workflows. citeturn5search1turn5search5turn5search9turn5search36

On feature/circuit infrastructure, **SAELens** is actively maintained (example: releases in March 2026), positioning itself as a standard library for training/analyzing sparse autoencoders. citeturn5search2turn5search6 Meanwhile, the field’s best “worked examples” for *circuits* have pivoted toward transcoders and (cross-layer) attribution graphs, pushed heavily by entity["organization","Anthropic","ai safety lab"] and the Decode/Neuronpedia ecosystem. citeturn7search13turn6search3turn6search2turn19search3

**Tuned Lens (2023)** is now broadly considered the correct baseline for “decode intermediate layers into vocab space” because it is explicitly designed to reduce the brittleness of raw logit lens by training per-layer affine probes on frozen models. citeturn6search0 **Backward Lens (2024)** extends “lens” thinking to gradients, relevant if you consider training-dynamics or fine-tuning effects. citeturn6search1turn6search5

### Mechanistic interpretability methods and benchmarks

The most important evaluation shift is the rise of **MIB: A Mechanistic Interpretability Benchmark (ICML 2025)**, which explicitly attempts lasting standards for comparing causal localization methods (circuit localization and causal variable localization). citeturn17view0turn23search2 The headline result you must treat as “reviewer ammunition” is that, under MIB’s causal variable localization track, supervised DAS performed best while SAE features were *not better than neurons* in that setting. citeturn17view0turn19search18 Even if you disagree with MIB’s task framing, reviewers will expect you to engage with it or justify why your evaluation is stronger/different.

Separately, **circuit tracing / attribution graphs (2025)** provide a strong methodological account of how to build “replacement models” (e.g., via cross-layer transcoders) and then do graph-based mechanistic analysis with causal interventions. citeturn7search13turn7search7 These papers/posts also emphasize that sparse coding (SAEs/transcoders) can be viewed as providing global weights between features that are prompt-independent in principle, a concept adjacent to your “routing surface” story. citeturn19search0

A parallel 2025–2026 trend is interpretability-by-design: **Weight‑sparse transformers have interpretable circuits (arXiv Nov 17, 2025)** proposes training with extreme weight sparsity to produce much simpler low-level circuits, with explicit discussion of faithfulness and limits. citeturn26search1turn21view0turn21view1turn21view2 This provides both (i) a major competing “interpretability surface,” and (ii) concrete critiques of patching-only approaches you can leverage to strengthen your tool-breakage lane. citeturn21view2

### Training dynamics resources beyond Pythia

**Pythia (paper 2023)** remains a flagship training-dynamics suite with many checkpoints per model and a major advantage: controlled training order across scales and dedup variants. citeturn9search16turn9search0turn9search20

But the “best choice” landscape changed: entity["organization","Allen Institute for AI","seattle, WA, US"] has released **OLMo 2 (2024)** and **OLMo 3 (Nov 2025)** as “fully open” flows including training code/data/logs and many intermediate checkpoints, explicitly encouraging mid-training intervention and analysis. citeturn9search17turn9search1turn9search4 For your project, OLMo’s openness is a major opportunity: with a modern architecture and open artifacts, you can examine whether your oracle‑alpha phenomena track known training events (SFT, RL, curriculum changes) rather than only pretraining-scale Pythia.

A major caution: **EvoLM (NeurIPS 2025 oral)** argues intermediate checkpoints from a longer schedule are not faithful proxies for fully trained smaller runs, warning against naïvely interpreting snapshots as “models trained for fewer tokens.” citeturn9search2turn9search6 This directly pressures your “Pythia checkpoints training-dynamics extension” lane: you need to state what kinds of training-dynamics claims you can and cannot support from checkpoint series.

Other notable open checkpoint suites include **YuLan‑Mini (ACL 2025)** (explicitly mentioning training-dynamics research using intermediate checkpoints) and several “reference model” releases in 2025. citeturn9search9turn9search15turn9search32

## What in your current plan still looks strong

Your plan has several structurally correct instincts that match where top mech‑interp review standards have moved.

**Oracle‑alpha against preregistered nulls** is the right kind of epistemic posture if your aim is “existence + structure” rather than “interpretability storytelling.” The broader scientific community’s replication concerns (p‑hacking, garden-of-forking-paths) make preregistered nulls, pilot/confirmatory splits, and sequence-level aggregation particularly defensible. citeturn10search2turn10search3turn10search12

**Softmax vs unconstrained vs top‑k comparison** is genuinely high-signal for the core thesis (“routing vs computation”), because it probes whether your recovered structure looks like a probabilistic router (softmax), a linear decomposition artifact (unconstrained), or a sparse selection mechanism (top‑k)—which directly interfaces with how many conditional-computation architectures constrain routers. citeturn22search1turn18search0turn2search2

**Block-structure / ~8-cluster test** is well motivated by the recent AttnRes “~8 blocks” claim: if a frozen standard transformer truly contains latent stage-like depth structure, it is plausible you’d recover a small number of stable “depth regimes.” citeturn22search1turn4search2

**Tool-breakage demonstration** is increasingly publishable because leading work now explicitly acknowledges that common intervention methods can miss mechanism components depending on how variability is distributed across prompts/tokens; you can ground your critique in recent primary sources rather than opinion. citeturn21view2turn7search13turn6search0

**Router training + w_l‑analog geometry** remains a plausible bridge from post-hoc oracle weights to learnable routers, and aligns with the broader shift toward “routing networks” in efficient computation (MoD/MoR). citeturn18search0turn18search11turn22search1

**Safety lane focused on refusal/honesty-related routing differences** is aligned with a large and growing body of mechanistic safety results. Work on refusal as a latent subspace and on explicit refusal circuits provides targets for “feature discovery/validation” rather than vague claims. citeturn7search0turn7search3turn7search7

## What looks weak, risky, outdated, or likely to be criticized

### The novelty/positioning risk is now higher

Since mid‑2024, multiple architecture papers have argued that learnable depth mixing is important and have introduced explicit mechanisms to do it (DenseFormer, Hyper‑Connections, DeepCrossAttention, AttnRes). citeturn2search0turn2search2turn4search3turn22search1 A reviewer can reasonably ask: *if these architectures explicitly learn depth-routing weights, why does recovering post-hoc “oracle alphas” from standard transformers constitute a new contribution rather than an analysis artifact?* You will need a crisp answer that is not “we saw patterns.”

### Oracle‑alpha is vulnerable to identifiability and “just linear algebra” critiques

Any method that fits mixture weights to reconstruct a downstream quantity in an overparameterized setting can appear to discover structure even when the solution is non-unique; this becomes especially plausible in transformer residual streams where sums and normalizations create many linear dependencies. Your own constraint list already anticipates some of this (e.g., normalization factor sharing), but reviewers will still demand that (a) your oracle‑alpha solution is stable under perturbations and (b) it generalizes out-of-sample. citeturn25search0turn25search2turn10search3

### “Figure 8 validation” is currently the most fragile lane

As of March 2026, AttnRes weights are not evidently released as open checkpoints; the public release is primarily code/paper materials. citeturn22search1turn12view0 If your Figure 8 claim depends on comparing recovered patterns to those “independently discovered in trained AttnRes models,” you risk a *non-reproducible validation* narrative (“trust us, it matches fig. 8”) unless you can (i) reproduce AttnRes training at a small scale, or (ii) validate against another open dynamic-mixing model family with released weights/checkpoints. The field is moving toward execution-grounded and benchmarked evaluation norms, which increases penalties for non-reproducible figure matching. citeturn17view0turn23academia43

### Tool-breakage must be grounded in modern standards, not “logit lens is bad”

You already flagged that “raw logit lens alone” is insufficient and tuned-lens-aware comparison is required. This is now close to mandatory because tuned lens is explicitly designed to avoid brittleness of the logit lens. citeturn6search0

More importantly, if you claim “current tools fail under dynamic routing,” reviewers will ask: *which tools, under what assumptions, and by what faithfulness criterion?* Recent work explicitly states that mean ablation is not a perfect faithfulness criterion and points toward causal scrubbing-style standards; it also argues activation patching can miss critical circuit components if relevant variability appears across prompt pairs rather than within them. citeturn21view1turn21view2turn7search13 Your breakage story will be attacked unless you operationalize “failure” with a metric and a controlled construction.

### Safety lane: you must avoid “feature confirmation bias”

The refusal literature provides both targets and traps. The “single refusal direction” result is strong, but the paper itself frames its extraction as heuristic and not necessarily optimal (and it explicitly demonstrates that mechanistic insights can be weaponized into jailbreaks). citeturn7search4turn7search0 If your safety lane is not careful, you can end up in a position where (a) you replicate a known vulnerability and (b) your new contribution is unclear, while also raising publication-risk questions. A safety reviewer may prefer you focus on understanding *why* routing differs rather than how to disable refusal.

### Statistical methodology will be scrutinized more than in 2021–2023 mech‑interp papers

As mech‑interp increasingly intersects with safety policy and high-stakes claims, the “latent structure exists” claim will be judged as an empirical discovery requiring multiple comparisons control, clear aggregation units, and an exploratory/confirmatory separation. These concerns are mainstream in other sciences (p‑hacking and forking paths), and explicitly endorsed by preregistration guidance emphasizing data splits as a way to enable exploration without contaminating confirmatory claims. citeturn10search2turn10search3turn10search12

## Best updated methodology recommendations

Below are the highest-signal updates I would make before implementation. Each item includes why it matters, a primary source, and a “must/should/optional” label.

### Make MIB a first-class evaluation backbone

**Recommendation (must-fix):** Integrate MIB tasks/metrics as part of your core evaluation suite (at least as a sanity anchor), even if your main claims extend beyond MIB’s scope. citeturn17view0turn23search2

**Why it matters:** MIB is explicitly motivated as a lasting standard for comparing causal localization, with two tracks and multi-task/multi-model coverage; ignoring it invites the critique “your evaluation is bespoke, likely overfit, and incomparable.” citeturn17view0turn23search22

**Implementation caveat:** Treat MIB’s result that SAEs were not better than neurons (in their causal variable localization track) as an empirical warning: if your project relies heavily on SAE features to validate routing claims, you need to justify why your setting differs (e.g., different causal variables, different intervention protocol). citeturn17view0turn19search18

### Define oracle‑alpha as a *predictive* object, not merely a reconstruction fit

**Recommendation (must-fix):** Precommit to at least one out-of-sample predictiveness test: e.g., learn a mapping from internal states (or prompt descriptors) → recovered oracle‑alpha, and evaluate whether predicted alphas (without access to the target) still reproduce the key routing phenomena on held-out prompts/sequences. citeturn10search12turn10search0

**Why it matters:** Without a predictiveness/stability story, oracle‑alpha can be dismissed as a non-identifiable decomposition (many solutions fit); predictiveness forces your α structure to “carry information” about the model’s computation rather than being a post-hoc artifact. This is the same philosophical motivation behind tuned lens as a learned, stable decoder vs brittle logit lens. citeturn6search0turn10search3

### Adopt a “faithfulness ladder” consistent with 2025 circuit work

**Recommendation (must-fix):** For any claim that a routed decomposition reveals mechanism, require at least one validation stronger than “logit attribution plots,” drawn from modern circuit standards (e.g., replacement-model interventions, scrubbing-style interchangeability constraints, or rigorous ablation protocols). citeturn7search13turn21view2turn26search19

**Why it matters:** Recent primary sources emphasize that explanations can look plausible but be unfaithful, and that mean ablation / patching choices can drive what you “find.” The weight‑sparse circuits paper explicitly states that mean ablation is not a perfect faithfulness measure and that some variant of causal scrubbing is needed for full confidence; it also points out activation patching can miss parts of circuits depending on how variability is distributed. citeturn21view1turn21view2

**Practical compromise:** You can define tiers:
- **Tier 0:** reconstruction accuracy only (exploratory).
- **Tier 1:** causal intervention that changes outputs in predicted ways (confirmatory).
- **Tier 2:** interchangeability / scrubbing-style constraints (strong confirmatory). citeturn21view2turn7search13

### Fix clustering methodology and pre-register distance/linkage choices

**Recommendation (must-fix):** If you do hierarchical clustering, do *not* use Ward with Jensen–Shannon distances unless you can guarantee Euclidean structure; instead use average/complete linkage for non-Euclidean distances, or embed to Euclidean and justify. citeturn24search4turn24search2turn24search0

**Why it matters:** This is an easy reviewer hit: SciPy and scikit-learn documentation explicitly warn Ward is defined correctly only for Euclidean distances and restrict Ward to Euclidean metrics. citeturn24search4turn24search2

**Recommendation (should-fix):** If you keep Jensen–Shannon, prefer **Jensen–Shannon distance** (square root of JSD) when you need a metric, and document the choice. citeturn24search3turn24search22

### Make normalization-aware decomposition an explicit, tested module

**Recommendation (must-fix):** Implement a “normalization-correct routed-logit decomposition” module with tests that enforce shared normalization factors where appropriate, and fail loudly if a user accidentally normalizes per-component. citeturn25search0turn25search1

**Why it matters:** In both LayerNorm and RMSNorm, the normalization statistic is computed from the *full* vector being normalized; this implies that any per-source “logit contribution” decomposition that applies normalization separately to each source will generally be wrong. PyTorch’s LayerNorm and RMSNorm docs make clear normalization depends on aggregated statistics. citeturn25search0turn25search1

### Update safety lane to use known mechanistic anchors and to avoid accidental “jailbreak contribution”

**Recommendation (must-fix):** Anchor refusal/honesty routing differences to at least one established mechanistic handle: (i) “refusal is mediated by a single direction” results, and/or (ii) circuit-tracing refusal mechanisms, and then test whether depth-routing patterns correlate with those features under causal intervention. citeturn7search0turn7search3turn7search7

**Why it matters:** Safety reviewers will reject “routing differs on refusal prompts” if it could be explained by generic difficulty/uncertainty or style shifts. The refusal-direction literature demonstrates that a small subspace can causally mediate refusal, and Anthropic’s circuit tracing suggests refusal circuits can be “default on” with competing inhibiting features; these provide concrete candidate mediators. citeturn7search0turn7search3

**Recommendation (should-fix):** Use GemmaScope resources (SAEs + transcoders) only after explicitly validating feature sensitivity/robustness on your distributions, because interpretability of features does not guarantee sensitivity or causal reliability. citeturn5search11turn8search4turn21view1

### Strengthen training-dynamics lane by expanding beyond Pythia and stating limits

**Recommendation (should-fix):** Add at least one “fully open modern” checkpoint suite (OLMo 2/3) alongside Pythia for training-dynamics extensions, and explicitly separate:
- claims about **within-run evolution** (safe),
- vs claims about **counterfactual training schedules** (not supported by checkpoints alone). citeturn9search21turn9search1turn9search2

**Why it matters:** OLMo explicitly releases data/code/checkpoints and motivates intervention at various points; EvoLM explicitly warns against treating intermediate checkpoints as substitutes for full smaller runs. citeturn9search1turn9search2turn9search6

### Reconsider “Figure 8 validation” sequencing

**Recommendation (should-fix):** Move “Figure 8 validation” later, and gate it on passing a “reproducible proxy” milestone: either (a) train small AttnRes models yourself, or (b) validate against an alternative open depth-mixing model with available weights. citeturn22search1turn9search1turn9search0

**Why it matters:** Without open AttnRes weights, direct comparison to trained AttnRes patterns risks being uncheckable; in 2025–2026, interpretability papers increasingly face demands for reproducible artifacts and benchmark grounding. citeturn17view0turn23academia43

## Best updated framing for a serious blog post or paper

A framing that is strong, honest, and resilient to 2026-era reviewer skepticism should (i) clearly delimit what your method can claim, (ii) connect to the now-mainstream “residual wiring is learnable,” and (iii) show value *even if* the latent-routing thesis is partially wrong.

### Proposed framing

**Core claim (confirmed by your experiments if executed well):**
Standard Pre‑LN transformers implement a token- and context-dependent *effective depth mixture* of intermediate computations that can be recovered by oracle‑alpha analysis. This depth-mixture is structured (e.g., low-rank / clustered regimes) and partly invisible to common layerwise interpretability views. citeturn6search0turn4search2turn22search1turn17view0

**Bridge claim (reasoned inference; must be tested):**
This effective depth-mixture resembles the kind of explicit depth routing learned in architectures that expose routing weights (AttnRes, DeepCrossAttention, Hyper‑Connections), suggesting that “separating routing from computation” is a promising *instrumentation surface* even when starting from standard models. citeturn22search1turn4search3turn2search2

**Negative result / critique claim (confirmed if you design it):**
Common interpretability methods that assume fixed layer contributions can fail under dynamic routing regimes—consistent with recent findings that patching/ablation choices can miss mechanism components and that stronger faithfulness tests are required. citeturn21view2turn7search13turn6search0

**Safety-relevance claim (must be scoped):**
Routing surfaces can help localize and compare safety-relevant behaviors (e.g., refusal vs non-refusal) when tied to explicit mechanistic anchors (refusal direction / refusal circuits), but are not automatically “safety guarantees.” citeturn7search0turn7search3turn21view1turn9search8

### What to avoid saying (likely to be attacked)

- “Standard transformers already contain AttnRes routers” (too strong; AttnRes is trained co-adaptively and explicitly changes the computation graph). citeturn22search1
- “Our decomposition is the mechanism” unless validated with scrubbing-like constraints or strong causal tests. citeturn21view1turn21view2
- “SAEs reveal refusal/honesty features” without sensitivity/causal evaluation (feature interpretability ≠ feature reliability). citeturn8search4turn17view0

## Concrete repo changes to make before implementation

The goal here is to force methodological discipline at the repository level so you cannot accidentally drift into exploratory storytelling.

### Experiment registry, splits, and prereg support

**Must-fix:** Add a versioned experiment registry (YAML/JSON) that defines:
- datasets/prompts,
- pilot vs confirmatory split,
- oracle‑alpha objective family (softmax / unconstrained / top‑k),
- aggregation unit (sequence-level by default),
- and all statistical tests / nulls. citeturn10search12turn10search3turn10search2

**Must-fix:** Enforce pilot/confirm split at the code level (e.g., “confirm set can only be accessed by a separate script that refuses to run if exploratory flags are enabled”). Relying on social discipline will fail under iteration pressure. citeturn10search12turn10search0

### Normalization-correct decomposition utilities

**Must-fix:** Implement a single source of truth for:
- extracting sublayer outputs (attention, MLP, resid_pre/post variants),
- recomposing routed residual streams,
- and applying the *shared* final normalization and unembedding.
Add unit tests that reconstruct the original model’s logits within tolerance under “uniform routing” and that fail if someone divides by L or normalizes per-source. citeturn25search0turn25search1turn22search1

### Tooling adapters

**Should-fix:** Create an abstraction layer so the same experiment can run via TransformerLens or NNsight backends. This reduces dependency risk given rapid tool evolution, and enables scaling from small (TL) to larger models (NNsight/NDIF). citeturn5search0turn5search5turn5search9

### Benchmark harness integration

**Should-fix:** Add MIB dataset/task loaders and baseline runners “in repo,” pinned to specific commits/versions, with cached artifacts. Include a script that reproduces at least one MIB baseline result as a sanity test. citeturn23search2turn17view0

### Safety lane hygiene

**Must-fix:** Add a “safety evaluation policy” doc in-repo that forbids:
- releasing prompt sets that directly enable bypassing safeguards,
- publishing operational jailbreak procedures,
- and mixing exploratory prompt crafting with confirmatory claims.
This aligns with the refusal-direction literature’s demonstrated dual-use risk. citeturn7search4turn7search0

### Reproducibility & compute ergonomics

**Should-fix:** Add deterministic caching for activations and routing outputs with content-addressed keys (model hash, tokenizer hash, prompt hash, code version). This is essential if you want sequence-level inference and multiple nulls without rerunning expensive forward passes. citeturn5search4turn5search5

## Concrete experiments to add, remove, or reorder

### Reorder: gate high-claim lanes behind “reconstruction + stability” milestones

**Must-fix reordering:** Put these first, before any “interpretability narrative”:
1) **Reconstruction correctness suite** (normalization-aware; uniform routing recovers logits; routed-logit decomposition correctness). citeturn25search0turn22search1
2) **Identifiability/stability suite** (seed stability, small perturbations, prompt paraphrases, and whether α clusters persist). This directly addresses the “oracle linear algebra” critique. citeturn10search3turn10search2
3) **Out-of-sample predictiveness** (learn to predict oracle α from internal states; evaluate on confirm set). citeturn10search12turn6search0

Only then run the “Figure 8 / block structure / safety” lanes.

### Add: benchmarked causal-localization baselines as controls

**Should-fix:** Add an explicit baseline where you test whether your oracle‑alpha clusters correspond to known circuit-localization signals on a controlled task (IOI, arithmetic), aligning with MIB’s circuit track. citeturn17view0turn23search7

### Add: a controlled dynamic-routing “counterfactual model” for tool-breakage

**Must-fix:** Build a minimal transformer wrapper that introduces synthetic dynamic depth mixing (e.g., a known router that flips between two depth-mixture patterns based on a trivial observable) so you can demonstrate precisely where standard tools fail and how tuned lens changes the picture. Recent work explicitly notes that patching can miss parts of circuits when variability is distributed across prompt pairs; you can turn that into a controlled, falsifiable demonstration. citeturn21view2turn6search0turn7search13

### Add or replace: “Figure 8 validation” with a reproducible proxy

**Should-fix:** Replace “Figure 8 validation” (as a core claim-bearing lane) with one of:
- **Small-scale AttnRes reproduction**: train tiny AttnRes and a standard baseline on the same data, then compare learned routing vs recovered oracle routing at matched scales. (Even if small, it addresses the “co-adaptation” critique.) citeturn22search1
- **Open depth-mixing baseline**: use a publicly released depth-mixing architecture (e.g., DeepCrossAttention if weights exist; otherwise train small). citeturn4search3

### Add: training-dynamics cross-check beyond Pythia

**Optional improvement:** Add an OLMo checkpoint sequence experiment where you test whether routing structure changes qualitatively across known stage transitions (pretrain → midtrain → long-context), leveraging explicit checkpoint naming conventions. citeturn9search4turn9search1

### Safety lane: tighten scope to “mechanistic correlation + causal test,” not “new jailbreak”

**Must-fix:** Explicitly pre-register you are measuring routing differences *conditional on* refusal-feature activation (direction/feature), and that your causal test is to intervene on the mediator and check whether routing differences persist. That converts “routing differs” into a causal question rather than a descriptive one. citeturn7search0turn7search3turn10search12

## Download list: papers, docs, repos to save locally

Organized by theme; each item is a primary source unless noted.

### Depth routing and residual mixing

- Attention Residuals (AttnRes) official repo (March 2026 release). citeturn22search1
- Hyper‑Connections (ICLR 2025). citeturn2search2
- DenseFormer (2024). citeturn2search0
- DeepCrossAttention (2025). citeturn4search3turn4search7
- The Curse of Depth in LLMs (arXiv Feb 2025; NeurIPS 2025). citeturn4search2turn4search6
- ShortGPT (arXiv Mar 2024; later venue pages). citeturn4search1turn4search5
- Mixture‑of‑Depths (Apr 2024). citeturn18search0
- Mixture‑of‑Recursions (2025). citeturn18search1turn18search11

### Interpretability tools and “lenses”

- TransformerLens docs + repo. citeturn5search0turn5search4
- NNsight docs + repo + 0.6 update (Feb 26, 2026). citeturn5search5turn5search1turn5search9
- Tuned Lens (arXiv 2023). citeturn6search0
- Backward Lens (arXiv 2024; EMNLP 2024). citeturn6search1turn6search5

### SAEs, transcoders, circuit tracing

- SAELens repo + release history. citeturn5search2turn5search6
- Gemma Scope docs (Gemma Scope 2, Dec 19 2025). citeturn5search11turn5search15
- Gemma Scope initial release landing (HF + blog entry describing “SAEs on every layer”). citeturn5search23turn5search7
- Transcoders Find Interpretable LLM Feature Circuits (June 2024; NeurIPS 2024). citeturn19search1turn19search4
- Transcoders Beat Sparse Autoencoders for Interpretability (Jan 2025). citeturn19academia40
- circuit-tracer repo + Anthropic open-source announcement (May 29, 2025). citeturn6search2turn6search3
- Circuit Tracing methods page (Mar 27, 2025). citeturn7search13turn7search7

### Benchmarks and evaluation

- MIB (OpenReview/ICML 2025) + GitHub + HF datasets. citeturn17view0turn23search2turn23search7
- BlackboxNLP 2025 shared task leveraging MIB. citeturn23search22
- Preference leakage in LLM-as-judge (relevant to evaluation leakage concerns). citeturn10search5

### Safety/refusal/honesty mechanistic anchors

- Refusal in Language Models Is Mediated by a Single Direction (June 2024; NeurIPS 2024). citeturn7search0turn7search4
- Anthropic: Tracing the thoughts of a large language model (refusal circuit narrative; Mar 27, 2025). citeturn7search3
- On the Biology of a Large Language Model (Mar 27, 2025) and associated updates. citeturn7search7turn7search5
- JailbreakLens (Nov 2024) as a representative mech‑interp framing of jailbreak mechanisms. citeturn7search1
- Recent safety‑mechanistic work on refusal boundaries (USENIX Security 2025). citeturn7search9

### Training dynamics suites

- Pythia paper + repo + HF checkpoint description. citeturn9search16turn9search0turn9search13
- OLMo 2 paper + OLMo 3 release post + HF checkpoint naming. citeturn9search21turn9search1turn9search4
- EvoLM (NeurIPS 2025) cautionary paper. citeturn9search5turn9search6

### Interpretability-by-design competitor/adjacent

- OpenAI blog: Understanding neural networks through sparse circuits (Nov 13, 2025). citeturn26search19
- Weight‑sparse transformers paper (arXiv Nov 17, 2025) + key screenshots on faithfulness limitations. citeturn26search1turn21view1turn21view2
- circuit_sparsity open-source repo + HF model card (Dec 2025). citeturn27search0turn27search1

## Open questions that still require human judgment

These are not “missing citations” problems; they are choices about scientific taste, positioning, and risk.

**How strong do you want the central ontological claim to be?**
Do you claim “latent routing exists” (implying something like an internal router variable), or do you claim “effective depth mixtures exist” (a weaker, more defensible statement)? The first is more exciting but will be attacked as over-interpretation; the second is more robust and still connects to AttnRes-style explicit routers. citeturn22search1turn2search2turn6search0

**Where do you want to sit on the “interpretability vs capabilities externalities” axis?**
Safety-focused mech‑interp publication has explicit concerns about externalities; refusal-direction work demonstrates dual-use in a very direct way. Decide what you will and will not release (datasets, prompts, code paths) before you have results. citeturn7search4turn10search17

**What is your primary success criterion: scientific insight, benchmark performance, or safety narrative?**
MIB is pushing the field toward benchmarked causal localization; circuit-tracing work pushes toward rich mechanistic narratives; safety work pushes toward actionable oversight. You can do all three, but only if you explicitly separate exploratory narratives from confirmatory claims. citeturn17view0turn7search13turn10search12

**Do you want to bet on SAEs, transcoders, or neither as the “feature substrate” for routing analysis?**
MIB suggests SAEs underperform neurons in at least one causal-variable setting, while transcoders are argued (in primary sources) to improve interpretability and circuit analysis through MLPs. Your choice changes both implementation complexity and credibility in 2026. citeturn17view0turn19academia40turn19search4

**Which model families are your primary empirical targets?**
Pythia remains clean for controlled training dynamics; OLMo offers modern fully open pipelines; GemmaScope offers rich safety‑interpretability tooling. Choosing one “spine” model family for coherence may matter more than breadth. citeturn9search16turn9search21turn5search11turn5search15
