# References

Curated paper and tooling library for the depth-routing experiment.

How to use this file:

1. Check here before searching the web for a cited paper.
2. Prefer the local paper cache in `background-work/papers/` when the file is already downloaded.
3. If a source is critical to a methodological decision, record the resulting choice in `DECISIONS.md`.

## Core Architecture and Thesis Papers

| Paper | Source URL | Local file | Why it matters |
|---|---|---|---|
| Attention Residuals | https://github.com/MoonshotAI/Attention-Residuals | `research/Attention_Residuals.pdf` and `background-work/papers/files/attention_residuals.pdf` | The motivating architecture and Figure 8 prediction surface. |
| DeepCrossAttention: Supercharging Transformer Residual Connections | https://arxiv.org/abs/2502.06785 | `background-work/papers/files/deepcrossattention_supercharging_transformer_residual_connections.pdf` | Closest dynamic residual-mixing comparison point outside AttnRes. |
| DenseFormer | https://proceedings.neurips.cc/paper_files/paper/2024/file/f67449c7ab72f441d3a713b046c6818c-Paper-Conference.pdf | `background-work/papers/files/denseformer.pdf` | Static depth-wise averaging baseline and cross-seed pattern reference. |
| Hyper-Connections | https://proceedings.iclr.cc/paper_files/paper/2025/file/f1e8ff97057e0c3f6bcb7018c78d4df1-Paper-Conference.pdf | `background-work/papers/files/hyper_connections.pdf` | Parallel residual-stream mixing baseline and stability comparison. |
| MUDDFormer | https://arxiv.org/abs/2502.12170 | `background-work/papers/files/muddformer.pdf` | Dynamic depth routing framed as multiway dense connectivity. |
| The Curse of Depth in Large Language Models | https://arxiv.org/abs/2502.05795 | `background-work/papers/files/the_curse_of_depth_in_large_language_models.pdf` | Main theoretical justification for expecting redundant or weak deep-layer contribution in standard Pre-LN transformers. |
| ShortGPT: Layers in Large Language Models are More Redundant Than You Expect | https://aclanthology.org/2025.findings-acl.1035/ | `background-work/papers/files/shortgpt_layers_in_large_language_models_are_more_redundant_than_you_expect.pdf` | Layer redundancy baseline and Block Influence comparison. |

## Interpretability and Tooling Papers

| Paper | Source URL | Local file | Why it matters |
|---|---|---|---|
| Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling | https://arxiv.org/abs/2304.01373 | `background-work/papers/files/pythia_a_suite_for_analyzing_large_language_models_across_training_and_scaling.pdf` | Training-dynamics lane and checkpoint-based validation. |
| Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2 | https://arxiv.org/abs/2408.05147 | `background-work/papers/files/gemma_scope_open_sparse_autoencoders_everywhere_all_at_once_on_gemma_2.pdf` | Main SAE feature source for Gemma-2-2B. |
| Route Sparse Autoencoder to Interpret Large Language Models | https://aclanthology.org/2025.emnlp-main.346/ | `background-work/papers/files/route_sparse_autoencoder_to_interpret_large_language_models.pdf` | Closest routing-aware SAE reference. |
| Backward Lens: Projecting Language Model Gradients into the Vocabulary Space | https://aclanthology.org/2024.emnlp-main.142/ | `background-work/papers/files/backward_lens_projecting_language_model_gradients_into_the_vocabulary_space.pdf` | Vocabulary-space attribution tool relevant to lens adaptation under routing. |
| Eliciting Latent Predictions from Transformers with the Tuned Lens | https://arxiv.org/abs/2303.08112 | `background-work/papers/files/eliciting_latent_predictions_from_transformers_with_the_tuned_lens.pdf` | Baseline lens methodology and layer-wise prediction framing. |
| LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding | https://arxiv.org/abs/2404.16710 | `background-work/papers/files/layerskip_enabling_early_exit_inference_and_self_speculative_decoding.pdf` | Dynamic depth-use reference for comparison and discussion. |
| Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small | https://arxiv.org/abs/2211.00593 | `background-work/papers/files/interpretability_in_the_wild_a_circuit_for_indirect_object_identification_in_gpt_2_small.pdf` | Canonical known-circuit reference if the tool-breakage lane uses IOI or induction-style tracing. |
| In-context Learning and Induction Heads | https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html | `background-work/papers/files/in_context_learning_and_induction_heads.html` | Primary induction-heads reference if the tool-breakage lane pivots to a known transformer circuit rather than factual recall. |
| Measuring Faithfulness in Chain-of-Thought Reasoning | https://arxiv.org/abs/2307.13702 | `background-work/papers/files/measuring_faithfulness_in_chain_of_thought_reasoning.pdf` | Useful for not overclaiming from surface-visible reasoning traces or simple lens plots. |
| Refusal in Language Models Is Mediated by a Single Direction | https://arxiv.org/abs/2406.11717 | `background-work/papers/files/refusal_in_language_models_is_mediated_by_a_single_direction.pdf` | Strong prior for the safety lane and a concrete comparison point for refusal-feature discovery. |

## Methodology and Numerical References

| Reference | Source URL | Local file | Why it matters |
|---|---|---|---|
| Root Mean Square Layer Normalization | https://arxiv.org/abs/1910.07467 | `background-work/papers/files/root_mean_square_layer_normalization.pdf` | Final-norm handling and the uniform-routing reconstruction sanity check. |
| scikit-learn AgglomerativeClustering documentation | https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html | none | Confirms that Ward linkage is only valid with Euclidean distances. |

## Local Cache

- Manifest: `background-work/papers/DOWNLOAD_MANIFEST.md`
- Download helper: `scripts/download_reference_papers.py`
- Downloaded files: `background-work/papers/files/`

Use the manifest for exact filenames and sizes.
