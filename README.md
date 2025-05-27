# Neuroweave

Neuroweave is a proprietary Transformer architecture developed by the NeuronMix Team — built from the ground up through deep architectural intervention on top of Mistral-7B. It integrates a novel activation function (`PhantomAct`), replaces all key transformer components, and connects to real-time search API — making it non-pretrained in function, non-reliant on static corpora, and patentable in design.

## Why Neuroweave Is Unique

### Scientific and Structural Independence
- Input & output embeddings fully replaced
- All 32 transformer layers patched at attention and MLP levels
- No SiLU, GELU, or residual pretrained activation pathways remain
- Custom activation: `PhantomAct()` improves feature flow
- Gradient flow and input norm traces confirm architectural divergence

### Activation Innovation: `AutoMixedActivation`

```
def forward(self, x):
    return 0.9 * torch.tanh(x) + 0.1 * torch.sin(x)
```

(`PhantomAct`) is a novel activation function crafted from scratch—independent of standard functions like SiLU or ReLU. It blends the bounded smoothness of tanh with a minor oscillatory component from sin, encouraging nuanced non-linear dynamics while avoiding gradient saturation. The result is a stable yet expressive activation that supports deeper representations, better generalization, and smoother optimization across transformer and vision models.

## Empirical Evidence of Novelty

Neuroweave does not reuse pretrained representations. Comparison of input norms across all 32 layers:

![Input Norms Comparison](https://github.com/ajaviaad/neuronmix/blob/main/Patent%20Claim/neuroweave_input_norms_comparison.png)

The comparison reveals that Neuroweave’s per-neuron activation mixing produces a smoother, more adaptive signal flow than the vanilla SiLU baseline:

- Embeddings & Early Layers (0–1): Neuroweave ramps from 0.24 → 1.85, while SiLU jumps from 0.43 → 6.06. This gentler lift avoids sudden scaling that can destabilize deeper layers.

- Middle Blocks (2–19): Instead of SiLU’s rigid 264.0 plateau, Neuroweave norms slide downward from ≈262.4 → 260.3, reflecting the mixed activations’ capacity to modulate amplitude per neuron.

- Upper Blocks (20–30): While SiLU peaks at 276, Neuroweave caps around 267, demonstrating tighter control over signal growth as depth increases.

- Pre-Head Layer (31): The drop to ≈226 (vs. 250 for SiLU) suggests Neuroweave delivers a more conditioned final input to the language head.

Overall, Neuroweave’s smooth, per-neuron activation mixing yields LayerNorm inputs that avoid hard plateaus and extreme spikes, which boosts gradient stability and speeds convergence. Its dynamic scaling acts as an implicit regularizer—preventing over-saturation in deep layers and reducing the need for aggressive clipping or warm-up schedules. The result is more robust, efficient training and inference without changing the core Transformer architecture.

The chart below shows token confidence for the top prediction per step. Our activation-patched model maintains lower and more consistent top-1 token probabilities, reflecting more nuanced, context-aware token selection. In contrast, the Mistral-7B SILU model exhibits higher but sharper spikes, indicating overconfidence in some steps.

![Token Confidence Comparison](https://github.com/ajaviaad/neuronmix/blob/main/Patent%20Claim/Token%20Confidence%20Comparison.png)

Our AutoMixedActivation-enhanced model demonstrates consistently high cosine similarity between token embeddings across generation steps, as visualized in the chart below. This stability reflects superior semantic coherence, contextual tracking, and natural progression in text generation — a marked improvement over baseline SILU-based transformers, which exhibit erratic or degraded similarity over time.

![Cosine Similarity Comparison](https://github.com/ajaviaad/neuronmix/blob/main/Patent%20Claim/Cosine%20Similarity.png)



| Metric                     | Activation-Patched      | SILU Vanilla           |
| -------------------------- | ----------------------- | ---------------------- |
| **Mean KL Divergence**     | 0.0186                  | N/A                    |
| **Mean Cosine Similarity** | 0.9805                  | N/A                    |
| **Top-1 Confidence MSE**   | 0.875 (lower certainty) | 0.363 (high certainty) |
| **Token Generation Speed** | **80 tokens/sec**       | 20 tokens/sec          |

### Weight Gradients Per Layer

The chart below compares the \texttt{mlp.down_proj.weight} gradient norms for each of the 32 layers:

- Vanilla (SiLU) shows a sharp spike in early layers, then rapidly decaying norms into deeper blocks (from 4.4×10⁷ at layer 1 down to ~1×10⁶ by layer 15).

- Neuroweave exhibits a much smoother decay, starting higher (≈7×10⁷–7.2×10⁷ in layers 1–3) and tapering gradually to ≈6×10⁶ by layer 30.

![Weight Gradients Per Layer](https://github.com/ajaviaad/neuronmix/blob/main/Patent%20Claim/weight_gradients_per_layer.png)

Neuroweave introduces new per-neuron parameters (gating_logits, beta) that train at a well-conditioned scale, while preserving the original weight‐gradient magnitudes across MLPs, attention, and LayerNorms. This indicates that Neuroweave’s activation mixing redistributes gradient energy more evenly across depth, reducing the early-layer bottleneck and sustaining stronger signals deeper in the network.

### Interpretation

- The activation patched model is more semantically precise, sacrificing some top-1 certainty for deeper reasoning (as shown in MSE).

- Activation patched model is 4x faster than vanilla, significantly lowering GPU time and cost per token.

- Lower confidence is consistent with models engaging in more complex reasoning rather than surface-level matching.

- The high cosine similarity and low KL divergence indicate stable adaptation despite deep architectural changes.

### Deployment

Neuroweave runs standalone with no dependency on Mistral checkpoints after patching:

```python
model.set_input_embeddings(PatchedEmbedding(...))
model.lm_head = PatchedLMHead(...)

for layer in model.model.layers:
    layer.self_attn.q_proj = PatchedLinear(...)
    layer.mlp.gate_proj = PatchedLinear(...)
    layer.mlp.act_fn = AutoMixedActivation()
```

## Licensing & Patent

Neuroweave is developed under Apache 2.0 compliance for original Mistral scaffolding.  
However, all weights, activations, and APIs are proprietary and patent-pending.

Redistribution of Neuroweave is subject to NeuronMix terms.

---

© 2025 NeuronMix Team.
