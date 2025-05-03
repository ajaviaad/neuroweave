# Neuroweave:

**Neuroweave** introduces a novel Transformer variant built on top of Mistral-7B, using a custom activation function, `AutoMixedActivation`, applied exclusively at the **embedding level**. This means the model’s behavior is modified at the entry and exit points — without altering the internal transformer block logic.

## Key Highlights

- ✅ **Input embedding patched**: Applies `AutoMixedActivation` to the token embedding vectors before they enter the transformer.
- ✅ **Output head patched**: Applies the same activation to the final hidden states before projecting to vocabulary logits.
- ✅ **No internal layer changes**: Transformer MLPs still use `SiLU`, ensuring stable forward dynamics.
- ✅ **Runtime-only patching**: No weight edits or retraining — changes are injected live via module wrapping.
- ✅ **Confirmed hook logs**: Verified that the activation is applied at both entry and exit points during generation.

## What AutoMixedActivation Does

```python
def forward(self, x):
    return torch.nn.functional.silu(x) + 0.1 * torch.tanh(x)
```

A custom blend of `SiLU` and `tanh`, this activation introduces a richer nonlinear signal early and late in the model’s processing.

## Evidence of Impact

- Hook logs confirm `AutoMixedActivation` is triggered on every forward pass at the embedding and output stages.
- Layer input norms show significant variation compared to baseline, indicating propagation of modified embeddings.
- Model responses show improved coherence and contextual grounding, particularly on scientific and technical prompts.

## Deployment

Runs from Google Drive using patched Hugging Face model components (`get_input_embeddings`, `lm_head`).
No reliance on external APIs. Entire inference process is local and self-contained.

---

© 2025 NeuronMix Team. The embedding-level activation mechanism in **Neuroweave** is proprietary and experimental.
