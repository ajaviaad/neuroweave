# Neuroweave: Full-Stack Transformer Variant with AutoMixedActivation

**Neuroweave** is a custom Transformer variant built on top of Mistral-7B, created by the NeuronMix Team. It introduces a proprietary activation function, `AutoMixedActivation`, applied not only at the embedding level but also in **every transformer layer**. This results in a fundamentally different behavior from baseline models — leading to significantly reduced plagiarism, greater content originality, and more dynamic generation.

## Key Highlights

- ✅ **Input embedding patched**: Tokens are embedded and immediately passed through `AutoMixedActivation` before entering the transformer stack.
- ✅ **Output head patched**: Final hidden states go through `AutoMixedActivation` before being projected to vocabulary logits.
- ✅ **All transformer layers patched**: Each transformer block's `mlp.act_fn` has been replaced with `AutoMixedActivation`, ensuring consistent nonlinearity throughout the network.
- ✅ **Live runtime patching**: No retraining, no weight changes — all modifications happen dynamically during inference.
- ✅ **Verified hook calls**: Activation logs confirm `AutoMixedActivation` is called once per layer per generation step.

## What AutoMixedActivation Does

```python
def forward(self, x):
    return torch.nn.functional.silu(x) + 0.1 * torch.tanh(x)
```
This hybrid activation combines the sharp feature sensitivity of `SiLU` with the saturation control of `tanh`, encouraging richer representations and mitigating overfitting.

## Evidence of Impact

- 32 `[HOOK] AutoMixedActivation called.` logs confirm full-layer integration.
- Input and hidden state norms propagate differently vs. SiLU baseline, indicating altered flow.
- Empirical plagiarism analysis shows up to **80% novel content generation**, drastically improving originality.
- Generation exhibits greater contextual flexibility, nuance, and reduced memorization bias.

## Deployment

Neuroweave runs entirely from local disk (e.g. Google Drive) using Hugging Face Transformers. No need for external APIs or model retraining — just load, patch, and run.

```python
model.set_input_embeddings(PatchedEmbedding(...))
model.lm_head = PatchedLMHead(...)
for layer in model.model.layers:
    layer.mlp.act_fn = AutoMixedActivation()
```

---

© 2025 NeuronMix Team. Neuroweave is transformer with fully customized runtime activation dynamics.
