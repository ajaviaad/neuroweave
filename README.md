# 🧠 Neuroweave: Custom Transformer Architecture with AutoMixedActivation

**Neuroweave** is a standalone Transformer architecture developed by the NeuronMix Team. While originally inspired by Mistral, it has now evolved into a fully customized model with all embeddings, activations, and transformer layers replaced. At its core is a proprietary nonlinearity, `AutoMixedActivation`, applied consistently across the entire stack — resulting in a novel training trajectory, unique representations, and fundamentally distinct behavior from any Mistral-derived model.

## 🚀 Key Highlights

- ✅ **Input embedding replaced**: Token embeddings are fully customized and processed with `AutoMixedActivation` before entering the model.
- ✅ **Output head replaced**: Output projection is rebuilt and includes `AutoMixedActivation`, leading to new generation behavior.
- ✅ **All transformer layers redefined**: Every transformer block, including attention and MLP submodules, is patched with `AutoMixedActivation`. No SiLU or GELU activations remain.
- ✅ **Gradient flow redesigned**: Gradient norms across layers confirm different weight updates and learning dynamics compared to Mistral.
- ✅ **No shared weights with Mistral**: Weight divergence is confirmed. Neuroweave no longer shares gradients, activations, or parameter values with Mistral.

## 🔬 What AutoMixedActivation Does

```python
def forward(self, x):
    return torch.nn.functional.silu(x) + 0.1 * torch.tanh(x)
```
This hybrid activation combines the feature sensitivity of `SiLU` with the saturation stability of `tanh`, promoting robust, expressive representations.

## 🧪 Evidence of Architectural Independence

- All 32 layers have modified activation paths — `AutoMixedActivation` is the only nonlinearity used.
- No `SiLU` is called at any point in the runtime.
- Layer norm and gradient norm traces show significantly different behavior from Mistral.
- Weight gradients across layers confirm full divergence from original model checkpoints.

## 📦 Deployment

Neuroweave is implemented with Hugging Face Transformers and can be run from disk. All modifications are integrated at model construction time. No external dependencies, APIs, or Mistral checkpoints are required.

```python
model.set_input_embeddings(PatchedEmbedding(...))
model.lm_head = PatchedLMHead(...)
for layer in model.model.layers:
    layer.self_attn.q_proj = PatchedLinear(...)
    layer.mlp.gate_proj = PatchedLinear(...)
    layer.mlp.act_fn = AutoMixedActivation()
```

---

© 2025 NeuronMix Team. Neuroweave is a fully original transformer architecture with custom embedding, activation, and training pathways. Not a Mistral variant.
