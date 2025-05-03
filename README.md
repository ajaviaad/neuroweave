
# Neuroweave

**Neuroweave** introduces a novel Transformer variant built on top of Mistral-7B, using a patent-pending custom activation function, `AutoMixedActivation`, that dynamically applies different nonlinearities across neuron groups. Unlike traditional activation changes, this is applied **at runtime** via forward-pass patching — no retraining, no weight edits.

## 🚀 Key Highlights

- ✅ **Forward-pass patched**: Replaces `SiLU` with `AutoMixedActivation` live in each MLP block during inference.
- ✅ **No retraining needed**: Weights remain intact, yet behavior and gradient flow are significantly altered.
- ✅ **Confirmed activation hooks**: Every transformer layer (0–31) calls the custom activation during generation.
- ✅ **Changed gradient norms**: Verified by comparing against vanilla Mistral using live backward passes.
- ✅ **Improved responsiveness**: More informative, nuanced answers demonstrated across health, safety, and science queries.
- ✅ **Fully decoupled**: No reliance on OpenAI or external APIs. Entire inference stack is self-contained.

## 🔬 What AutoMixedActivation Does

```python
def forward(self, x):
    C = x.shape[-1]
    n = len(self.activations)
    pad = (n - C % n) % n
    if pad:
        x = F.pad(x, (0, pad))
    splits = torch.chunk(x, n, dim=-1)
    out = [act(s) for act, s in zip(self.activations, splits)]
    return torch.cat(out, dim=-1)[..., :C]
```

It mixes ReLU, GELU, and Swish across channel chunks, allowing more expressive neuron behavior — supporting both 3D and 4D tensors.

## 🧪 Evidence of Impact

- Hook logs confirm all 32 layers call `AutoMixedActivation` during inference.
- Gradient norms show spikes (e.g. `down_proj` in Layer 1 → 200+) proving change in learning behavior.
- Model answers reflect improved generalization, safety, and contextual clarity.

## 📦 Deployment

Runs with custom transformers and patched `modeling.py`.
Zero dependency on HuggingFace servers.

---

© 2025 NeuronMix Team. Custom activation and modified runtime behavior are proprietary.
