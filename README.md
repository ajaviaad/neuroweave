# Neuroweave (Mistral Variant with AutoMixedActivation)

Neuroweave is a transformer variant based on [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) that introduces a novel **AutoMixedActivation** module in the MLP layers. This change allows each hidden dimension group to be routed through a different activation function, enhancing nonlinearity and output diversity — all without retraining.

---

## 🔍 Key Differences from Mistral

| Feature               | Mistral              | Neuroweave               |
|----------------------|----------------------|---------------------------|
| Activation Function  | SiLU (Swish)         | ReLU + GELU + Swish mix  |
| Retraining Required? | ❌ No                | ❌ No                    |
| Behavior Change      | 🔁 Minor variation    | ✅ Substantial improvement |
| Hugging Face Format  | ✅ Yes               | ✅ Yes                    |

---

## 🧠 AutoMixedActivation

Instead of using a single activation, the hidden dimension is split into chunks and each chunk is passed through a different function (ReLU, GELU, Swish).

```python
splits = torch.chunk(x, 3, dim=-1)
out = torch.cat([relu(splits[0]), gelu(splits[1]), swish(splits[2])], dim=-1)
```

This is applied **only during inference**, and the model remains fully compatible with Mistral weights.

---

## 🔧 Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from modeling_auto import AutoMixedActivation

model = AutoModelForCausalLM.from_pretrained("path/to/neuroweave", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("path/to/neuroweave")

# Apply activation patch
for layer in model.model.layers:
    layer.mlp.act_fn = AutoMixedActivation()
```

---

## 🧪 Performance

Empirical prompt testing shows:
- Better contextuality
- More complete answers
- Improved ethical responses

All achieved without any additional training.

---

## 📁 Files

- `modeling_auto.py` — defines the AutoMixedActivation
- `config.json` — config compatible with Hugging Face loading
- Tokenizer and generation configs included for ease of use

---

## 🤝 License

This work is released under the Apache 2.0 License (see LICENSE.md).

---

## ⚠️ Licensing Note

While this repository uses the Apache 2.0 license for standard model loading files and configuration, the following components are **not** released under Apache 2.0 and are instead the **sole property of the NeuronMix Team**:

- `AutoMixedActivation` class
- Any modified model weights resulting from use of this activation

Reproduction, redistribution, or reuse of these components outside this repository requires **explicit written permission** from the NeuronMix Team.
