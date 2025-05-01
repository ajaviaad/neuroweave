# Neuroweave: Transformer with Custom Activations

Neuroweave is a fully custom transformer model architecture utilizing dynamic activation routing across all feed-forward (MLP) blocks. Each block custom activation at the neuron level, enhancing gradient diversity and representational power.

## Key Features

- 32-layer transformer with AutoMixedActivation in every MLP
- All 14336 neurons per layer patched
- Optimized for TPU (V8) and PyTorch/XLA
- Custom tokenizer (SentencePiece-based)
- Safetensor-sharded model weights

## Files & Structure

```
neuroweave/
├── config.json                   # Model architecture config
├── generation_config.json       # Optional decoding parameters
├── model-00001-of-00006.safetensors  # Sharded model weights
├── model.safetensors.index.json      # Shard index map
├── tokenizer.model              # SentencePiece tokenizer model
├── tokenizer.json               # Tokenizer config
├── special_tokens_map.json      # Special token settings
├── added_tokens.json            # Custom token list
├── tokenizer_config.json        # Tokenizer metadata
├── modeling_auto.py             # Custom model logic 
```

The hidden layer is split evenly, each segment passed through one activation, then concatenated. This creates highly expressive per-neuron behavior and improves learning dynamics.

## Model Weights

Model weights are hosted on Google Drive in `.safetensors` format. You can download them using:

```bash
wget "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID" -O neuroweave_weights.zip
unzip neuroweave_weights.zip
```

## Usage

```python
from torch_xla.core.xla_model import xla_device
device = xla_device()

model = NeuroWeaveModel(config).to(device)
state_dict = load_file("model-00001-of-00003.safetensors", device=device)
model.load_state_dict(state_dict)
```

## Note on Architecture Modification

This model is a heavily modified derivative of Mistral-7B. It introduces a novel AutoMixedActivation mechanism and customized transformer behavior with proven architectural and gradient-level differences. While based on open-source foundations, the resulting model behavior, weights, and activation logic reflect a distinct innovation.

Neuroweave is no longer distributing the original Mistral weights. The model incorporates architectural modifications and altered activations, resulting in derivative weights generated through modified computation. As a result, Neuroweave weights are no longer bitwise identical to those of Mistral.

## Attribution

This model is based on [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1), which is licensed under the Apache License 2.0.

Modifications have been made, including the integration of a proprietary activation mechanism ("AutoMixedActivation"). These modifications are not part of the original Mistral release and are the intellectual property of the author.
