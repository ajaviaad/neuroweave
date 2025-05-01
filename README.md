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
├── model-00001-of-00003.safetensors  # Sharded model weights
├── model.safetensors.index.json      # Shard index map
├── tokenizer.model              # SentencePiece tokenizer model
├── tokenizer.json               # Tokenizer config
├── special_tokens_map.json      # Special token settings
├── added_tokens.json            # Custom token list
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

This model introduces a fully patched MLP architecture over the Mistral base, replacing all original activation functions with the custom AutoMixedActivation, a fusion of ReLU and SiLU. Corresponding weight gradients and safetensors partitions have been updated across layers (down_proj, gate_proj, up_proj), resulting in a substantially modified model distinct from the original.

Weight changes are reflected in: model.layers.*.mlp.*.weight across all shards (model-00001/02/03-of-00003.safetensors).
