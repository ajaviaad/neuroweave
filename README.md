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

## Note

Neuroweave is a custom transformer variant derived from an open-source Mistral base. It features a complete patching of all MLP activations with the novel AutoMixedActivation module and has undergone substantial architectural modifications and gradient-based weight transformations across all layers. As a result, the current model exhibits distinct behavior and is functionally decoupled from the original pretrained variant.
