# Neuroweave

Neuroweave is a proprietary Transformer architecture developed by the NeuronMix Team — built from the ground up through deep architectural intervention on top of Mistral‑7B. It integrates a novel activation function (AutoMixedActivation), replaces all key transformer components, and connects to real‑time search via DuckDuckGo — making it non‑pretrained in function, non‑reliant on static corpora, and patent‑pending in design.

## Features

- **AutoMixedActivation**: Neuron‑level combined ReLU, GELU, and Swish for dynamic activations.
- **Component Replacement**: Custom MLPs and attention projections throughout the transformer.
- **Real‑Time Search Integration**: DuckDuckGo API for live knowledge augmentation.
- **Patent‑Pending**: Unique architectural interventions and activation functions.

## Installation

```bash
git clone https://github.com/NeuronMix/Neuroweave.git
cd Neuroweave
pip install -r requirements.txt
```

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("NeuronMix/Neuroweave", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("NeuronMix/Neuroweave", trust_remote_code=True)
```

## License

The core Neuroweave codebase is licensed under the Apache License 2.0.

### Post‑Mutation Weights

**Important**: Any weights that have been reinitialized, patched, or mutated post‑checkpoint load—including all activation‑patched and fully mutated tensors—are proprietary to the NeuronMix Team and **are not** subject to the Apache License 2.0. Use of these custom weights is governed by separate, proprietary terms.

