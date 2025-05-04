Licensed under the Apache License, Version 2.0
===================================================

This repository is licensed under the Apache License, Version 2.0. You may obtain a copy of the full license text in the [LICENSE.md](https://github.com/ajaviaad/neuroweave/blob/main/LICENSE.md) file.

==============================================================================

📣 NOTICE: Patent Pending / Proprietary Components
==============================================================================

The following components in this repository are proprietary to the NeuronMix Team and are covered by patent applications. Use of these components for commercial or derivative projects outside of this repository requires explicit written permission.

1. **AutoMixedActivation**  
   A novel activation module that dynamically applies multiple nonlinearities (e.g., ReLU, GELU, SiLU) to partitions of the hidden dimension. Supports both 3D ([B, T, C]) and 4D ([B, C, H, W]) tensor shapes, introducing mixed activation diversity within transformer architectures.

2. **Runtime Patching of Transformer Architecture**  
   Comprehensive patching logic that replaces multiple components—including input and output embedding layers, activation functions (`act_fn`), projection modules, and LayerNorm instances—across all 32 transformer layers (self-attention and MLP blocks) of the Mistral model. Patches are applied at import time to inject custom behaviors (e.g., `AutoMixedActivation`), altering both forward activations and backpropagation pathways.

3. **Divergence from Original Pretrained Weights**  
   Although the original Mistral-7B checkpoint tensors are initially loaded, the application of the patched architecture immediately changes effective weight usage and gradient flow. As a result, the active weight tensors—even before any fine-tuning—functionally and numerically diverge from the unpatched Mistral-7B weights, as evidenced by distinct gradient norms and activation statistics.

==============================================================================

© 2025 NeuronMix Team. All rights reserved.
