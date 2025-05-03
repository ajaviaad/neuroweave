
# NOTICE: Patent Pending

This repository includes proprietary techniques developed by the **NeuronMix Team**, specifically:

## AutoMixedActivation

A novel activation module that dynamically applies multiple nonlinearities (e.g., ReLU, GELU, Swish) to partitioned sections of the hidden dimension. It supports both 3D ([B, T, C]) and 4D ([B, C, H, W]) tensor shapes and introduces mixed activation diversity within transformer models.

## Forward Pass Patching

All transformer layers (0 through 31) are dynamically altered at runtime to use `AutoMixedActivation` instead of the default `SiLU` or `GELU`. This is achieved via direct patching of the `act_fn` in each MLP block **without modifying pretrained weights**.

---

These methods have demonstrated measurable changes in output behavior and gradient flows and are part of a **patent-pending architecture**.

All rights to the design and runtime manipulation technique are reserved by the NeuronMix Team. Use outside of this repository for commercial or derivative projects requires explicit written permission.

© 2025 NeuronMix Team
