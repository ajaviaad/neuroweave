import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Custom Activation Only ---
class AutoMixedActivation(nn.Module):
    def __init__(self, activations=None):
        super().__init__()
        self.activations = activations or [
            lambda x: F.relu(x),
            lambda x: F.gelu(x),
            lambda x: x * torch.sigmoid(x)  # Swish
        ]

    def forward(self, x):
        C = x.shape[-1]
        n = len(self.activations)
        pad = (n - C % n) % n
        if pad:
            x = F.pad(x, (0, pad))
        splits = torch.chunk(x, n, dim=-1)
        out = [act(s) for act, s in zip(self.activations, splits)]
        return torch.cat(out, dim=-1)[..., :C]
