
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


class AutoMixedActivation(nn.Module):
    def __init__(self, activations=None):
        super().__init__()
        self.activations = activations or [
            lambda x: F.relu(x),
            lambda x: F.gelu(x),
            lambda x: x * torch.sigmoid(x)  # Swish
        ]

    def forward(self, x):
        B, T, C = x.shape
        n = len(self.activations)
        if C % n != 0:
            raise ValueError(f"Hidden dim {C} must be divisible by number of activations ({n})")
        chunks = torch.chunk(x, n, dim=-1)
        activated_chunks = [act(chunk) for act, chunk in zip(self.activations, chunks)]
        return torch.cat(activated_chunks, dim=-1)


# Optional: Re-export the model class for AutoClass loading
AutoModelForCausalLM = AutoModelForCausalLM
