
import torch
import torch.nn as nn

class AutoMixedActivation(nn.Module):
    def forward(self, x):
        return torch.nn.functional.silu(x) + 0.1 * torch.tanh(x)

class PatchedEmbedding(nn.Module):
    def __init__(self, base_embedding):
        super().__init__()
        self.base_embedding = base_embedding
        self.activation = AutoMixedActivation()

    def forward(self, input_ids):
        return self.activation(self.base_embedding(input_ids))

class PatchedLMHead(nn.Module):
    def __init__(self, base_lm_head):
        super().__init__()
        self.base_lm_head = base_lm_head
        self.activation = AutoMixedActivation()

    def forward(self, hidden_states):
        return self.base_lm_head(self.activation(hidden_states))

def apply_neuroweave_patch(model):
    model.set_input_embeddings(PatchedEmbedding(model.get_input_embeddings()))
    model.lm_head = PatchedLMHead(model.lm_head)
    for layer in model.model.layers:
        layer.mlp.act_fn = AutoMixedActivation()
    return model
