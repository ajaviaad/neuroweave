import torch

# pick one down_proj weight parameter
for name, param in model.named_parameters():
    if "layers.31.mlp.down_proj.weight" in name:
        # snapshot before
        before = param.data.clone()

        # mutate
        with torch.no_grad():
            param.data += torch.randn_like(param) * 0.01

        # snapshot after
        after = param.data

        print(torch.allclose(before, after))  # prints False
        break
