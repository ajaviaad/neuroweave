# Apply full mutation
with torch.no_grad():
    for name, param in model.named_parameters():
        param.data = torch.randn_like(param) * 0.01

# Save the mutated model (proof of mutation)
torch.save(model.state_dict(), "mutated_model.pt")
