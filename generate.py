import torch
from model import TinyGPT

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the trained model
model = TinyGPT().to(device)
model.load_state_dict(torch.load("tinygpt.pt"))
model.eval()

# Starting token (you can change this to 1 too)
start_token = 0
context = torch.tensor([[start_token]], dtype=torch.long).to(device)

# Generate tokens
num_tokens = 20
generated = [start_token]

with torch.no_grad():
    for _ in range(num_tokens):
        logits = model(context)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.argmax(probs, dim=-1).item()
        generated.append(next_token)
        context = torch.tensor([[next_token]], dtype=torch.long).to(device)

print("Generated sequence:")
print(" ".join(map(str, generated)))
