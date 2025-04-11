import torch
import torch.nn.functional as F
from model import TinyGPT

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = TinyGPT().to(device)

with open("data.txt", "r") as f:
    tokens = list(map(int, f.read().strip().split()))

X = torch.tensor([[tokens[i]] for i in range(len(tokens) - 1)], dtype=torch.long).to(device)
Y = torch.tensor([[tokens[i + 1]] for i in range(len(tokens) - 1)], dtype=torch.long).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-1)

for step in range(100):
    logits = model(X)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")

torch.save(model.state_dict(), "tinygpt.pt")
print("Model saved to tinygpt.pt")
