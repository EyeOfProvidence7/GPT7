import torch
import torch.nn.functional as F
from model import GPT77k  

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model config (match with your model.py)
vocab_size = 96
context_size = 64  # upgraded context size
model = GPT77k(vocab_size=vocab_size, context_size=context_size, d_model=64, n_layers=2, n_heads=4).to(device)

# Load data and encode to ASCII tokens
with open("data.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

def encode(text):
    return [ord(c) - 32 for c in text if 32 <= ord(c) < 128]

tokens = encode(raw_text)

# Create training sequences (X: input, Y: shifted next-token target)
X = []
Y = []
for i in range(len(tokens) - context_size):
    X.append(tokens[i:i + context_size])
    Y.append(tokens[i + 1:i + context_size + 1])

X = torch.tensor(X, dtype=torch.long).to(device)
Y = torch.tensor(Y, dtype=torch.long).to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Training loop
for step in range(200):
    logits = model(X)  # [B, T, vocab]
    loss = F.cross_entropy(logits.view(-1, vocab_size), Y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")

torch.save(model.state_dict(), "gpt77k.pt")
print("✅ Model saved to gpt77k.pt")
