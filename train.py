import torch
import torch.nn.functional as F
from model import TinyGPT10k  # Make sure you renamed your model class accordingly

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model config
vocab_size = 96
context_size = 32
model = TinyGPT10k(vocab_size=vocab_size, context_size=context_size, d_model=16).to(device)

# Load data and encode to integers
with open("data.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Simple ASCII tokenizer (printable characters only)
def encode(text):
    return [ord(c) - 32 for c in text if 32 <= ord(c) < 128]  # maps ' ' to 0, '~' to 95

tokens = encode(raw_text)

# Create input-target pairs: X = context window, Y = next token
X = []
Y = []
for i in range(len(tokens) - context_size):
    X.append(tokens[i:i + context_size])
    Y.append(tokens[i + 1:i + context_size + 1])  # shifted right by 1

X = torch.tensor(X, dtype=torch.long).to(device)  # shape [N, T]
Y = torch.tensor(Y, dtype=torch.long).to(device)  # shape [N, T]

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

for step in range(100):
    logits = model(X)  # shape [N, T, vocab_size]
    loss = F.cross_entropy(logits.view(-1, vocab_size), Y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")

torch.save(model.state_dict(), "tinygpt10k.pt")
print("Model saved to tinygpt10k.pt")
