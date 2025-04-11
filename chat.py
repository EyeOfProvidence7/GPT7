import torch
from model import GPT77k  # Use your upgraded 77k-param model

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model config (must match training!)
vocab_size = 96
context_size = 64
model = GPT77k(
    vocab_size=vocab_size,
    context_size=context_size,
    d_model=64,      
    n_layers=2,
    n_heads=4
).to(device)

model.load_state_dict(torch.load("gpt77k.pt"))
model.eval()

total, trainable = count_parameters(model)
print(f"📊 Total parameters: {total}")
print(f"📦 Trainable parameters: {trainable}")

# Encoding and decoding
def encode(text):
    return [ord(c) - 32 for c in text if 32 <= ord(c) < 128]

def decode(indices):
    return ''.join(chr(i + 32) for i in indices)

print(f"\n🤖 Welcome to GPT-77k! Type an ASCII prompt, and it will complete it with 100 generated characters.")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("🧠 You: ")

    if user_input.strip().lower() == "exit":
        break

    try:
        tokens = encode(user_input)
        if not tokens:
            raise ValueError
    except ValueError:
        print("⚠️ Please enter printable ASCII characters.\n")
        continue

    # Truncate or pad to context size
    if len(tokens) < context_size:
        context = [0] * (context_size - len(tokens)) + tokens
    else:
        context = tokens[-context_size:]

    generated = []

    for _ in range(100):
        x = torch.tensor([context], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(x)  # [1, T, vocab_size]
            temperature = 1.0  # try 0.7–1.3 for creativity
            probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_token)
        context = context[1:] + [next_token]  # Slide window

    output = decode(generated)
    print(f"🤖 GPT-77k says:\n{output}\n")
