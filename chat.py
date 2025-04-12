import torch
from model import GPT558k

# Standard ASCII vocab (printable characters 32 to 127)
VOCAB_SIZE = 96
CONTEXT_SIZE = 128  # match the scaled model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the larger GPT558k model
model = GPT558k(
    vocab_size=VOCAB_SIZE,
    context_size=CONTEXT_SIZE,
    d_model=128,
    n_layers=4,
    n_heads=4
).to(device)

model.load_state_dict(torch.load("gpt558k.pt", map_location=device))
model.eval()

# Debug param count
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

total, trainable = count_parameters(model)
print(f"📊 Total parameters: {total:,}")
print(f"📦 Trainable parameters: {trainable:,}")

# Encoding and decoding for ASCII vocab (ord 32–127)
def encode(text):
    return [ord(c) - 32 for c in text if 32 <= ord(c) < 128]

def decode(indices):
    return ''.join(chr(i + 32) for i in indices)

print("\n🤖 GPT-558k is ready! Type any ASCII prompt.")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("🧠 You: ").strip()
    if user_input.lower() == "exit":
        break

    try:
        tokens = encode(user_input)
        if not tokens:
            raise ValueError
    except ValueError:
        print("⚠️ Please enter printable ASCII characters (32–127).\n")
        continue

    # Pad or trim context
    if len(tokens) < CONTEXT_SIZE:
        context = [0] * (CONTEXT_SIZE - len(tokens)) + tokens
    else:
        context = tokens[-CONTEXT_SIZE:]

    generated = []
    for _ in range(100):  # Max generated tokens
        x = torch.tensor([context], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_token)
        context = context[1:] + [next_token]

        if chr(next_token + 32) in ['.', '\n']:
            break

    output = decode(generated)
    print(f"🤖 GPT-558k says: {output}\n")
