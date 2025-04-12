import torch
from model import GPT77k

# Define custom vocab
CUSTOM_VOCAB = ['0', '1', '+', '=']
CHAR_TO_INDEX = {c: i for i, c in enumerate(CUSTOM_VOCAB)}
INDEX_TO_CHAR = {i: c for c, i in CHAR_TO_INDEX.items()}
VOCAB_SIZE = len(CUSTOM_VOCAB)
CONTEXT_SIZE = 64

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = GPT77k(vocab_size=VOCAB_SIZE, context_size=CONTEXT_SIZE, d_model=64, n_layers=2, n_heads=4).to(device)
model.load_state_dict(torch.load("gpt77k.pt"))
model.eval()

# Debug param count
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

total, trainable = count_parameters(model)
print(f"📊 Total parameters: {total}")
print(f"📦 Trainable parameters: {trainable}")

# Encoding/decoding for 4-token vocab
def encode(text):
    return [CHAR_TO_INDEX[c] for c in text if c in CHAR_TO_INDEX]

def decode(indices):
    return ''.join(INDEX_TO_CHAR[i] for i in indices)

print("\n🤖 GPT-77k is ready to compute binary string addition!")
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
        print("⚠️ Please enter only characters: 0, 1, +, =\n")
        continue

    # Pad context
    if len(tokens) < CONTEXT_SIZE:
        context = [0] * (CONTEXT_SIZE - len(tokens)) + tokens
    else:
        context = tokens[-CONTEXT_SIZE:]

    generated = []
    for _ in range(10):  # up to 10 binary digits
        x = torch.tensor([context], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_token)
        context = context[1:] + [next_token]

        # stop generation once we see a token that wouldn't occur in a result (optional)
        if len(generated) > 2 and next_token in [2]:  # '+' shouldn't appear after '='
            break

    output = decode(generated)
    print(f"🤖 GPT-77k says: {output}\n")
