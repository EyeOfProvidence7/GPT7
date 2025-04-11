import torch
from model import TinyGPT

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model and infer context size
model = TinyGPT().to(device)
model.load_state_dict(torch.load("tinygpt.pt"))
model.eval()

context_size = model.position_embedding.shape[0]

print(f"🤖 Welcome to GPT-7 (7 params, context size = {context_size}). Type 0s and 1s separated by spaces.")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("🧠 You: ")

    if user_input.strip().lower() == "exit":
        break

    try:
        tokens = list(map(int, user_input.strip().split()))
        if not tokens or any(t not in [0, 1] for t in tokens):
            raise ValueError
    except ValueError:
        print("⚠️ Please enter a space-separated sequence of 0s and 1s.\n")
        continue

    # Adjust input to match context size
    if len(tokens) <= context_size:
        # Pad with zeros on the left
        padded_tokens = [0] * (context_size - len(tokens)) + tokens
        
    else:
        # Truncate from the left
        padded_tokens = tokens[-context_size:]
        print(f"⚠️ Input too long. Truncating to last {context_size} tokens.\n")

    context = torch.tensor([padded_tokens], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(context)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.argmax(probs, dim=-1).item()

    print(f"🤖 GPT-7 says: {next_token}\n")
