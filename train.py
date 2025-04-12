import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from model import GPT77k
import json

# ─────────────────────────────────────────────
# ⚙️ Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    print(f"🚀 Using CUDA: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ Using CPU (no CUDA found)")

# ─────────────────────────────────────────────
# 🧠 Model config
CUSTOM_VOCAB = ['0', '1', '+', '=']
CHAR_TO_INDEX = {c: i for i, c in enumerate(CUSTOM_VOCAB)}
INDEX_TO_CHAR = {i: c for c, i in CHAR_TO_INDEX.items()}
vocab_size = len(CUSTOM_VOCAB)

context_size = 64
model = GPT77k(vocab_size=vocab_size, context_size=context_size, d_model=64, n_layers=2, n_heads=4).to(device)

# ─────────────────────────────────────────────
# 📚 Load and encode data
with open("data.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

def encode(text):
    return [CHAR_TO_INDEX[c] for c in text if c in CHAR_TO_INDEX]

tokens = encode(raw_text)

# 🔄 Create input-output sequences
X, Y = [], []
for i in range(len(tokens) - context_size):
    X.append(tokens[i:i + context_size])
    Y.append(tokens[i + 1:i + context_size + 1])

X = torch.tensor(X, dtype=torch.long)
Y = torch.tensor(Y, dtype=torch.long)

# ─────────────────────────────────────────────
# 🧺 Batch & shuffle with DataLoader
BATCH_SIZE = 32
dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ─────────────────────────────────────────────
# 🚀 Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# ─────────────────────────────────────────────
# 🧠 Training loop with epoch loss + save logic
best_loss = float('inf')
SAVE_DELTA = 0.01
LOSS_THRESHOLD = 0.2
MAX_STEPS = 1000
MODEL_PATH = "gpt77k.pt"
META_PATH = "gpt77k-meta.json"

print("🧪 Starting training...")
for step in range(MAX_STEPS):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), batch_y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    epoch_loss = total_loss / num_batches

    print(f"Step {step}: loss = {epoch_loss:.4f}")

    if epoch_loss < best_loss - SAVE_DELTA:
        best_loss = epoch_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"💾 Best model saved (loss = {best_loss:.4f})")

        with open(META_PATH, "w") as meta_file:
            json.dump({"loss": round(best_loss, 4), "step": step}, meta_file, indent=4)
        print(f"📝 Metadata saved to {META_PATH}")

    if epoch_loss <= LOSS_THRESHOLD:
        print(f"🧠 Early stopping triggered (loss = {epoch_loss:.4f})")
        break

print("✅ Training complete.")
