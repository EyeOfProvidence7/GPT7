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
VOCAB_SIZE = 96  # ASCII printable characters (32–127)
CONTEXT_SIZE = 128

model = GPT77k(
    vocab_size=VOCAB_SIZE,
    context_size=CONTEXT_SIZE,
    d_model=128,
    n_layers=4,
    n_heads=4
).to(device)

# ─────────────────────────────────────────────
# 📚 Load and encode data
with open("data.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

def encode(text):
    return [ord(c) - 32 for c in text if 32 <= ord(c) < 128]

tokens = encode(raw_text)

# 🔄 Create input-output sequences
X, Y = [], []
for i in range(len(tokens) - CONTEXT_SIZE):
    X.append(tokens[i:i + CONTEXT_SIZE])
    Y.append(tokens[i + 1:i + CONTEXT_SIZE + 1])

X = torch.tensor(X, dtype=torch.long)
Y = torch.tensor(Y, dtype=torch.long)

# ─────────────────────────────────────────────
# 🧺 DataLoader
BATCH_SIZE = min(512, len(X))  # auto-adjust if dataset is small
dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

# ─────────────────────────────────────────────
# 🚀 Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# ─────────────────────────────────────────────
# 🧠 Training loop with save
best_loss = float('inf')
SAVE_DELTA = 0.01
LOSS_THRESHOLD = 0.0
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
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), batch_y.view(-1))

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
