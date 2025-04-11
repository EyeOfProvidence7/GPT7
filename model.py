import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyGPT(nn.Module):
    def __init__(self, vocab_size=2, context_size=1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, 1)  # 2 params
        self.position_embedding = nn.Parameter(torch.randn(context_size, 1))  # 1 param
        self.output_proj = nn.Linear(1, vocab_size, bias=True)  # 2 weights + 2 bias = 4 params (we’ll trim later)

        # Freeze all but 7 parameters
        with torch.no_grad():
            self.output_proj.weight.zero_()  # 2
            self.output_proj.bias.zero_()    # 2
        self.output_proj.weight.requires_grad = True
        self.output_proj.bias.requires_grad = True

    def forward(self, idx):
        tok_emb = self.token_embedding(idx)  # [B, T, 1]
        pos_emb = self.position_embedding[:idx.size(1)]  # [T, 1]
        x = tok_emb + pos_emb  # [B, T, 1]
        logits = self.output_proj(x)  # [B, T, vocab_size]
        return logits
