import torch
import torch.nn as nn
import torch.nn.functional as F

class GPT100k(nn.Module):
    def __init__(self, vocab_size=96, context_size=64, d_model=32, n_layers=2, n_heads=2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Parameter(torch.randn(context_size, d_model))

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.token_embedding.weight  # tie weights

    def forward(self, idx):
        B, T = idx.shape
        x = self.token_embedding(idx) + self.position_embedding[:T]

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.output_proj(x)
        return logits

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x):
        T = x.size(1)
        attn_mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        attn_mask = ~attn_mask  # MHA expects True = masked

        x_res = x
        x = self.ln1(x)
        x, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = x_res + x  # residual

        x = x + self.ffn(self.ln2(x))  # another residual
        return x