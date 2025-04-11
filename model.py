import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyGPT10k(nn.Module):
    def __init__(self, vocab_size=96, context_size=32, d_model=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.d_model = d_model

        # Token and Position Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Parameter(torch.randn(context_size, d_model))

        # Self-Attention (built-in, single-head)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=1, batch_first=True)

        # Feedforward
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        # Optional: Tie output projection weights to token embedding
        self.output_proj.weight = self.token_embedding.weight

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)                         # [B, T, d_model]
        pos_emb = self.position_embedding[:T]                       # [T, d_model]
        x = tok_emb + pos_emb                                       # [B, T, d_model]

        x = self.ln1(x)

        # Causal attention mask
        attn_mask = torch.tril(torch.ones(T, T, device=idx.device)).bool()
        attn_mask = ~attn_mask  # MHA expects True for masked positions

        x, _ = self.attn(x, x, x, attn_mask=attn_mask)              # [B, T, d_model]
        x = self.ln2(x)
        x = x + self.ffn(x)                                         # Residual connection
        logits = self.output_proj(x)                                # [B, T, vocab_size]
        return logits
