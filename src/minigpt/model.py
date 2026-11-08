"""Build a decoder-only Transformer here.

Implementation exercises:
  1. CausalSelfAttention
  2. FeedForward
  3. TransformerBlock (pre-LayerNorm)
  4. GPTLanguageModel (token/position embeddings, output projection)

Suggested contract: forward(input_ids: Tensor[B,T], targets: Tensor[B,T] | None)
returns logits Tensor[B,T,V] and optional scalar cross-entropy loss.
"""

import math
import torch
from torch import nn
import torch.nn.functional as F


class GPTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size,embed_dim)
        self.positional_embedding = nn.Embedding(block_size, embed_dim)






    def forward(self, input_ids):
        B, T = input_ids.shape

        if T > self.positional_embedding.num_embeddings:
            raise ValueError(
                f"Sequence length {T} exceeds block_size"
                f"{self.positional_embedding}"
            )
        emb1 = self.token_embedding(input_ids)

        positions = torch.arange(T, device=input_ids.device)
        emb2 = self.positional_embedding(positions)

        outputs = emb1 + emb2

        return outputs


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4,bias=False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
      

        self.W_q = nn.LazyLinear(embed_dim, bias=bias)
        self.W_k = nn.LazyLinear(embed_dim, bias=bias)
        self.W_v = nn.LazyLinear(embed_dim, bias=bias)

        self.w_o = nn.LazyLinear(embed_dim, bias=bias)


    def forward(self, x):
        B, T, C = x.shape

        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        queries = queries.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (queries@keys.transpose(-2, -1) / math.sqrt(self.head_dim))
        mask = torch.tril(
            torch.ones(T, T, device=x.device, dtype=torch.bool)
        )

        scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        output = attn_weights @ values

        output = output.transpose(1, 2)
        output = output.contiguous().reshape(B, T, C)

        output = self.w_o(output)

        return output


        
class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.GELU(),
            nn.Linear(4*embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn(x)
    

class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size, num_heads, num_blocks, dropout=0.1):
        super().__init__()

        self.embedding = GPTEmbedding(
            vocab_size=vocab_size, embed_dim=embed_dim, block_size=block_size
        )

        self.blocks = nn.Sequential(
            *[
                TransformerDecoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout
                )
                for _ in range(num_blocks)
            ]
        )

        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(
            embed_dim, vocab_size, bias=False
        )     

    def forward(self, input_ids, targets=None):
        x = self.embedding(input_ids)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        B, T, V = logits.shape

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(B*T, V),
                targets.reshape(B*T)
            )

        return logits, loss


            



if __name__ == "__main__":
    model = GPTLanguageModel(
        vocab_size=8000,
        embed_dim=256,
        block_size=256,
        num_heads=4,
        num_blocks=4,
        dropout=0.1
    )

    input_ids = torch.randint(0, 8000, (4, 128))
    targets = torch.randint(0, 8000, (4, 128))

    # 1. 学習時のForward
    logits, loss = model(input_ids, targets)

    assert logits.shape == (4, 128, 8000)
    assert loss.ndim == 0
    print("Forward test passed!")

    # 2. Backward
    loss.backward()
    assert all(
        p.grad is not None
        for p in model.parameters()
        if p.requires_grad
    )
    print("Backward test passed!")

    # 3. 推論（targetsなし）
    model.eval()
    with torch.no_grad():
        logits, loss = model(input_ids)

    assert logits.shape == (4, 128, 8000)
    assert loss is None
    print("Inference test passed!")

    print("Number of parameters:",
          sum(p.numel() for p in model.parameters()))