"""Build a decoder-only Transformer here.

Implementation exercises:
  1. CausalSelfAttention
  2. FeedForward
  3. TransformerBlock (pre-LayerNorm)
  4. GPTLanguageModel (token/position embeddings, output projection)

Suggested contract: forward(input_ids: Tensor[B,T], targets: Tensor[B,T] | None)
returns logits Tensor[B,T,V] and optional scalar cross-entropy loss.
"""
