# Mini-GPT from Scratch

Build and train a small decoder-only Transformer language model **from random initialization** using PyTorch.

## Learning goals

- Implement causal multi-head self-attention, feed-forward layers, and pre-LayerNorm Transformer blocks.
- Train a byte-pair encoding (BPE) tokenizer and inspect tokenization results.
- Train with next-token cross entropy; track validation loss and perplexity.
- Generate text with temperature and top-k sampling.
- Compare architectural and training choices through reproducible experiments.

## Initial specification

| Setting | Value |
|---|---|
| Dataset | TinyStories (subset for initial experiment) |
| Layers | 4 |
| Embedding dimension | 256 |
| Attention heads | 4 |
| FFN hidden dimension | 1024 |
| Context length | 256 |
| Vocabulary size | 8,000 (BPE) |
| Positional embeddings | Learned |
| Layer norm | Pre-LN |
| Activation | GELU |
| Dropout | 0.1 |
| Weight tying | Yes |
| Optimizer | AdamW |
| Initial learning rate | 3e-4 |

*This is a starting point, not a finalized experiment configuration.*

## Setup

Python 3.10+ recommended. Install a CUDA-enabled PyTorch build matching your GPU first if needed; see the official PyTorch installation instructions.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## Project layout

```text
configs/          Model and training settings
src/minigpt/      Model, tokenizer, and training components (to implement)
scripts/          Data preparation, training, and text generation (to implement)
tests/            Unit tests for model behavior
```

## Milestones

- [ ] Implement and verify causal masking and tensor shapes.
- [ ] Implement GPT model and overfit a tiny minibatch.
- [ ] Prepare TinyStories subset and BPE tokenizer.
- [ ] Train baseline, track training/validation curves.
- [ ] Generate text and evaluate qualitatively.
- [ ] Investigate improvements (KV cache, RoPE, LoRA).

## Responsible dataset usage

Check the dataset license and terms before redistribution. Do not commit raw corpora, checkpoints, API keys, or experiment artifacts to the repository.

## Status

Initial scaffold. Core model and scripts are intentionally left as exercises.
