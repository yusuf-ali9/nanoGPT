# nanoGPT

A decoder-only transformer language model, built from scratch in PyTorch.

This project is the final stage of a journey through the defining machine-learning
architectures of the last century — perceptrons, convolutional networks, recurrent
networks, and finally the transformer. The goal here was not to use a transformer,
but to understand it: to build one from first principles, line by line, and know
exactly where every piece comes from and why it exists.

The transformer, introduced in *Attention Is All You Need* (Vaswani et al., 2017),
is the architecture underneath modern large language models such as **ChatGPT**,
**Claude**, and **Gemini**. This repository implements the same core ideas at
miniature scale: a character-level GPT trained on Shakespeare that learns to
generate text one character at a time.

## What it does

The model is trained on the [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
corpus (`input.txt`, ~1.1M characters). It reads the raw text, learns a character
vocabulary, and trains autoregressively to predict the next character given the
preceding context. After training it samples new, Shakespeare-flavored text from
scratch.

## Architecture

Everything lives in a single file, [`bigram.py`](bigram.py). The name is a nod to
the starting point of the build — a trivial bigram lookup — which is then grown
into a full GPT. The final model implements, from scratch:

- **Token & positional embeddings** — each character and each position in the
  context window gets a learned vector.
- **Multi-head self-attention** — each `Head` computes queries, keys, and values,
  applies scaled dot-product attention with a causal (triangular) mask so a
  position can only attend to earlier positions, and `MultiHead` runs several
  heads in parallel and projects the result.
- **Feed-forward networks** — a two-layer MLP that expands the embedding by 4×
  (as in the original paper) with a ReLU non-linearity before compressing back.
- **Transformer blocks** — `Block` stacks attention and feed-forward sublayers
  with **pre-layer-norm** and **residual connections**, repeated `n_layer` times.
- **Dropout** — regularization throughout attention and the feed-forward layers.
- **Final layer norm + linear head** — projects the hidden state onto the
  vocabulary to produce next-character logits.
- **Autoregressive generation** — `generate()` samples characters one at a time,
  feeding each prediction back in, cropped to the block size.

## Hyperparameters

| Parameter        | Value | Meaning                                   |
| ---------------- | ----- | ----------------------------------------- |
| `batch_size`     | 16    | sequences per training step               |
| `block_size`     | 64    | context length (characters)               |
| `n_embd`         | 128   | embedding / hidden dimension              |
| `num_heads`      | 4     | attention heads per block                 |
| `n_layer`        | 6     | number of transformer blocks              |
| `dropout`        | 0.2   | dropout probability                       |
| `learning_rate`  | 1e-3  | AdamW learning rate                       |
| `max_iters`      | 1000  | training iterations                       |

The model runs on GPU (`cuda`) automatically when available, otherwise on CPU.

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/)

```bash
pip install torch
```

## Usage

```bash
python bigram.py
```

This trains the model, printing training and validation loss periodically, then
prints 500 characters of generated text at the end.

## Acknowledgements

Inspired by Andrej Karpathy's ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY)
and the *Attention Is All You Need* paper. Written from scratch as a learning
exercise to understand the transformer inside and out.
