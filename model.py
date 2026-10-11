"""Character-level GPT (decoder-only Transformer).

Architecture matches the one in experiments/02_gpt_v2.py and the weights in
checkpoints/mythology_model_v2.pt (a bare state_dict; v1 is mythology_model.pt). Based on Andrej Karpathy's
"Let's build GPT" video / nanoGPT.
"""
import os
from dataclasses import dataclass
from typing import Iterator, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(ROOT, 'data', 'input.txt')
DEFAULT_CHECKPOINT = os.path.join(ROOT, 'checkpoints', 'mythology_model_v2.pt')


@dataclass
class GPTConfig:
    vocab_size: int = 90
    block_size: int = 256
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2


class CharTokenizer:
    """Character-level tokenizer; vocab is the sorted set of chars in the corpus."""

    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def from_file(cls, path: str = DEFAULT_DATA) -> 'CharTokenizer':
        with open(path, 'r', encoding='utf-8') as f:
            return cls(f.read())

    def encode(self, s: str, skip_unknown: bool = False) -> list:
        if skip_unknown:
            return [self.stoi[c] for c in s if c in self.stoi]
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return ''.join(self.itos[int(i)] for i in ids)


class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, config: GPTConfig, head_size: int):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, config: GPTConfig, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
        self.proj = nn.Linear(head_size * config.n_head, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedFoward(nn.Module):
    """Linear layer followed by a non-linearity."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.sa = MultiHeadAttention(config, head_size)
        self.ffwd = FeedFoward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self, config: Optional[GPTConfig] = None):
        super().__init__()
        self.config = config or GPTConfig()
        c = self.config
        self.token_embedding_table = nn.Embedding(c.vocab_size, c.n_embd)
        self.position_embedding_table = nn.Embedding(c.block_size, c.n_embd)
        self.blocks = nn.Sequential(*[Block(c) for _ in range(c.n_layer)])
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens: int, temperature: float = 1.0,
                        top_k: Optional[int] = None) -> Iterator[torch.Tensor]:
        """Yield one sampled token (shape (B, 1)) at a time.

        temperature=1.0 and top_k=None reproduces plain multinomial sampling.
        """
        block_size = self.config.block_size
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None and 0 < top_k < logits.size(-1):
                v, _ = torch.topk(logits, top_k)
                logits = logits.masked_fill(logits < v[:, [-1]], float('-inf'))
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            yield idx_next

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0,
                 top_k: Optional[int] = None):
        for tok in self.generate_stream(idx, max_new_tokens, temperature, top_k):
            idx = torch.cat((idx, tok), dim=1)
        return idx

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def load_model(checkpoint: str = DEFAULT_CHECKPOINT, data_path: str = DEFAULT_DATA,
               device: str = 'cpu'):
    """Load tokenizer + model from a state_dict checkpoint. Returns (model, tokenizer)."""
    tok = CharTokenizer.from_file(data_path)
    model = GPTLanguageModel(GPTConfig(vocab_size=tok.vocab_size))
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, tok
