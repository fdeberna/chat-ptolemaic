from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    repetition_penalty: float,
) -> torch.Tensor:
    if repetition_penalty <= 1.0:
        return logits

    for batch_idx in range(logits.size(0)):
        unique_tokens = torch.unique(generated_tokens[batch_idx])
        token_logits = logits[batch_idx, unique_tokens]
        token_logits = torch.where(
            token_logits > 0,
            token_logits / repetition_penalty,
            token_logits * repetition_penalty,
        )
        logits[batch_idx, unique_tokens] = token_logits
    return logits


def get_banned_tokens_ngram(tokens: list[int], ngram_size: int) -> set[int]:
    if ngram_size <= 0:
        return set()
    if len(tokens) + 1 < ngram_size:
        return set()
    if ngram_size == 1:
        return set(tokens)

    ngram_prefixes: dict[tuple[int, ...], set[int]] = {}
    for i in range(len(tokens) - ngram_size + 1):
        ngram = tuple(tokens[i : i + ngram_size])
        prefix = ngram[:-1]
        ngram_prefixes.setdefault(prefix, set()).add(ngram[-1])

    current_prefix = tuple(tokens[-(ngram_size - 1) :])
    return ngram_prefixes.get(current_prefix, set())


def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = -float("inf")
    return logits


def has_repeated_ngram(tokens: list[int], ngram_size: int) -> bool:
    if len(tokens) < ngram_size:
        return False

    last_ngram = tuple(tokens[-ngram_size:])
    for i in range(len(tokens) - ngram_size):
        if tuple(tokens[i : i + ngram_size]) == last_ngram:
            return True
    return False


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True
    weight_tying: bool = True
    layer_norm_eps: float = 1e-5
    pad_token_id: Optional[int] = None


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, channels = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(bsz, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)
        q = q.view(bsz, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.weight_tying:
            self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        eps: float = 1e-8,
    ) -> torch.optim.Optimizer:
        decay_params = []
        no_decay_params = []
        module_lookup = dict(self.named_modules())
        for param_name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            module_name = ".".join(param_name.split(".")[:-1])
            module = module_lookup.get(module_name, self)
            if param_name.endswith("bias"):
                no_decay_params.append(param)
            elif isinstance(module, (nn.LayerNorm, nn.Embedding)):
                no_decay_params.append(param)
            elif param_name.endswith("weight") and isinstance(module, nn.Linear):
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps)
        return optimizer

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len = idx.size()
        if seq_len > self.config.block_size:
            raise ValueError(f"Cannot forward sequence of length {seq_len}, block size is {self.config.block_size}")

        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            ignore_index = self.config.pad_token_id if self.config.pad_token_id is not None else -100
            loss = F.cross_entropy(
                logits.view(batch_size * seq_len, -1),
                targets.view(batch_size * seq_len),
                ignore_index=ignore_index,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be >= 1.0")
        if not 0 < top_p <= 1.0:
            raise ValueError("top_p must be in the range (0, 1]")
        if no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size must be >= 0")

        prompt_length = idx.size(1)

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if repetition_penalty > 1.0 and idx.size(1) > prompt_length:
                generated_part = idx[:, prompt_length:]
                logits = apply_repetition_penalty(logits, generated_part, repetition_penalty)

            if no_repeat_ngram_size > 0:
                for batch_idx in range(idx.size(0)):
                    banned_tokens = get_banned_tokens_ngram(idx[batch_idx].tolist(), no_repeat_ngram_size)
                    if banned_tokens:
                        logits[batch_idx, list(banned_tokens)] = -float("inf")

            if top_k is not None:
                current_top_k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, current_top_k)
                threshold = values[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < threshold,
                    torch.full_like(logits, -float("inf")),
                    logits,
                )

            logits = top_p_filtering(logits, top_p)

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            if eos_token_id is not None and torch.all(idx_next == eos_token_id):
                break
        return idx
