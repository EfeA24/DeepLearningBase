# Purpose: Provides a self-contained small GPT/Llama-style causal decoder LM (embeddings, decoder stack, final norm, LM head) with KV caching.
"""Self-contained reference causal decoder language model.

Architecture (defaults): token embedding -> N x [RMSNorm -> GQA causal attention with RoPE
-> residual -> RMSNorm -> SwiGLU -> residual] -> RMSNorm -> LM head (optionally tied).
Minimal components are defined locally (``_RMSNorm``, ``_SwiGLU``, ``_Attention``, ``_Block``)
so this single file can be copied into a project; the separately reusable versions live in
models/normalization, models/feedforward, models/attention. A custom stack can be injected
with ``block_factory(layer_idx) -> nn.Module`` following the block contract of
``models/blocks/decoder_block.py``; such blocks must handle positional information
themselves (e.g. an injected RoPE) unless ``position_embedding="learned"``.

forward(input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False)
    input_ids:       LongTensor (B, T) -- only the NEW tokens when a cache is supplied.
    attention_mask:  optional, polarity True/1 = real token:
                       (B, T_past + T) key padding mask over cached + new tokens, or
                       (B, 1, T, T_past + T) bool final mask (caller includes causality,
                       e.g. packed block-diagonal masks; True = allowed).
                     None = every token is real (pure causal attention).
    position_ids:    optional (B, T) absolute positions. Default: from a 2D mask as
                     ``cumsum(mask) - 1`` (correct for LEFT padding), else
                     ``T_past + arange(T)``.
    past_key_values: None or a list (one entry per layer) of ``(k, v)`` tuples or cache
                     objects (``seq_len``/``update``), k/v shaped (B, n_kv_heads, T_past, head_dim),
                     keys stored after RoPE.
    returns DecoderOutput(logits (B, T, vocab_size) raw logits,
                          past_key_values (list per layer) if use_cache else None,
                          hidden_states (B, T, d_model) after the final norm).

Label shift: this module does not compute a loss. With labels aligned to input_ids
(template convention) the loss compares ``logits[:, :-1]`` with ``labels[:, 1:]``, ignoring
-100 (see training/causal_lm_trainer.py).

Requires: torch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class DecoderConfig:
    """Hyper-parameters of :class:`TransformerDecoderLM`."""

    vocab_size: int
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: Optional[int] = None
    max_positions: int = 1024
    ffn_hidden_dim: Optional[int] = None
    multiple_of: int = 64
    dropout: float = 0.0
    norm_eps: float = 1e-6
    rope_base: float = 10000.0
    position_embedding: str = "rope"
    tie_embeddings: bool = True
    bias: bool = False
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if min(self.vocab_size, self.d_model, self.n_layers, self.n_heads, self.n_kv_heads, self.max_positions) <= 0:
            raise ValueError("vocab_size, d_model, n_layers, n_heads, n_kv_heads, max_positions must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        if self.position_embedding not in ("rope", "learned"):
            raise ValueError("position_embedding must be 'rope' or 'learned'")
        if self.position_embedding == "rope" and (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("RoPE requires an even head_dim")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


@dataclass
class DecoderOutput:
    logits: torch.Tensor
    past_key_values: Optional[List[Any]] = None
    hidden_states: Optional[torch.Tensor] = None


def _past_len(past_kv: Any) -> int:
    if past_kv is None:
        return 0
    if hasattr(past_kv, "update"):
        return int(past_kv.seq_len)
    return int(past_kv[0].shape[2])


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.to(x.dtype) * self.weight.to(x.dtype)


class _SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, bias: bool) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _Attention(nn.Module):
    """GQA causal self-attention; expects attention_mask None or 4D (normalized by the model)."""

    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout
        self.use_rope = cfg.position_embedding == "rope"
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=cfg.bias)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=cfg.bias)
        if self.use_rope:
            inv_freq = 1.0 / (cfg.rope_base ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float32) / cfg.head_dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        freqs = position_ids.to(torch.float32).unsqueeze(-1) * self.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(1).to(x.dtype)
        sin = emb.sin().unsqueeze(1).to(x.dtype)
        return x * cos + _rotate_half(x) * sin

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_kv: Any = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Any]:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        past = _past_len(past_kv)
        if self.use_rope:
            if position_ids is None:
                position_ids = torch.arange(past, past + t, device=x.device).unsqueeze(0).expand(b, t)
            q = self._rope(q, position_ids)
            k = self._rope(k, position_ids)

        if past_kv is None:
            k_all, v_all = k, v
            present = (k, v) if use_cache else None
        elif hasattr(past_kv, "update"):
            k_all, v_all = past_kv.update(k, v)
            present = past_kv
        else:
            k_all = torch.cat((past_kv[0].to(k.dtype), k), dim=2)
            v_all = torch.cat((past_kv[1].to(v.dtype), v), dim=2)
            present = (k_all, v_all) if use_cache else None

        kv_len = k_all.shape[2]
        is_causal = False
        if attention_mask is None:
            if t > 1 and kv_len == t:
                is_causal = True
            elif t > 1:
                attention_mask = torch.ones(t, kv_len, dtype=torch.bool, device=x.device).tril(kv_len - t)
        elif attention_mask.is_floating_point():
            attention_mask = attention_mask.to(q.dtype)

        n_rep = self.n_heads // self.n_kv_heads
        if n_rep > 1:
            k_all = k_all.repeat_interleave(n_rep, dim=1)
            v_all = v_all.repeat_interleave(n_rep, dim=1)
        out = F.scaled_dot_product_attention(
            q, k_all, v_all,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1)), present


class _Block(nn.Module):
    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__()
        hidden = cfg.ffn_hidden_dim
        if hidden is None:
            hidden = int(2 * 4 * cfg.d_model / 3)
            hidden = cfg.multiple_of * ((hidden + cfg.multiple_of - 1) // cfg.multiple_of)
        self.attn_norm = _RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attention = _Attention(cfg)
        self.ffn_norm = _RMSNorm(cfg.d_model, cfg.norm_eps)
        self.feed_forward = _SwiGLU(cfg.d_model, hidden, cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_kv: Any = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Any]:
        attn_out, present = self.attention(
            self.attn_norm(hidden_states), attention_mask, position_ids, past_kv, use_cache
        )
        hidden_states = hidden_states + self.dropout(attn_out)
        hidden_states = hidden_states + self.dropout(self.feed_forward(self.ffn_norm(hidden_states)))
        return hidden_states, present


class TransformerDecoderLM(nn.Module):
    """Small GPT/Llama-style causal LM (see module docstring for the full contract).

    Args:
        config: :class:`DecoderConfig`.
        block_factory: Optional ``layer_idx -> nn.Module`` overriding the local block.
    """

    def __init__(
        self, config: DecoderConfig, block_factory: Optional[Callable[[int], nn.Module]] = None
    ) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = (
            nn.Embedding(config.max_positions, config.d_model)
            if config.position_embedding == "learned"
            else None
        )
        self.drop = nn.Dropout(config.dropout)
        factory = block_factory if block_factory is not None else (lambda _idx: _Block(config))
        self.layers = nn.ModuleList([factory(i) for i in range(config.n_layers)])
        self.norm = _RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self._init_weights()
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

    def _init_weights(self) -> None:
        std = self.config.init_std
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
        residual_std = std / math.sqrt(2 * self.config.n_layers)
        for name, param in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=residual_std)

    def _position_ids(
        self, attention_mask: Optional[torch.Tensor], b: int, t: int, past: int, device: torch.device
    ) -> torch.Tensor:
        if attention_mask is not None and attention_mask.dim() == 2:
            positions = attention_mask.long().cumsum(-1) - 1
            positions = positions.clamp(min=0)
            return positions[:, past:]
        return torch.arange(past, past + t, device=device).unsqueeze(0).expand(b, t)

    @staticmethod
    def _expand_mask(
        attention_mask: Optional[torch.Tensor], b: int, t: int, kv_len: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if attention_mask.dim() == 2:
            if tuple(attention_mask.shape) != (b, kv_len):
                raise ValueError(
                    f"2D attention_mask must be (B, T_past + T) = {(b, kv_len)}, got {tuple(attention_mask.shape)}"
                )
            causal = torch.ones(t, kv_len, dtype=torch.bool, device=device).tril(kv_len - t)
            mask = causal[None, None] & attention_mask.to(device=device, dtype=torch.bool)[:, None, None, :]
        elif attention_mask.dim() == 4:
            if tuple(attention_mask.shape[-2:]) != (t, kv_len):
                raise ValueError(f"4D attention_mask must end with {(t, kv_len)}, got {tuple(attention_mask.shape)}")
            if attention_mask.dtype != torch.bool:
                raise TypeError("4D attention_mask must be bool (True = allowed)")
            mask = attention_mask.to(device)
        else:
            raise ValueError(f"attention_mask must be 2D or 4D, got {attention_mask.dim()}D")
        return mask | ~mask.any(dim=-1, keepdim=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Any]] = None,
        use_cache: bool = False,
    ) -> DecoderOutput:
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be (B, T), got {tuple(input_ids.shape)}")
        b, t = input_ids.shape
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError(f"past_key_values must have {len(self.layers)} entries, got {len(past_key_values)}")
        past = _past_len(past_key_values[0]) if past_key_values is not None else 0
        kv_len = past + t
        if position_ids is None:
            position_ids = self._position_ids(attention_mask, b, t, past, input_ids.device)
        if self.pos_emb is not None and int(position_ids.max()) >= self.config.max_positions:
            raise ValueError(f"positions exceed max_positions={self.config.max_positions}")

        hidden = self.tok_emb(input_ids)
        if self.pos_emb is not None:
            hidden = hidden + self.pos_emb(position_ids)
        hidden = self.drop(hidden)
        mask = self._expand_mask(attention_mask, b, t, kv_len, input_ids.device)

        presents: Optional[List[Any]] = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            hidden, present = layer(
                hidden, attention_mask=mask, position_ids=position_ids, past_kv=layer_past, use_cache=use_cache
            )
            if presents is not None:
                presents.append(present)
        hidden = self.norm(hidden)
        return DecoderOutput(logits=self.lm_head(hidden), past_key_values=presents, hidden_states=hidden)

    def generation_step(
        self,
        input_ids: torch.Tensor,
        past: Optional[List[Any]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Any]]]:
        """Adapter for the generation contract ``model_fn(input_ids, past, attention_mask) -> (logits, past)``."""
        out = self.forward(input_ids, attention_mask=attention_mask, past_key_values=past, use_cache=True)
        return out.logits, out.past_key_values
