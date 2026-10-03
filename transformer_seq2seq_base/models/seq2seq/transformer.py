# Purpose: Provides a self-contained encoder-decoder Transformer reference for text-to-text tasks returning raw target-vocab logits with an encode()/decode() split for generation.
"""Self-contained seq2seq Transformer (Vaswani et al. style, pre-norm by default).

This file defines its own minimal components so it can be copied alone; the modular files
under models/ provide the same pieces separately.

Contracts (batch-first, int64 ids):
    forward(input_ids (B, S), decoder_input_ids (B, T), attention_mask=None (B, S),
            decoder_attention_mask=None (B, T)) -> logits (B, T, tgt_vocab_size), raw (no softmax).
    encode(input_ids, attention_mask=None) -> memory (B, S, d_model).
    decode(decoder_input_ids (B, T_new), memory, memory_attention_mask=None,
           decoder_attention_mask=None (B, past + T_new), past_key_values=None,
           use_cache=False) -> (logits (B, T_new, V), present_key_values or None).
Masks: padding masks are 1/True = real token, 0/False = pad. When ``attention_mask`` is
None in encode/forward it is derived from ``src_pad_token_id``. Internally boolean
attention masks use True = may attend. ``decoder_input_ids`` = shift_right(labels) (see
data/sequences/target_shift.py); labels use -100 for padding and are not consumed here.
Cache: one dict per decoder layer {"self": (k, v) (B, H, past, Dh), "cross": (k, v) (B, H, S, Dh)};
with a cache pass ONLY the new decoder tokens; positions continue at the cached length.

Requires: torch only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

KeyValue = Tuple[torch.Tensor, torch.Tensor]
LayerCache = Dict[str, KeyValue]


@dataclass
class Seq2SeqTransformerConfig:
    """Hyperparameters; all vocab/pad ids are explicit."""

    src_vocab_size: int
    tgt_vocab_size: int
    src_pad_token_id: int
    tgt_pad_token_id: int
    d_model: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    activation: str = "relu"
    norm_first: bool = True
    max_position: int = 1024
    layer_norm_eps: float = 1e-5
    scale_embeddings: bool = True
    share_embeddings: bool = False
    tie_output_projection: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by num_heads={self.num_heads}")
        if self.activation not in ("relu", "gelu"):
            raise ValueError(f"activation must be 'relu' or 'gelu', got {self.activation!r}")
        if min(self.src_vocab_size, self.tgt_vocab_size, self.num_encoder_layers,
               self.num_decoder_layers, self.d_ff, self.max_position) < 1:
            raise ValueError("vocab sizes, layer counts, d_ff and max_position must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0 <= self.src_pad_token_id < self.src_vocab_size:
            raise ValueError("src_pad_token_id outside source vocabulary")
        if not 0 <= self.tgt_pad_token_id < self.tgt_vocab_size:
            raise ValueError("tgt_pad_token_id outside target vocabulary")
        if self.share_embeddings and (
            self.src_vocab_size != self.tgt_vocab_size or self.src_pad_token_id != self.tgt_pad_token_id
        ):
            raise ValueError("share_embeddings requires identical source/target vocab size and pad id")


def _sinusoidal_table(max_len: int, d_model: int) -> torch.Tensor:
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    table = torch.zeros(max_len, d_model)
    table[:, 0::2] = torch.sin(position * div_term)
    table[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
    return table


class _Attention(nn.Module):
    """Minimal MHA: boolean masks with True = may attend; uses SDPA."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

    def project_kv(self, source: torch.Tensor) -> KeyValue:
        return self._split(self.k_proj(source)), self._split(self.v_proj(source))

    def forward(self, query: torch.Tensor, kv: KeyValue, mask: Optional[torch.Tensor]) -> torch.Tensor:
        q = self._split(self.q_proj(query))
        k, v = kv
        if mask is not None:
            mask = mask | ~mask.any(dim=-1, keepdim=True)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        b, _, t, _ = out.shape
        return self.out_proj(out.transpose(1, 2).reshape(b, t, -1))


class _FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, activation: str, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class _EncoderLayer(nn.Module):
    def __init__(self, cfg: Seq2SeqTransformerConfig) -> None:
        super().__init__()
        self.norm_first = cfg.norm_first
        self.self_attn = _Attention(cfg.d_model, cfg.num_heads, cfg.dropout)
        self.ffn = _FeedForward(cfg.d_model, cfg.d_ff, cfg.activation, cfg.dropout)
        self.norm1 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.norm2 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.dropout)

    def _sa(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        return self.dropout(self.self_attn(x, self.self_attn.project_kv(x), mask))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.norm_first:
            x = x + self._sa(self.norm1(x), mask)
            return x + self.dropout(self.ffn(self.norm2(x)))
        x = self.norm1(x + self._sa(x, mask))
        return self.norm2(x + self.dropout(self.ffn(x)))


class _DecoderLayer(nn.Module):
    def __init__(self, cfg: Seq2SeqTransformerConfig) -> None:
        super().__init__()
        self.norm_first = cfg.norm_first
        self.self_attn = _Attention(cfg.d_model, cfg.num_heads, cfg.dropout)
        self.cross_attn = _Attention(cfg.d_model, cfg.num_heads, cfg.dropout)
        self.ffn = _FeedForward(cfg.d_model, cfg.d_ff, cfg.activation, cfg.dropout)
        self.norm1 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.norm2 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.norm3 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_mask: torch.Tensor,
        memory_mask: Optional[torch.Tensor],
        past: Optional[LayerCache],
    ) -> Tuple[torch.Tensor, LayerCache]:
        h = self.norm1(x) if self.norm_first else x
        k, v = self.self_attn.project_kv(h)
        if past is not None:
            k = torch.cat([past["self"][0], k], dim=2)
            v = torch.cat([past["self"][1], v], dim=2)
        h = self.dropout(self.self_attn(h, (k, v), self_mask))
        x = x + h if self.norm_first else self.norm1(x + h)

        h = self.norm2(x) if self.norm_first else x
        cross_kv = past["cross"] if past is not None else self.cross_attn.project_kv(memory)
        h = self.dropout(self.cross_attn(h, cross_kv, memory_mask))
        x = x + h if self.norm_first else self.norm2(x + h)

        h = self.norm3(x) if self.norm_first else x
        h = self.dropout(self.ffn(h))
        x = x + h if self.norm_first else self.norm3(x + h)
        return x, {"self": (k, v), "cross": cross_kv}


class Seq2SeqTransformer(nn.Module):
    """Encoder-decoder Transformer returning raw target-vocabulary logits."""

    def __init__(self, config: Seq2SeqTransformerConfig) -> None:
        super().__init__()
        cfg = config
        self.config = cfg
        self.embed_scale = math.sqrt(cfg.d_model) if cfg.scale_embeddings else 1.0
        self.src_embed = nn.Embedding(cfg.src_vocab_size, cfg.d_model, padding_idx=cfg.src_pad_token_id)
        self.tgt_embed = (
            self.src_embed
            if cfg.share_embeddings
            else nn.Embedding(cfg.tgt_vocab_size, cfg.d_model, padding_idx=cfg.tgt_pad_token_id)
        )
        self.register_buffer("position_table", _sinusoidal_table(cfg.max_position, cfg.d_model), persistent=False)
        self.embed_dropout = nn.Dropout(cfg.dropout)
        self.encoder_layers = nn.ModuleList(_EncoderLayer(cfg) for _ in range(cfg.num_encoder_layers))
        self.decoder_layers = nn.ModuleList(_DecoderLayer(cfg) for _ in range(cfg.num_decoder_layers))
        final = (lambda: nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)) if cfg.norm_first else nn.Identity
        self.encoder_norm = final()
        self.decoder_norm = final()
        self.output_projection = nn.Linear(cfg.d_model, cfg.tgt_vocab_size, bias=False)
        self.reset_parameters()
        if cfg.tie_output_projection:
            self.output_projection.weight = self.tgt_embed.weight

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for embedding in {self.src_embed, self.tgt_embed}:
            nn.init.normal_(embedding.weight, mean=0.0, std=self.config.d_model ** -0.5)
            with torch.no_grad():
                embedding.weight[embedding.padding_idx].zero_()

    def _embed(self, embedding: nn.Embedding, ids: torch.Tensor, offset: int = 0) -> torch.Tensor:
        if ids.dim() != 2:
            raise ValueError(f"token ids must be (B, T), got {tuple(ids.shape)}")
        length = ids.size(1)
        if offset + length > self.config.max_position:
            raise ValueError(f"positions up to {offset + length} exceed max_position={self.config.max_position}")
        x = embedding(ids) * self.embed_scale
        x = x + self.position_table[offset : offset + length].to(dtype=x.dtype)
        return self.embed_dropout(x)

    def encode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Source ids (B, S) -> memory (B, S, d_model)."""
        if attention_mask is None:
            attention_mask = input_ids != self.config.src_pad_token_id
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids shape (B, S)")
        mask = attention_mask.to(torch.bool)[:, None, None, :]
        x = self._embed(self.src_embed, input_ids)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return self.encoder_norm(x)

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[LayerCache]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[LayerCache, ...]]]:
        """Decoder ids (new tokens only when cached) -> (logits (B, T_new, V), presents)."""
        if past_key_values is not None and len(past_key_values) != len(self.decoder_layers):
            raise ValueError(f"expected {len(self.decoder_layers)} layer caches, got {len(past_key_values)}")
        past_len = int(past_key_values[0]["self"][0].size(2)) if past_key_values else 0
        new_len = decoder_input_ids.size(1)
        total = past_len + new_len
        self_mask = torch.ones((new_len, total), dtype=torch.bool, device=decoder_input_ids.device)
        self_mask = torch.tril(self_mask, diagonal=past_len)[None, None]
        if decoder_attention_mask is not None:
            if decoder_attention_mask.shape != (decoder_input_ids.size(0), total):
                raise ValueError(f"decoder_attention_mask must be (B, past+T)=(B, {total})")
            self_mask = self_mask & decoder_attention_mask.to(torch.bool)[:, None, None, :]
        memory_mask = None
        if memory_attention_mask is not None:
            if memory_attention_mask.shape != memory.shape[:2]:
                raise ValueError("memory_attention_mask must be (B, S) matching memory")
            memory_mask = memory_attention_mask.to(torch.bool)[:, None, None, :]

        x = self._embed(self.tgt_embed, decoder_input_ids, offset=past_len)
        presents: List[LayerCache] = []
        for i, layer in enumerate(self.decoder_layers):
            layer_past = past_key_values[i] if past_key_values else None
            x, present = layer(x, memory, self_mask, memory_mask, layer_past)
            presents.append(present)
        logits = self.output_projection(self.decoder_norm(x))
        return logits, (tuple(presents) if use_cache else None)

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Teacher-forced forward pass -> logits (B, T, tgt_vocab_size)."""
        if attention_mask is None:
            attention_mask = input_ids != self.config.src_pad_token_id
        memory = self.encode(input_ids, attention_mask)
        logits, _ = self.decode(
            decoder_input_ids, memory, memory_attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        return logits
