from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int
    hidden_size: int
    n_layers: int
    n_heads: int
    n_kv_heads: Optional[int] = None
    hidden_dim: Optional[int] = None
    norm_eps: float = 1e-06
    rope_base: int = 10000
    bias: bool = False
    dropout: float = 0.0
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.hidden_size % self.n_heads != 0:
            raise ValueError(
                f'hidden_size must be divisible by n_heads, '
                f'got hidden_size={self.hidden_size}, n_heads={self.n_heads}'
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f'n_heads must be divisible by n_kv_heads, '
                f'got n_heads={self.n_heads}, n_kv_heads={self.n_kv_heads}'
            )
        if self.hidden_dim is None:
            self.hidden_dim = 4 * self.hidden_size
        if self.head_dim % 2 != 0:
            raise ValueError(
                f'head_dim must be even for this simple RoPE implementation, '
                f'got head_dim={self.head_dim}'
            )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.n_heads

    @property
    def n_rep(self) -> int:
        assert self.n_kv_heads is not None
        return self.n_heads // self.n_kv_heads

    @property
    def max_position_embeddings(self) -> int:
        return self.block_size


# RMS(x) = sqrt(mean(x^2) + eps)
# RMSNorm(x) = weight * x / RMS(x)
class RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-06) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        rms_square = torch.mean(x * x, dim=-1, keepdim=True)
        x_normed = x / torch.sqrt(rms_square + self.eps)
        out = x_normed * self.weight.float()
        return out.to(dtype=input_dtype)


def build_rope_cache(
    block_size: int,
    head_dim: int,
    base: int = 10000,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError(f'head_dim must be even for RoPE, got {head_dim}')

    inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    positions = torch.arange(block_size, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    freqs = torch.cat([freqs, freqs], dim=-1)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    return (cos, sin)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    head_dim = x.size(-1)
    half = head_dim // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)

# q_i' = q_i*cos(mθ) - q_(i+D/2)*sin(mθ)
# q_(i+D/2)' = q_i*sin(mθ) + q_(i+D/2)*cos(mθ)
def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if cos.dim() != 3 or sin.dim() != 3:
        raise ValueError(f'cos and sin must be 3D: [1, T, head_dim], got cos={cos.shape}, sin={sin.shape}')
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    out = x * cos + rotate_half(x) * sin
    return out.to(dtype=x.dtype)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    B, n_kv_heads, T, head_dim = x.shape
    x = x[:, :, None, :, :]
    x = x.expand(B, n_kv_heads, n_rep, T, head_dim)
    x = x.reshape(B, n_kv_heads * n_rep, T, head_dim)
    return x


class KVCache(nn.Module):

    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        n_kv_heads: int,
        block_size: int,
        head_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.n_kv_heads = n_kv_heads
        self.block_size = block_size
        self.head_dim = head_dim
        self.register_buffer(
            'k_cache',
            torch.zeros(
                n_layers,
                batch_size,
                n_kv_heads,
                block_size,
                head_dim,
                device=device,
                dtype=dtype,
            ),
            persistent=False,
        )
        self.register_buffer(
            'v_cache',
            torch.zeros(
                n_layers,
                batch_size,
                n_kv_heads,
                block_size,
                head_dim,
                device=device,
                dtype=dtype,
            ),
            persistent=False,
        )

    def reset(self) -> None:
        self.k_cache.zero_()
        self.v_cache.zero_()

    def update(
        self,
        layer_idx: int,
        input_pos: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_len: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_pos.dim() != 1:
            raise ValueError(
                f'This teaching KVCache only supports 1D input_pos [T], '
                f'got input_pos.shape={input_pos.shape}'
            )
        if not 0 <= layer_idx < self.n_layers:
            raise ValueError(f'Invalid layer_idx={layer_idx}')
        B, n_kv_heads, T_new, head_dim = k.shape

        # B < self.batch_size 主要表示“实际使用量可以小于预分配容量”。
        # 它可以支持结束序列移除后的较小 batch，但不会自动处理从 batch 中间删除序列造成的缓存槽位错位。
        if B > self.batch_size:
            raise ValueError(f'Input batch size B={B} exceeds cache batch_size={self.batch_size}')
        if n_kv_heads != self.n_kv_heads:
            raise ValueError(f'Expected n_kv_heads={self.n_kv_heads}, got {n_kv_heads}')
        if head_dim != self.head_dim:
            raise ValueError(f'Expected head_dim={self.head_dim}, got {head_dim}')
        if input_pos.numel() != T_new:
            raise ValueError(
                f'input_pos length must equal T_new, '
                f'got input_pos.numel()={input_pos.numel()}, T_new={T_new}'
            )
        if cache_len is None:
            max_pos = int(input_pos.max().item())
            if max_pos >= self.block_size:
                raise ValueError(f'input_pos max={max_pos} exceeds cache block_size={self.block_size}')
            cache_len = max_pos + 1
        elif not 1 <= cache_len <= self.block_size:
            raise ValueError(
                f'cache_len must be in [1, {self.block_size}], got {cache_len}'
            )
        if self.k_cache.device != k.device or self.k_cache.dtype != k.dtype:
            self.k_cache = self.k_cache.to(device=k.device, dtype=k.dtype)
            self.v_cache = self.v_cache.to(device=v.device, dtype=v.dtype)
        input_pos = input_pos.to(device=k.device)
        layer_k_cache = self.k_cache[layer_idx, :B]
        layer_v_cache = self.v_cache[layer_idx, :B]
        layer_k_cache.index_copy_(dim=2, index=input_pos, source=k)
        layer_v_cache.index_copy_(dim=2, index=input_pos, source=v)
        k_full = layer_k_cache[:, :, :cache_len, :]
        v_full = layer_v_cache[:, :, :cache_len, :]
        return (k_full, v_full)


class CausalSelfAttention(nn.Module):

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.n_heads = config.n_heads
        assert config.n_kv_heads is not None
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = config.n_rep
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.n_heads * config.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * config.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * config.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.hidden_size, bias=config.bias)
        self.dropout_p = config.dropout

    def _shape_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = q.shape
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        return (q, k, v)

    def _should_use_causal_mask(
        self,
        T: int,
        kv_cache: Optional[KVCache],
        input_pos: Optional[torch.Tensor],
    ) -> bool:
        if kv_cache is None:
            return True
        if input_pos is None:
            raise ValueError('input_pos must be provided when kv_cache is used')
        if T == 1:
            return False
        expected = torch.arange(T, device=input_pos.device)
        if not torch.equal(input_pos, expected):
            raise NotImplementedError(
                'This teaching implementation only supports KV-cache prefill '
                'with input_pos = [0, 1, ..., T-1] when T > 1. '
                'For decode, use T=1.'
            )
        return True

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        input_pos: Optional[torch.Tensor] = None,
        cache_len: Optional[int] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        if C != self.hidden_size:
            raise ValueError(f'Expected x.size(-1)={self.hidden_size}, got {C}')
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q, k, v = self._shape_qkv(q, k, v)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if kv_cache is not None:
            if input_pos is None:
                raise ValueError('input_pos must be provided when using KV Cache')
            k, v = kv_cache.update(
                layer_idx=self.layer_idx,
                input_pos=input_pos,
                k=k,
                v=v,
                cache_len=cache_len,
            )
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        use_causal_mask = self._should_use_causal_mask(T=T, kv_cache=kv_cache, input_pos=input_pos)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=use_causal_mask,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        out = self.o_proj(y)
        return out


class SwiGLUMLP(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        assert config.hidden_dim is not None
        self.gate_proj = nn.Linear(config.hidden_size, config.hidden_dim, bias=config.bias)
        self.up_proj = nn.Linear(config.hidden_size, config.hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(config.hidden_dim, config.hidden_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        out = self.down_proj(hidden)
        return out


class DecoderBlock(nn.Module):

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.self_attn = CausalSelfAttention(config, layer_idx=layer_idx)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        input_pos: Optional[torch.Tensor] = None,
        cache_len: Optional[int] = None,
    ) -> torch.Tensor:
        normed_x = self.attn_norm(x)
        attn_out = self.self_attn(
            normed_x,
            cos=cos,
            sin=sin,
            kv_cache=kv_cache,
            input_pos=input_pos,
            cache_len=cache_len,
        )
        x = x + attn_out
        normed_x = self.mlp_norm(x)
        mlp_out = self.mlp(normed_x)
        x = x + mlp_out
        return x


class MinimalLlamaStyleCausalLM(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderBlock(config, layer_idx=i) for i in range(config.n_layers)])
        self.final_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        cos, sin = build_rope_cache(
            block_size=config.block_size,
            head_dim=config.head_dim,
            base=config.rope_base,
            device=None,
        )
        self.register_buffer('cos_cache', cos, persistent=False)
        self.register_buffer('sin_cache', sin, persistent=False)
        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def create_kv_cache(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> KVCache:
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        assert self.config.n_kv_heads is not None
        return KVCache(
            n_layers=self.config.n_layers,
            batch_size=batch_size,
            n_kv_heads=self.config.n_kv_heads,
            block_size=self.config.block_size,
            head_dim=self.config.head_dim,
            device=device,
            dtype=dtype,
        )

    def _get_rope_for_positions(
        self,
        seq_len: int,
        device: torch.device,
        input_pos: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[int]]:
        cache_len: Optional[int] = None
        if input_pos is None:
            if seq_len > self.config.block_size:
                raise ValueError(f'seq_len={seq_len} exceeds block_size={self.config.block_size}')
            cos = self.cos_cache[:seq_len]
            sin = self.sin_cache[:seq_len]
        else:
            if input_pos.dim() != 1:
                raise ValueError(
                    f'This teaching implementation only supports 1D input_pos [T], '
                    f'got input_pos.shape={input_pos.shape}'
                )
            if input_pos.numel() != seq_len:
                raise ValueError(
                    f'input_pos length must equal seq_len, '
                    f'got input_pos.numel()={input_pos.numel()}, seq_len={seq_len}'
                )
            max_pos = int(input_pos.max().item())
            if max_pos >= self.config.block_size:
                raise ValueError(f'input_pos max={max_pos} exceeds block_size={self.config.block_size}')
            cache_len = max_pos + 1
            input_pos = input_pos.to(device=self.cos_cache.device)
            cos = self.cos_cache.index_select(dim=0, index=input_pos)
            sin = self.sin_cache.index_select(dim=0, index=input_pos)
        cos = cos.to(device=device)
        sin = sin.to(device=device)
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        return (cos, sin, cache_len)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        input_pos: Optional[torch.Tensor] = None,
        return_last_logits_only: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, T = input_ids.shape
        if T > self.config.block_size:
            raise ValueError(f'Input sequence length T={T} exceeds block_size={self.config.block_size}')
        if kv_cache is not None and input_pos is None:
            raise ValueError('input_pos must be provided when kv_cache is used')
        if labels is not None and kv_cache is not None:
            raise ValueError(
                'Training loss with kv_cache is intentionally not supported '
                'in this teaching version. '
                'Use full-sequence training without kv_cache.'
            )
        if labels is not None and return_last_logits_only:
            raise ValueError(
                'return_last_logits_only is only supported when labels is None'
            )
        x = self.token_embedding(input_ids)
        cos, sin, cache_len = self._get_rope_for_positions(
            seq_len=T,
            device=input_ids.device,
            input_pos=input_pos,
        )
        for layer in self.layers:
            x = layer(
                x,
                cos=cos,
                sin=sin,
                kv_cache=kv_cache,
                input_pos=input_pos,
                cache_len=cache_len,
            )
        x = self.final_norm(x)
        if return_last_logits_only:
            x = x[:, -1:, :]
        logits = self.lm_head(x)
        if labels is None:
            return logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1), ignore_index=-100)
        return (logits, loss)
