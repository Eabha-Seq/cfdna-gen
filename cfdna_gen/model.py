"""
Conditional Causal Transformer for cfDNA sequence generation.

A 120M parameter transformer with RoPE, SwiGLU, RMSNorm, and Flash Attention
for generating realistic cell-free DNA sequences.

Architecture:
    - Token embedding (vocab -> hidden)
    - Continuous embeddings for length, GC, and fetal fraction
    - 14 transformer blocks with RoPE, SwiGLU, RMSNorm
    - Output projection (weight-tied with embeddings)

Example:
    >>> from cfdna_gen import CfDNACausalLM, CfDNAConfig
    >>> config = CfDNAConfig()
    >>> model = CfDNACausalLM(config)
    >>> # Or load pretrained
    >>> model = CfDNACausalLM.from_pretrained("eabhaseq/cfdna-gen")
"""

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokens import VOCAB_SIZE, TOKEN_BOS, TOKEN_EOS, TOKEN_PAD

__all__ = ["CfDNAConfig", "CfDNACausalLM"]


@dataclass
class CfDNAConfig:
    """
    Configuration for cfDNA Causal LM.

    Default configuration creates a ~120M parameter model optimized for
    cfDNA sequence generation.

    Attributes:
        vocab_size: Size of token vocabulary (default: 64)
        hidden_dim: Hidden dimension size (default: 768)
        num_layers: Number of transformer layers (default: 14)
        num_heads: Number of attention heads (default: 12)
        ffn_dim: Feed-forward network dimension (default: 3072)
        max_seq_len: Maximum sequence length (default: 256)
        dropout: Dropout probability (default: 0.1)
        max_fragment_length: Maximum fragment length for embedding (default: 300)
    """

    vocab_size: int = VOCAB_SIZE
    hidden_dim: int = 768
    num_layers: int = 14
    num_heads: int = 12
    ffn_dim: int = 3072
    max_seq_len: int = 256
    dropout: float = 0.1
    max_fragment_length: int = 300

    def __post_init__(self):
        assert self.hidden_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = self.hidden_dim // self.num_heads

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CfDNAConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k != "head_dim"})

    def save(self, path: Union[str, Path]) -> None:
        """Save config to JSON file."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CfDNAConfig":
        """Load config from JSON file."""
        path = Path(path)
        with open(path) as f:
            return cls.from_dict(json.load(f))


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 512, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos and sin
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to Q and K."""
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """SwiGLU activation function for FFN."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.w3 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and SDPA (Flash Attention)."""

    def __init__(self, config: CfDNAConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_dim = config.hidden_dim
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        self.resid_dropout = nn.Dropout(config.dropout)
        self.rotary_emb = RotaryPositionEmbedding(config.head_dim, config.max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Handle KV cache
        if past_kv is not None:
            past_k, past_v = past_kv
            seq_len = past_k.shape[2] + L
            cos, sin = self.rotary_emb(q, seq_len)
            cos = cos[past_k.shape[2] : seq_len]
            sin = sin[past_k.shape[2] : seq_len]
        else:
            seq_len = L
            cos, sin = self.rotary_emb(q, seq_len)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(
            q, k, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
        )

        # Concatenate past KV
        if past_kv is not None:
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v) if use_cache else None

        # Use PyTorch's scaled_dot_product_attention (SDPA)
        dropout_p = self.dropout if self.training else 0.0

        if past_kv is not None and L == 1:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=False)
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.resid_dropout(self.o_proj(out))

        return out, new_kv


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""

    def __init__(self, config: CfDNAConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_dim)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_dim)
        self.ffn = SwiGLU(config.hidden_dim, config.ffn_dim, config.hidden_dim, config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, new_kv = self.attn(self.attn_norm(x), past_kv, use_cache)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_kv


class LengthEmbedding(nn.Module):
    """Continuous length embedding for capturing smooth length-feature correlations."""

    def __init__(self, hidden_dim: int, max_length: int = 300):
        super().__init__()
        self.proj = nn.Linear(1, hidden_dim)
        self.max_length = max_length

    def forward(self, length: torch.Tensor) -> torch.Tensor:
        normalized = length.float() / self.max_length
        return self.proj(normalized.unsqueeze(-1)).unsqueeze(1)


class GCEmbedding(nn.Module):
    """Continuous GC content embedding for per-sequence GC conditioning."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )

    def forward(self, gc: torch.Tensor) -> torch.Tensor:
        normalized = (gc.float() - 0.42) * 5.0
        return self.proj(normalized.unsqueeze(-1)).unsqueeze(1)


class FFEmbedding(nn.Module):
    """Continuous fetal fraction embedding for per-sequence FF conditioning."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )

    def forward(self, ff: torch.Tensor) -> torch.Tensor:
        normalized = (ff.float() - 0.10) * 10.0
        return self.proj(normalized.unsqueeze(-1)).unsqueeze(1)


class CfDNACausalLM(nn.Module):
    """
    Conditional Causal Language Model for cfDNA generation.

    A 120M parameter transformer that generates realistic cell-free DNA
    sequences conditioned on fragment length, GC content, and fetal fraction.

    Architecture:
        - Token embedding (vocab -> hidden)
        - Continuous embeddings for length, GC, and fetal fraction
        - 14 transformer blocks with RoPE, SwiGLU, RMSNorm
        - Output projection (weight-tied with embeddings)

    Example:
        >>> from cfdna_gen import CfDNACausalLM, CfDNAConfig
        >>> config = CfDNAConfig()
        >>> model = CfDNACausalLM(config)
        >>> # Or load pretrained
        >>> model = CfDNACausalLM.from_pretrained("path/to/model")
    """

    def __init__(self, config: CfDNAConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Continuous condition embeddings
        self.length_embed = LengthEmbedding(config.hidden_dim, config.max_fragment_length)
        self.gc_embed = GCEmbedding(config.hidden_dim)
        self.ff_embed = FFEmbedding(config.hidden_dim)

        # Dropout
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])

        # Final norm
        self.norm = RMSNorm(config.hidden_dim)

        # Output projection (weight-tied with token embedding)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Report param count
        n_params = sum(p.numel() for p in self.parameters())
        print(f"CfDNACausalLM: {n_params / 1e6:.2f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: Union[str, Path],
        device: Optional[str] = None,
        **kwargs,
    ) -> "CfDNACausalLM":
        """
        Load a pretrained model from a local path or HuggingFace Hub.

        Args:
            path_or_repo: Local path to model directory, or HuggingFace repo ID
            device: Device to load model on ('cpu', 'cuda', 'auto')
            **kwargs: Additional arguments passed to config

        Returns:
            Loaded CfDNACausalLM model

        Example:
            >>> # From local path
            >>> model = CfDNACausalLM.from_pretrained("./models/v15")
            >>> # From HuggingFace Hub
            >>> model = CfDNACausalLM.from_pretrained("eabhaseq/cfdna-gen")
        """
        path = Path(path_or_repo)

        # Try to load from HuggingFace Hub if not a local path
        if not path.exists():
            try:
                from huggingface_hub import snapshot_download

                path = Path(snapshot_download(str(path_or_repo)))
            except ImportError:
                raise ImportError(
                    "huggingface_hub is required to download models. "
                    "Install with: pip install huggingface-hub"
                )
            except Exception as e:
                raise ValueError(f"Could not find model at {path_or_repo}: {e}")

        # Load config
        config_path = path / "config.json"
        if config_path.exists():
            config = CfDNAConfig.load(config_path)
        else:
            config = CfDNAConfig(**kwargs)

        # Create model
        model = cls(config)

        # Load weights
        weights_path = path / "model.safetensors"
        if weights_path.exists():
            try:
                from safetensors.torch import load_model

                # Use load_model to handle tied weights properly
                load_model(model, weights_path)

                # Move to device and return early
                if device is None:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                if device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                model = model.to(device)
                model.eval()
                return model
            except ImportError:
                raise ImportError(
                    "safetensors is required to load model weights. "
                    "Install with: pip install safetensors"
                )
        else:
            # Try PyTorch format
            pt_path = path / "model.pt"
            if pt_path.exists():
                checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)
                state_dict = checkpoint.get("model_state_dict", checkpoint)
            else:
                # Try checkpoint-epoch format
                import glob
                checkpoint_files = glob.glob(str(path / "checkpoint-epoch*.pt"))
                if checkpoint_files:
                    # Use the latest checkpoint
                    checkpoint_files.sort()
                    checkpoint = torch.load(checkpoint_files[-1], map_location="cpu", weights_only=False)
                    state_dict = checkpoint.get("model_state_dict", checkpoint)
                else:
                    # Try best_model.pt
                    best_path = path / "best_model.pt"
                    if best_path.exists():
                        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
                        state_dict = checkpoint.get("model_state_dict", checkpoint)
                    else:
                        raise FileNotFoundError(f"No model weights found in {path}")

        # Strip '_orig_mod.' prefix if present (from torch.compile)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("_orig_mod."):
                cleaned_state_dict[key[len("_orig_mod."):]] = value
            else:
                cleaned_state_dict[key] = value
        state_dict = cleaned_state_dict

        model.load_state_dict(state_dict)

        # Move to device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        return model

    def save_pretrained(self, path: Union[str, Path]) -> None:
        """
        Save model and config to a directory.

        Args:
            path: Directory to save model to

        Example:
            >>> model.save_pretrained("./my_model")
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.save(path / "config.json")

        # Save weights in safetensors format
        try:
            from safetensors.torch import save_model

            # Use save_model to handle tied weights properly
            save_model(self, path / "model.safetensors")
        except ImportError:
            # Fallback to PyTorch format
            torch.save({"model_state_dict": self.state_dict()}, path / "model.pt")

    def forward(
        self,
        input_ids: torch.Tensor,
        fragment_length: Optional[torch.Tensor] = None,
        target_gc: Optional[torch.Tensor] = None,
        target_ff: Optional[torch.Tensor] = None,
        past_kv: Optional[list] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass.

        Args:
            input_ids: [B, L] token IDs
            fragment_length: [B] optional fragment lengths for continuous embedding
            target_gc: [B] optional target GC content for conditioning
            target_ff: [B] optional target fetal fraction for conditioning
            past_kv: List of (k, v) tuples from previous steps (for generation)
            use_cache: Whether to return new KV cache

        Returns:
            logits: [B, L, vocab_size]
            new_past_kv: List of (k, v) tuples (if use_cache=True)
        """
        B, L = input_ids.shape

        # Token embeddings
        h = self.token_embed(input_ids)

        # Add continuous condition embeddings if provided
        if fragment_length is not None:
            h = h + self.length_embed(fragment_length)
        if target_gc is not None:
            h = h + self.gc_embed(target_gc)
        if target_ff is not None:
            h = h + self.ff_embed(target_ff)

        h = self.drop(h)

        # Transformer blocks
        new_past_kv = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_past = past_kv[i] if past_kv is not None else None
            h, new_kv = block(h, layer_past, use_cache)
            if use_cache:
                new_past_kv.append(new_kv)

        # Final norm and projection
        h = self.norm(h)
        logits = self.lm_head(h)

        return logits, new_past_kv

    @torch.no_grad()
    def generate(
        self,
        condition_tokens: torch.Tensor,
        fragment_length: torch.Tensor,
        target_gc: Optional[torch.Tensor] = None,
        target_ff: Optional[torch.Tensor] = None,
        max_length: int = 200,
        temperature: float = 0.95,
        top_p: float = 0.96,
        enforce_length: bool = True,
    ) -> torch.Tensor:
        """
        Generate sequences autoregressively with conditioning.

        Args:
            condition_tokens: [B, num_conditions] condition token IDs
            fragment_length: [B] fragment lengths (controls output length)
            target_gc: [B] optional target GC content (0.0-1.0)
            target_ff: [B] optional target fetal fraction (0.0-0.5)
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold
            enforce_length: If True, stop each sequence at its target fragment_length

        Returns:
            generated: [B, max_length] generated token IDs
        """
        device = condition_tokens.device
        B = condition_tokens.shape[0]

        # Start with conditions + BOS
        bos = torch.full((B, 1), TOKEN_BOS, dtype=torch.long, device=device)
        input_ids = torch.cat([condition_tokens, bos], dim=1)

        # KV cache
        past_kv = None

        # Generated tokens storage
        generated = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        tokens_generated = torch.zeros(B, dtype=torch.long, device=device)

        for step in range(max_length + 1):
            logits, past_kv = self.forward(
                input_ids,
                fragment_length=fragment_length,
                target_gc=target_gc,
                target_ff=target_ff,
                past_kv=past_kv,
                use_cache=True,
            )

            next_logits = logits[:, -1, :] / temperature

            # Top-p (nucleus) sampling
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            cumsum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_mask = cumsum_probs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False

            mask = torch.zeros_like(sorted_mask)
            mask.scatter_(1, sorted_idx, sorted_mask)
            next_logits[mask] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Enforce length
            if enforce_length:
                reached_length = tokens_generated >= fragment_length
                next_token = torch.where(
                    reached_length,
                    torch.full_like(next_token, TOKEN_EOS),
                    next_token,
                )

            generated.append(next_token)
            tokens_generated = tokens_generated + (~finished).long()

            finished = finished | (next_token == TOKEN_EOS)
            if finished.all():
                break

            input_ids = next_token.unsqueeze(-1)

        result = torch.stack(generated, dim=1)

        if result.shape[1] < max_length:
            padding = torch.full(
                (B, max_length - result.shape[1]),
                TOKEN_PAD,
                dtype=torch.long,
                device=device,
            )
            result = torch.cat([result, padding], dim=1)

        return result
