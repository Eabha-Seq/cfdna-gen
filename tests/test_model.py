"""Tests for the CfDNACausalLM model."""

import pytest
import torch
import json
import tempfile
from pathlib import Path

from cfdna_gen.model import CfDNAConfig, CfDNACausalLM
from cfdna_gen.tokens import VOCAB_SIZE, TOKEN_BOS


class TestCfDNAConfig:
    """Test model configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CfDNAConfig()
        assert config.vocab_size == VOCAB_SIZE
        assert config.hidden_dim == 768
        assert config.num_layers == 14
        assert config.num_heads == 12
        assert config.ffn_dim == 3072
        assert config.max_seq_len == 256
        assert config.dropout == 0.1

    def test_config_head_dim(self):
        """Test that head_dim is computed correctly."""
        config = CfDNAConfig()
        assert config.head_dim == config.hidden_dim // config.num_heads

    def test_config_to_dict(self):
        """Test config serialization."""
        config = CfDNAConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["hidden_dim"] == 768
        assert d["num_layers"] == 14

    def test_config_from_dict(self):
        """Test config deserialization."""
        d = {"hidden_dim": 512, "num_layers": 8, "num_heads": 8}
        config = CfDNAConfig.from_dict(d)
        assert config.hidden_dim == 512
        assert config.num_layers == 8
        assert config.num_heads == 8

    def test_config_save_load(self):
        """Test config save and load."""
        config = CfDNAConfig(hidden_dim=512, num_layers=8, num_heads=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            config.save(path)

            loaded = CfDNAConfig.load(path)
            assert loaded.hidden_dim == 512
            assert loaded.num_layers == 8

    def test_config_invalid_head_dim(self):
        """Test that invalid head_dim raises error."""
        with pytest.raises(AssertionError):
            CfDNAConfig(hidden_dim=100, num_heads=12)  # 100 not divisible by 12


class TestCfDNACausalLM:
    """Test the causal language model."""

    @pytest.fixture
    def small_config(self):
        """Create a small config for testing."""
        return CfDNAConfig(
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            ffn_dim=128,
            max_seq_len=64,
            dropout=0.0,
        )

    @pytest.fixture
    def small_model(self, small_config):
        """Create a small model for testing."""
        return CfDNACausalLM(small_config)

    def test_model_creation(self, small_model):
        """Test model can be created."""
        assert small_model is not None

    def test_model_param_count(self, small_model):
        """Test model parameter count is reasonable."""
        n_params = sum(p.numel() for p in small_model.parameters())
        # Small model should have significantly fewer params than full model
        assert n_params < 10_000_000  # Less than 10M for small model

    def test_forward_pass(self, small_model, small_config):
        """Test forward pass produces correct shapes."""
        batch_size = 4
        seq_len = 16

        input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
        fragment_length = torch.randint(100, 200, (batch_size,))

        logits, _ = small_model(
            input_ids,
            fragment_length=fragment_length,
        )

        assert logits.shape == (batch_size, seq_len, VOCAB_SIZE)

    def test_forward_with_gc_ff(self, small_model):
        """Test forward pass with GC and FF conditioning."""
        batch_size = 4
        seq_len = 16

        input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
        fragment_length = torch.randint(100, 200, (batch_size,))
        target_gc = torch.full((batch_size,), 0.42)
        target_ff = torch.full((batch_size,), 0.10)

        logits, _ = small_model(
            input_ids,
            fragment_length=fragment_length,
            target_gc=target_gc,
            target_ff=target_ff,
        )

        assert logits.shape == (batch_size, seq_len, VOCAB_SIZE)

    def test_forward_with_cache(self, small_model):
        """Test forward pass with KV cache."""
        batch_size = 2
        seq_len = 8

        input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))

        # First pass without cache
        logits, past_kv = small_model(input_ids, use_cache=True)

        assert past_kv is not None
        assert len(past_kv) == small_model.config.num_layers

        # Second pass with cache (single token)
        new_token = torch.randint(0, VOCAB_SIZE, (batch_size, 1))
        logits2, new_past_kv = small_model(
            new_token, past_kv=past_kv, use_cache=True
        )

        assert logits2.shape == (batch_size, 1, VOCAB_SIZE)

    def test_generate(self, small_model):
        """Test sequence generation."""
        batch_size = 2
        fragment_length = torch.tensor([20, 25])

        # Condition tokens (length bin)
        condition_tokens = torch.tensor([[7], [7]])  # Length bin tokens

        generated = small_model.generate(
            condition_tokens=condition_tokens,
            fragment_length=fragment_length,
            max_length=30,
            temperature=1.0,
            top_p=0.9,
        )

        assert generated.shape[0] == batch_size
        assert generated.shape[1] <= 30

    def test_save_load(self, small_model):
        """Test model save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model"

            # Save
            small_model.save_pretrained(path)

            # Check files exist
            assert (path / "config.json").exists()
            # Either safetensors or pt should exist
            assert (path / "model.safetensors").exists() or (path / "model.pt").exists()

            # Load (force CPU for test consistency)
            loaded = CfDNACausalLM.from_pretrained(path, device="cpu")

            # Compare outputs (both on CPU)
            small_model_cpu = small_model.to("cpu")
            input_ids = torch.randint(0, VOCAB_SIZE, (2, 8))
            with torch.no_grad():
                orig_out, _ = small_model_cpu(input_ids)
                loaded_out, _ = loaded(input_ids)

            assert torch.allclose(orig_out, loaded_out, atol=1e-5)


class TestModelComponents:
    """Test individual model components."""

    def test_rms_norm(self):
        """Test RMSNorm."""
        from cfdna_gen.model import RMSNorm

        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64)
        y = norm(x)

        assert y.shape == x.shape
        # Output should have roughly unit variance
        assert y.std().item() < 2.0

    def test_swiglu(self):
        """Test SwiGLU activation."""
        from cfdna_gen.model import SwiGLU

        ffn = SwiGLU(64, 128, 64, dropout=0.0)
        x = torch.randn(2, 8, 64)
        y = ffn(x)

        assert y.shape == x.shape

    def test_length_embedding(self):
        """Test length embedding."""
        from cfdna_gen.model import LengthEmbedding

        embed = LengthEmbedding(64, max_length=300)
        lengths = torch.tensor([100, 150, 200])
        y = embed(lengths)

        assert y.shape == (3, 1, 64)

    def test_gc_embedding(self):
        """Test GC content embedding."""
        from cfdna_gen.model import GCEmbedding

        embed = GCEmbedding(64)
        gc = torch.tensor([0.40, 0.42, 0.45])
        y = embed(gc)

        assert y.shape == (3, 1, 64)

    def test_ff_embedding(self):
        """Test fetal fraction embedding."""
        from cfdna_gen.model import FFEmbedding

        embed = FFEmbedding(64)
        ff = torch.tensor([0.05, 0.10, 0.15])
        y = embed(ff)

        assert y.shape == (3, 1, 64)
