"""Tests for the CfDNAGenerator high-level API."""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path

from cfdna_gen.generate import CfDNAGenerator
from cfdna_gen.model import CfDNACausalLM, CfDNAConfig


class TestCfDNAGenerator:
    """Test the high-level generator API."""

    @pytest.fixture
    def small_generator(self):
        """Create a small generator for testing."""
        config = CfDNAConfig(
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            ffn_dim=128,
            max_seq_len=64,
            dropout=0.0,
        )
        model = CfDNACausalLM(config)
        return CfDNAGenerator(model, device="cpu")

    def test_generator_creation(self, small_generator):
        """Test generator can be created."""
        assert small_generator is not None
        assert small_generator.device == "cpu"

    def test_generate_basic(self, small_generator):
        """Test basic sequence generation."""
        sequences = small_generator.generate(
            n_sequences=5,
            fragment_lengths=30,
        )

        assert len(sequences) == 5
        assert all(isinstance(s, str) for s in sequences)
        assert all(set(s).issubset({"A", "C", "G", "T"}) for s in sequences)

    def test_generate_with_gc_ff(self, small_generator):
        """Test generation with GC and FF conditioning."""
        sequences = small_generator.generate(
            n_sequences=5,
            fragment_lengths=30,
            target_gc=0.42,
            target_ff=0.10,
        )

        assert len(sequences) == 5

    def test_generate_variable_lengths(self, small_generator):
        """Test generation with variable fragment lengths."""
        lengths = [25, 30, 35, 40, 45]
        sequences = small_generator.generate(
            n_sequences=5,
            fragment_lengths=lengths,
        )

        assert len(sequences) == 5

    def test_generate_numpy_lengths(self, small_generator):
        """Test generation with numpy array lengths."""
        lengths = np.array([25, 30, 35, 40, 45])
        sequences = small_generator.generate(
            n_sequences=5,
            fragment_lengths=lengths,
        )

        assert len(sequences) == 5

    def test_generate_single_length_broadcast(self, small_generator):
        """Test that single length is broadcast to all sequences."""
        sequences = small_generator.generate(
            n_sequences=10,
            fragment_lengths=30,
        )

        assert len(sequences) == 10

    def test_generate_length_mismatch_error(self, small_generator):
        """Test that mismatched lengths raise error."""
        with pytest.raises(ValueError):
            small_generator.generate(
                n_sequences=10,
                fragment_lengths=[25, 30, 35],  # Only 3 lengths for 10 sequences
            )

    def test_generate_with_metadata(self, small_generator):
        """Test generation with metadata."""
        results = small_generator.generate_with_metadata(
            n_sequences=5,
            fragment_lengths=30,
            target_gc=0.42,
            target_ff=0.10,
        )

        assert len(results) == 5
        for r in results:
            assert "sequence" in r
            assert "length" in r
            assert "gc_content" in r
            assert "target_gc" in r
            assert "target_ff" in r
            assert r["length"] == len(r["sequence"])
            assert 0 <= r["gc_content"] <= 1

    def test_generate_batched(self, small_generator):
        """Test that batching works correctly."""
        sequences = small_generator.generate(
            n_sequences=100,
            fragment_lengths=30,
            batch_size=32,  # Will require multiple batches
        )

        assert len(sequences) == 100

    def test_generate_temperature(self, small_generator):
        """Test that different temperatures produce different outputs."""
        # Set seed for reproducibility
        torch.manual_seed(42)
        seq_low_temp = small_generator.generate(
            n_sequences=1,
            fragment_lengths=30,
            temperature=0.5,
        )[0]

        torch.manual_seed(42)
        seq_high_temp = small_generator.generate(
            n_sequences=1,
            fragment_lengths=30,
            temperature=1.5,
        )[0]

        # With same seed, different temperatures should produce different results
        # (though this isn't guaranteed, it's likely)
        # At minimum, both should be valid sequences
        assert set(seq_low_temp).issubset({"A", "C", "G", "T"})
        assert set(seq_high_temp).issubset({"A", "C", "G", "T"})

    def test_generate_fastq(self, small_generator):
        """Test FASTQ file generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.fastq"

            n_written = small_generator.generate_fastq(
                n_sequences=10,
                fragment_lengths=30,
                output_path=output_path,
            )

            assert n_written == 10
            assert output_path.exists()

            # Check FASTQ format
            with open(output_path) as f:
                lines = f.readlines()

            # FASTQ has 4 lines per record
            assert len(lines) == 40

            # Check format of first record
            assert lines[0].startswith("@synthetic_cfdna_")
            assert set(lines[1].strip()).issubset({"A", "C", "G", "T"})
            assert lines[2].strip() == "+"
            # Quality line should have same length as sequence
            assert len(lines[3].strip()) == len(lines[1].strip())

    def test_generate_fastq_gzip(self, small_generator):
        """Test gzipped FASTQ file generation."""
        import gzip

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.fastq.gz"

            n_written = small_generator.generate_fastq(
                n_sequences=10,
                fragment_lengths=30,
                output_path=output_path,
            )

            assert n_written == 10
            assert output_path.exists()

            # Check it's valid gzip
            with gzip.open(output_path, "rt") as f:
                lines = f.readlines()

            assert len(lines) == 40


class TestGeneratorFromPretrained:
    """Test loading generator from pretrained models."""

    def test_from_pretrained_local(self):
        """Test loading from local path."""
        # Create and save a small model
        config = CfDNAConfig(
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            ffn_dim=128,
            max_seq_len=64,
            dropout=0.0,
        )
        model = CfDNACausalLM(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model"
            model.save_pretrained(path)

            # Load using from_pretrained
            generator = CfDNAGenerator.from_pretrained(path, device="cpu")

            # Should work
            sequences = generator.generate(
                n_sequences=3,
                fragment_lengths=20,
            )
            assert len(sequences) == 3

    def test_from_pretrained_invalid_path(self):
        """Test that invalid path raises error."""
        with pytest.raises((ValueError, FileNotFoundError)):
            CfDNAGenerator.from_pretrained("/nonexistent/path")
