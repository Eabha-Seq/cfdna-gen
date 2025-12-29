"""Tests for token utilities."""

import pytest
from cfdna_gen.tokens import (
    TOKEN_A, TOKEN_C, TOKEN_G, TOKEN_T,
    TOKEN_BOS, TOKEN_EOS, TOKEN_PAD,
    VOCAB_SIZE,
    sequence_to_tokens,
    tokens_to_sequence,
    get_len_bin_token,
    get_gc_bin_token,
    get_ff_bin_token,
    decode_len_bin_token,
    decode_gc_bin_token,
    LEN_TOKEN_START, LEN_TOKEN_END,
    GC_TOKEN_START, GC_TOKEN_END,
    FF_TOKEN_START, FF_TOKEN_END,
)


class TestBasicTokens:
    """Test basic token definitions."""

    def test_nucleotide_tokens(self):
        """Test nucleotide token values."""
        assert TOKEN_A == 0
        assert TOKEN_C == 1
        assert TOKEN_G == 2
        assert TOKEN_T == 3

    def test_special_tokens(self):
        """Test special token values."""
        assert TOKEN_BOS == 4
        assert TOKEN_EOS == 5
        assert TOKEN_PAD == 6

    def test_vocab_size(self):
        """Test vocabulary size."""
        assert VOCAB_SIZE == 64


class TestSequenceConversion:
    """Test sequence to/from token conversion."""

    def test_sequence_to_tokens_basic(self):
        """Test basic sequence tokenization."""
        tokens = sequence_to_tokens("ACGT")
        assert tokens == [0, 1, 2, 3]

    def test_sequence_to_tokens_longer(self):
        """Test longer sequence tokenization."""
        tokens = sequence_to_tokens("AACCGGTT")
        assert tokens == [0, 0, 1, 1, 2, 2, 3, 3]

    def test_sequence_to_tokens_case_insensitive(self):
        """Test case insensitivity."""
        assert sequence_to_tokens("acgt") == sequence_to_tokens("ACGT")

    def test_sequence_to_tokens_invalid(self):
        """Test invalid character handling."""
        with pytest.raises(ValueError):
            sequence_to_tokens("ACGTN")  # N is invalid

    def test_tokens_to_sequence_basic(self):
        """Test basic token to sequence conversion."""
        seq = tokens_to_sequence([0, 1, 2, 3])
        assert seq == "ACGT"

    def test_tokens_to_sequence_with_special(self):
        """Test that special tokens are skipped."""
        seq = tokens_to_sequence([0, 1, 5, 6])  # A, C, EOS, PAD
        assert seq == "AC"

    def test_roundtrip(self):
        """Test sequence -> tokens -> sequence roundtrip."""
        original = "ACGTACGTACGT"
        tokens = sequence_to_tokens(original)
        recovered = tokens_to_sequence(tokens)
        assert recovered == original


class TestLengthBinTokens:
    """Test length bin token functions."""

    def test_length_bin_range(self):
        """Test that length bins cover expected range."""
        # 20 bins from 50 to 250bp
        assert get_len_bin_token(50) == LEN_TOKEN_START
        assert get_len_bin_token(245) == LEN_TOKEN_END - 1

    def test_length_bin_typical(self):
        """Test typical cfDNA lengths."""
        token_144 = get_len_bin_token(144)
        token_167 = get_len_bin_token(167)
        # These should be in different bins
        assert token_144 != token_167

    def test_length_bin_clipping(self):
        """Test that extreme values are clipped."""
        # Very short should map to first bin
        assert get_len_bin_token(10) == LEN_TOKEN_START
        # Very long should map to last bin
        assert get_len_bin_token(500) == LEN_TOKEN_END - 1

    def test_decode_length_bin(self):
        """Test decoding length bin tokens."""
        token = get_len_bin_token(165)
        min_len, max_len = decode_len_bin_token(token)
        assert min_len <= 165 < max_len


class TestGCBinTokens:
    """Test GC content bin token functions."""

    def test_gc_bin_range(self):
        """Test that GC bins cover expected range."""
        # 20 bins from 0.25 to 0.65
        assert get_gc_bin_token(0.25) == GC_TOKEN_START
        assert get_gc_bin_token(0.64) == GC_TOKEN_END - 1

    def test_gc_bin_typical(self):
        """Test typical cfDNA GC content."""
        token = get_gc_bin_token(0.42)
        assert GC_TOKEN_START <= token < GC_TOKEN_END

    def test_gc_bin_clipping(self):
        """Test that extreme values are clipped."""
        assert get_gc_bin_token(0.0) == GC_TOKEN_START
        assert get_gc_bin_token(1.0) == GC_TOKEN_END - 1

    def test_decode_gc_bin(self):
        """Test decoding GC bin tokens."""
        token = get_gc_bin_token(0.42)
        min_gc, max_gc = decode_gc_bin_token(token)
        assert min_gc <= 0.42 < max_gc


class TestFFBinTokens:
    """Test fetal fraction bin token functions."""

    def test_ff_bin_range(self):
        """Test that FF bins cover expected range."""
        assert get_ff_bin_token(0.0) == FF_TOKEN_START
        assert get_ff_bin_token(0.40) == FF_TOKEN_END - 1

    def test_ff_bin_typical(self):
        """Test typical fetal fractions."""
        # Clinical range is typically 4-20%
        token_4 = get_ff_bin_token(0.04)
        token_10 = get_ff_bin_token(0.10)
        token_20 = get_ff_bin_token(0.20)

        assert FF_TOKEN_START <= token_4 < FF_TOKEN_END
        assert FF_TOKEN_START <= token_10 < FF_TOKEN_END
        assert FF_TOKEN_START <= token_20 < FF_TOKEN_END

    def test_ff_bins_non_uniform(self):
        """Test that FF bins have finer resolution at low values."""
        # Low FF (clinical range) should have finer bins
        diff_low = get_ff_bin_token(0.06) - get_ff_bin_token(0.04)
        # High FF should have coarser bins
        # At high end, bins are wider (0.32-0.36, 0.36-0.40)
        diff_high = get_ff_bin_token(0.38) - get_ff_bin_token(0.34)
        # Low should have at least as fine resolution
        assert diff_low >= diff_high
