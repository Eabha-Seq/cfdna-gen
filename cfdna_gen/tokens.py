"""
Token definitions for cfDNA sequence generation.

This module defines the vocabulary used by the cfDNA causal language model,
including nucleotide tokens (A, C, G, T), special tokens (BOS, EOS, PAD),
and conditioning tokens (length bins, GC bins, fetal fraction bins).

Vocabulary Layout (64 tokens total):
    0-3:   Nucleotides (A, C, G, T)
    4:     BOS (beginning of sequence)
    5:     EOS (end of sequence)
    6:     PAD (padding)
    7-26:  Length bin tokens (20 bins covering 50-250bp)
    27-46: GC content bin tokens (20 bins covering 0.25-0.65)
    47-63: Fetal fraction bin tokens (17 bins covering 0.0-0.40)
"""

from typing import List, Sequence

__all__ = [
    "TOKEN_A",
    "TOKEN_C",
    "TOKEN_G",
    "TOKEN_T",
    "TOKEN_BOS",
    "TOKEN_EOS",
    "TOKEN_PAD",
    "VOCAB_SIZE",
    "tokens_to_sequence",
    "sequence_to_tokens",
    "get_len_bin_token",
    "get_gc_bin_token",
    "get_ff_bin_token",
]

# =============================================================================
# Core Token Definitions
# =============================================================================

# Nucleotide tokens
TOKEN_A = 0
TOKEN_C = 1
TOKEN_G = 2
TOKEN_T = 3

# Special tokens
TOKEN_BOS = 4  # Beginning of sequence
TOKEN_EOS = 5  # End of sequence
TOKEN_PAD = 6  # Padding

# Conditioning token ranges
LEN_TOKEN_START = 7
LEN_TOKEN_END = 27  # 20 length bins

GC_TOKEN_START = 27
GC_TOKEN_END = 47  # 20 GC bins

FF_TOKEN_START = 47
FF_TOKEN_END = 64  # 17 FF bins

VOCAB_SIZE = 64

# Token mappings
_NUCLEOTIDE_TO_TOKEN = {"A": TOKEN_A, "C": TOKEN_C, "G": TOKEN_G, "T": TOKEN_T}
_TOKEN_TO_NUCLEOTIDE = {TOKEN_A: "A", TOKEN_C: "C", TOKEN_G: "G", TOKEN_T: "T"}


# =============================================================================
# Sequence Conversion Functions
# =============================================================================


def sequence_to_tokens(sequence: str) -> List[int]:
    """
    Convert a DNA sequence string to a list of token IDs.

    Args:
        sequence: DNA sequence string containing only A, C, G, T characters

    Returns:
        List of integer token IDs

    Raises:
        ValueError: If sequence contains invalid characters

    Example:
        >>> sequence_to_tokens("ACGT")
        [0, 1, 2, 3]
    """
    tokens = []
    for i, char in enumerate(sequence.upper()):
        if char not in _NUCLEOTIDE_TO_TOKEN:
            raise ValueError(f"Invalid nucleotide '{char}' at position {i}")
        tokens.append(_NUCLEOTIDE_TO_TOKEN[char])
    return tokens


def tokens_to_sequence(tokens: Sequence[int]) -> str:
    """
    Convert a list of token IDs back to a DNA sequence string.

    Only nucleotide tokens (0-3) are converted; special tokens are skipped.

    Args:
        tokens: Sequence of integer token IDs

    Returns:
        DNA sequence string

    Example:
        >>> tokens_to_sequence([0, 1, 2, 3, 5])  # includes EOS token
        'ACGT'
    """
    sequence = []
    for token in tokens:
        if token in _TOKEN_TO_NUCLEOTIDE:
            sequence.append(_TOKEN_TO_NUCLEOTIDE[token])
    return "".join(sequence)


# =============================================================================
# Conditioning Token Functions
# =============================================================================


def get_len_bin_token(length: int) -> int:
    """
    Get the token ID for a fragment length bin.

    Fragment lengths are binned into 20 categories covering 50-250bp.
    Each bin spans 10bp (50-60, 60-70, ..., 240-250).

    Args:
        length: Fragment length in base pairs

    Returns:
        Token ID for the corresponding length bin

    Example:
        >>> get_len_bin_token(165)  # Typical cfDNA fragment
        18  # Bin for 160-170bp
    """
    # 20 bins from 50bp to 250bp, each bin is 10bp
    bin_idx = min(max((length - 50) // 10, 0), 19)
    return LEN_TOKEN_START + bin_idx


def get_gc_bin_token(gc_content: float) -> int:
    """
    Get the token ID for a GC content bin.

    GC content is binned into 20 categories covering 0.25-0.65 (25%-65%).
    Each bin spans 0.02 (2 percentage points).

    Args:
        gc_content: GC content as a fraction (0.0-1.0)

    Returns:
        Token ID for the corresponding GC bin

    Example:
        >>> get_gc_bin_token(0.42)  # Typical cfDNA GC content
        35  # Bin for 0.40-0.42
    """
    # 20 bins from 0.25 to 0.65, each bin is 0.02
    bin_idx = min(max(int((gc_content - 0.25) / 0.02), 0), 19)
    return GC_TOKEN_START + bin_idx


def get_ff_bin_token(fetal_fraction: float) -> int:
    """
    Get the token ID for a fetal fraction bin.

    Fetal fraction is binned into 17 categories covering 0.0-0.40 (0%-40%).
    Bins are non-uniform: finer resolution at low FF (clinical range).

    Bin boundaries:
        0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18,
        0.20, 0.22, 0.25, 0.28, 0.32, 0.36, 0.40

    Args:
        fetal_fraction: Fetal fraction as a fraction (0.0-0.5)

    Returns:
        Token ID for the corresponding FF bin

    Example:
        >>> get_ff_bin_token(0.10)  # 10% fetal fraction
        52  # Bin for 0.10-0.12
    """
    # Non-uniform bins: finer at low FF (clinical range 4-20%)
    boundaries = [
        0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18,
        0.20, 0.22, 0.25, 0.28, 0.32, 0.36, 0.40
    ]

    # Find the bin
    bin_idx = 0
    for i, boundary in enumerate(boundaries[1:], 1):
        if fetal_fraction < boundary:
            break
        bin_idx = i

    bin_idx = min(bin_idx, 16)  # 17 bins (0-16)
    return FF_TOKEN_START + bin_idx


def decode_len_bin_token(token: int) -> tuple:
    """
    Decode a length bin token to its range.

    Args:
        token: Length bin token ID

    Returns:
        Tuple of (min_length, max_length) for the bin

    Example:
        >>> decode_len_bin_token(18)
        (160, 170)
    """
    if not (LEN_TOKEN_START <= token < LEN_TOKEN_END):
        raise ValueError(f"Token {token} is not a length bin token")
    bin_idx = token - LEN_TOKEN_START
    min_len = 50 + bin_idx * 10
    max_len = min_len + 10
    return (min_len, max_len)


def decode_gc_bin_token(token: int) -> tuple:
    """
    Decode a GC bin token to its range.

    Args:
        token: GC bin token ID

    Returns:
        Tuple of (min_gc, max_gc) for the bin

    Example:
        >>> decode_gc_bin_token(35)
        (0.41, 0.43)
    """
    if not (GC_TOKEN_START <= token < GC_TOKEN_END):
        raise ValueError(f"Token {token} is not a GC bin token")
    bin_idx = token - GC_TOKEN_START
    min_gc = 0.25 + bin_idx * 0.02
    max_gc = min_gc + 0.02
    return (min_gc, max_gc)
