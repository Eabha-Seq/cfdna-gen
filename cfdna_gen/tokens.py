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

Fetal-fraction bins are library-level style tokens. The left tail is coarse:
0.00–0.02 share a single bin. See docs/FF_CONDITIONING_FIX.md for the v15
weight collapse and what a proper continued train must change.
"""

import warnings
from collections.abc import Sequence

__all__ = [
    "TOKEN_A",
    "TOKEN_C",
    "TOKEN_G",
    "TOKEN_T",
    "TOKEN_BOS",
    "TOKEN_EOS",
    "TOKEN_PAD",
    "VOCAB_SIZE",
    "FF_BIN_BOUNDARIES",
    "FF_DOCUMENTED_MIN",
    "FF_DOCUMENTED_MAX",
    "tokens_to_sequence",
    "sequence_to_tokens",
    "get_len_bin_token",
    "get_gc_bin_token",
    "get_ff_bin_token",
    "decode_len_bin_token",
    "decode_gc_bin_token",
    "decode_ff_bin_token",
    "validate_fetal_fraction",
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

# Documented API range for target_ff. Bin tokens only resolve 0.00–0.40;
# values in (0.40, 0.50] share the last bin.
FF_DOCUMENTED_MIN = 0.0
FF_DOCUMENTED_MAX = 0.5

# Non-uniform FF bin edges (17 bins). Left tail is coarse: [0.00, 0.02) is one bin.
# Finer 2% steps from 2–22%, then coarser above that.
FF_BIN_BOUNDARIES = (
    0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18,
    0.20, 0.22, 0.25, 0.28, 0.32, 0.36, 0.40,
)

# Token mappings
_NUCLEOTIDE_TO_TOKEN = {"A": TOKEN_A, "C": TOKEN_C, "G": TOKEN_G, "T": TOKEN_T}
_TOKEN_TO_NUCLEOTIDE = {TOKEN_A: "A", TOKEN_C: "C", TOKEN_G: "G", TOKEN_T: "T"}


# =============================================================================
# Sequence Conversion Functions
# =============================================================================


def sequence_to_tokens(sequence: str) -> list[int]:
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


def validate_fetal_fraction(fetal_fraction: float) -> float:
    """
    Warn if a fetal fraction is outside the documented API range [0.0, 0.5].

    Out-of-range values are still accepted: ``get_ff_bin_token`` clips them
    into the 17-bin map (everything below 0.02 shares the left-tail bin;
    everything at or above 0.40 shares the last bin).

    Args:
        fetal_fraction: Fetal fraction as a fraction

    Returns:
        The same value, unchanged
    """
    if fetal_fraction < FF_DOCUMENTED_MIN or fetal_fraction > FF_DOCUMENTED_MAX:
        warnings.warn(
            f"fetal fraction {fetal_fraction} is outside the documented range "
            f"[{FF_DOCUMENTED_MIN}, {FF_DOCUMENTED_MAX}]. Values are still mapped "
            f"into the 17 FF bins (0–40%); 0–2% share a single left-tail bin.",
            UserWarning,
            stacklevel=3,
        )
    return fetal_fraction


def get_ff_bin_token(fetal_fraction: float) -> int:
    """
    Get the token ID for a fetal fraction bin.

    Fetal fraction is binned into 17 categories covering 0.0-0.40 (0%-40%).
    Bins are non-uniform: 2% steps from 2–22%, coarser above that.
    The left tail is coarse — 0.00–0.02 share one bin (e.g. 0.5% and 1.9%
    are indistinguishable at the token path).

    This is library-level style conditioning, not a per-fragment origin flag.
    On published v15 weights the paired continuous ``FFEmbedding`` path is
    collapsed; do not treat ``target_ff`` alone as a fetal-fraction simulator.
    See docs/FF_CONDITIONING_FIX.md.

    Bin boundaries:
        0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18,
        0.20, 0.22, 0.25, 0.28, 0.32, 0.36, 0.40

    Args:
        fetal_fraction: Fetal fraction as a fraction (documented 0.0-0.5)

    Returns:
        Token ID for the corresponding FF bin

    Example:
        >>> get_ff_bin_token(0.10)  # 10% fetal fraction
        52  # Bin for 0.10-0.12
        >>> get_ff_bin_token(0.005) == get_ff_bin_token(0.019)
        True
    """
    validate_fetal_fraction(fetal_fraction)

    # Find the bin: first edge strictly greater than ff, else last bin.
    bin_idx = 0
    for i, boundary in enumerate(FF_BIN_BOUNDARIES[1:], 1):
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


def decode_ff_bin_token(token: int) -> tuple:
    """
    Decode a fetal-fraction bin token to its range.

    Ranges are half-open ``[min_ff, max_ff)`` except the last bin, which
    is ``[0.40, 0.50]`` (everything at or above 0.40 shares that bin).
    The left tail ``[0.00, 0.02)`` is a single coarse bin.

    Args:
        token: Fetal-fraction bin token ID

    Returns:
        Tuple of (min_ff, max_ff) for the bin

    Example:
        >>> decode_ff_bin_token(get_ff_bin_token(0.005))
        (0.0, 0.02)
    """
    if not (FF_TOKEN_START <= token < FF_TOKEN_END):
        raise ValueError(f"Token {token} is not a fetal-fraction bin token")
    bin_idx = token - FF_TOKEN_START
    min_ff = FF_BIN_BOUNDARIES[bin_idx]
    if bin_idx + 1 < len(FF_BIN_BOUNDARIES):
        max_ff = FF_BIN_BOUNDARIES[bin_idx + 1]
    else:
        max_ff = FF_DOCUMENTED_MAX
    return (min_ff, max_ff)
