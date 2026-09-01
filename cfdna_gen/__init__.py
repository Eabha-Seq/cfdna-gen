"""
cfDNA-Gen: Conditional Causal Transformer for Cell-Free DNA Sequence Generation.

Generate realistic synthetic cell-free DNA (cfDNA) sequences for NIPT simulation,
benchmark development, and research.

Example:
    >>> from cfdna_gen import CfDNAGenerator
    >>> generator = CfDNAGenerator.from_pretrained("eabhaseq/cfdna-gen")
    >>> sequences = generator.generate(
    ...     n_sequences=100,
    ...     fragment_lengths=165,
    ...     target_gc=0.42,
    ...     target_ff=0.10,
    ... )
"""

from .generate import CfDNAGenerator
from .model import CfDNACausalLM, CfDNAConfig
from .tokens import (
    TOKEN_A,
    TOKEN_BOS,
    TOKEN_C,
    TOKEN_EOS,
    TOKEN_G,
    TOKEN_PAD,
    TOKEN_T,
    VOCAB_SIZE,
    get_ff_bin_token,
    get_gc_bin_token,
    get_len_bin_token,
    sequence_to_tokens,
    tokens_to_sequence,
)

__version__ = "1.0.0"
__author__ = "Kyle Redelinghuys"
__email__ = "kyle@eabhaseq.com"

__all__ = [
    # Main API
    "CfDNAGenerator",
    "CfDNACausalLM",
    "CfDNAConfig",
    # Tokens
    "TOKEN_A",
    "TOKEN_C",
    "TOKEN_G",
    "TOKEN_T",
    "TOKEN_BOS",
    "TOKEN_EOS",
    "TOKEN_PAD",
    "VOCAB_SIZE",
    # Token utilities
    "tokens_to_sequence",
    "sequence_to_tokens",
    "get_ff_bin_token",
    "get_len_bin_token",
    "get_gc_bin_token",
    # Version
    "__version__",
]
