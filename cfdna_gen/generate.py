"""
High-level API for cfDNA sequence generation.

This module provides a simple interface for generating synthetic cfDNA
sequences using the pretrained CfDNACausalLM model.

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

from pathlib import Path
from typing import List, Optional, Union

import torch
import numpy as np

from .model import CfDNACausalLM, CfDNAConfig
from .tokens import (
    get_len_bin_token,
    get_gc_bin_token,
    get_ff_bin_token,
    TOKEN_EOS,
    TOKEN_PAD,
)

__all__ = ["CfDNAGenerator"]


class CfDNAGenerator:
    """
    High-level API for generating synthetic cfDNA sequences.

    This class provides a simple interface for generating realistic cell-free
    DNA sequences with controllable properties like fragment length, GC content,
    and fetal fraction.

    Attributes:
        model: The underlying CfDNACausalLM model
        device: Device the model is running on

    Example:
        >>> from cfdna_gen import CfDNAGenerator
        >>>
        >>> # Load pretrained model
        >>> generator = CfDNAGenerator.from_pretrained("eabhaseq/cfdna-gen")
        >>>
        >>> # Generate sequences
        >>> sequences = generator.generate(
        ...     n_sequences=100,
        ...     fragment_lengths=165,
        ...     target_gc=0.42,
        ...     target_ff=0.10,
        ... )
        >>>
        >>> for seq in sequences[:5]:
        ...     print(seq)
    """

    def __init__(
        self,
        model: CfDNACausalLM,
        device: Optional[str] = None,
        dtype: Union[torch.dtype, str] = "auto",
        compile: Optional[bool] = None,
    ):
        """
        Initialize the generator with a model.

        Args:
            model: A CfDNACausalLM model instance
            device: Device to run generation on ('cpu', 'cuda', 'auto')
            dtype: Model dtype. 'auto' uses BF16 on CUDA, FP32 on CPU.
                Accepts torch.bfloat16, torch.float16, torch.float32, or 'auto'.
            compile: Whether to torch.compile the generation forward pass.
                Default: False. The autoregressive token-by-token loop with
                varying cache positions defeats graph capture, so compilation
                adds overhead without speedup.
        """
        self.model = model

        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Resolve dtype
        if dtype == "auto":
            self._dtype = torch.bfloat16 if "cuda" in self.device else torch.float32
        else:
            self._dtype = dtype

        self.model = self.model.to(device=self.device, dtype=self._dtype)
        self.model.eval()

        # Resolve compile (lazy — actual compilation on first generate call)
        if compile is None:
            self._compile = False
        else:
            self._compile = compile
        self._compiled = False

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: Union[str, Path],
        device: Optional[str] = None,
        dtype: Union[torch.dtype, str] = "auto",
        compile: Optional[bool] = None,
    ) -> "CfDNAGenerator":
        """
        Load a generator from a pretrained model.

        Args:
            path_or_repo: Local path to model directory, or HuggingFace repo ID
            device: Device to run generation on ('cpu', 'cuda', 'auto')
            dtype: Model dtype. 'auto' uses BF16 on CUDA, FP32 on CPU.
                Accepts torch.bfloat16, torch.float16, torch.float32, or 'auto'.
            compile: Whether to torch.compile the generation forward pass.
                Default: False. The autoregressive token-by-token loop with
                varying cache positions defeats graph capture, so compilation
                adds overhead without speedup.

        Returns:
            CfDNAGenerator instance with loaded model

        Example:
            >>> # From local path
            >>> generator = CfDNAGenerator.from_pretrained("./models/v15")
            >>> # From HuggingFace Hub
            >>> generator = CfDNAGenerator.from_pretrained("eabhaseq/cfdna-gen")
        """
        # Load to CPU first; __init__ handles device/dtype cast
        model = CfDNACausalLM.from_pretrained(path_or_repo, device="cpu")
        return cls(model, device=device, dtype=dtype, compile=compile)

    def generate(
        self,
        n_sequences: int,
        fragment_lengths: Union[int, List[int], np.ndarray],
        target_gc: Optional[float] = 0.42,
        target_ff: Optional[float] = 0.10,
        temperature: float = 0.95,
        top_p: float = 0.96,
        batch_size: int = 512,
        show_progress: bool = False,
    ) -> List[str]:
        """
        Generate synthetic cfDNA sequences.

        Args:
            n_sequences: Number of sequences to generate
            fragment_lengths: Target fragment length(s) in base pairs.
                Can be a single int (same for all), list, or numpy array.
            target_gc: Target GC content (0.0-1.0). Default is 0.42 (typical cfDNA).
            target_ff: Target fetal fraction (0.0-0.5). Default is 0.10 (10%).
            temperature: Sampling temperature. Higher = more random. Default 0.95.
            top_p: Nucleus sampling threshold. Default 0.96.
            batch_size: Sequences per forward-batch. Default 512. Pass 128
                on GPUs with less than ~8 GB.
            show_progress: Whether to show a progress bar. Default False.

        Returns:
            List of DNA sequence strings (A, C, G, T)

        Example:
            >>> # Generate 100 sequences of ~165bp
            >>> sequences = generator.generate(
            ...     n_sequences=100,
            ...     fragment_lengths=165,
            ...     target_gc=0.42,
            ...     target_ff=0.10,
            ... )
            >>>
            >>> # Generate with varying lengths
            >>> import numpy as np
            >>> lengths = np.random.normal(167, 12, size=100).astype(int)
            >>> sequences = generator.generate(
            ...     n_sequences=100,
            ...     fragment_lengths=lengths,
            ... )
        """
        # Handle fragment lengths
        if isinstance(fragment_lengths, int):
            lengths = np.full(n_sequences, fragment_lengths)
        elif isinstance(fragment_lengths, list):
            lengths = np.array(fragment_lengths)
            if len(lengths) == 1:
                lengths = np.full(n_sequences, lengths[0])
        else:
            lengths = np.asarray(fragment_lengths)

        if len(lengths) != n_sequences:
            raise ValueError(
                f"fragment_lengths has {len(lengths)} elements but n_sequences={n_sequences}"
            )

        # Lazy torch.compile on first call
        if self._compile and not self._compiled:
            self.model._forward_with_static_cache = torch.compile(
                self.model._forward_with_static_cache,
                mode="default",
                dynamic=True,
            )
            self._compiled = True

        # Generate in batches
        all_sequences = []
        n_batches = (n_sequences + batch_size - 1) // batch_size

        iterator = range(n_batches)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Generating", unit="batch")
            except ImportError:
                pass

        for batch_idx in iterator:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_sequences)
            batch_lengths = lengths[start_idx:end_idx]

            batch_sequences = self._generate_batch(
                batch_lengths=batch_lengths,
                target_gc=target_gc,
                target_ff=target_ff,
                temperature=temperature,
                top_p=top_p,
            )
            all_sequences.extend(batch_sequences)

        return all_sequences

    def _generate_batch(
        self,
        batch_lengths: np.ndarray,
        target_gc: Optional[float],
        target_ff: Optional[float],
        temperature: float,
        top_p: float,
    ) -> List[str]:
        """Generate a batch of sequences."""
        batch_size = len(batch_lengths)
        device = self.device

        # Build condition tokens
        condition_tokens = []
        for length in batch_lengths:
            tokens = [
                get_len_bin_token(int(length)),
            ]
            if target_gc is not None:
                tokens.append(get_gc_bin_token(target_gc))
            if target_ff is not None:
                tokens.append(get_ff_bin_token(target_ff))
            condition_tokens.append(tokens)

        # Pad to same length
        max_cond_len = max(len(t) for t in condition_tokens)
        condition_tokens = [
            t + [0] * (max_cond_len - len(t)) for t in condition_tokens
        ]

        condition_tokens = torch.tensor(condition_tokens, dtype=torch.long, device=device)
        fragment_lengths = torch.tensor(batch_lengths, dtype=torch.long, device=device)

        target_gc_tensor = None
        target_ff_tensor = None
        if target_gc is not None:
            target_gc_tensor = torch.full((batch_size,), target_gc, device=device)
        if target_ff is not None:
            target_ff_tensor = torch.full((batch_size,), target_ff, device=device)

        max_length = int(batch_lengths.max()) + 10
        with torch.inference_mode():
            generated_tokens = self.model.generate(
                condition_tokens=condition_tokens,
                fragment_length=fragment_lengths,
                target_gc=target_gc_tensor,
                target_ff=target_ff_tensor,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                enforce_length=True,
            )

        return batch_tokens_to_sequences(generated_tokens)

    def generate_with_metadata(
        self,
        n_sequences: int,
        fragment_lengths: Union[int, List[int], np.ndarray],
        target_gc: Optional[float] = 0.42,
        target_ff: Optional[float] = 0.10,
        **kwargs,
    ) -> List[dict]:
        """
        Generate sequences with metadata.

        Returns a list of dicts with 'sequence', 'length', 'gc_content', etc.

        Args:
            n_sequences: Number of sequences to generate
            fragment_lengths: Target fragment length(s) in base pairs
            target_gc: Target GC content (0.0-1.0)
            target_ff: Target fetal fraction (0.0-0.5)
            **kwargs: Additional arguments passed to generate()

        Returns:
            List of dicts with sequence and metadata

        Example:
            >>> results = generator.generate_with_metadata(
            ...     n_sequences=10,
            ...     fragment_lengths=165,
            ... )
            >>> for r in results[:3]:
            ...     print(f"Length: {r['length']}, GC: {r['gc_content']:.2f}")
        """
        sequences = self.generate(
            n_sequences=n_sequences,
            fragment_lengths=fragment_lengths,
            target_gc=target_gc,
            target_ff=target_ff,
            **kwargs,
        )

        results = []
        for seq in sequences:
            gc_count = seq.count('G') + seq.count('C')
            gc_content = gc_count / len(seq) if len(seq) > 0 else 0.0

            results.append({
                'sequence': seq,
                'length': len(seq),
                'gc_content': gc_content,
                'target_gc': target_gc,
                'target_ff': target_ff,
            })

        return results

    def generate_fastq(
        self,
        n_sequences: int,
        fragment_lengths: Union[int, List[int], np.ndarray],
        output_path: Union[str, Path],
        target_gc: Optional[float] = 0.42,
        target_ff: Optional[float] = 0.10,
        quality_score: int = 30,
        **kwargs,
    ) -> int:
        """
        Generate sequences and write directly to a FASTQ file.

        Args:
            n_sequences: Number of sequences to generate
            fragment_lengths: Target fragment length(s) in base pairs
            output_path: Path to output FASTQ file (.fastq or .fastq.gz)
            target_gc: Target GC content (0.0-1.0)
            target_ff: Target fetal fraction (0.0-0.5)
            quality_score: Phred quality score for all bases (default: 30)
            **kwargs: Additional arguments passed to generate()

        Returns:
            Number of sequences written

        Example:
            >>> generator.generate_fastq(
            ...     n_sequences=10000,
            ...     fragment_lengths=165,
            ...     output_path="synthetic.fastq.gz",
            ... )
        """
        import gzip

        output_path = Path(output_path)
        sequences = self.generate(
            n_sequences=n_sequences,
            fragment_lengths=fragment_lengths,
            target_gc=target_gc,
            target_ff=target_ff,
            **kwargs,
        )

        quality_char = chr(quality_score + 33)

        open_fn = gzip.open if str(output_path).endswith('.gz') else open
        mode = 'wt' if str(output_path).endswith('.gz') else 'w'

        with open_fn(output_path, mode) as f:
            for i, seq in enumerate(sequences):
                f.write(f"@synthetic_cfdna_{i:08d}\n")
                f.write(f"{seq}\n")
                f.write("+\n")
                f.write(f"{quality_char * len(seq)}\n")

        return len(sequences)


# Token 0-3 → ASCII A/C/G/T. One byte lookup so a row becomes one decode.
_TOKEN_TO_BYTE = np.array([ord("A"), ord("C"), ord("G"), ord("T")], dtype=np.uint8)


def batch_tokens_to_sequences(generated_tokens: torch.Tensor) -> List[str]:
    """Convert a [B, L] token tensor to DNA strings.

    One host copy for the whole batch. Stops at the first EOS or PAD per row,
    then keeps nucleotide tokens (0-3) only — same result as the old
    per-row ``tokens_to_sequence`` loop.
    """
    tokens_np = generated_tokens.detach().to(device="cpu", dtype=torch.int64).numpy()
    stop = (tokens_np == TOKEN_EOS) | (tokens_np == TOKEN_PAD)
    has_stop = stop.any(axis=1)
    stop_idx = np.where(has_stop, stop.argmax(axis=1), tokens_np.shape[1])

    sequences: List[str] = []
    for i, row in enumerate(tokens_np):
        valid = row[: stop_idx[i]]
        nuc = valid[(valid >= 0) & (valid <= 3)]
        if nuc.size == 0:
            sequences.append("")
        else:
            sequences.append(_TOKEN_TO_BYTE[nuc].tobytes().decode("ascii"))
    return sequences
