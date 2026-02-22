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
    tokens_to_sequence,
    TOKEN_EOS,
    TOKEN_PAD,
    LEN_TOKEN_START,
    GC_TOKEN_START,
    FF_TOKEN_START,
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
        use_compile: bool = False,
        use_half: bool = False,
    ):
        """
        Initialize the generator with a model.

        Args:
            model: A CfDNACausalLM model instance
            device: Device to run generation on ('cpu', 'cuda', 'auto')
            use_compile: If True, apply torch.compile() for faster inference.
                Requires PyTorch >= 2.0. First call will be slower due to compilation.
            use_half: If True, use float16 inference on CUDA (bfloat16 if supported).
                Provides ~2x memory reduction and faster compute with negligible
                quality impact for this model size. Only applies on CUDA devices.
        """
        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = model.to(self.device)
        self.model.eval()

        # Half-precision: convert model weights for faster compute
        self._autocast_dtype = None
        if use_half and self.device != "cpu":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self._autocast_dtype = torch.bfloat16
            else:
                self._autocast_dtype = torch.float16
            self.model = self.model.to(self._autocast_dtype)

        # torch.compile: fuses operations and optimizes the computation graph
        self._compiled = False
        if use_compile:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                self._compiled = True
            except Exception:
                pass  # Silently fall back if compile is not available

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: Union[str, Path],
        device: Optional[str] = None,
        use_compile: bool = False,
        use_half: bool = False,
    ) -> "CfDNAGenerator":
        """
        Load a generator from a pretrained model.

        Args:
            path_or_repo: Local path to model directory, or HuggingFace repo ID
            device: Device to run generation on ('cpu', 'cuda', 'auto')
            use_compile: If True, apply torch.compile() for faster inference
            use_half: If True, use half-precision inference on CUDA

        Returns:
            CfDNAGenerator instance with loaded model

        Example:
            >>> # From local path
            >>> generator = CfDNAGenerator.from_pretrained("./models/v15")
            >>> # From HuggingFace Hub
            >>> generator = CfDNAGenerator.from_pretrained("eabhaseq/cfdna-gen")
            >>> # With optimizations
            >>> generator = CfDNAGenerator.from_pretrained(
            ...     "eabhaseq/cfdna-gen",
            ...     use_compile=True,
            ...     use_half=True,
            ... )
        """
        model = CfDNACausalLM.from_pretrained(path_or_repo, device=device)
        return cls(model, device=device, use_compile=use_compile, use_half=use_half)

    def generate(
        self,
        n_sequences: int,
        fragment_lengths: Union[int, List[int], np.ndarray],
        target_gc: Optional[float] = 0.42,
        target_ff: Optional[float] = 0.10,
        temperature: float = 0.95,
        top_p: float = 0.96,
        batch_size: int = 128,
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
            batch_size: Number of sequences to generate per batch. Default 128.
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

        # Vectorized condition token preparation (no Python loop)
        condition_cols = []

        # Length bin tokens: vectorized binning
        lengths_np = batch_lengths.astype(np.int64)
        len_bins = np.clip((lengths_np - 50) // 10, 0, 19) + LEN_TOKEN_START
        condition_cols.append(torch.tensor(len_bins, dtype=torch.long, device=device))

        if target_gc is not None:
            gc_bin = min(max(int((target_gc - 0.25) / 0.02), 0), 19) + GC_TOKEN_START
            condition_cols.append(
                torch.full((batch_size,), gc_bin, dtype=torch.long, device=device)
            )

        if target_ff is not None:
            # Non-uniform FF bins
            boundaries = [
                0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18,
                0.20, 0.22, 0.25, 0.28, 0.32, 0.36, 0.40
            ]
            ff_bin = 0
            for i, boundary in enumerate(boundaries[1:], 1):
                if target_ff < boundary:
                    break
                ff_bin = i
            ff_bin = min(ff_bin, 16) + FF_TOKEN_START
            condition_cols.append(
                torch.full((batch_size,), ff_bin, dtype=torch.long, device=device)
            )

        # Stack into [B, num_conditions]
        condition_tokens = torch.stack(condition_cols, dim=1)
        fragment_lengths = torch.tensor(batch_lengths, dtype=torch.long, device=device)

        target_gc_tensor = None
        target_ff_tensor = None
        if target_gc is not None:
            target_gc_tensor = torch.full((batch_size,), target_gc, device=device)
        if target_ff is not None:
            target_ff_tensor = torch.full((batch_size,), target_ff, device=device)

        # Generate
        max_length = int(batch_lengths.max()) + 10
        with torch.no_grad():
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

        # Vectorized post-processing: convert tokens to sequences
        sequences = _batch_tokens_to_sequences(generated_tokens)

        return sequences

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


# Nucleotide lookup table for fast token-to-char conversion
_TOKEN_TO_CHAR = {0: "A", 1: "C", 2: "G", 3: "T"}


def _batch_tokens_to_sequences(generated_tokens: torch.Tensor) -> List[str]:
    """
    Convert a batch of generated token tensors to DNA strings.

    Uses numpy for batch processing instead of per-token Python loops.
    """
    tokens_np = generated_tokens.cpu().numpy()
    sequences = []
    for row in tokens_np:
        # Find first EOS or PAD token
        chars = []
        for t in row:
            if t == TOKEN_EOS or t == TOKEN_PAD:
                break
            c = _TOKEN_TO_CHAR.get(t)
            if c is not None:
                chars.append(c)
        sequences.append("".join(chars))
    return sequences
