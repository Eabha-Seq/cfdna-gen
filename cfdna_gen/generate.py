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

import numpy as np
import torch

from .model import CfDNACausalLM
from .tokens import (
    LEN_TOKEN_START,
    TOKEN_EOS,
    TOKEN_PAD,
    get_ff_bin_token,
    get_gc_bin_token,
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
        device: str | None = None,
        use_compile: bool = False,
        use_half: bool | None = None,
    ):
        """
        Initialize the generator with a model.

        Args:
            model: A CfDNACausalLM model instance
            device: Device to run generation on ('cpu', 'cuda', 'auto')
            use_compile: If True, apply torch.compile() to the model. First
                batch is slower; later batches are faster. Opt-in because
                compile is environment-dependent.
            use_half: If True, run in bfloat16 (or float16) on CUDA. If None
                (default), half precision is enabled automatically on CUDA.
                Same sampling distribution; halves weight-read traffic.
        """
        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = model.to(self.device)
        self.model.eval()

        if use_half is None:
            use_half = self.device != "cpu"

        self._autocast_dtype = None
        if use_half and self.device != "cpu":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self._autocast_dtype = torch.bfloat16
            else:
                self._autocast_dtype = torch.float16
            self.model = self.model.to(self._autocast_dtype)

        self._compiled = False
        if use_compile:
            try:
                self.model = torch.compile(self.model, mode="default")  # type: ignore[assignment]
                self._compiled = True
            except Exception:
                pass

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str | Path,
        device: str | None = None,
        use_compile: bool = False,
        use_half: bool | None = None,
    ) -> "CfDNAGenerator":
        """
        Load a generator from a pretrained model.

        Args:
            path_or_repo: Local path to model directory, or HuggingFace repo ID
            device: Device to run generation on ('cpu', 'cuda', 'auto')
            use_compile: If True, apply torch.compile() for faster inference
            use_half: If True, use half-precision on CUDA. Default: on for CUDA.

        Returns:
            CfDNAGenerator instance with loaded model

        Example:
            >>> # From local path
            >>> generator = CfDNAGenerator.from_pretrained("./models/v15")
            >>> # From HuggingFace Hub
            >>> generator = CfDNAGenerator.from_pretrained("eabhaseq/cfdna-gen")
            >>> # Maximum throughput on GPU
            >>> generator = CfDNAGenerator.from_pretrained(
            ...     "eabhaseq/cfdna-gen",
            ...     use_compile=True,
            ... )
        """
        model = CfDNACausalLM.from_pretrained(path_or_repo, device=device)
        return cls(
            model, device=device, use_compile=use_compile, use_half=use_half
        )

    def generate(
        self,
        n_sequences: int,
        fragment_lengths: int | list[int] | np.ndarray,
        target_gc: float | None = 0.42,
        target_ff: float | None = 0.10,
        temperature: float = 0.95,
        top_p: float = 0.96,
        batch_size: int = 512,
        show_progress: bool = False,
    ) -> list[str]:
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
            batch_size: Sequences per forward-batch. Default 512. Lower this
                (e.g. 128) on GPUs with less than ~8 GB.
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
        # Handle fragment lengths. Always go through asarray so mypy sees one ndarray type.
        if isinstance(fragment_lengths, int):
            lengths = np.asarray([fragment_lengths] * n_sequences, dtype=np.int64)
        else:
            lengths = np.asarray(fragment_lengths, dtype=np.int64).reshape(-1)
            if lengths.size == 1:
                lengths = np.full(n_sequences, int(lengths[0]), dtype=np.int64)

        if lengths.size != n_sequences:
            raise ValueError(
                f"fragment_lengths has {lengths.size} elements but n_sequences={n_sequences}"
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
        target_gc: float | None,
        target_ff: float | None,
        temperature: float,
        top_p: float,
    ) -> list[str]:
        """Generate a batch of sequences."""
        batch_size = len(batch_lengths)
        device = self.device

        condition_cols = []
        len_bins = np.clip((batch_lengths.astype(np.int64) - 50) // 10, 0, 19) + LEN_TOKEN_START
        condition_cols.append(torch.tensor(len_bins, dtype=torch.long, device=device))

        if target_gc is not None:
            gc_bin = get_gc_bin_token(target_gc)
            condition_cols.append(
                torch.full((batch_size,), gc_bin, dtype=torch.long, device=device)
            )
        if target_ff is not None:
            ff_bin = get_ff_bin_token(target_ff)
            condition_cols.append(
                torch.full((batch_size,), ff_bin, dtype=torch.long, device=device)
            )

        condition_tokens = torch.stack(condition_cols, dim=1)
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
        fragment_lengths: int | list[int] | np.ndarray,
        target_gc: float | None = 0.42,
        target_ff: float | None = 0.10,
        **kwargs,
    ) -> list[dict]:
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
            gc_count = seq.count("G") + seq.count("C")
            gc_content = gc_count / len(seq) if len(seq) > 0 else 0.0

            results.append({
                "sequence": seq,
                "length": len(seq),
                "gc_content": gc_content,
                "target_gc": target_gc,
                "target_ff": target_ff,
            })

        return results

    def generate_fastq(
        self,
        n_sequences: int,
        fragment_lengths: int | list[int] | np.ndarray,
        output_path: str | Path,
        target_gc: float | None = 0.42,
        target_ff: float | None = 0.10,
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

        open_fn = gzip.open if str(output_path).endswith(".gz") else open
        mode = "wt" if str(output_path).endswith(".gz") else "w"

        with open_fn(output_path, mode) as f:
            for i, seq in enumerate(sequences):
                f.write(f"@synthetic_cfdna_{i:08d}\n")  # type: ignore[arg-type]
                f.write(f"{seq}\n")  # type: ignore[arg-type]
                f.write("+\n")  # type: ignore[arg-type]
                f.write(f"{quality_char * len(seq)}\n")  # type: ignore[arg-type]

        return len(sequences)


# Token 0-3 → ASCII A/C/G/T. Used as a byte lookup so a row becomes one decode.
_TOKEN_TO_BYTE = np.array([ord("A"), ord("C"), ord("G"), ord("T")], dtype=np.uint8)


def batch_tokens_to_sequences(generated_tokens: torch.Tensor) -> list[str]:
    """
    Convert a [B, L] token tensor to DNA strings.

    One host copy for the whole batch. Stops at the first EOS or PAD per row.
    """
    tokens_np = generated_tokens.detach().to(device="cpu", dtype=torch.int64).numpy()
    stop = (tokens_np == TOKEN_EOS) | (tokens_np == TOKEN_PAD)
    has_stop = stop.any(axis=1)
    stop_idx = np.where(has_stop, stop.argmax(axis=1), tokens_np.shape[1])

    sequences: list[str] = []
    for i, row in enumerate(tokens_np):
        valid = row[: stop_idx[i]]
        nuc = valid[(valid >= 0) & (valid <= 3)]
        if nuc.size == 0:
            sequences.append("")
        else:
            sequences.append(_TOKEN_TO_BYTE[nuc].tobytes().decode("ascii"))
    return sequences
