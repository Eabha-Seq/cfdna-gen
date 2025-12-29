#!/usr/bin/env python3
"""
Large-scale batch generation example.

This script demonstrates how to generate large numbers of sequences
efficiently using batching and shows performance metrics.
"""

import time
import numpy as np
from cfdna_gen import CfDNAGenerator


def main():
    print("Loading model...")
    generator = CfDNAGenerator.from_pretrained("eabhaseq/cfdna-gen")

    # Generate different batch sizes
    for n_sequences in [100, 1000, 10000]:
        print(f"\n{'='*60}")
        print(f"Generating {n_sequences:,} sequences...")

        start_time = time.time()
        sequences = generator.generate(
            n_sequences=n_sequences,
            fragment_lengths=165,
            target_gc=0.42,
            target_ff=0.10,
            batch_size=256,
            show_progress=True,
        )
        elapsed = time.time() - start_time

        # Calculate statistics
        lengths = [len(s) for s in sequences]
        gc_contents = [(s.count('G') + s.count('C')) / len(s) for s in sequences]

        print(f"\nResults:")
        print(f"  Time: {elapsed:.2f}s ({n_sequences/elapsed:.0f} seq/s)")
        print(f"  Length: mean={np.mean(lengths):.1f}, std={np.std(lengths):.1f}")
        print(f"  GC: mean={np.mean(gc_contents):.3f}, std={np.std(gc_contents):.3f}")

    # Generate to FASTQ file
    print(f"\n{'='*60}")
    print("Generating FASTQ file...")

    start_time = time.time()
    n_written = generator.generate_fastq(
        n_sequences=10000,
        fragment_lengths=165,
        output_path="/tmp/synthetic_cfdna.fastq.gz",
        target_gc=0.42,
        target_ff=0.10,
        batch_size=256,
        show_progress=True,
    )
    elapsed = time.time() - start_time

    print(f"\nFASTQ generation:")
    print(f"  Sequences: {n_written:,}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Output: /tmp/synthetic_cfdna.fastq.gz")


if __name__ == "__main__":
    main()
