#!/usr/bin/env python3
"""
Generate cfDNA sequences with variable fragment lengths.

This example demonstrates how to generate sequences with a realistic
fragment length distribution, mimicking real cfDNA data.
"""

import numpy as np
from cfdna_gen import CfDNAGenerator


def main():
    print("Loading model...")
    generator = CfDNAGenerator.from_pretrained("eabhaseq/cfdna-gen")

    # Real cfDNA has a bimodal distribution:
    # - Fetal fragments: ~144bp (shorter)
    # - Maternal fragments: ~167bp (longer)

    n_sequences = 100
    fetal_fraction = 0.10  # 10% fetal

    # Sample fragment lengths from bimodal distribution
    n_fetal = int(n_sequences * fetal_fraction)
    n_maternal = n_sequences - n_fetal

    # Fetal fragments (shorter, ~144bp mean)
    fetal_lengths = np.random.normal(144, 8, size=n_fetal).astype(int)
    fetal_lengths = np.clip(fetal_lengths, 100, 180)

    # Maternal fragments (longer, ~167bp mean)
    maternal_lengths = np.random.normal(167, 12, size=n_maternal).astype(int)
    maternal_lengths = np.clip(maternal_lengths, 140, 220)

    # Combine and shuffle
    all_lengths = np.concatenate([fetal_lengths, maternal_lengths])
    np.random.shuffle(all_lengths)

    print(f"\nFragment length statistics:")
    print(f"  Mean: {all_lengths.mean():.1f}bp")
    print(f"  Std:  {all_lengths.std():.1f}bp")
    print(f"  Min:  {all_lengths.min()}bp")
    print(f"  Max:  {all_lengths.max()}bp")

    # Generate with variable lengths
    print(f"\nGenerating {n_sequences} sequences with variable lengths...")
    sequences = generator.generate(
        n_sequences=n_sequences,
        fragment_lengths=all_lengths,
        target_gc=0.42,
        target_ff=fetal_fraction,
        show_progress=True,
    )

    # Verify output lengths match input
    actual_lengths = [len(s) for s in sequences]
    print(f"\nOutput length statistics:")
    print(f"  Mean: {np.mean(actual_lengths):.1f}bp")
    print(f"  Std:  {np.std(actual_lengths):.1f}bp")

    # Show a few examples
    print("\nSample sequences:")
    for i in range(min(5, len(sequences))):
        seq = sequences[i]
        gc = (seq.count('G') + seq.count('C')) / len(seq)
        print(f"  {i+1}. len={len(seq):3d}, GC={gc:.2f}: {seq[:40]}...")


if __name__ == "__main__":
    main()
