#!/usr/bin/env python3
"""
Basic cfDNA sequence generation example.

This script demonstrates the simplest use case: generating synthetic
cfDNA sequences with default parameters.
"""

from cfdna_gen import CfDNAGenerator


def main():
    # Load the pretrained model
    # First run will download from HuggingFace Hub (~500MB)
    print("Loading model...")
    generator = CfDNAGenerator.from_pretrained("eabhaseq/cfdna-gen")
    print("Model loaded!")

    # Generate 10 sequences with typical cfDNA parameters
    print("\nGenerating 10 cfDNA sequences...")
    sequences = generator.generate(
        n_sequences=10,
        fragment_lengths=165,  # Typical cfDNA fragment length
        target_gc=0.42,        # Typical cfDNA GC content
        target_ff=0.10,        # 10% fetal fraction
    )

    # Print the generated sequences
    print("\nGenerated sequences:")
    print("-" * 80)
    for i, seq in enumerate(sequences):
        gc = (seq.count('G') + seq.count('C')) / len(seq)
        print(f"Seq {i+1}: len={len(seq):3d}, GC={gc:.2f}")
        print(f"        {seq[:50]}...")
    print("-" * 80)

    # Generate with metadata for more details
    print("\nGenerating with metadata...")
    results = generator.generate_with_metadata(
        n_sequences=5,
        fragment_lengths=167,
        target_gc=0.42,
        target_ff=0.08,
    )

    print("\nResults with metadata:")
    for r in results:
        print(f"  Length: {r['length']}, GC: {r['gc_content']:.3f}, "
              f"Target GC: {r['target_gc']}, Target FF: {r['target_ff']}")


if __name__ == "__main__":
    main()
