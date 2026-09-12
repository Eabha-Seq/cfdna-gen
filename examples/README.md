# cfDNA-Gen Examples

This directory contains example scripts demonstrating how to use cfDNA-Gen.

## Examples

### basic_generation.py

The simplest example showing how to generate cfDNA sequences with default parameters.

```bash
python basic_generation.py
```

### variable_lengths.py

Demonstrates a mixer-style bimodal length mix (shorter fetal-like + longer
maternal-like fragments). On published v15 weights that look comes from
`fragment_lengths`, not from `target_ff`.

```bash
python variable_lengths.py
```

### batch_generation.py

Shows how to efficiently generate large numbers of sequences using batching, with performance metrics and FASTQ output.

```bash
python batch_generation.py
```

## Running Examples

1. First, install cfDNA-Gen with hub support:
   ```bash
   pip install cfdna-gen[hub]
   ```

2. Run any example:
   ```bash
   cd examples
   python basic_generation.py
   ```

The first run will download the pretrained model from HuggingFace (~500MB).
