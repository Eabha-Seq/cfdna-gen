# cfDNA-Gen

[![PyPI version](https://badge.fury.io/py/cfdna-gen.svg)](https://badge.fury.io/py/cfdna-gen)
[![License: PolyForm Noncommercial](https://img.shields.io/badge/License-PolyForm%20Noncommercial-red.svg)](https://polyformproject.org/licenses/noncommercial/1.0.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Conditional Causal Transformer for Cell-Free DNA Sequence Generation**

Generate realistic synthetic cell-free DNA (cfDNA) sequences for NIPT
simulation, benchmark development, and genomics research.

## Overview

cfDNA-Gen is a 120M parameter causal transformer trained on real cell-free
DNA data. It generates synthetic cfDNA sequences with controllable
**fragment length** and **GC content**. That is what the published v15
weights actually do well:

- Length-conditioned sequence realism (typically 50–250 bp)
- GC-conditioned composition (typical cfDNA ~42%)
- Nucleosome-associated 10 bp periodicity and characteristic end motifs

**Fetal fraction is library-level style conditioning**, not a
per-fragment origin switch. The API takes `target_ff` (default **0.10**)
on a dual path: an FF **bin token** plus a continuous `FFEmbedding`.
Bins are non-uniform; the left tail is coarse (**0–2% share one bin** —
0.5% and 1.9% are the same token).

### Published v15 weights (`eabhaseq/cfdna-gen`)

The continuous FF path is **effectively collapsed** (near-constant), and
the 17 FF bin token embeddings are nearly identical. **Do not claim that
setting `target_ff` alone simulates different fetal fractions or bimodal
low-FF libraries.** Low-FF packs must define that look in a **mixer**
(length mix, maternal vs fetal origin, coverage) — for example
`eabhaseq-synthetic` — not via `target_ff` alone.

A bimodal length mix (fetal-like ~144 bp, maternal-like ~167 bp) is
created by **sampling `fragment_lengths`**, not by changing `target_ff`.
See [docs/FF_CONDITIONING_FIX.md](docs/FF_CONDITIONING_FIX.md) for what a
proper continued train would need. This public repo is inference-only.

## Installation

```bash
pip install cfdna-gen

# With HuggingFace Hub support (recommended)
pip install cfdna-gen[hub]
```

## Quick Start

```python
from cfdna_gen import CfDNAGenerator

# Load pretrained model (downloads automatically from HuggingFace)
generator = CfDNAGenerator.from_pretrained("eabhaseq/cfdna-gen")

# Generate 100 cfDNA sequences
sequences = generator.generate(
    n_sequences=100,
    fragment_lengths=165,  # Target length in bp
    target_gc=0.42,        # Target GC content
    target_ff=0.10,        # Default library-level FF token (v15: collapsed)
)

for seq in sequences[:5]:
    print(seq)
```

## Features

### Basic Generation

```python
# Generate sequences with specific length
sequences = generator.generate(
    n_sequences=100,
    fragment_lengths=165,
)
```

### Variable Lengths

```python
import numpy as np

# Bimodal fetal/maternal *look* comes from the length mix, not target_ff
lengths = np.random.normal(167, 12, size=100).astype(int)
lengths = np.clip(lengths, 100, 250)

sequences = generator.generate(
    n_sequences=100,
    fragment_lengths=lengths,
)
```

### With Metadata

```python
results = generator.generate_with_metadata(
    n_sequences=100,
    fragment_lengths=165,
    target_gc=0.42,
    target_ff=0.10,
)

for r in results[:5]:
    print(f"Length: {r['length']}, GC: {r['gc_content']:.2f}")
```

### Direct FASTQ Output

```python
# Generate and write directly to FASTQ
generator.generate_fastq(
    n_sequences=10000,
    fragment_lengths=165,
    output_path="synthetic_cfdna.fastq.gz",
    target_gc=0.42,
    target_ff=0.10,
)
```

### Sampling Parameters

```python
# More diverse sequences (higher temperature)
sequences = generator.generate(
    n_sequences=100,
    fragment_lengths=165,
    temperature=1.0,    # Default: 0.95
    top_p=0.98,         # Default: 0.96
)

# More deterministic sequences (lower temperature)
sequences = generator.generate(
    n_sequences=100,
    fragment_lengths=165,
    temperature=0.7,
    top_p=0.90,
)
```

### Batch Processing

```python
# Large-scale generation with progress bar
sequences = generator.generate(
    n_sequences=100000,
    fragment_lengths=167,
    batch_size=256,
    show_progress=True,
)
```

## Model Architecture

```
Input (Condition Tokens + Sequence)
    |
Token Embedding (64 -> 768)
    |
+ Length Embedding (continuous)
+ GC Embedding (continuous)
+ FF Embedding (continuous; collapsed on published v15)
    |
14 x TransformerBlock (Pre-norm)
  |-- RMSNorm
  |-- CausalSelfAttention (12 heads, RoPE, Flash Attention)
  +-- SwiGLU FFN (768 -> 3072 -> 768)
    |
RMSNorm
    |
Output Projection (768 -> 64)
```

**Key features:**
- **RoPE (Rotary Position Embeddings)**: Better handling of sequence positions
- **SwiGLU activation**: Improved feed-forward network performance
- **RMSNorm**: Efficient layer normalization
- **Flash Attention (SDPA)**: Fast and memory-efficient attention

## Model Variants

| Model | Size | Description |
|-------|------|-------------|
| `eabhaseq/cfdna-gen` | ~500MB | v15 public checkpoint. Strong length/GC-conditioned sequence realism. FF path present but collapsed — see above. |

There is no separate in-tree Hugging Face model card; treat this README
and [docs/FF_CONDITIONING_FIX.md](docs/FF_CONDITIONING_FIX.md) as the
source of truth until a post-retrain card is published.

## Local Model Loading

```python
# Load from local path
generator = CfDNAGenerator.from_pretrained("./path/to/model")

# Or use the model directly
from cfdna_gen import CfDNACausalLM

model = CfDNACausalLM.from_pretrained("./path/to/model")
```

Loading a checkpoint whose continuous FF embed barely moves between
0.01 and 0.10 emits a `UserWarning`. To run the same check on real
weights (not executed in CI):

```bash
python scripts/check_ff_embedding_collapse.py --model eabhaseq/cfdna-gen
```

## Validation Results

The v15 numbers below measure **length/GC-conditioned sequence realism**
against real cfDNA. They are **not** FF-stratified and do **not** show
that `target_ff` changes generated sequences.

| Metric | Score |
|--------|-------|
| Overall Similarity | 92.9% |
| Fragment Length Match | 98.4% |
| GC Content Match | 93.3% |
| Nucleotide Frequency | 99.0% |
| Bimodal Peaks Detection | 100% |
| Nucleosome Periodicity | 100% |

Bimodal peak detection reflects mixed `fragment_lengths` in the
validation suite (short fetal-like + longer maternal-like), not
`target_ff`.

## Use Cases

- **NIPT simulation (mixer-defined mix):** Generate length/GC-realistic
  fragments; put maternal vs fetal origin, coverage, and low-FF look in
  the caller or `eabhaseq-synthetic`, not in `target_ff` alone
- **Benchmarking:** Create standardized test datasets for cfDNA analysis pipelines
- **Training data:** Augment real datasets for machine learning applications
- **Method development:** Test new analysis methods on controlled synthetic data

## Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- (Optional) huggingface-hub >= 0.20.0
- (Optional) safetensors >= 0.4.0

## Citation

If you use cfDNA-Gen in your research, please cite:

```bibtex
@software{cfdna_gen,
  title={cfDNA-Gen: Conditional Causal Transformer for cfDNA Sequence Generation},
  author={Redelinghuys, Kyle},
  year={2025},
  url={https://github.com/eabhaseq/cfdna-gen}
}
```

## License

This software is dual-licensed:

- **Free** for academic, research, educational, and personal non-commercial use under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0).
- **Commercial use** requires a separate paid license. Contact [kyle@eabhaseq.com](mailto:kyle@eabhaseq.com) for details.

See [LICENSE](LICENSE) for full terms.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

This repository is the **inference** package. Training and data
pipelines are private; see [docs/FF_CONDITIONING_FIX.md](docs/FF_CONDITIONING_FIX.md)
if you are planning a continued-train FF fix.

## Acknowledgments

This work was developed for advancing non-invasive prenatal testing (NIPT) research and synthetic genomics data generation.
