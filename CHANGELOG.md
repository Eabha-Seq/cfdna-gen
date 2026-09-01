# Changelog

All notable changes to cfDNA-Gen will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Decode uses a pre-allocated static KV cache (no `torch.cat` per token).
- `generate()` no longer calls `finished.all()` each step (CUDA sync).
- Default `batch_size` is 512 (was 128). Pass 128 on smaller GPUs.
- On CUDA, `CfDNAGenerator` defaults to bfloat16. Pass `dtype=torch.float32` for FP32.
- Token-to-sequence conversion is vectorized (one host copy per batch).
- Sampling is unchanged: same temperature, top-p scatter, length/GC/FF conditioning.

### Added
- `dtype` / `compile` flags on `CfDNAGenerator`. `compile` stays off by default.

## [1.0.0] - 2025-01-01

### Added
- Initial public release
- CfDNAGenerator high-level API for sequence generation
- CfDNACausalLM 120M parameter transformer model
- Support for conditioning on fragment length, GC content, and fetal fraction
- HuggingFace Hub integration for model loading
- FASTQ output support
- Comprehensive documentation and examples

### Model
- v15 model with per-sequence GC/FF conditioning
- 92.9% overall similarity to real cfDNA data
- Captures bimodal length distribution and nucleosome periodicity
