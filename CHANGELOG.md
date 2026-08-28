# Changelog

All notable changes to cfDNA-Gen will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Autoregressive decode now uses a pre-allocated `StaticKVCache` instead of `torch.cat` every token.
- `CfDNACausalLM.generate` no longer calls `finished.all()` each step (that was a CUDA sync).
- Default `generate(batch_size=...)` is 512 (was 128). Pass 128 on smaller GPUs.
- On CUDA, `CfDNAGenerator` defaults to bfloat16/float16. Pass `use_half=False` for FP32.
- Condition-token setup and token-to-sequence conversion are vectorized (one host copy per batch).

### Added
- `use_compile` / `use_half` flags on `CfDNAGenerator` and `from_pretrained`.
- `StaticKVCache` for compile-friendly decode.

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
