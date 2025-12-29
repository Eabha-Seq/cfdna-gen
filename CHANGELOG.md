# Changelog

All notable changes to cfDNA-Gen will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
