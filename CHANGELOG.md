# Changelog

All notable changes to cfDNA-Gen will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- README and API docs no longer claim that `target_ff` alone simulates
  fetal fraction or bimodal low-FF libraries on published v15 weights.
  FF is documented as library-level style conditioning (bin token +
  continuous embed); default remains `target_ff=0.10`; left tail 0–2%
  shares one bin.
- Condition-token batch padding uses `TOKEN_PAD` instead of `TOKEN_A`.

### Added
- `decode_ff_bin_token` and `validate_fetal_fraction` (warns outside 0.0–0.5)
- Serve-time warning when a loaded checkpoint’s continuous FF embed is
  collapsed; `scripts/check_ff_embedding_collapse.py` for real weights
  (not run in CI)
- [docs/FF_CONDITIONING_FIX.md](docs/FF_CONDITIONING_FIX.md) — training
  brief for a proper continued-train FF fix
- Unit tests for the left-tail bin map (`0.005` vs `0.019` vs `0.02`)

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
- v15 model with per-sequence GC conditioning and an FF dual path
- 92.9% overall similarity to real cfDNA data (length/GC realism; not FF-stratified)
- Captures bimodal length distribution (via mixed `fragment_lengths`) and nucleosome periodicity
- Published v15 continuous FF path is collapsed; see Unreleased notes and docs/FF_CONDITIONING_FIX.md
