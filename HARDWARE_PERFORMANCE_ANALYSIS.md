# cfDNA-Gen Hardware Performance Analysis

## Model Profile Summary

| Property | Value |
|---|---|
| Architecture | Decoder-only causal transformer |
| Parameters | ~132M (reported as "120M", weight-tied) |
| Hidden dim | 768 |
| Layers | 14 |
| Attention heads | 12 (head_dim=64) |
| FFN dim | 3072 (SwiGLU) |
| Max sequence length | 256 tokens |
| Vocab size | 64 |
| Default precision | FP32 |
| Attention | PyTorch SDPA (Flash Attention kernel) |
| KV cache | Yes (autoregressive generation) |
| Default batch size | 128 (examples use 256) |

## Why GPU Choice Matters: The Bottleneck Is Memory Bandwidth

This model's primary use case is **autoregressive generation** — producing DNA sequences token-by-token. This is fundamentally different from training or batch inference workloads.

During autoregressive generation, each step:
1. Reads the **entire model weights** (~530 MB in FP32, ~265 MB in FP16) from GPU memory
2. Reads the **entire KV cache** (grows with sequence position) from GPU memory
3. Performs a relatively small amount of arithmetic (batched matrix-vector products)

The arithmetic intensity (FLOPs per byte read) is extremely low — approximately **2 FLOPs/byte**. This means the GPU's compute cores (TFLOPS) sit mostly idle, waiting for data to arrive from memory. The generation speed is almost entirely determined by **memory bandwidth** (GB/s), not TFLOPS.

### Memory Footprint (Very Small)

| Component | FP32 | FP16/BF16 |
|---|---|---|
| Model weights | ~530 MB | ~265 MB |
| KV cache (batch=128, full seq=256) | ~1.4 GB | ~700 MB |
| Activations + overhead | ~200 MB | ~100 MB |
| **Total** | **~2.1 GB** | **~1.1 GB** |

This model fits comfortably on **any** modern GPU. Even a 4 GB card has headroom. VRAM size is not a differentiator here — you will never come close to filling even a 24 GB card.

## RunPod GPU Comparison for This Workload

The table below ranks GPUs by what actually matters for this model: **memory bandwidth** and **bandwidth per dollar**.

| GPU | VRAM | Mem BW (GB/s) | FP16 TFLOPS | On-Demand $/hr | BW per $ (GB/s/$/hr) | Verdict |
|---|---|---|---|---|---|---|
| **H100 SXM** | 80 GB | 3,350 | 990 | $2.69 | 1,245 | Fastest absolute speed, overkill VRAM |
| **A100 SXM** | 80 GB | 2,039 | 312 | $1.39 | 1,467 | Great BW/$, overkill VRAM |
| **A100 PCIe** | 80 GB | 2,039 | 312 | $1.19 | 1,714 | Best BW/$ among datacenter GPUs |
| **RTX 4090** | 24 GB | 1,008 | 330 | ~$0.44 | 2,291 | **Best BW/$ overall** |
| **RTX 6000 Ada** | 48 GB | 960 | 91 | $0.74 | 1,297 | Good BW, reasonable cost |
| **RTX 3090** | 24 GB | 936 | 71 | ~$0.22 | 4,255 | **Best budget option** |
| **L40S** | 48 GB | 864 | 366 | $0.79 | 1,094 | Balanced but not ideal |
| **A40** | 48 GB | 696 | 150 | $0.35 | 1,989 | Good value, lower BW |
| **L4** | 24 GB | 300 | 121 | ~$0.24 | 1,250 | Too low BW for good throughput |
| **A4000** | 16 GB | 448 | 76 | ~$0.16 | 2,800 | Budget option, decent BW/$ |

> Note: RunPod pricing varies between Secure and Community Cloud. Community Cloud prices are typically 10-30% lower. Prices above are approximate Secure Cloud on-demand rates.

## Recommendations

### Best Overall: RTX 4090 (~$0.44/hr community)

- **1,008 GB/s** memory bandwidth — more than L40S, close to RTX 6000 Ada
- 24 GB VRAM is more than enough (model uses ~2 GB in FP32, ~1 GB in FP16)
- Best bandwidth-per-dollar ratio among readily available GPUs
- Excellent PyTorch support, good SDPA/Flash Attention performance

### Best Absolute Speed: H100 SXM ($2.69/hr)

- **3,350 GB/s** — 3.3x the bandwidth of an RTX 4090
- HBM3 memory delivers the highest throughput for bandwidth-bound workloads
- Only worth it if you need maximum generation speed regardless of cost
- With FP16, this could generate sequences at roughly 3x the rate of an RTX 4090

### Best Budget: RTX 3090 (~$0.22/hr community)

- **936 GB/s** — nearly as fast as an RTX 4090
- At roughly half the price, it has the best bandwidth-per-dollar
- 24 GB VRAM, more than sufficient
- Older generation but the memory bus is still wide (384-bit GDDR6X)

### Best Datacenter-Grade Value: A100 PCIe ($1.19/hr)

- **2,039 GB/s** HBM2e — 2x the bandwidth of consumer cards
- Reliable, well-tested in production environments
- If running long jobs or in production pipelines, the stability and HBM memory are worth it

### Avoid for This Workload

- **L4** (300 GB/s): Too little bandwidth, meant for light inference
- **A4000/RTX 4000** (448 GB/s): Low bandwidth, only if budget is extremely tight
- Any multi-GPU setup: The model is too small to benefit from multi-GPU; the overhead of synchronization would hurt

## Minimal Code Changes for Better Hardware Utilization

The model currently runs entirely in **FP32**. This means every generation step reads 530 MB of weights when it could read 265 MB. The following changes would roughly **double generation throughput** on any GPU without affecting output quality.

### 1. FP16/BF16 Inference (Biggest Win — ~2x speedup)

The model has no numerical sensitivity that requires FP32 during inference. The vocabulary is only 64 tokens, hidden dim is 768, and the values are well-scaled. BF16 is preferred on Ampere+ GPUs (A100, RTX 3090/4090, H100, L40S) as it has wider dynamic range.

**Where to change:** `CfDNAGenerator.from_pretrained()` or after loading:

```python
# After loading the model, cast to half precision
model = model.half()  # FP16
# or
model = model.to(torch.bfloat16)  # BF16, preferred on Ampere+
```

This halves memory bandwidth demand per token step, directly translating to ~2x generation speed on bandwidth-bound workloads.

**Quality impact:** None measurable. DNA token logits over 64 classes with temperature 0.95 sampling are well within FP16 precision. BF16 has even less risk due to the same exponent range as FP32.

### 2. torch.compile() (~10-30% additional speedup)

The model already supports `torch.compile` (the loading code strips `_orig_mod.` prefixes). Compiling fuses operations and reduces kernel launch overhead, which matters when each step's compute is small.

```python
model = torch.compile(model, mode="reduce-overhead")
```

The `"reduce-overhead"` mode uses CUDA graphs under the hood, which eliminates per-step kernel launch latency — significant when each generation step is microseconds of compute but has 14 layers × multiple kernels.

**Quality impact:** None. `torch.compile` is numerically identical.

### 3. Increase Batch Size (if generating many sequences)

Larger batches improve arithmetic intensity, pushing the workload slightly more toward compute-bound territory. The model uses ~2 GB at batch=128 in FP32, so on a 24 GB card you can comfortably run batch=512 or even batch=1024 in FP16.

```python
sequences = generator.generate(
    n_sequences=10000,
    fragment_lengths=165,
    batch_size=512,  # or 1024 in FP16
)
```

**Quality impact:** None. Batch size does not affect the model's output distribution.

### 4. Combined Effect Estimate

| Configuration | Relative Speed | Notes |
|---|---|---|
| FP32, no compile (current) | 1.0x | Baseline |
| FP16, no compile | ~1.8-2.0x | Halved weight reads |
| FP16 + torch.compile | ~2.2-2.6x | Fused kernels + reduced overhead |
| FP16 + compile + batch=512 | ~2.5-3.0x | Better arithmetic intensity |

These are conservative estimates. On high-bandwidth GPUs (A100, H100), the gains from FP16 are particularly clean because HBM bandwidth scales linearly with transfer size.

## Putting It Together: Cost Efficiency Scenarios

Assuming you need to generate 1 million cfDNA sequences:

| GPU | Config | Est. seq/s | Time | Cost |
|---|---|---|---|---|
| RTX 3090 | FP16 + compile | ~700 | ~24 min | ~$0.09 |
| RTX 4090 | FP16 + compile | ~750 | ~22 min | ~$0.16 |
| A100 PCIe | FP16 + compile | ~1,400 | ~12 min | ~$0.24 |
| H100 SXM | FP16 + compile | ~2,200 | ~7.5 min | ~$0.34 |
| RTX 4090 | FP32 (current) | ~300 | ~55 min | ~$0.40 |

> Throughput estimates based on memory bandwidth ratios and typical autoregressive transformer performance. Actual numbers will vary with PyTorch version, CUDA version, and driver.

## Summary

1. **This model is memory-bandwidth bound**, not compute bound. GPU TFLOPS barely matter; memory bandwidth (GB/s) is what determines speed.
2. **The model is tiny** (~2 GB in FP32). VRAM size above 8 GB is irrelevant. Don't pay for 48 or 80 GB cards.
3. **RTX 4090 is the sweet spot** on RunPod — best bandwidth per dollar with plenty of VRAM.
4. **RTX 3090 is the budget pick** — nearly the same bandwidth at half the price.
5. **A100 PCIe for production** — 2x the bandwidth of consumer cards, stable HBM memory.
6. **Switch to FP16/BF16** for an easy ~2x speed gain with no quality impact. This is the single highest-value change.
7. **Add torch.compile** for another 10-30% on top.
8. Multi-GPU is not beneficial — the model is too small and generation is sequential per batch.
