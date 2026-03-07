# Research Bundle: AMD ROCm vs CUDA for Local AI in 2026

---

## Key Facts & Data Points

| Claim | Source URL |
|-------|------------|
| CUDA powers over 90% of AI frameworks with mature tooling and broad support | [technolynx.com](https://www.technolynx.com/post/cuda-vs-rocm-choosing-for-modern-ai) |
| ROCm outperforms by only ~10–30% gap vs CUDA historically 40–50% | [ingoampt.com](https://ingoampt.com/3-choices-1-apple-mlx-2-nvidia-cuda-3-amd-rocm-choosing-the-best-ai-platform-for-a-solo-startup/) |
| ROCm 6.4.0 provides forward/backward compatibility up to a year apart; earlier releases ±2 releases | [rocm.docs.amd.com](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) |
| AMD RX 7900 XTX delivers 85–95% of RTX 4090 performance with 24GB VRAM at $700–$950 | [insiderllm.com](https://www.insiderllm.com/guides/amd-vs-nvidia-local-ai-rocm/) |
| ROCm 7.2 released Jan 21, 2026 adds proper support for RDNA3 RX 7700 series and RDNA4 9000 series GPUs | [phoronix.com](https://www.phoronix.com/news/AMD-ROCm-7.2-Released) |
| CUDA typically outperforms ROCm by 10% to 30% in compute-intensive workloads | [thundercompute.com](https://www.thundercompute.com/blog/rocm-vs-cuda-gpu-computing) |
| AMD's Instinct MI325X represents a turning point for ROCm performance | [thundercompute.com](https://www.thundercompute.com/blog/rocm-vs-cuda-gpu-computing) |
| RX 7900 XTX finally works with ROCm 6.x on Linux (honest benchmarks and compatibility gaps exist) | [insiderllm.com](https://www.insiderllm.com/guides/amd-vs-nvidia-local-ai-rocm/) |
| ROCm Core SDK 7.11.0 supports GPU virtualization on MI355X, MI350X, MI325X, MI300X | [rocm.docs.amd.com](https://rocm.docs.amd.com/en/7.11.0-preview/about/release-notes.html) |
| AMD ROCm does not support Radeon 890M (Ryzen AI 300 series); third-party patches required | [frame.work community](https://community.frame.work/t/amd-rocm-does-not-support-the-amd-ryzen-ai-300-series-gpus/68767) |
| For local LLMs, llama.cpp supports AMD GPUs via ROCm or Vulkan | [dev.to](https://dev.to/sienna/qwen3-coder-next-the-complete-2026-guide-to-running-powerful-ai-coding-agents-locally-1k95) |

---

## Price Data (2026 Used Market - eBay Hammer Prices)

| GPU Model | VRAM | New MSRP (if known) | Used Price Range | Typical Used Price | Source |
|-----------|------|---------------------|------------------|-------------------|---------|
| NVIDIA RTX 3090 | 24GB GDDR6X | N/A | $950–$1,125 | ~$1,040 | InsiderLLM reference data |
| NVIDIA RTX 3080 12GB | 12GB GDDR6X | N/A | $230–$380 | ~$305 | InsiderLLM reference data |
| AMD RX 7900 XT | 20GB GDDR6 | N/A | $500–$700 | ~$600 | InsiderLLM reference data |
| AMD RX 7800 XT | 16GB GDDR6 | N/A | $380–$550 | ~$465 | InsiderLLM reference data |
| NVIDIA RTX 4060 Ti 16GB | 16GB GDDR6 | N/A | $380–$480 | ~$430 | InsiderLLM reference data |
| AMD RX 7900 GRE | 16GB GDDR6 | N/A | $400–$550 | ~$475 | InsiderLLM reference data |
| NVIDIA RTX 3060 12GB | 12GB GDDR6 | N/A | $170–$380 | ~$275 | InsiderLLM reference data |
| AMD RX 7700 XT | 12GB GDDR6 | N/A | $300–$350 | ~$325 | InsiderLLM reference data |
| NVIDIA RTX 3070 | 8GB GDDR6 | N/A | $210–$300 | ~$255 | InsiderLLM reference data |
| AMD RX 7600 | 8GB GDDR6 | N/A | $170–$225 | ~$200 | InsiderLLM reference data |

**DRAM Pricing for Offloading:**
- DDR3 (up to 1866MHz): ~$15–$30 per 16GB used
- DDR4 (up to 3600MHz): ~$50–$80 per 16GB, ~$100–$170 per 32GB used
- DDR5 (up to 6400MHz): ~$200–$300 per 32GB, ~$320–$450 per 64GB used

---

## Benchmark Data

### Memory Bandwidth & Token Generation Correlation

| Card | Bandwidth (GB/s) | llama3 8B Q4 tok/s | Notes | Source |
|------|------------------|---------------------|-------|--------|
| NVIDIA RTX 3090 | 936 | 112 | Budget local AI king | InsiderLLM reference data |
| AMD RX 7900 XT | 800 | 116 (llama2 7B Q4) | ROCm compatibility varies | InsiderLLM reference data |
| NVIDIA RTX 3080 12GB | 912 | 107 | Sleeper pick | InsiderLLM reference data |
| AMD RX 7800 XT | 624 | 39 (llama3 8B Q4) | ROCm compatibility varies | InsiderLLM reference data |
| NVIDIA RTX 3070 Ti | 608 | N/A | GDDR6X bandwidth advantage | InsiderLLM reference data |
| AMD RX 7700 XT | 432 | 96 (llama2 7B Q4) | ROCm compatible | InsiderLLM reference data |
| NVIDIA RTX 3060 12GB | 360 | 51 (llama3 8B Q4) | Entry-level workhorse | InsiderLLM reference data |
| AMD RX 7600 | 288 | N/A | ROCm support improving | InsiderLLM reference data |

### CUDA vs ROCm Performance Gap

- **AMD cards achieve ~0.06 tok/s per GB/s** of bandwidth due to less optimized kernels
- **NVIDIA cards average ~0.13 tok/s per GB/s** of bandwidth for Llama 3 8B Q4
- This means **~2x slower token generation** on ROCm for same bandwidth tier

### Model Capacity by VRAM (VRAM → Model size guide)

| VRAM | Max Model Size | Notes | Source |
|------|----------------|-------|--------|
| 6GB | 7B Q4 max, tight | No tensor cores on GTX 1660 Super | InsiderLLM reference data |
| 8GB | 8B Q4 comfortable, 14B Q2 possible | Limited by VRAM | InsiderLLM reference data |
| 10GB | 8B Q6, 14B Q3 possible | Bandwidth bottleneck | InsiderLLM reference data |
| 12GB | 14B Q4, 8B Q8 or FP16 | Sweet spot for budget builds | InsiderLLM reference data |
| 16GB | 30B Q3, 14B Q6 | Great for model capacity | InsiderLLM reference data |
| 20GB | 30B Q4, some 70B Q2 | Between 3090 and 4060 Ti 16GB | InsiderLLM reference data |
| 24GB | 30B Q5, 70B Q2-Q3 | Runs large models comfortably | InsiderLLM reference data |

---

## Key Specs (VRAM, TDP, Architecture)

### NVIDIA Cards (CUDA)

| GPU | VRAM | Bandwidth | TDP | Architecture | Notes |
|-----|------|-----------|-----|--------------|--------|
| GTX 1660 Super | 6GB GDDR6 | 336 GB/s | 125W | Turing | No tensor cores, limited to experimentation |
| RTX 2060 12GB | 12GB GDDR6 | 336 GB/s | 185W | Turing | Tensor cores help some workloads |
| RTX 3060 12GB | 12GB GDDR6 | 360 GB/s | 170W | Ampere | Entry-level workhorse, bandwidth bottleneck |
| RTX 3070 | 8GB GDDR6 | 448 GB/s | 220W | Ampere | Fast but VRAM limited |
| RTX 3070 Ti | 8GB GDDR6X | 608 GB/s | 290W | Ampere | GDDR6X bandwidth advantage |
| RTX 3080 10GB | 10GB GDDR6X | 760 GB/s | 320W | Ampere | 2x 3060 bandwidth |
| RTX 3080 12GB | 12GB GDDR6X | 912 GB/s | 350W | Ampere | Sleeper pick, excellent value |
| RTX 3090 | 24GB GDDR6X | 936 GB/s | 350W | Ampere | Budget local AI king, runs hot |
| RTX 4060 | 8GB GDDR6 | 272 GB/s | 115W | Ada Lovelace | Lowest bandwidth, avoid for AI |
| RTX 4060 Ti 8GB | 8GB GDDR6 | 288 GB/s | 160W | Ada Lovelace | Bandwidth-starved despite compute |
| RTX 4060 Ti 16GB | 16GB GDDR6 | 288 GB/s | 165W | Ada Lovelace | Trade VRAM for speed vs 3080 12GB |

### AMD Cards (ROCm)

| GPU | VRAM | Bandwidth | TDP | Architecture | Notes |
|-----|------|-----------|-----|--------------|--------|
| RX 7600 | 8GB GDDR6 | 288 GB/s | 165W | RDNA 3 | ROCm support improving but behind CUDA |
| RX 7700 XT | 12GB GDDR6 | 432 GB/s | 245W | RDNA 3 | Competitive if ROCm works for your stack |
| RX 7800 XT | 16GB GDDR6 | 624 GB/s | 263W | RDNA 3 | AMD's best AI value if ROCm works |
| RX 7900 GRE | 16GB GDDR6 | 576 GB/s | 260W | RDNA 3 | More compute but lower bandwidth than 7800 XT |
| RX 7900 XT | 20GB GDDR6 | 800 GB/s | 315W | RDNA 3 | AMD's best current option for local AI |

**Upcoming/Rumored:**
- **NVIDIA RTX 5070**: 12GB GDDR7, 672 GB/s bandwidth, MSRP $549 (Blackwell)
- **AMD RX 9070**: 16GB GDDR6, 608 GB/s bandwidth, MSRP $549 (RDNA 4) — ROCm support unknown
- **AMD RX 9070 XT**: 16GB GDDR6, 608 GB/s bandwidth, MSRP $599 (RDNA 4)

---

## Expert Opinions & Analysis

### Technolynx (Jan 2026)
> "CUDA remains widely adopted due to its robust developer community, unified device architecture, and deep integration with modern Linux environments. On the other hand, AMD hardware has become a viable alternative due to ROCm's open-source nature, rapid improvements in ROCm support, and increasingly comparable performance in real AI applications."

### AIMultiple (Jan 2026)
> "CUDA typically outperforms ROCm by 10% to 30% in compute-intensive workloads. AMD's MI325X represents a turning point for ROCm performance."

### Thunder Compute (Oct 2025)
> "For years, CUDA has been the default for AI teams, mainly because there were no serious alternatives. But the hardware landscape has changed. So has the software. Enter ROCm, AMD's open-source compute platform. Combined with the latest MI325X GPUs, ROCm is no longer just 'an alternative' but now it's a real performance contender."

### Ingo A. Mpt (2025/2026)
> "By 2025/2026, AMD's solution has narrowed the performance gap with CUDA considerably. Recent tests show that CUDA outperforms ROCm by only about 10–30% now, whereas a few years ago the gap was 40–50%."

### InsiderLLM Analysis
> "NVIDIA cards average ~0.13 tok/s per GB/s of bandwidth for Llama 3 8B Q4. AMD ROCm cards achieve ~0.06 tok/s per GB/s due to less optimized kernels."

### Phoronix (Jan 21, 2026)
> "ROCm 7.2 released with proper support for RDNA3 RX 7700 series and RDNA4 9000 series GPUs. Beyond these two RDNA4 graphics cards added to the official support matrix with ROCm 7.2, the ROCm 7.2 release also (finally) adds proper support for the RDNA3-based Radeon RX 7700 series."

---

## Gaps (Missing Data)

1. **Specific local AI benchmark comparisons**: Need direct side-by-side tok/s comparisons between same model on CUDA vs ROCm for identical hardware tiers
2. **ROCm compatibility matrix completeness**: Not all RX 7000 series cards are equally supported; need detailed list of which specific models work with which ROCm versions
3. **Framework-specific support**: Which LLM frameworks (llama.cpp, Ollama, vLLM, etc.) work best with ROCm vs CUDA? What version requirements exist?
4. **Power consumption data for AMD cards in AI workloads**: TDP specs available but actual power draw during inference unknown
5. **Cooling considerations**: Dual-slot blower cooler RTX 3090s run hot; need AMD equivalents' thermal characteristics
6. **Upcoming hardware confirmation**: RTX 5070, RX 9070 specs and local AI performance not yet confirmed
7. **CPU offloading efficiency**: Need DDR4 vs DDR5 offloading benchmarks for large models that don't fit in VRAM
8. **Driver version requirements**: Specific amdgpu driver versions needed for ROCm compatibility

---

## Suggested Angle for InsiderLLM Coverage

**"ROCm is Finally Viable — But With Caveats: The 2026 Local AI Reality Check"**

### Key Story Angles:

1. **The Performance Gap Reality**: CUDA still wins on raw speed (~2x tok/s per GB/s), but ROCm closing the gap to 10-30% makes AMD viable for budget-conscious builders who prioritize VRAM capacity over absolute speed.