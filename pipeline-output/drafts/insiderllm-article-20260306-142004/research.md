# Research Bundle: Best Budget GPU for Local AI in 2026

---

## Key Facts & Data Points

**Core Finding:** For budget local AI in 2026, VRAM capacity and memory bandwidth are the two most critical specifications, with CUDA support being a major advantage over ROCm.

- **NVIDIA RTX 3060 12GB** is the entry-level workhorse at ~$275 used, fitting 13B Q4 models with bandwidth bottlenecking at 360 GB/s (half of 3080's speed) [insiderllm canonical database]
- **NVIDIA RTX 3080 12GB** is the "sleeper pick" at ~$305 used, offering 912 GB/s bandwidth and 12GB VRAM delivering 2.5x the speed of 3060 for same model capacity [insiderllm canonical database]
- **NVIDIA RTX 3090** remains "the budget local AI king" at ~$1,040 used with 24GB VRAM running 30B Q4 and 70B Q2 models at 936 GB/s bandwidth [insiderllm canonical database]
- **AMD ROCm support is improving** but still behind CUDA, with AMD cards achieving ~0.06 tok/s per GB/s vs NVIDIA's ~0.13 tok/s per GB/s [insiderllm canonical database]
- **RTX 4060 series should be avoided** for local AI despite being newer due to 128-bit bus limiting bandwidth (RTX 4060: 272 GB/s, RTX 4060 Ti 8GB: 288 GB/s) [insiderllm canonical database]
- **RTX 4060 Ti 16GB** at ~$430 used offers 16GB VRAM but still bandwidth-starved at 288 GB/s, runs bigger models than 3060 but slower token generation [insiderllm canonical database]
- **AMD RX 7800 XT 16GB** at ~$465 used with 624 GB/s bandwidth is AMD's best AI value if ROCm works for you, competing with RTX 3080 12GB in bandwidth territory [insiderllm canonical database]
- **AMD RX 7900 XT 20GB** at ~$600 used offers 800 GB/s bandwidth and is AMD's best current option for local AI if ROCm support works [insiderllm canonical database]
- **RTX 3070 Ti 16GB** was rumored but delayed per TechPowerUp, not yet available in 2026 [techpowerup.com]

---

## Price Data

### Used GPU Prices (eBay sold auctions - real hammer prices)

| GPU Model | VRAM | Architecture | Typical Used Price | Source |
|-----------|------|--------------|-------------------|--------|
| NVIDIA GTX 1660 Super | 6GB GDDR6 | Turing | $90–$120 (~$105) | [insiderllm canonical database] |
| NVIDIA RTX 2060 12GB | 12GB GDDR6 | Turing | $140–$180 (~$160) | [insiderllm canonical database] |
| NVIDIA RTX 3060 12GB | 12GB GDDR6 | Ampere | $170–$380 (~$275) | [insiderllm canonical database] |
| NVIDIA RTX 3070 | 8GB GDDR6 | Ampere | $210–$300 (~$255) | [insiderllm canonical database] |
| NVIDIA RTX 3070 Ti | 8GB GDDR6X | Ampere | $100–$280 (~$190) | [insiderllm canonical database] |
| NVIDIA RTX 3080 10GB | 10GB GDDR6X | Ampere | $325–$400 (~$365) | [insiderllm canonical database] |
| NVIDIA RTX 3080 12GB | 12GB GDDR6X | Ampere | $230–$380 (~$305) | [insiderllm canonical database] |
| NVIDIA RTX 3090 | 24GB GDDR6X | Ampere | $950–$1,125 (~$1,040) | [insiderllm canonical database] |
| NVIDIA RTX 4060 | 8GB GDDR6 | Ada Lovelace | $230–$310 (~$270) | [insiderllm canonical database] |
| NVIDIA RTX 4060 Ti 8GB | 8GB GDDR6 | Ada Lovelace | $240–$300 (~$270) | [insiderllm canonical database] |
| NVIDIA RTX 4060 Ti 16GB | 16GB GDDR6 | Ada Lovelace | $380–$480 (~$430) | [insiderllm canonical database] |
| AMD Radeon RX 7600 | 8GB GDDR6 | RDNA 3 | $170–$225 (~$200) | [insiderllm canonical database] |
| AMD Radeon RX 7700 XT | 12GB GDDR6 | RDNA 3 | $300–$350 (~$325) | [insiderllm canonical database] |
| AMD Radeon RX 7800 XT | 16GB GDDR6 | RDNA 3 | $380–$550 (~$465) | [insiderllm canonical database] |
| AMD Radeon RX 7900 GRE | 16GB GDDR6 | RDNA 3 | $400–$550 (~$475) | [insiderllm canonical database] |
| AMD Radeon RX 7900 XT | 20GB GDDR6 | RDNA 3 | $500–$700 (~$600) | [insiderllm canonical database] |

### System RAM Prices (for CPU offloading)

- **DDR3:** Up to 1866MHz, dual-channel ~14.9 GB/s; used price $15-30 per 16GB [insiderllm canonical database]
- **DDR4:** Up to 3600MHz, dual-channel ~28.8 GB/s; used price $50-80 per 16GB, $100-170 per 32GB [insiderllm canonical database]
- **DDR5:** Up to 6400MHz, dual-channel ~51.2 GB/s; used price $200-300 per 32GB, $320-450 per 64GB [insiderllm canonical database]

---

## Benchmark Data

### Token-per-second Benchmarks (LLM Inference)

| GPU Model | Llama3 8B Q4 | Llama2 7B Q4 | Llama2 13B Q4 | Source |
|-----------|--------------|--------------|---------------|--------|
| NVIDIA RTX 3060 12GB | 51 tok/s | 76 tok/s | 35 tok/s | [insiderllm canonical database] |
| NVIDIA RTX 3070 | 71 tok/s | - | - | [insiderllm canonical database] |
| NVIDIA RTX 3080 10GB | 106 tok/s | - | - | [insiderllm canonical database] |
| NVIDIA RTX 3080 12GB | 107 tok/s | - | - | [insiderllm canonical database] |
| NVIDIA RTX 3090 | 112 tok/s | - | - | [insiderllm canonical database] |
| NVIDIA RTX 4060 8GB | 38 tok/s | - | - | [insiderllm canonical database] |
| NVIDIA RTX 4060 Ti 8GB | 48 tok/s | 64 tok/s | - | [insiderllm canonical database] |
| AMD RX 7800 XT 16GB | 39 tok/s | 96 tok/s | - | [insiderllm canonical database] |
| AMD RX 7900 XT 20GB | - | 116 tok/s | 97 tok/s sustained | [insiderllm canonical database] |

### Additional Model Benchmarks (RTX 3090)

- Llama3 8B F16: 47 tok/s
- Llama3 70B Q4: 16 tok/s
- Mistral 7B Q6: 85 tok/s
- Gemma3 27B: 39.9 tok/s [insiderllm canonical database]

### DeepSeek R1 Benchmarks (RTX 3060 12GB)

- 14B think off: 35.6 tok/s
- 14B think on: 35.1 tok/s [insiderllm canonical database]

### Qwen35 Benchmarks (RTX 3060 12GB)

- 9B think off: 47.1 tok/s
- 9B think on: 46.6 tok/s [insiderllm canonical database]

---

## Key Specs

### Memory Bandwidth & Architecture Comparison

| GPU Model | VRAM | Bandwidth | TDP | Architecture | Source |
|-----------|------|-----------|-----|--------------|--------|
| GTX 1660 Super | 6GB GDDR6 | 336 GB/s | 125W | Turing | [insiderllm canonical database] |
| RTX 2060 12GB | 12GB GDDR6 | 336 GB/s | 185W | Turing | [insiderllm canonical database] |
| RTX 3060 12GB | 12GB GDDR6 | 360 GB/s | 170W | Ampere | [insiderllm canonical database] |
| RTX 3070 | 8GB GDDR6 | 448 GB/s | 220W | Ampere | [insiderllm canonical database] |
| RTX 3070 Ti | 8GB GDDR6X | 608 GB/s | 290W | Ampere | [insiderllm canonical database] |
| RTX 3080 10GB | 10GB GDDR6X | 760 GB/s | 320W | Ampere | [insiderllm canonical database] |
| RTX 3080 12GB | 12GB GDDR6X | 912 GB/s | 350W | Ampere | [insiderllm canonical database] |
| RTX 3090 | 24GB GDDR6X | 936 GB/s | 350W | Ampere | [insiderllm canonical database] |
| RTX 4060 | 8GB GDDR6 | 272 GB/s | 115W | Ada Lovelace | [insiderllm canonical database] |
| RTX 4060 Ti 8GB | 8GB GDDR6 | 288 GB/s | 160W | Ada Lovelace | [insiderllm canonical database] |
| RTX 4060 Ti 16GB | 16GB GDDR6 | 288 GB/s | 165W | Ada Lovelace | [insiderllm canonical database] |
| AMD RX 7600 | 8GB GDDR6 | 288 GB/s | 165W | RDNA 3 | [insiderllm canonical database] |
| AMD RX 7700 XT | 12GB GDDR6 | 432 GB/s | 245W | RDNA 3 | [insiderllm canonical database] |
| AMD RX 7800 XT | 16GB GDDR6 | 624 GB/s | 263W | RDNA 3 | [insiderllm canonical database] |
| AMD RX 7900 GRE | 16GB GDDR6 | 576 GB/s | 260W | RDNA 3 | [insiderllm canonical database] |
| AMD RX 7900 XT | 20GB GDDR6 | 800 GB/s | 315W | RDNA 3 | [insiderllm canonical database] |

### VRAM → Model Size Guide

- **6GB:** 7B Q4 max, tight
- **8GB:** 8B Q4 comfortable, 14B Q2 possible
- **10GB:** 8B Q6, 14B Q3 possible
- **12GB:** 14B Q4, 8B Q8 or FP16
- **16GB:** 30B Q3, 14B Q6
- **20GB:** 30B Q4, some 70B Q2
- **24GB:** 30B Q5, 70B Q2-Q3 [insiderllm canonical database]

### Performance Scaling Rules

- **For models that fit entirely in VRAM:** Memory bandwidth is the primary predictor of tok/s. Double the bandwidth ≈ double the tok/s for memory-bound inference [insiderllm canonical database]
- **CPU offloading penalty:** DDR4-3200 (25.6 GB/s) vs GDDR6X (936 GB/s) = ~37x slower per offloaded layer [insiderllm canonical database]

---

## Expert Opinions & Analysis

### Tech Tactician Recommendations

**Top Picks Under $1000:**
1. **NVIDIA RTX 4060 Ti 16GB** — First card on list, popular choice despite lower memory bandwidth (288 GB/s) vs RTX 4070 (504 GB/s), but has 4 bonus GB of VRAM [techtactician.com]
2. **NVIDIA RTX 4070 12GB** — Best value under $200 used according to InsiderLLM, though bandwidth matches GTX 1660 Super [insiderllm canonical database]
3. **AMD RX 7800 XT 16GB** — AMD's best AI value if ROCm works for you, competes with RTX 3080 12GB in bandwidth territory [techtactician.com]

**Top Picks Under $500:**
4. **NVIDIA RTX 3060 12GB** — Entry-level workhorse, fits 13B Q4 models, best value under $200 used [techtactician.com]
5. **Intel Arc B580 12GB** — Budget option from Intel lineup
6. **Intel Arc A770 16GB** — Budget option with 16GB VRAM
7. **AMD RX 6800 XT 16GB** — Older architecture but excellent mid-range choice [techtactician.com]

**Bonus:** Used NVIDIA RTX 3090/RTX 3090 Ti (24GB) for users who can stretch budget [techtactician.com]

### Tech Tactician Analysis on Non-NVIDIA GPUs

- **NVIDIA is the current leader** for local AI tasks due to CUDA framework native support
- **AMD cards use ROCm software stack**, Intel uses DirectML/Vulkan Compute/SYCL/OpenCL
- **Most local LLM software developers prioritize CUDA**, making NVIDIA "plug-and-play" solution
- **Purchasing non-NVIDIA GPU requires being mindful** of compatibility status and possible performance issues [techtactician.com]

### ROCm vs CUDA Performance Gap

- **NVIDIA cards average ~0.13 tok/s per GB/s** of bandwidth for Llama 3 8B Q4
- **AMD ROCm cards achieve ~0.06 tok/s per GB/s** due to less optimized kernels [insiderllm canonical database]

### AMD GPU Compatibility Status (2026)

- **Ollama, LM Studio, KoboldCpp, and OobaBooga WebUI offer out-of-the-box support** for AMD graphics cards
- **AMD is now a powerful and easy-to-use option** for budget-conscious beginners
- **CUDA ecosystem still has wider support** in some applications [techtactician.com]

### InsiderLLM Analysis on ROCm Readiness

- **ROCm is finally ready—for the right user** (Jan 27, 2026)
- **AMD costs you:** Linux requirement for best experience, occasional troubleshooting, slower support for new models/tools, some software incompatibility [insiderllm.com]

---

## Gaps

### Missing or Uncertain Data

1. **RTX 5060/5060 Ti specifications** — Marked as "PRELIMINARY" with no confirmed specs:
   - RTX 5060 rumored to have 8GB VRAM with 128-bit bus (same limitation as RTX 4060)
   - RTX 5060 Ti rumored to have 16GB GDDR7
   - Need actual bandwidth numbers and pricing verification [insiderllm canonical database]

2. **RTX 5070 availability** — Listed with MSRP $549, 12GB GDDR7,