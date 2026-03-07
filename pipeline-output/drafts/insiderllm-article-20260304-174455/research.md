# Research Bundle: AMD ROCm vs CUDA for Local AI in 2026

## Key Facts & Data Points

| Claim | Source |
|-------|--------|
| AMD ROCm support is improving but still behind CUDA for local AI workloads | InsiderLLM canonical database (verified reference) |
| NVIDIA cards average ~0.13 tok/s per GB/s of bandwidth for Llama 3 8B Q4 | InsiderLLM canonical database (verified reference) |
| AMD ROCm cards achieve ~0.06 tok/s per GB/s due to less optimized kernels | InsiderLLM canonical database (verified reference) |
| Each layer offloaded to CPU RAM runs at system RAM bandwidth instead of VRAM bandwidth | InsiderLLM canonical database (verified reference) |
| DDR4-3200 (25.6 GB/s) vs GDDR6X (936 GB/s) = ~37x slower per offloaded layer | InsiderLLM canonical database (verified reference) |
| Always verify ROCm compatibility for specific use case before buying AMD for AI | InsiderLLM canonical database (verified reference) |

## Price Data

### NVIDIA Cards (Used Prices - eBay Sold Auctions, 2026-03-02)

| GPU | VRAM | Used Price Range | Typical Price | Source |
|-----|------|------------------|---------------|--------|
| GTX 1660 Super | 6GB GDDR6 | $90–$120 | ~$105 | InsiderLLM DB |
| RTX 2060 12GB | 12GB GDDR6 | $140–$180 | ~$160 | InsiderLLM DB |
| RTX 3060 12GB | 12GB GDDR6 | $170–$380 | ~$275 | InsiderLLM DB |
| RTX 3070 | 8GB GDDR6 | $210–$300 | ~$255 | InsiderLLM DB |
| RTX 3070 Ti | 8GB GDDR6X | $100–$280 | ~$190 | InsiderLLM DB |
| RTX 3080 10GB | 10GB GDDR6X | $325–$400 | ~$365 | InsiderLLM DB |
| RTX 3080 12GB | 12GB GDDR6X | $230–$380 | ~$305 | InsiderLLM DB |
| RTX 3090 | 24GB GDDR6X | $950–$1125 | ~$1040 | InsiderLLM DB |
| RTX 4060 | 8GB GDDR6 | $230–$310 | ~$270 | InsiderLLM DB |
| RTX 4060 Ti 8GB | 8GB GDDR6 | $240–$300 | ~$270 | InsiderLLM DB |
| RTX 4060 Ti 16GB | 16GB GDDR6 | $380–$480 | ~$430 | InsiderLLM DB |

### AMD Cards (Used Prices - eBay Sold Auctions, 2026-03-02)

| GPU | VRAM | Used Price Range | Typical Price | Source |
|-----|------|------------------|---------------|--------|
| RX 7600 | 8GB GDDR6 | $170–$225 | ~$200 | InsiderLLM DB |
| RX 7700 XT | 12GB GDDR6 | $300–$350 | ~$325 | InsiderLLM DB |
| RX 7800 XT | 16GB GDDR6 | $380–$550 | ~$465 | InsiderLLM DB |
| RX 7900 GRE | 16GB GDDR6 | $400–$550 | ~$475 | InsiderLLM DB |
| RX 7900 XT | 20GB GDDR6 | $500–$700 | ~$600 | InsiderLLM DB |

### System RAM Prices (Used)

| RAM Type | Speed | Used Price (16GB/32GB/64GB) | Source |
|----------|-------|------------------------------|--------|
| DDR3 | up to 1866MHz | $15-30 per 16GB | InsiderLLM DB |
| DDR4 | up to 3600MHz | $50-80 (16GB), $100-170 (32GB) | InsiderLLM DB |
| DDR5 | up to 6400MHz | $200-300 (32GB), $320-450 (64GB) | InsiderLLM DB |

## Benchmark Data

### NVIDIA GPU Inference Benchmarks (Llama 3/2 Models, Q4 Quantization)

| GPU | Llama 3 8B Q4 | Llama 2 7B Q4 | Llama 2 13B Q4 | Source |
|-----|---------------|---------------|----------------|--------|
| GTX 1660 Super | N/A | 76 tok/s | 35 tok/s | InsiderLLM DB |
| RTX 3060 12GB | 51 tok/s | 76 tok/s | N/A | InsiderLLM DB |
| RTX 3070 | 71 tok/s | N/A | N/A | InsiderLLM DB |
| RTX 3080 10GB | 106 tok/s | N/A | N/A | InsiderLLM DB |
| RTX 3080 12GB | 107 tok/s | N/A | N/A | InsiderLLM DB |
| RTX 3090 | 112 tok/s | N/A | 16 tok/s (70B Q4) | InsiderLLM DB |
| RTX 4060 Ti 16GB | 48 tok/s | 64 tok/s | N/A | InsiderLLM DB |

### AMD GPU Inference Benchmarks

| GPU | Llama 2 7B Q4 | Llama 2 7B Q4 Sustained | Source |
|-----|---------------|-------------------------|--------|
| RX 7800 XT | 96 tok/s | N/A | InsiderLLM DB |
| RX 7900 XT | N/A | 116 tok/s (sustained: 97 tok/s) | InsiderLLM DB |

### Additional Model Benchmarks (NVIDIA RTX 3090)

| Model | GPU | Benchmark | Source |
|-------|-----|-----------|--------|
| Mistral 7B Q6 | RTX 3090 | 85 tok/s | InsiderLLM DB |
| Gemma3 27B | RTX 3090 | 39.9 tok/s | InsiderLLM DB |
| DeepSeek R1 14B Think Off | RTX 3060 12GB | 35.6 tok/s | InsiderLLM DB |
| DeepSeek R1 14B Think On | RTX 3060 12GB | 35.1 tok/s | InsiderLLM DB |
| Qwen35 9B Think Off | RTX 3060 12GB | 47.1 tok/s | InsiderLLM DB |
| Qwen35 9B Think On | RTX 3060 12GB | 46.6 tok/s | InsiderLLM DB |

### Bandwidth Comparison & Speed Ratios

| GPU | Bandwidth | Typical Use Case | Source |
|-----|-----------|------------------|--------|
| GTX 1660 Super | 336 GB/s | 7B Q4 max | InsiderLLM DB |
| RTX 2060 12GB | 336 GB/s | Budget option | InsiderLLM DB |
| RTX 3060 12GB | 360 GB/s | Entry-level workhorse | InsiderLLM DB |
| RTX 3070 Ti | 608 GB/s | High bandwidth, 8GB limited | InsiderLLM DB |
| RTX 3080 12GB | 912 GB/s | Sleeper pick (2.5x speed vs 3060) | InsiderLLM DB |
| RTX 3090 | 936 GB/s | Budget local AI king | InsiderLLM DB |
| RX 7800 XT | 624 GB/s | AMD's best AI value (if ROCm works) | InsiderLLM DB |
| RX 7900 XT | 800 GB/s | Competitive with RTX 3080 12GB | InsiderLLM DB |

## Key Specs

### Memory Bandwidth Predictors (VRAM → Model Size Guide)

| VRAM | Max Model Size | Notes | Source |
|------|----------------|-------|--------|
| 6GB | 7B Q4 max, tight | No tensor cores | InsiderLLM DB |
| 8GB | 8B Q4 comfortable, 14B Q2 possible | Bandwidth varies by GPU | InsiderLLM DB |
| 10GB | 8B Q6, 14B Q3 possible | Fits some larger models | InsiderLLM DB |
| 12GB | 14B Q4, 8B Q8 or FP16 | Sweet spot for many users | InsiderLLM DB |
| 16GB | 30B Q3, 14B Q6 | More capacity, bandwidth varies | InsiderLLM DB |
| 20GB | 30B Q4, some 70B Q2 | Between 3090 and 4060 Ti 16GB | InsiderLLM DB |
| 24GB | 30B Q5, 70B Q2-Q3 | Runs large models comfortably | InsiderLLM DB |

### Architecture & TDP Data

| GPU | Architecture | Bandwidth | TDP | Source |
|-----|--------------|-----------|-----|--------|
| GTX 1660 Super | Turing | 336 GB/s | 125W | InsiderLLM DB |
| RTX 2060 12GB | Turing | 336 GB/s | 185W | InsiderLLM DB |
| RTX 3060 12GB | Ampere | 360 GB/s | 170W | InsiderLLM DB |
| RTX 3070 | Ampere | 448 GB/s | 220W | InsiderLLM DB |
| RTX 3070 Ti | Ampere | 608 GB/s | 290W | InsiderLLM DB |
| RTX 3080 10GB | Ampere | 760 GB/s | 320W | InsiderLLM DB |
| RTX 3080 12GB | Ampere | 912 GB/s | 350W | InsiderLLM DB |
| RTX 3090 | Ampere | 936 GB/s | 350W | InsiderLLM DB |
| RX 7600 | RDNA 3 | 288 GB/s | 165W | InsiderLLM DB |
| RX 7700 XT | RDNA 3 | 432 GB/s | 245W | InsiderLLM DB |
| RX 7800 XT | RDNA 3 | 624 GB/s | 263W | InsiderLLM DB |
| RX 7900 GRE | RDNA 3 | 576 GB/s | 260W | InsiderLLM DB |
| RX 7900 XT | RDNA 3 | 800 GB/s | 315W | InsiderLLM DB |

### Pre-2026 Hardware (RTX 50-series & RX 90-series - Preliminary)

| GPU | VRAM | Bus Width | Bandwidth | TDP | Architecture | Notes | Source |
|-----|------|-----------|-----------|-----|--------------|-------|--------|
| RTX 5060 | 8GB GDDR7 | 128-bit (likely) | N/A | N/A | Blackwell | Preliminary - likely same VRAM limitation as 4060 | InsiderLLM DB |
| RTX 5060 Ti 16GB | 16GB GDDR7 | N/A | N/A | N/A | Blackwell | Rumored - could be solid mid-range AI card if bandwidth improves | InsiderLLM DB |
| RTX 5070 | 12GB GDDR7 | 192-bit | 672 GB/s | 250W | Blackwell | MSRP $549, nearly 2x bandwidth of 3060 | InsiderLLM DB |
| RX 9070 | 16GB GDDR6 | 256-bit | 608 GB/s | 220W | RDNA 4 | Preliminary - ROCm status unknown | InsiderLLM DB |
| RX 9070 XT | 16GB GDDR6 | N/A | 608 GB/s | 250W | RDNA 4 | Preliminary - same ROCm uncertainty as 9070 | InsiderLLM DB |

## Expert Opinions & Analysis

### Key Recommendations from Verified Sources

1. **RTX 3060 12GB**: "The entry-level workhorse... Best value under $200 used"
   - *Analysis*: Despite bandwidth being half of RTX 3080, 12GB VRAM allows running larger models
   - *Tradeoff*: Bandwidth is the bottleneck at 360 GB/s

2. **RTX 3080 12GB**: "The sleeper pick... Hard to find but excellent value"
   - *Analysis*: 12GB VRAM + 912 GB/s = 3060's model capacity at 2.5x speed
   - *Note*: Better than RTX 4060 Ti 16GB for actual inference speed

3. **RTX 3090**: "The budget local AI king"
   - *Analysis*: 24GB runs 30B Q4, even 70B Q2
   - *Warning*: Power hungry and runs hot; dual-slot blower coolers run especially hot — look for triple-fan models

4. **AMD RX 7800 XT**: "16GB VRAM + 624 GB/s bandwidth makes this AMD's best AI value if ROCm works for you"
   - *Analysis*: Competes with RTX 3080 12GB in bandwidth territory
   - *Critical Caveat*: "ROCm compatibility is the main risk. If AMD software works for your stack, this is competitive."

5. **AMD RX 7900 XT**: "AMD's best current option for local AI if ROCm support works"
   - *Analysis*: 20GB VRAM is between 3090 and 4060 Ti 16GB in capacity
   - *Critical Caveat*: Same ROCm compatibility warnings

### Performance Efficiency Analysis

- **NVIDIA Efficiency**: ~0.13 tok/s per GB/s of bandwidth (Llama 3 8B Q4)
- **AMD ROCm Efficiency**: ~0.06 tok/s per GB/s (50% less efficient due to kernel optimization)
- **Implication**: For equal VRAM, NVIDIA will typically be ~2x faster in actual token generation

### System RAM Offloading Performance

- DDR3: CPU offloading at 3-4x slower than DDR5
- DDR4: Sweet spot for budget builds (Optiplex 5060/7060)
- DDR5: Best for CPU-heavy workloads or large models needing lots of offloaded layers
- *