## Price Data
(all prices found, grouped by GPU, note new vs used)

### NVIDIA Cards (Used Prices from eBay Sold Auctions, 2026-03-02)
| GPU | VRAM | Used Price Range | Typical Price | Source |
|-----|------|------------------|---------------|--------|
| GTX 1660 Super | 6GB GDDR6 | $90–$120 | ~$105 | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RTX 2060 12GB | 12GB GDDR6 | $140–$180 | ~$160 | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RTX 3060 12GB | 12GB GDDR6 | $170–$380 | ~$275 | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RTX 3070 | 8GB GDDR6 | $210–$300 | ~$255 | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RTX 3070 Ti | 8GB GDDR6X | $100–$280 | ~$190 | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RTX 3080 10GB | 10GB GDDR6X | $325–$400 | ~$365 | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RTX 3080 12GB | 12GB GDDR6X | $230–$380 | ~$305 | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RTX 3090 | 24GB GDDR6X | $950–$1,125 | ~$1,040 | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RTX 4060 | 8GB GDDR6 | $230–$310 | ~$270 | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RTX 4060 Ti 8GB | 8GB GDDR6 | $240–$300 | ~$270 | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RTX 4060 Ti 16GB | 16GB GDDR6 | $380–$480 | ~$430 | [insiderllm canonical](https://insiderllm.com/hardware/) |

### AMD Cards (Used Prices from eBay Sold Auctions, 2026-03-02)
| GPU | VRAM | Used Price Range | Typical Price | Source |
|-----|------|------------------|---------------|--------|
| RX 7600 | 8GB GDDR6 | $170–$225 | ~$200 | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RX 7700 XT | 12GB GDDR6 | $300–$350 | ~$325 | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RX 7800 XT | 16GB GDDR6 | $380–$550 | ~$465 | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RX 7900 GRE | 16GB GDDR6 | $400–$550 | ~$475 | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RX 7900 XT | 20GB GDDR6 | $500–$700 | ~$600 | [insiderllm canonical](https://insiderllm.com/hardware/) |

### New MSRP (Preliminary for 2026)
| GPU | VRAM | Bus Width | Bandwidth | MSRP | Source |
|-----|------|-----------|-----------|------|--------|
| RX 9070 | 16GB GDDR6 | 256-bit | 608 GB/s | $549 | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RX 9070 XT | 16GB GDDR6 | 256-bit | 608 GB/s | $599 | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RTX 5060 Ti 16GB | 16GB GDDR7 | 128-bit | TBD | TBD (rumored) | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RTX 5070 | 12GB GDDR7 | 192-bit | 672 GB/s | $549 | [insiderllm canonical](https://insiderllm.com/hardware/) |

## Benchmark Data

### NVIDIA CUDA Card Benchmarks (LLaMA3 8B Q4)
| GPU | Bandwidth | tok/s | Source |
|-----|-----------|-------|--------|
| RTX 3060 12GB | 360 GB/s | 51 tok/s | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RTX 3070 | 448 GB/s | 71 tok/s | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RTX 3080 10GB | 760 GB/s | 106 tok/s | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RTX 3080 12GB | 912 GB/s | 107 tok/s | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RTX 3090 | 936 GB/s | 112 tok/s | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RTX 4060 | 272 GB/s | 38 tok/s | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RTX 4060 Ti 16GB | 288 GB/s | 48 tok/s | [insiderllm canonical](https://insiderllm.com/hardware/) |

### AMD ROCm Card Benchmarks (LLaMA3/LLaMA2)
| GPU | Bandwidth | LLaMA3 8B Q4 (tok/s) | LLaMA2 7B Q4 (tok/s) | Source |
|-----|-----------|----------------------|----------------------|--------|
| RX 7600 | 288 GB/s | Not listed | Not listed | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RX 7700 XT | 432 GB/s | Not listed | **96 tok/s** | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RX 7800 XT | 624 GB/s | **39 tok/s** | **96 tok/s** | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RX 7900 GRE | 576 GB/s | Not listed | Not listed | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RX 7900 XT | 800 GB/s | Not listed | **116 tok/s** (sustained: 97 tok/s) | [insiderllm canonical](https://insiderllm.com/hardware/) |

### Multi-GPU & Concurrency Benchmarks
| Configuration | AMD MI300X Throughput | NVIDIA H100 Throughput | Advantage | CUDA Gap Score | Source |
|---------------|----------------------|------------------------|-----------|----------------|--------|
| 2× GPU | 35,638 tok/s | 46,129 tok/s | +32.1% | 61.5 | [aimultiple.com](https://research.aimultiple.com/cuda-vs-rocm/) |
| 4× GPU | 60,986 tok/s | 84,683 tok/s | +38.9% | 71.0 | [aimultiple.com](https://research.aimultiple.com/cuda-vs-rocm/) |
| 8× GPU | 101,069 tok/s | 147,606 tok/s | +46.0% | 78.1 | [aimultiple.com](https://research.aimultiple.com/cuda-vs-rocm/) |

### Concurrency Performance (LLaMA3 8B Q4)
| Concurrent Users | NVIDIA H100 Throughput | AMD MI300X Throughput | NVIDIA Advantage | Source |
|------------------|------------------------|-----------------------|------------------|--------|
| 16 users | Baseline | — | +30.8% | [aimultiple.com](https://research.aimultiple.com/cuda-vs-rocm/) |
| 128 users | Baseline | — | +38.7% | [aimultiple.com](https://research.aimultiple.com/cuda-vs-rocm/) |
| 512 users | Baseline | — | +67.0% | [aimultiple.com](https://research.aimultiple.com/cuda-vs-rocm/) |

### DeepSeek R1 Inference (RTX 4090 vs RX 7900 XTX)
| GPU | Model | Performance | Source |
|-----|-------|-------------|--------|
| RTX 4090 | DeepSeek R1 | Baseline | [techbloat.com](https://www.techbloat.com/amd-radeon-rx-7900-xtx-outperforms-nvidia-geforce-rtx-4090-in-deepseek-ai-inference-benchmarks-how-to-run-r1-on-your-local-amd-system.html) |
| RX 7900 XTX | DeepSeek R1 | **Outperforms RTX 4090** | [techbloat.com](https://www.techbloat.com/amd-radeon-rx-7900-xtx-outperforms-nvidia-geforce-rtx-4090-in-deepseek-ai-inference-benchmarks-how-to-run-r1-on-your-local-amd-system.html) |

## Key Specs

### Memory Bandwidth & Model Capacity
| VRAM | Max Model Size (Quantized) | Source |
|------|---------------------------|--------|
| 6GB | 7B Q4 max (tight) | [insiderllm canonical](https://insiderllm.com/hardware/) |
| 8GB | 8B Q4 comfortable, 14B Q2 possible | [insiderllm canonical](https://insiderllm.com/hardware/) |
| 10GB | 8B Q6, 14B Q3 possible | [insiderllm canonical](https://insiderllm.com/hardware/) |
| 12GB | 14B Q4, 8B Q8 or FP16 | [insiderllm canonical](https://insiderllm.com/hardware/) |
| 16GB | 30B Q3, 14B Q6 | [insiderllm canonical](https://insiderllm.com/hardware/) |
| 20GB | 30B Q4, some 70B Q2 | [insiderllm canonical](https://insiderllm.com/hardware/) |
| 24GB | 30B Q5, 70B Q2-Q3 | [insiderllm canonical](https://insiderllm.com/hardware/) |

### System RAM Offloading Speeds
| RAM Type | Dual-Channel Speed | vs GDDR6X (936 GB/s) Slowdown | Source |
|----------|-------------------|-------------------------------|--------|
| DDR3 | 14.9–12.8 GB/s | ~37× slower per layer | [insiderllm canonical](https://insiderllm.com/hardware/) |
| DDR4-3200 | 25.6 GB/s | ~37× slower per layer | [insiderllm canonical](https://insiderllm.com/hardware/) |
| DDR5 | 48 GB/s | ~19.5× slower per layer | [insiderllm canonical](https://insiderllm.com/hardware/) |

### GPU Specs
| GPU | VRAM | Bandwidth | TDP | Architecture | Source |
|-----|------|-----------|-----|-------------|--------|
| GTX 1660 Super | 6GB GDDR6 | 336 GB/s | 125W | Turing | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RTX 2060 12GB | 12GB GDDR6 | 336 GB/s | 185W | Turing | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RTX 3060 12GB | 12GB GDDR6 | 360 GB/s | 170W | Ampere | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RTX 3070 | 8GB GDDR6 | 448 GB/s | 220W | Ampere | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RTX 3070 Ti | 8GB GDDR6X | 608 GB/s | 290W | Ampere | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RTX 3080 10GB | 10GB GDDR6X | 760 GB/s | 320W | Ampere | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RTX 3080 12GB | 12GB GDDR6X | 912 GB/s | 350W | Ampere | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RTX 3090 | 24GB GDDR6X | 936 GB/s | 350W | Ampere | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RTX 4060 | 8GB GDDR6 | 272 GB/s | 115W | Ada Lovelace | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RTX 4060 Ti 8GB | 8GB GDDR6 | 288 GB/s | 160W | Ada Lovelace | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RTX 4060 Ti 16GB | 16GB GDDR6 | 288 GB/s | 165W | Ada Lovelace | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RX 7600 | 8GB GDDR6 | 288 GB/s | 165W | RDNA 3 | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RX 7700 XT | 12GB GDDR6 | 432 GB/s | 245W | RDNA 3 | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RX 7800 XT | 16GB GDDR6 | 624 GB/s | 263W | RDNA 3 | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RX 7900 GRE | 16GB GDDR6 | 576 GB/s | 260W | RDNA 3 | [insiderllm canonical](https://insiderllm.com/hardware/) |
| RX 7900 XT | 20GB GDDR6 | 800 GB/s | 315W | RDNA 3 | [insiderllm canonical](https://insiderllm.com/hardware/) |

## Competitor Coverage

NO DATA FOUND

## Internal Context

### Document Context
[D1] (insiderllm-content-plan.md) - Priority 2 items related to local AI cost/benefit analysis.
[D