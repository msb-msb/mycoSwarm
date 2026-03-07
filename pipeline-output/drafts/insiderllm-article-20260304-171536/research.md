# Research Bundle: AMD ROCm vs CUDA for Local AI in 2026

## Key Facts & Data Points

| Claim | Source URL |
|-------|------------|
| NVIDIA CUDA dominates with ~85-90% market share in data center GPUs for AI workloads | [droid4x.com](https://droid4x.com/cuda-vs-alternatives-for-local-llms/) |
| AMD ROCm 7.2 officially supports RDNA 3 (RX 7000 series) and select RDNA 4 (RX 9000) consumer GPUs | [insiderllm.com](https://www.insiderllm.com/guides/amd-vs-nvidia-local-ai-rocm/) |
| PyTorch now officially supports AMD on Windows via ROCm 6.4.4 public preview (Sept 2025) | [techpowerup.com](https://www.techpowerup.com/341329/amd-enables-pytorch-on-radeon-rx-7000-9000-gpus-with-windows-and-linux-preview) |
| CUDA gap: NVIDIA's software ecosystem delivers 28.7-99.1% performance advantage beyond hardware specs | [aimultiple.com](https://research.aimultiple.com/cuda-vs-rocm/) |
| AMD ROCm achieves ~0.06 tok/s per GB/s bandwidth vs NVIDIA's ~0.13 tok/s per GB/s (2x efficiency penalty) | [insiderllm.com](https://insiderllm.com/guides/rocm-vs-cuda-local-ai-2026/) |
| ROCm 7.2 on RDNA 4 (RX 9070) has Vulkan beating ROCm HIP by 14-30% in llama.cpp benchmarks | [insiderllm.com](https://insiderllm.com/guides/rocm-vs-cuda-local-ai-2026/) |
| vLLM integration on AMD: 93% of test groups pass on ROCm CI pipeline as of Jan 2026 (up from 37% in Nov 2025) | [insiderllm.com](https://www.insiderllm.com/guides/amd-vs-nvidia-local-ai-rocm/) |
| CUDA gap increases with scale: At 512 concurrent users, NVIDIA H100 delivers +67% more throughput vs AMD MI300X | [aimultiple.com](https://research.aimultiple.com/cuda-vs-rocm/) |
| ROCm works best on Linux; Windows support is limited and not recommended for serious work | [insiderllm.com](https://www.insiderllm.com/guides/amd-vs-nvidia-local-ai-rocm/) |
| HIP backend allows converting CUDA code to run on AMD hardware with 80-90% success rate | [tillcode.com](https://tillcode.com/amd-rocm-vs-nvidia-cuda-which-gpu-should-developers-choose/) |

## Price Data

### NVIDIA GPU Prices (Used - eBay sold auctions, typical)
| GPU | VRAM | Bandwidth | TDP | Used Price Range | Typical Price | Source |
|-----|------|-----------|-----|------------------|---------------|---------|
| GTX 1660 Super | 6GB GDDR6 | 336 GB/s | 125W | $90–$120 | ~$105 | [insiderllm.com] |
| RTX 2060 12GB | 12GB GDDR6 | 336 GB/s | 185W | $140–$180 | ~$160 | [insiderllm.com] |
| RTX 3060 12GB | 12GB GDDR6 | 360 GB/s | 170W | $170–$380 | ~$275 | [insiderllm.com] |
| RTX 3070 | 8GB GDDR6 | 448 GB/s | 220W | $210–$300 | ~$255 | [insiderllm.com] |
| RTX 3070 Ti | 8GB GDDR6X | 608 GB/s | 290W | $100–$280 | ~$190 | [insiderllm.com] |
| RTX 3080 10GB | 10GB GDDR6X | 760 GB/s | 320W | $325–$400 | ~$365 | [insiderllm.com] |
| RTX 3080 12GB | 12GB GDDR6X | 912 GB/s | 350W | $230–$380 | ~$305 | [insiderllm.com] |
| RTX 3090 | 24GB GDDR6X | 936 GB/s | 350W | $950–$1125 | ~$1,040 | [insiderllm.com] |
| RTX 4060 | 8GB GDDR6 | 272 GB/s | 115W | $230–$310 | ~$270 | [insiderllm.com] |
| RTX 4060 Ti 8GB | 8GB GDDR6 | 288 GB/s | 160W | $240–$300 | ~$270 | [insiderllm.com] |
| RTX 4060 Ti 16GB | 16GB GDDR6 | 288 GB/s | 165W | $380–$480 | ~$430 | [insiderllm.com] |

### AMD GPU Prices (Used - typical)
| GPU | VRAM | Bandwidth | TDP | Used Price Range | Typical Price | Source |
|-----|------|-----------|-----|------------------|---------------|---------|
| RX 7600 | 8GB GDDR6 | 288 GB/s | 165W | $170–$225 | ~$200 | [insiderllm.com] |
| RX 7700 XT | 12GB GDDR6 | 432 GB/s | 245W | $300–$350 | ~$325 | [insiderllm.com] |
| RX 7800 XT | 16GB GDDR6 | 624 GB/s | 263W | $380–$550 | ~$465 | [insiderllm.com] |
| RX 7900 GRE | 16GB GDDR6 | 576 GB/s | 260W | $400–$550 | ~$475 | [insiderllm.com] |
| RX 7900 XT | 20GB GDDR6 | 800 GB/s | 315W | $500–$700 | ~$600 | [insiderllm.com] |
| RX 7900 XTX | 24GB GDDR6 | 960 GB/s | 355W | — | ~$750-950 new, ~$750 used | [insiderllm.com] |

### New MSRP (Preliminary)
| GPU | VRAM | Bandwidth | TDP | MSRP | Source |
|-----|------|-----------|-----|------|---------|
| RTX 5060 | 8GB GDDR7 | N/A | N/A | N/A | [insiderllm.com] |
| RTX 5060 Ti | 16GB GDDR7 | N/A | N/A | N/A | [insiderllm.com] |
| RTX 5070 | 12GB GDDR7 | 672 GB/s | 250W | $549 | [insiderllm.com] |
| RX 9070 | 16GB GDDR6 | 608 GB/s | 220W | $549 | [insiderllm.com] |
| RX 9070 XT | 16GB GDDR6 | 608 GB/s | 250W | $599 | [insiderllm.com] |

### Price-Per-VRAM Comparison
| GPU | VRAM | Street Price | $/GB | Source |
|-----|------|--------------|------|---------|
| RTX 4090 | 24GB | $1,800+ | $75/GB | [insiderllm.com] |
| RTX 3090 (used) | 24GB | $750-900 | $31-37/GB | [insiderllm.com] |
| RX 7900 XTX | 24GB | $750-950 | $31-40/GB | [insiderllm.com] |
| RX 7900 XT | 20GB | $530-675 | $26-34/GB | [insiderllm.com] |

## Benchmark Data

### Llama 3 8B Q4 Inference (tok/s)
| GPU | Platform | tok/s | Source |
|-----|----------|-------|---------|
| RTX 3060 12GB | CUDA | 51 tok/s | [insiderllm.com] |
| RTX 3080 10GB | CUDA | 106 tok/s | [insiderllm.com] |
| RTX 3080 12GB | CUDA | 107 tok/s | [insiderllm.com] |
| RTX 3090 24GB | CUDA | 112 tok/s | [insiderllm.com] |
| RTX 4060 Ti 16GB | CUDA | 48 tok/s | [insiderllm.com] |
| RX 7800 XT 16GB | ROCm | 39 tok/s | [insiderllm.com] |
| RX 7900 XT 20GB | ROCm | ~45 tok/s* | [insiderllm.com] |

### Llama 3 8B Q4 Multi-GPU Comparison (throughput)
| Configuration | AMD MI300X | NVIDIA H100 | NVIDIA Advantage | Source |
|---------------|------------|-------------|------------------|---------|
| 2x GPU | 35,638 tok/s | 46,129 tok/s | +32.1% | [aimultiple.com] |
| 4x GPU | 60,986 tok/s | 84,683 tok/s | +38.9% | [aimultiple.com] |
| 8x GPU | 101,069 tok/s | 147,606 tok/s | +46% | [aimultiple.com] |

### AMD vs NVIDIA Head-to-Head Benchmarks (llama.cpp)
| Model | RTX 4090 (CUDA) | RX 7900 XTX (ROCm) | Winner | Source |
|-------|-----------------|--------------------|--------|---------|
| Llama 3 8B Q4_K_M | 142 tok/s | 89 tok/s | NVIDIA (+60%) | [insiderllm.com] |
| Llama 3 70B Q4_K_M | 38 tok/s | 23 tok/s | NVIDIA (+65%) | [insiderllm.com] |
| DeepSeek R1 7B | 100% | 113% | AMD (+13%) | [insiderllm.com] |
| DeepSeek R1 14B | 100% | 102% | AMD (+2%) | [insiderllm.com] |

### RDNA 4 Vulkan vs ROCm HIP Comparison (RX 9070 XT)
| Model (Q8_0) | ROCm HIP | Vulkan | Advantage | Source |
|--------------|----------|--------|-----------|---------|
| Llama 3.1 8B | 60.9 tok/s | 69.2 tok/s | +14% | [insiderllm.com] |
| Qwen3 8B | 58.3 tok/s | 66.2 tok/s | +13% | [insiderllm.com] |
| Mistral 7B | 63.3 tok/s | 72.6 tok/s | +15% | [insiderllm.com] |
| GPT-OSS 20B | 117.2 tok/s | 152.7 tok/s | +30% | [insiderllm.com] |

### Prompt Processing Benchmarks (tok/s)
| Backend | Llama 3.1 8B (512-token prompts) | Source |
|---------|----------------------------------|---------|
| Vulkan | 2,888 tok/s | [insiderllm.com] |
| ROCm HIP | 1,149 tok/s | [insiderllm.com] |

### Model Size to VRAM Requirements (from reference data)
| VRAM | Max Model Size | Source |
|------|----------------|---------|
| 6GB | 7B Q4 max, tight | [reference] |
| 8GB | 8B Q4 comfortable, 14B Q2 possible | [reference] |
| 10GB | 8B Q6, 14B Q3 possible | [reference] |
| 12GB | 14B Q4, 8B Q8 or FP16 | [reference] |
| 16GB | 30B Q3, 14B Q6 | [reference] |
| 20GB | 30B Q4, some 70B Q2 | [reference] |
| 24GB | 30B Q5, 70B Q2-Q3 | [reference] |

### System RAM Offloading Speed Penalty
| Memory Type | Bandwidth | Offload Speed vs VRAM | Source |
|-------------|-----------|----------------------|---------|
| DDR3 (1866MHz) | 14.9 GB/s dual-channel | Baseline | [reference] |
| DDR4 (3200MHz) | 25.6 GB/s dual-channel | ~2x DDR3 | [reference] |
| DDR5 (6000MHz) | 48 GB/s dual-channel | ~4x DDR3 | [reference] |

## Key Specs

### NVIDIA GPU Specifications
| GPU | Architecture | VRAM Type | Memory Bus | Bandwidth | TDP | Source |
|-----|--------------|-----------|-------------|-----------|-----|---------|
| GTX 1660 Super | Turing | GDDR6 | 192-bit | 336 GB/s | 125W | [insiderllm.com] |
| RTX 2060 12GB | Turing | GDDR6 | 192-bit | 336 GB/s | 185W | [insiderllm.com] |
| RTX 3060 12GB | Ampere | GDDR6 | 192-bit | 360 GB/s | 170W | [insiderllm.com] |
| RTX 3070 | Ampere | GDDR6 | 256-bit | 448 GB/s | 220W | [insiderllm.com] |
| RTX 3070 Ti | Ampere | GDDR6X | 256-bit | 608 GB/s | 290W | [insiderllm.com] |
| RTX 3080 10GB | Ampere | GDDR6X | 320-bit | 760 GB/s | 320W | [insiderllm.com] |
| RTX 3080 12GB | Ampere | GDDR6X | 320-bit | 912 GB/s | 350W | [insiderllm.com] |
| RTX 3090 | Ampere | GDDR6X | 384-bit | 936 GB/s | 350W | [insiderllm.com] |
| RTX 4060 | Ada Lovelace | GDDR6 | 128-bit | 272 GB/s | 115W | [insiderllm.com] |
| RTX 4060 Ti 8GB | Ada Lovelace | GDDR6 | 128-bit | 288 GB/s | 160W | [insiderllm.com] |
| RTX 4060 Ti 16GB | Ada Lovelace | GDDR6 | 12