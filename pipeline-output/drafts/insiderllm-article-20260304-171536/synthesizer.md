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

### New MSRP (Preliminary)
| GPU | VRAM | Bandwidth | TDP | MSRP | Source |
|-----|------|-----------|-----|------|---------|
| RTX 5060 | 8GB GDDR7 | N/A | N/A | N/A | [insiderllm.com] |
| RTX 5060 Ti | 16GB GDDR7 | N/A | N/A | N/A | [insiderllm.com] |
| RTX 5070 | 12GB GDDR7 | 672 GB/s | 250W | $549 | [insiderllm.com] |
| RX 9070 | 16GB GDDR6 | 608 GB/s | 220W | $549 | [insiderllm.com] |
| RX 9070 XT | 16GB GDDR6 | 608 GB/s | 250W | $599 | [insiderllm.com] |

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
| RTX 4060 Ti 16GB | Ada Lovelace | GDDR6 | 128-bit | 288 GB/s | 165W | [insiderllm.com] |

### AMD GPU Specifications
| GPU | Architecture | VRAM Type | Memory Bus | Bandwidth | TDP | Source |
|-----|--------------|-----------|-------------|-----------|-----|---------|
| RX 7600 | RDNA 3 | GDDR6 | 128-bit | 288 GB/s | 165W | [insiderllm.com] |
| RX 7700 XT | RDNA 3 | GDDR6 | 256-bit | 432 GB/s | 245W | [insiderllm.com] |
| RX 7800 XT | RDNA 3 | GDDR6 | 256-bit | 624 GB/s | 263W | [insiderllm.com] |
| RX 7900 GRE | RDNA 3 | GDDR6 | 256-bit | 576 GB/s | 260W | [insiderllm.com] |
| RX 7900 XT | RDNA 3 | GDDR6 | 320-bit | 800 GB/s | 315W | [insiderllm.com] |

## Competitor Coverage

No competitor coverage data found in provided sources.

## Internal Context

*   Priority content areas: Anthropic vs OpenAI vs Local, When to Use Cloud vs Local, Privacy: Local vs Cloud AI, Cost Calculator: Local vs API. (D1)
*   Published content on related topics: GPU Buying Guide, RTX 5060 Ti 16GB News, VRAM Requirements Guide, Used RTX 3090 Buying Guide, NVIDIA Price Hikes Analysis, AMD vs NVIDIA for Local AI, Budget AI PC Under $500, What Can You Run on X GB VRAM, CPU-Only LLMs, Mac vs PC for Local AI, Best Models Under 3B Parameters, Running LLMs on Mac M-Series (D2, D3).
*   Local LLMs vs ChatGPT/Claude, OpenClaw vs Commercial AI Agents, Cost to Run LLMs Locally also published (D4).
*   Daily Reddit engagement on r/LocalLLaMA, 2-4 articles published daily (D5).

## Gaps

*   **CRITICAL:** RTX 5060/5060 Ti/5070 GDDR7 bandwidth specs are missing.
*   RX 9070/9070 XT specs are preliminary and need confirmation.
*   Performance benchmarks for newer GPUs (RTX 50 series, RX 90 series) are entirely missing.
*   Detailed ROCm compatibility list for specific models/frameworks is needed.
*   Competitor analysis: What are other sites saying about AMD vs NVIDIA for local AI?
*   More detailed ROCm performance data beyond the limited llama.cpp benchmarks.
*   Any data on power consumption differences between AMD and NVIDIA cards during LLM inference.
*   Data on the effectiveness of Vulkan on AMD GPUs vs ROCm.

## Suggested Angle

InsiderLLM should focus on **practical performance and real-world usability**, going beyond raw benchmarks. The current data shows NVIDIA maintains a significant performance lead, but AMD is closing the gap, especially with Vulkan.  Instead of simply stating "NVIDIA is faster," the article should:

1.  **Quantify the *cost* of that performance advantage.** How much more do you pay for NVIDIA to get X% faster inference?
2.  **Highlight the trade-offs.** AMD offers more VRAM for the price, allowing larger models.  Is that worth the performance hit, *especially* if you're willing to optimize and use Vulkan?
3.  **Emphasize the ROCm ecosystem's ongoing improvements.**  Acknowledge the historical issues but focus on the progress being made and the potential for future gains.  Provide clear guidance on verifying ROCm compatibility.
4.  **Provide a tiered recommendation system.**  "Best for absolute performance," "Best value," "Best for large models," "Best if you’re already an AMD user."