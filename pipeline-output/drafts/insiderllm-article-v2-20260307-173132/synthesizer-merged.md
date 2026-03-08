## Price Data

**GPU Pricing (from InsiderLLM canonical hardware database):**

| GPU | VRAM | Used Price Range | Typical Used Price |
|-----|------|------------------|-------------------|
| GTX 1660 Super | 6GB GDDR6 | $90–$120 | ~$105 (2026-03-02) |
| RTX 2060 12GB | 12GB GDDR6 | $140–$180 | ~$160 (2026-03-02) |
| RTX 3060 12GB | 12GB GDDR6 | $170–$380 | ~$275 (2026-03-02) |
| RTX 3070 | 8GB GDDR6 | $210–$300 | ~$255 (2026-03-02) |
| RTX 3070 Ti | 8GB GDDR6X | $100–$280 | ~$190 (2026-03-02) |
| RTX 3080 10GB | 10GB GDDR6X | $325–$400 | ~$365 (2026-03-02) |
| RTX 3080 12GB | 12GB GDDR6X | $230–$380 | ~$305 (2026-03-02) |
| RTX 3090 | 24GB GDDR6X | $950–$1125 | ~$1040 (2026-03-02) |
| RTX 4060 | 8GB GDDR6 | $230–$310 | ~$270 (2026-03-02) |
| RTX 4060 Ti 8GB | 8GB GDDR6 | $240–$300 | ~$270 (2026-03-02) |
| RTX 4060 Ti 16GB | 16GB GDDR6 | $380–$480 | ~$430 (2026-03-02) |
| AMD RX 7600 | 8GB GDDR6 | $170–$225 | ~$200 (2026-03-02) |
| AMD RX 7700 XT | 12GB GDDR6 | $300–$350 | ~$325 (2026-03-02) |
| AMD RX 7800 XT | 16GB GDDR6 | $380–$550 | ~$465 (2026-03-02) |
| AMD RX 7900 GRE | 16GB GDDR6 | $400–$550 | ~$475 (2026-03-02) |
| AMD RX 7900 XT | 20GB GDDR6 | $500–$700 | ~$600 (2026-03-02) |

**RAM Pricing:**
- DDR4: $50-80 per 16GB used, $100-170 per 32GB used
- DDR5: $200-300 per 32GB used, $320-450 per 64GB used

**Cloud/API Pricing:**
- DeepSeek V3.2 API: ~$0.27–0.55 per million tokens

**MSRP for New Cards (Preliminary):**
- NVIDIA RTX 5070: $549 MSRP
- AMD RX 9070: $549 MSRP
- AMD RX 9070 XT: $599 MSRP

## Benchmark Data

**HumanEval Scores:**
| Model | HumanEval % | FIM Support | VRAM Q4 | Best For |
|-------|-------------|-------------|---------|----------|
| Qwen 2.5 Coder 7B | 88.4% | Yes | ~5 GB | Autocomplete/FIM |
| Qwen 2.5 Coder 14B | ~89% | Yes | ~9 GB | Best at 16GB tier |
| Qwen 2.5 Coder 32B | 92.7% | Yes | ~20 GB | Best FIM/autocomplete at 24GB |
| DeepSeek Coder V2 Lite | 81.1% | Yes | ~5 GB | Reasoning-heavy tasks |
| DeepSeek Coder 33B (Q3) | 70% / 78% | Yes | ~16-20 GB | Squeezed into 16GB/with CodeFuse |
| GLM-4.7 | N/A | N/A | ~5 GB | Agent coordination |

**SWE-bench Verified Scores (February 2026):**
| Model | Score | License |
|-------|-------|---------|
| Claude Opus 4.6 (Proprietary) | 80.9% | Proprietary |
| GPT-5.2 (Proprietary) | 80.0% | Proprietary |
| Kimi K2.5 | 76.8% | MIT* |
| GLM-4.7 | 73.8% | MIT |
| DeepSeek V3.2 | 73.1% | MIT |
| Qwen3-Coder-Next | 70.6% | Apache 2.0 |

*MIT with commercial restrictions

**AIME 2025 Mathematical Reasoning:**
| Model | Score |
|-------|-------|
| GPT-5.2 (Proprietary) | 99.0% |
| Gemini 2.0 Flash Thinking (Proprietary) | 97.0% |
| Gemini 2.0 Pro Thinking (Proprietary) | 95.7% |
| GLM-4.7 | 95.7% |
| DeepSeek V3.2 | 93.1% |
| Qwen2.5-Max | 92.3% |

**GPQA Diamond Scientific Reasoning:**
| Model | Score |
|-------|-------|
| Gemini 3 Pro (Proprietary) | 90.8% |
| GPT-5.2 (Proprietary) | 90.3% |
| GLM-4.7 | 85.7% |
| DeepSeek V3.2 | ~85–88% (estimated) |
| Qwen3 variants | ~84–87% |

**τ²-Bench Agent Coordination:**
| Model | Score |
|-------|-------|
| GLM-4.7 | 87.4% |

**LiveCodeBench v6:**
| Model | Score |
|-------|-------|
| Kimi K2.5 | 85.0% |
| GLM-4.7 | 84.9% |

**Terminal-Bench 2.0:**
| Model | Score |
|-------|-------|
| Kimi K2.5 | 40.45% |
| GLM-4.7 | 41.0% |

**GPU Token Generation Benchmarks:**
| GPU | llama3 8b Q4 | llama2 7b Q4 |
|-----|--------------|--------------|
| RTX 3060 12GB | 51 tok/s | 76 tok/s |
| RTX 3070 | 71 tok/s | N/A |
| RTX 3090 | 112 tok/s | N/A |
| AMD RX 7900 XT | 116 tok/s sustained | N/A |

## Key Specs

**GPU Specifications:**

| GPU | VRAM | Memory Bandwidth | TDP | Architecture | Best For |
|-----|------|------------------|-----|--------------|----------|
| GTX 1660 Super | 6GB GDDR6 | 336 GB/s | 125W | Turing | Experimentation |
| RTX 2060 12GB | 12GB GDDR6 | 336 GB/s | 185W | Turing | Budget option |
| RTX 3060 12GB | 12GB GDDR6 | 360 GB/s | 170W | Ampere | Entry-level workhorse |
| RTX 3070 | 8GB GDDR6 | 448 GB/s | 220W | Ampere | Fast, VRAM limited |
| RTX 3070 Ti | 8GB GDDR6X | 608 GB/s | 290W | Ampere | Better bandwidth, VRAM limited |
| RTX 3080 10GB | 10GB GDDR6X | 760 GB/s | 320W | Ampere | High bandwidth |
| RTX 3080 12GB | 12GB GDDR6X | 912 GB/s | 350W | Ampere | Value, excellent bandwidth |
| RTX 3090 | 24GB GDDR6X | 936 GB/s | 350W | Ampere | Budget local AI king |
| RTX 4060 | 8GB GDDR6 | 272 GB/s | 115W | Ada Lovelace | Avoid for local AI |
| RTX 4060 Ti 8GB | 8GB GDDR6 | 288 GB/s | 160W | Ada Lovelace | Bandwidth-starved |
| RTX 4060 Ti 16GB | 16GB GDDR6 | 288 GB/s | 165W | Ada Lovelace | Trade VRAM for speed |
| AMD RX 7600 | 8GB GDDR6 | 288 GB/s | 165W | RDNA 3 | AMD, tradeoffs |
| AMD RX 7700 XT | 12GB GDDR6 | 432 GB/s | 245W | RDNA 3 | Competitive, ROCm risk |
| AMD RX 7800 XT | 16GB GDDR6 | 624 GB/s | 263W | RDNA 3 | Best AMD value, ROCm risk |
| AMD RX 7900 GRE | 16GB GDDR6 | 576 GB/s | 260W | RDNA 3 | Competitive with RTX 3080 12GB |
| AMD RX 7900 XT | 20GB GDDR6 | 800 GB/s | 315W | RDNA 3 | AMD's best, ROCm risk |

## Competitor Coverage

No competitor coverage data was provided in the inputs.

## Internal Context

The InsiderLLM content plan indicates a focus on local AI, with several articles already published covering topics like GPU buying guides, model guides (DeepSeek, Qwen), and setup instructions for tools like Ollama, LM Studio, and OpenWebUI. There is an emphasis on cost-effective solutions and making local AI accessible.

## New Data

- **Qwen3-Coder-Next:** A new 80B MoE model with 3B active parameters per token. Requires 35-40GB VRAM and outperforms DeepSeek V3.2 on SWE-bench. Licensed under Apache 2.0.
- **Kimi K2.5:** A 1 trillion parameter model with 32B active parameters per token. Requires 240GB VRAM. Licensed under MIT with commercial restrictions.
- **GLM-4.7:** Runs on a single RTX 4090 and boasts high scores for mathematical reasoning and agent coordination.
- **Performance Data:** Detailed token/second benchmarks for several GPUs with different LLama versions.
- **VRAM → Model size guide:** A helpful guide to matching VRAM capacity with model size and quantization levels.

## Gaps

- **Competitor Coverage:** What are other sites saying about these models and GPUs? (Specifically, what angle are they taking?)
- **Cloud GPU Rental Costs:** Provide more detailed pricing for cloud GPU instances for models like Kimi K2.5 and DeepSeek V3.2.
- **ROCm Compatibility:** A deeper dive into the current state of ROCm support for these models and which frameworks work best.
- **More detailed benchmarks:** Benchmarks on more models and with different quantization methods.
- **Power Consumption:** Actual power draw numbers for each GPU during LLM inference.

## Suggested Angle

InsiderLLM should focus on *practicality and value* for local AI development. Many sites will hype the largest models, but InsiderLLM should emphasize models that deliver the best performance *within the constraints of consumer hardware*. Focus on:

- **Real-world performance on common GPUs:** Highlight the RTX 3090 as the best value for serious local LLM work and the RTX 3060 as the entry point.
- **Quantization and optimization techniques:** Explain how to maximize performance on limited hardware.
- **The trade-offs between model size, performance, and cost:** Help readers make informed decisions about what they can realistically run on their hardware.
- **AMD ROCm support:** Provide honest and up-to-date information on the state of ROCm compatibility.
- **Highlighting the Qwen family:** Given the strong performance and open-source license, Qwen models are a good fit for InsiderLLM's focus on accessibility and value.