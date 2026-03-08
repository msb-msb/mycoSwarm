# Research Bundle: Best Open Source Coding Models for Local Development 2026

---

## Key Facts & Data Points

**Qwen 2.5 Coder Dominance:**
- Qwen 2.5 Coder 7B achieves 88.4% HumanEval with FIM support, making it the best autocomplete model at 7B parameters, beating CodeStral-22B and DeepSeek Coder 33B V1 (Source: insiderllm.com)
- Qwen 2.5 Coder 14B achieves ~89% HumanEval with FIM support, best model at 16GB VRAM tier (Source: insiderllm.com)
- Qwen 2.5 Coder 32B achieves 92.7% HumanEval at 24GB VRAM, the FIM king for autocomplete (Source: insiderllm.com)

**Qwen3-Coder-Next:**
- Released February 3, 2026 by Alibaba's Qwen team (Source: dev.to)
- 80B MoE with only 3B active parameters per token (Source: insiderllm.com)
- SWE-bench Verified score of 70.6% — ahead of DeepSeek V3.2 (70.2%) on real-world GitHub issue resolution (Source: insiderllm.com)
- Requires ~35-40GB VRAM at Q4 quantization; runs natively on Mac with 48GB+ unified memory (Source: insiderllm.com)
- Apache 2.0 license, fully open-source for commercial use (Source: dev.to)

**Kimi K2.5:**
- Achieves 76.8% SWE-bench Verified — highest open-source score (Source: dev.to)
- Released January 27, 2026 by Moonshot AI (Source: dev.to)
- 1 trillion parameters with 32B active per token (Source: dev.to)
- Requires ~240GB VRAM for INT4 quantization — practically requires cloud GPU rental or API access (Source: dev.to)
- MIT license with commercial restrictions (free for companies under 100M monthly active users) (Source: dev.to)

**DeepSeek V3.2:**
- Achieves 73.1% SWE-bench Verified, ranking #6 open-source on leaderboard (Source: dev.to)
- 671 billion parameters with 37B active via MoE (Source: dev.to)
- Requires 336GB VRAM with 4-bit quantization — most users access via API at $0.27–0.55 per million tokens (Source: dev.to)
- AIME 2025 score of 93.1% for mathematical reasoning (Source: dev.to)

**GLM-4.7:**
- Achieves 73.8% SWE-bench Verified, ranking #5 open-source (Source: dev.to)
- Runs on single RTX 4090 (24GB VRAM) using GLM-4.7-Flash variant (Source: dev.to)
- 30B total parameters, 3B active (efficient!) (Source: dev.to)
- τ²-Bench score of 87.4% — highest verified open-source agent coordination benchmark (Source: dev.to)
- AIME 2025 score of 95.7% for mathematical reasoning — top open-source (Source: dev.to)

**Gemma 3:**
- Ranked 4th in Index.dev testing with more specialized strengths but lower overall scores (Source: index.dev)
- Failed on punctuation handling in palindrome detection task (Source: index.dev)

**Cohere:**
- Ranked 5th in Index.dev testing, scoring Medium across tasks (Source: index.dev)
- Good at concise explanations but lacks user input flexibility (Source: index.dev)

**LLaMA 4:**
- Ranked 3rd in Index.dev testing with more specialized strengths (Source: index.dev)
- Failed Task 1 due to hardcoded lists instead of user input (Source: index.dev)
- Achieved High scores on palindrome detection and code explanation tasks (Source: index.dev)

**Performance Benchmarks:**
- NVIDIA RTX 3060 12GB: llama3 8b Q4 = 51 tok/s, llama2 7b Q4 = 76 tok/s, llama2 13b Q4 = 35 tok/s (Source: insiderllm.com reference data)
- NVIDIA RTX 3090: llama3 8b Q4 = 112 tok/s, mistral 7b Q6 = 85 tok/s, gemma3 27b = 39.9 tok/s (Source: insiderllm.com reference data)
- AMD RX 7900 XT: llama2 7b Q4 = 116 tok/s sustained = 97 tok/s (Source: insiderllm.com reference data)
- NVIDIA cards average ~0.13 tok/s per GB/s of bandwidth for Llama 3 8B Q4 (Source: insiderllm.com reference data)
- AMD ROCm cards achieve ~0.06 tok/s per GB/s due to less optimized kernels (Source: insiderllm.com reference data)

---

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
- DDR4: $50-80 per 16GB used, $100-170 per 32GB used (Source: insiderllm.com reference data)
- DDR5: $200-300 per 32GB used, $320-450 per 64GB used (Source: insiderllm.com reference data)

**Cloud/API Pricing:**
- DeepSeek V3.2 API: ~$0.27–0.55 per million tokens (Source: dev.to)

**MSRP for New Cards (Preliminary):**
- NVIDIA RTX 5070: $549 MSRP (Source: insiderllm.com reference data)
- AMD RX 9070: $549 MSRP (Source: insiderllm.com reference data)
- AMD RX 9070 XT: $599 MSRP (Source: insiderllm.com reference data)

---

## Benchmark Data

**HumanEval Scores:**
| Model | HumanEval % | FIM Support | VRAM Q4 | Best For |
|-------|-------------|-------------|---------|----------|
| Qwen 2.5 Coder 7B | 88.4% | Yes | ~5 GB | Autocomplete/FIM |
| Qwen 3.5 9B | N/A (not coding-specific) | No | 6.6 GB | Chat coding, multimodal |
| DeepSeek Coder V2 Lite | 81.1% | Yes | ~5 GB | Reasoning-heavy tasks |
| CodeLlama 7B | ~30% | Yes | ~4.5 GB | Legacy (skip) |
| Qwen 2.5 Coder 14B | ~89% | Yes | ~9 GB | Best at 16GB tier |
| DeepSeek Coder 33B (Q3) | 70% | Yes | ~16 GB | Squeezed into 16GB |
| Qwen 2.5 Coder 32B | 92.7% | Yes | ~20 GB | Best FIM/autocomplete at 24GB |
| DeepSeek Coder 33B (Q3) | 78%* | Yes | ~20 GB | With CodeFuse fine-tuning |

*CodeLlama 34B: 53.7% HumanEval (Source: insiderllm.com)

**SWE-bench Verified Scores (February 2026):**
| Model | Score | License |
|-------|-------|---------|
| Claude Opus 4.6 (Proprietary) | 80.9% | Proprietary |
| GPT-5.2 (Proprietary) | 80.0% | Proprietary |
| Kimi K2.5 | 76.8% | MIT* |
| GLM-4.7 | 73.8% | MIT |
| DeepSeek V3.2 | 73.1% | MIT |
| Qwen3-Coder-Next | 70.6% | Apache 2.0 |

*MIT with commercial restrictions (Source: dev.to)

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
| Many proprietary models collapse here | N/A |

**LiveCodeBench v6:**
| Model | Score |
|-------|-------|
| Kimi K2.5 | 85.0% |
| GLM-4.7 | 84.9% |
| DeepSeek V3.2 | N/A (not reported) |

**Terminal-Bench 2.0:**
| Model | Score |
|-------|-------|
| Kimi K2.5 | 40.45% |
| GLM-4.7 | 41.0% |

**GPU Token Generation Benchmarks (from reference data):**
| GPU | llama3 8b Q4 | llama2 7b Q4 |
|-----|--------------|--------------|
| RTX 3060 12GB | 51 tok/s | 76 tok/s |
| RTX 3070 | 71 tok/s | N/A |
| RTX 3090 | 112 tok/s | N/A |
| AMD RX 7900 XT | 116 tok/s sustained (llama2 7b Q4) |

---

## Key Specs

**GPU Specifications:**

| GPU | VRAM | Memory Bandwidth | TDP | Architecture | Best For |
|-----|------|------------------|-----|--------------|----------|
| GTX 1660 Super | 6GB GDDR6 | 336 GB/s | 125W | Turing | Experimentation, not daily use |
| RTX 2060 12GB | 12GB GDDR6 | 336 GB/s | 185W | Turing | Budget option if found cheap |
| RTX 3060 12GB | 12GB GDDR6 | 360 GB/s | 170W | Ampere | Entry-level workhorse, best value under $200 used |
| RTX 3070 | 8GB GDDR6 | 448 GB/s | 220W | Ampere | Fast but VRAM limited |
| RTX 3070 Ti | 8GB GDDR6X | 608 GB/s | 290W | Ampere | Better bandwidth, still VRAM limited |
| RTX 3080 10GB | 10GB GDDR6X | 760 GB/s | 320W | Ampere | 2x bandwidth of 3060 |
| RTX 3080 12GB | 12GB GDDR6X | 912 GB/s | 350W | Ampere | Sleeper pick, excellent value |
| RTX 3090 | 24GB GDDR6X | 936 GB/s | 350W | Ampere | Budget local AI king |
| RTX 4060 | 8GB GDDR6 | 272 GB/s | 115W | Ada Lovelace | Avoid for local AI (lowest bandwidth) |
| RTX 4060 Ti 8GB | 8GB GDDR6 | 288 GB/s | 160W | Ada Lovelace | Bandwidth-starved despite more compute |
| RTX 4060 Ti 16GB | 16GB GDDR6 | 288 GB/s | 165W | Ada Lovelace | Trade VRAM for speed vs 3080 12GB |
| AMD RX 7600 | 8GB GDDR6 | 288 GB/s | 165W | RDNA 3 | Only if you want AMD and accept tradeoffs |
| AMD RX 7700 XT | 12GB GDDR6 | 432 GB/s | 245W | RDNA 3 | Competitive if ROCm works |
| AMD RX 7800 XT | 16GB GDDR6 | 624 GB/s | 263W | RDNA 3 | AMD's best AI value if ROCm works |
| AMD RX 7900 GRE | 16GB GDDR6 | 576 GB/s | 260W | RDNA 3 | Competitive with RTX 3080 12GB in bandwidth |
| AMD RX 7900 XT | 20GB GDDR6 | 800 GB/s | 315W | RDNA 3 | AMD's best current option if ROCm works |

**Memory → Model Size Guide:**
- 6GB VRAM: 7B Q4 max (tight)
- 8GB VRAM: 8B Q4 comfortable, 14B Q2 possible
- 10GB VRAM: 8B Q6, 14B Q