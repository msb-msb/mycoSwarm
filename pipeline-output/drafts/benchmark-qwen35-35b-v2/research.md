# Research Bundle: Best Budget GPU for Local AI in 2026

---

## ## Key Facts & Data Points

**VRAM is the #1 Priority:**
- "For smooth and efficient LLM inference, having enough VRAM is essential" — Tech Tactician (best-amd-cards-for-local-ai-and-llms-this-year/)
- "In an ideal situation, the loaded model should fit entirely within the GPU's memory to avoid unwanted slowdowns caused by data being offloaded to system RAM" — Tech Tactician
- Reference: 6GB = 7B Q4 max; 8GB = 8B Q4 comfortable; 12GB = 14B Q4; 16GB = 30B Q3; 24GB = 70B Q2-Q3

**NVIDIA Cards (CUDA Native Support):**
- RTX 3060 12GB: "The entry-level workhorse. 12GB fits 13B Q4 models" — InsiderLLM database
- RTX 3080 12GB: "The sleeper pick. 12GB VRAM + 912 GB/s bandwidth = 3060's model capacity at 2.5x the speed" — InsiderLLM database
- RTX 4060 Ti 16GB: "16GB VRAM is great for model capacity but 128-bit bus caps bandwidth at 288 GB/s" — InsiderLLM database

**AMD Cards (ROCm Support Improving):**
- "For the vast majority of local AI enthusiasts, and especially for budget-conscious beginners, AMD is now a powerful and easy-to-use option" — Tech Tactician (best-amd-cards-for-local-ai-and-llms-this-year/)
- Most popular projects: Ollama, LM Studio, KoboldCpp, OobaBooga WebUI "offer out-of-the-box support for AMD graphics cards" — Tech Tactician
- ROCm compatibility varies by model and software stack — always test before committing

**Memory Bandwidth is Critical:**
- "For models that fit entirely in VRAM, memory bandwidth is the primary predictor of tok/s. Double the bandwidth ≈ double the tok/s for memory-bound inference" — Reference data
- NVIDIA cards: ~0.13 tok/s per GB/s of bandwidth (Llama 3 8B Q4) — Reference
- AMD ROCm cards: ~0.06 tok/s per GB/s due to less optimized kernels — Reference

**System RAM Offloading:**
- DDR3 (14.9 GB/s dual-channel): "Fine for models that fit entirely in VRAM. Painful for offloaded layers" — Reference
- DDR4 (28.8 GB/s dual-channel at 3200MHz): "Sweet spot for budget builds — Optiplex 5060/7060 are cheap and take DDR4" — Reference
- DDR5 (51.2 GB/s dual-channel at 6000MHz): "Best for CPU-heavy workloads or large models that need lots of offloaded layers" — Reference
- Each offloaded layer runs at system RAM bandwidth instead of VRAM bandwidth — Reference

---

## ## Price Data

**NVIDIA Used Prices (eBay sold auctions, 2026-03-02):**
| GPU | VRAM | Used Price Range | Typical Price | Source |
|-----|------|------------------|---------------|--------|
| GTX 1660 Super | 6GB GDDR6 | $90–$120 | ~$105 | InsiderLLM database |
| RTX 2060 12GB | 12GB GDDR6 | $140–$180 | ~$160 | InsiderLLM database |
| RTX 3060 12GB | 12GB GDDR6 | $170–$380 | ~$275 | InsiderLLM database |
| RTX 3070 | 8GB GDDR6 | $210–$300 | ~$255 | InsiderLLM database |
| RTX 3070 Ti | 8GB GDDR6X | $100–$280 | ~$190 | InsiderLLM database |
| RTX 3080 10GB | 10GB GDDR6X | $325–$400 | ~$365 | InsiderLLM database |
| RTX 3080 12GB | 12GB GDDR6X | $230–$380 | ~$305 | InsiderLLM database |
| RTX 3090 | 24GB GDDR6X | $950–$1125 | ~$1040 | InsiderLLM database |
| RTX 4060 | 8GB GDDR6 | $230–$310 | ~$270 | InsiderLLM database |
| RTX 4060 Ti 8GB | 8GB GDDR6 | $240–$300 | ~$270 | InsiderLLM database |
| RTX 4060 Ti 16GB | 16GB GDDR6 | $380–$480 | ~$430 | InsiderLLM database |

**AMD Used Prices (eBay sold auctions, 2026-03-02):**
| GPU | VRAM | Used Price Range | Typical Price | Source |
|-----|------|------------------|---------------|--------|
| RX 7600 | 8GB GDDR6 | $170–$225 | ~$200 | InsiderLLM database |
| RX 7700 XT | 12GB GDDR6 | $300–$350 | ~$325 | InsiderLLM database |
| RX 7800 XT | 16GB GDDR6 | $380–$550 | ~$465 | InsiderLLM database |
| RX 7900 GRE | 16GB GDDR6 | $400–$550 | ~$475 | InsiderLLM database |
| RX 7900 XT | 20GB GDDR6 | $500–$700 | ~$600 | InsiderLLM database |

**MSRP Prices (New Cards):**
- RTX 5070: $549 (12GB GDDR7) — Reference data (PRELIMINARY)
- RX 9070: $549 (16GB GDDR6) — Reference data (PRELIMINARY)
- RX 9070 XT: $599 (16GB GDDR6) — Reference data (PRELIMINARY)

**System RAM Used Prices:**
- DDR3 ($15–$30 per 16GB) — Reference
- DDR4 ($50–$80 per 16GB; $100–$170 per 32GB) — Reference
- DDR5 ($200–$300 per 32GB; $320–$450 per 64GB) — Reference

---

## ## Benchmark Data

**NVIDIA RTX 3060 12GB Benchmarks (Llama 3 8B Q4):**
- llama3 8b Q4: 51 tok/s — InsiderLLM database
- llama2 7b Q4: 76 tok/s — InsiderLLM database
- llama2 13b Q4: 35 tok/s — InsiderLLM database
- qwen35 9b think off: 47.1 tok/s — InsiderLLM database
- qwen35 9b think on: 46.6 tok/s — InsiderLLM database
- deepseek r1 14b think off: 35.6 tok/s — InsiderLLM database
- deepseek r1 14b think on: 35.1 tok/s — InsiderLLM database

**NVIDIA RTX 3070 Benchmarks:**
- llama3 8b Q4: 71 tok/s — InsiderLLM database

**NVIDIA RTX 3070 Ti (8GB GDDR6X):**
- Faster bandwidth (608 GB/s vs 336 GB/s) but still 8GB VRAM limited
- Power hungry for what you get (290W TDP)

**NVIDIA RTX 3080 10GB Benchmarks:**
- llama3 8b Q4: 106 tok/s — InsiderLLM database

**NVIDIA RTX 3080 12GB Benchmarks:**
- llama3 8b Q4: 107 tok/s — InsiderLLM database
- "2x the bandwidth of 3060" — Reference data
- Speed upgrade is dramatic if models fit in VRAM

**NVIDIA RTX 3090 Benchmarks:**
- llama3 8b Q4: 112 tok/s — InsiderLLM database
- llama3 8b F16: 47 tok/s — InsiderLLM database
- llama3 70b Q4: 16 tok/s — InsiderLLM database
- mistral 7b Q6: 85 tok/s — InsiderLLM database
- gemma3 27b: 39.9 tok/s — InsiderLLM database

**NVIDIA RTX 4060 Benchmarks (Avoid for AI):**
- llama3 8b Q4: 38 tok/s — InsiderLLM database
- "Lowest bandwidth of any current card" — Reference data

**NVIDIA RTX 4060 Ti 8GB Benchmarks:**
- llama3 8b Q4: 48 tok/s — InsiderLLM database
- llama2 7b Q4: 64 tok/s — InsiderLLM database
- "Same 128-bit bus problem as 4060" — Reference data

**NVIDIA RTX 4060 Ti 16GB Benchmarks:**
- llama3 8b Q4: 48 tok/s — InsiderLLM database
- llama2 7b Q4: 64 tok/s — InsiderLLM database
- "Runs bigger models than 3060 but slower token generation" — Reference data

**AMD RX 7600 Benchmarks:**
- llama3 8b Q4: 39 tok/s (ROCm) — InsiderLLM database
- llama2 7b Q4: 96 tok/s — InsiderLLM database

**AMD RX 7800 XT Benchmarks:**
- llama3 8b Q4: 39 tok/s (ROCm) — InsiderLLM database
- llama2 7b Q4: 96 tok/s — InsiderLLM database

**AMD RX 7900 XT Benchmarks:**
- llama2 7b Q4: 116 tok/s — InsiderLLM database
- llama2 7b Q4 sustained: 97 tok/s — InsiderLLM database

**Performance Comparison Insights:**
- "NVIDIA cards average ~0.13 tok/s per GB/s of bandwidth for Llama 3 8B Q4" — Reference
- "AMD ROCm cards achieve ~0.06 tok/s per GB/s due to less optimized kernels" — Reference
- This means AMD needs ~2x more bandwidth to match NVIDIA performance

**Stable Diffusion Performance:**
- "There exist quite a few useful AI benchmark resources online, for example ones showing local Stable Diffusion performance of different GPUs" — Tech Tactician (best-budget-gpus-for-local-ai-workflows/)

---

## ## Key Specs

**NVIDIA Cards:**
| GPU | VRAM | Memory Type | Bandwidth | TDP | Architecture | Source |
|-----|------|-------------|-----------|-----|--------------|--------|
| GTX 1660 Super | 6GB GDDR6 | 336 GB/s | 125W | Turing | InsiderLLM database |
| RTX 2060 12GB | 12GB GDDR6 | 336 GB/s | 185W | Turing | InsiderLLM database |
| RTX 3060 12GB | 12GB GDDR6 | 360 GB/s | 170W | Ampere | InsiderLLM database |
| RTX 3070 | 8GB GDDR6 | 448 GB/s | 220W | Ampere | InsiderLLM database |
| RTX 3070 Ti | 8GB GDDR6X | 608 GB/s | 290W | Ampere | InsiderLLM database |
| RTX 3080 10GB | 10GB GDDR6X | 760 GB/s | 320W | Ampere | InsiderLLM database |
| RTX 3080 12GB | 12GB GDDR6X | 912 GB/s | 350W | Ampere | InsiderLLM database |
| RTX 3090 | 24GB GDDR6X | 936 GB/s | 350W | Ampere | InsiderLLM database |
| RTX 4060 | 8GB GDDR6 | 272 GB/s | 115W | Ada Lovelace | InsiderLLM database |
| RTX 4060 Ti 8GB | 8GB GDDR6 | 288 GB/s | 160W | Ada Lovelace | InsiderLLM database |
| RTX 4060 Ti 16GB | 16GB GDDR6 | 288 GB/s | 165W | Ada Lovelace | InsiderLLM database |
| RTX 5060 (PRELIMINARY) | 8GB GDDR7 | None GB/s | NoneW | Blackwell | Reference data |
| RTX 5060 Ti (PRELIMINARY) | 16GB GDDR7 | None GB/s | NoneW | Blackwell | Reference data |
| RTX 5070 (MSRP $549) | 12GB GDDR7 | 672 GB/s | 250W | Blackwell | Reference data |

**AMD Cards:**
| GPU | VRAM | Memory Type | Bandwidth | TDP | Architecture | Source |
|-----|------|-------------|-----------|-----|--------------|--------|
| RX 7600 | 8GB GDDR6 | 288 GB/s | 165W | RDNA 3 | InsiderLLM database |
| RX 7700 XT | 12GB GDDR6 | 432 GB/s | 245W | RDNA 3 | InsiderLLM database |
| RX 7800 XT | 16GB GDDR6 | 624 GB/s | 263W | RDNA 3 | InsiderLLM database |
| RX 7900 GRE | 16GB GDDR6 | 576 GB/s | 260W | RDNA 3 | InsiderLLM database |
| RX 7900 XT | 20GB GDDR6 | 800 GB/s | 315W | RDNA 3 | InsiderLLM database |

**VRAM → Model Size Guide:**
- 6GB: 7B Q4 max, tight — Reference data
- 8GB: 8B Q4 comfortable, 14B Q2 possible — Reference data
- 10GB: 8B Q6, 14B Q3 possible — Reference data
- 12GB: 14B Q4, 8B Q8 or FP16 — Reference data
- 16GB: 30B Q3, 14B Q6 — Reference data
- 20GB: 30B Q4, some 70B Q2 — Reference data
- 24GB: 30B Q5, 70B Q2-Q3 — Reference data

**System RAM Offloading Performance:**
| RAM Type | Speed | Bandwidth (dual-channel) | Relative Speed vs GDDR6X | Source |
|----------|-------|--------------------------|---------------------------|--------|
| DDR3 | up to 1866MHz | 14.9 GB/s | ~2% of GDDR6X (37x slower) | Reference data |
| DDR4 | up to 3600MHz | 28.8 GB/s | ~5% of GDDR6X | Reference data |
| DDR5 | up to 6400MHz | 51.2 GB/s | ~9% of GDDR6X | Reference data |

---

## ## Expert Opinions & Analysis

**Tech Tactician (Tom Smigla) - Top Budget Picks:**

**Under $1000 Options:**
1. **NVIDIA RTX 4060 Ti 16GB** — "Alongside the RTX 4070, one of the better budget picks from the 4th gen" — Tech Tactician (best-budget-gpus-for-local-ai-workflows/)
   - "It does feature 4 bonus GB of VRAM on board" — Tech Tactician
   - Tradeoff: "2x lower memory bandwidth (288 GB/s) than the NVIDIA RTX 4070 (504 GB/s)" — Tech Tactician

2. **NVIDIA RTX 4070 12GB** — Coming up next in list, higher bandwidth but less VRAM

3. **AMD RX 7800 XT 16GB** — "Best AMD GPU for local AI" — Tech Tactician (best-amd-cards-for-local-ai-and-llms-this-year/)
   - "In terms of performance, it's closest NVIDIA counterpart seems to be the