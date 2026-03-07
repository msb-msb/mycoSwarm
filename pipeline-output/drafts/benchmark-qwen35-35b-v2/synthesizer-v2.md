# Research Bundle: Best Budget GPU for Local AI in 2026 (Updated)

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
- DDR3 (up to 1866MHz, 14.9 GB/s dual-channel): "Fine for models that fit entirely in VRAM. Painful for offloaded layers" — Reference
- DDR4 (up to 3600MHz, 28.8 GB/s dual-channel at 3200MHz): "Sweet spot for budget builds — Optiplex 5060/7060 are cheap and take DDR4" — Reference
- DDR5 (up to 6400MHz, 51.2 GB/s dual-channel at 6000MHz): "Best for CPU-heavy workloads or large models that need lots of offloaded layers" — Reference
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
| RTX 5070 (MSRP $549) | 12GB GDDR7 | 672 GB/s | 250W | Blackwell | Reference data |

**AMD Cards:**
| GPU | VRAM | Memory Type | Bandwidth | TDP | Architecture | Source |
|-----|------|-------------|-----------|-----|--------------|--------|
| RX 7600 | 8GB GDDR6 | 288 GB/s | 165W | RDNA 3 | InsiderLLM database |
| RX 7700 XT | 12GB GDDR6 | 432 GB/s | 245W | RDNA 3 | InsiderLLM database |
| RX 7800 XT | 16GB GDDR6 | 624 GB/s | 263W | RDNA 3 | InsiderLLM database |
| RX 7900 GRE | 16GB GDDR6 | 576 GB/s | 260W | RDNA 3 | InsiderLLM database |
| RX 7900 XT | 20GB GDDR6 | 800 GB/s | 315W | RDNA 3 | InsiderLLM database |
| RX 9070 (MSRP $549) | 16GB GDDR6 | 608 GB/s | 220W | RDNA 4 | Reference data |
| RX 9070 XT (MSRP $599) | 16GB GDDR6 | 608 GB/s | 250W | RDNA 4 | Reference data |

**Memory Type Differences:**
- GDDR6 vs GDDR6X: GDDR6X provides ~2x bandwidth of standard GDDR6
- GDDR7: New generation with higher bandwidth efficiency (RTX 5070 has 672 GB/s on 192-bit bus)

---

## ## Analysis & Insights

### **Best Budget GPU Under $300 for Local AI**

**Top Recommendation: RTX 3070 Ti (Used ~$190)**
- **Why:** Best bandwidth/price ratio in the under-$300 category
- **Pros:** 608 GB/s bandwidth, 8GB VRAM fits 8B Q4 models comfortably
- **Cons:** 290W TDP is high for power efficiency; 8GB limits larger models
- **Best for:** Users who prioritize speed over model size and have good cooling

**Second Choice: RTX 3060 12GB (Used ~$275)**
- **Why:** Best VRAM capacity in budget category
- **Pros:** 12GB VRAM runs 13B Q4 models; widely available used
- **Cons:** Lower bandwidth (360 GB/s) means slower token generation
- **Best for:** Users who need to run larger models and can accept slower speeds

**Third Choice: RTX 3080 12GB (Used ~$305)**
- **Why:** "Sleeper pick" with excellent performance/price ratio
- **Pros:** 912 GB/s bandwidth, 12GB VRAM runs 14B Q4 models at high speed
- **Cons:** Higher power consumption (350W TDP); harder to find used
- **Best for:** Users who want premium performance and can handle higher power draw

### **AMD vs NVIDIA for Budget AI**

**When to Choose AMD:**
- "For the vast majority of local AI enthusiasts, and especially for budget-conscious beginners, AMD is now a powerful and easy-to-use option" — Tech Tactician
- Best value when ROCm compatibility is confirmed for your specific use case
- RX 7800 XT (16GB, $465) offers competitive performance to RTX 3080 12GB at lower price
- "Most popular projects: Ollama, LM Studio, KoboldCpp,