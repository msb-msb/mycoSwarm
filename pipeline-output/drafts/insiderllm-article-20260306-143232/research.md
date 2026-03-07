# Research Bundle: Best Budget GPU for Local AI in 2026

---

## Key Facts & Data Points

**RTX 30-Series Dominance:** The RTX 30-series is becoming the best GPU value in 2026, with used market prices finally normalizing after crypto and pandemic-era inflation. Ampere cards offer genuine gaming and AI performance without AI-inflated pricing. [Source: https://www.xda-developers.com/the-rtx-30-series-is-quietly-becoming-the-best-gpu-value-in-2026/]

**VRAM Requirements:** 
- 6GB VRAM limits to 7B quantized models
- 8GB VRAM fits 8B Q4 comfortably, 14B Q2 possible
- 10GB VRAM fits 8B Q6, 14B Q3 possible
- 12GB VRAM fits 14B Q4, 8B Q8 or FP16
- 16GB VRAM fits 30B Q3, 14B Q6
- 20GB VRAM fits 30B Q4, some 70B Q2
- 24GB VRAM fits 30B Q5, 70B Q2-Q3 [Source: Reference data]

**Bandwidth Matters:** For models that fit entirely in VRAM, memory bandwidth is the primary predictor of tok/s. Double the bandwidth ≈ double the tok/s for memory-bound inference. DDR4-3200 (25.6 GB/s) vs GDDR6X (936 GB/s) = ~37x slower per offloaded layer. [Source: Reference data]

**NVIDIA CUDA Advantage:** NVIDIA GPUs run CUDA, which most AI tools are built around. AMD uses ROCm, which has improved, but is still behind in compatibility and ease of use. NVIDIA cards average ~0.13 tok/s per GB/s of bandwidth for Llama 3 8B Q4. AMD ROCm cards achieve ~0.06 tok/s per GB/s due to less optimized kernels. [Source: Reference data]

**RTX 4060 Series Warning:** The RTX 4060 and 4060 Ti have the lowest bandwidth of any current card (272-288 GB/s). The 128-bit bus makes them worse than an RTX 3060 12GB for AI despite being newer. [Source: Reference data]

---

## Price Data

**RTX 30-Series Used Market Prices (2026-03-02):**
- GTX 1660 Super: $90–$120 (typical ~$105) [Source: Reference data]
- RTX 2060 12GB: $140–$180 (typical ~$160) [Source: Reference data]
- RTX 3060 12GB: $170–$380 (typical ~$275) [Source: Reference data]
- RTX 3070: $210–$300 (typical ~$255) [Source: Reference data]
- RTX 3070 Ti: $100–$280 (typical ~$190) [Source: Reference data]
- RTX 3080 10GB: $325–$400 (typical ~$365) [Source: Reference data]
- RTX 3080 12GB: $230–$380 (typical ~$305) [Source: Reference data]
- RTX 3090: $950–$1125 (typical ~$1040) [Source: Reference data]

**RTX 40-Series Used Market Prices:**
- RTX 4060: $230–$310 (typical ~$270) [Source: Reference data]
- RTX 4060 Ti 8GB: $240–$300 (typical ~$270) [Source: Reference data]
- RTX 4060 Ti 16GB: $380–$480 (typical ~$430) [Source: Reference data]

**AMD Radeon Used Market Prices:**
- RX 7600: $170–$225 (typical ~$200) [Source: Reference data]
- RX 7700 XT: $300–$350 (typical ~$325) [Source: Reference data]
- RX 7800 XT: $380–$550 (typical ~$465) [Source: Reference data]
- RX 7900 GRE: $400–$550 (typical ~$475) [Source: Reference data]
- RX 7900 XT: $500–$700 (typical ~$600) [Source: Reference data]

**TechRadar Deal Prices (April 2022 - may be outdated):**
- RTX 3060: $389 at Newegg / £379 at OCUK [Source: https://www.techradar.com/deals/the-rtx-3060-is-finally-affordable-with-crashing-gpu-prices-this-week]
- RTX 3070: $713 at Newegg / £559 at OCUK [Source: https://www.techradar.com/deals/the-rtx-3060-is-finally-affordable-with-crashing-gpu-prices-this-week]

**XDA Developers Used Market Summary (2026):**
- RTX 3070: $250–$300 [Source: https://www.xda-developers.com/the-rtx-30-series-is-quietly-becoming-the-best-gpu-value-in-2026/]
- RTX 3080: $350–$400 [Source: https://www.xda-developers.com/the-rtx-30-series-is-quietly-becoming-the-best-gpu-value-in-2026/]

---

## Benchmark Data

**RTX 3060 12GB Benchmarks:**
- llama3 8b Q4: 51 tok/s
- llama2 7b Q4: 76 tok/s
- llama2 13b Q4: 35 tok/s
- qwen35 9b think off: 47.1 tok/s
- qwen35 9b think on: 46.6 tok/s
- deepseek r1 14b think off: 35.6 tok/s
- deepseek r1 14b think on: 35.1 tok/s [Source: Reference data]

**RTX 3070 Benchmarks:**
- llama3 8b Q4: 71 tok/s [Source: Reference data]

**RTX 3080 10GB Benchmarks:**
- llama3 8b Q4: 106 tok/s [Source: Reference data]

**RTX 3080 12GB Benchmarks:**
- llama3 8b Q4: 107 tok/s [Source: Reference data]

**RTX 3090 Benchmarks:**
- llama3 8b Q4: 112 tok/s
- llama3 8b F16: 47 tok/s
- llama3 70b Q4: 16 tok/s
- mistral 7b Q6: 85 tok/s
- gemma3 27b: 39.9 tok/s [Source: Reference data]

**RTX 4060 Benchmarks:**
- llama3 8b Q4: 38 tok/s [Source: Reference data]

**RTX 4060 Ti 16GB Benchmarks:**
- llama3 8b Q4: 48 tok/s
- llama2 7b Q4: 64 tok/s [Source: Reference data]

**AMD RX 7900 XT Benchmarks:**
- llama2 7b Q4: 116 tok/s (sustained: 97 tok/s) [Source: Reference data]

**AMD RX 7800 XT Benchmarks:**
- llama3 8b Q4: 39 tok/s
- llama2 7b Q4: 96 tok/s [Source: Reference data]

---

## Key Specs

**NVIDIA RTX 30-Series:**
- GTX 1660 Super: 6GB GDDR6, 336 GB/s, TDP: 125W, Arch: Turing, No tensor cores [Source: Reference data]
- RTX 2060 12GB: 12GB GDDR6, 336 GB/s, TDP: 185W, Arch: Turing, Tensor cores present [Source: Reference data]
- RTX 3060 12GB: 12GB GDDR6, 360 GB/s, TDP: 170W, Arch: Ampere [Source: Reference data]
- RTX 3070: 8GB GDDR6, 448 GB/s, TDP: 220W, Arch: Ampere [Source: Reference data]
- RTX 3070 Ti: 8GB GDDR6X, 608 GB/s, TDP: 290W, Arch: Ampere [Source: Reference data]
- RTX 3080 10GB: 10GB GDDR6X, 760 GB/s, TDP: 320W, Arch: Ampere [Source: Reference data]
- RTX 3080 12GB: 12GB GDDR6X, 912 GB/s, TDP: 350W, Arch: Ampere [Source: Reference data]
- RTX 3090: 24GB GDDR6X, 936 GB/s, TDP: 350W, Arch: Ampere [Source: Reference data]

**NVIDIA RTX 40-Series:**
- RTX 4060: 8GB GDDR6, 272 GB/s, TDP: 115W, Arch: Ada Lovelace, 128-bit bus [Source: Reference data]
- RTX 4060 Ti 8GB: 8GB GDDR6, 288 GB/s, TDP: 160W, Arch: Ada Lovelace, 128-bit bus [Source: Reference data]
- RTX 4060 Ti 16GB: 16GB GDDR6, 288 GB/s, TDP: 165W, Arch: Ada Lovelace, 128-bit bus [Source: Reference data]

**AMD Radeon RX 7000 Series:**
- RX 7600: 8GB GDDR6, 288 GB/s, TDP: 165W, Arch: RDNA 3 [Source: Reference data]
- RX 7700 XT: 12GB GDDR6, 432 GB/s, TDP: 245W, Arch: RDNA 3 [Source: Reference data]
- RX 7800 XT: 16GB GDDR6, 624 GB/s, TDP: 263W, Arch: RDNA 3 [Source: Reference data]
- RX 7900 GRE: 16GB GDDR6, 576 GB/s, TDP: 260W, Arch: RDNA 3 [Source: Reference data]
- RX 7900 XT: 20GB GDDR6, 800 GB/s, TDP: 315W, Arch: RDNA 3 [Source: Reference data]

**Preliminary RTX 50-Series (Blackwell):**
- RTX 5060: 8GB GDDR7, None GB/s bandwidth, TDP: NoneW (PRELIMINARY) [Source: Reference data]
- RTX 5060 Ti: 16GB GDDR7, None GB/s bandwidth (PRELIMINARY) [Source: Reference data]
- RTX 5070: 12GB GDDR7, 672 GB/s, TDP: 250W, Arch: Blackwell, MSRP: $549 [Source: Reference data]

**Preliminary AMD RX 9000 Series (RDNA 4):**
- RX 9070: 16GB GDDR6, 608 GB/s, TDP: 220W, Arch: RDNA 4, MSRP: $549 (PRELIMINARY) [Source: Reference data]
- RX 9070 XT: 16GB GDDR6, 608 GB/s, TDP: 250W, Arch: RDNA 4, MSRP: $599 (PRELIMINARY) [Source: Reference data]

---

## Expert Opinions & Analysis

**XDA Developers Analysis:**
"RTX 30-series cards are the biggest winners here. RTX 3070s and RTX 3080s are now landing at fractions of their launch price. That alone makes them incredibly compelling." [Source: https://www.xda-developers.com/the-rtx-30-series-is-quietly-becoming-the-best-gpu-value-in-2026/]

"Ampere cards are continuing to deliver what most gamers actually need, which is dependable performance at prices that make sense. In a hardware landscape defined by excess, the RTX 30-series stands out for being sensible, and right now, that makes it one of the smartest GPU buys you can make." [Source: https://www.xda-developers.com/the-rtx-30-series-is-quietly-becoming-the-best-gpu-value-in-2026/]

"Even something like an RTX 3060 Ti still delivers rock-solid raster performance in modern games. Strip away the marketing buzzwords and AI features for a second, and you're left with GPUs that were built during a performance-first generation." [Source: https://www.xda-developers.com/the-rtx-30-series-is-quietly-becoming-the-best-gpu-value-in-2026/]

**TechRadar Deal Analysis:**
"The RTX 3060 is finally affordable with crashing GPU prices. Even though both these listings are still above MSRP, they collectively represent what's quite possibly the best opportunity in the past two years to bag a decent 1080p card without breaking the bank." [Source: https://www.techradar.com/deals/the-rtx-3060-is-finally-affordable-with-crashing-gpu-prices-this-week]

**Tech Tactician Analysis:**
"VRAM – For training AI models, fine-tuning and doing any calculations on large batches of data at the same time efficiently, you need as much VRAM as you can get. In general, you don't want less than 12GB of video memory, and the 24GB models would be ideal if only you can afford to get them." [Source: https://techtactician.com/best-gpu-for-local-ai-software-this-year/]

"AMD's cards look more tempting with more VRAM and stronger raster performance. Cards like the RX 6800 XT or RX 6900 XT do come close to RTX 3080-level raster performance, even when the NVIDIA card carries 6GB less VRAM. Still, the story doesn't end there, and that's all because of NVIDIA's Deep Learning Super Sampling." [Source: https://www.xda-developers.com/the-rtx-30-series-is-quietly-becoming-the-best-gpu-value-in-2026/]

**Northflank Analysis:**
"The RTX 4060 Ti 16GB works well with all the mainstream AI tools that you can use today, offering power efficiency and small form factor." [Source: https://northflank.com/blog/best-gpu-for-ai]

"VRAM for 70B Models: Why 16GB GPU Is the Minimum in 2026 — For budget buyers, the RTX 4060 Ti 16GB remains the most accessible 16GB option on the NVIDIA side. AMD's RX 7800 XT with 16GB competes well on price." [Source: https://www.sitepoint.com/vram-requirements-70b-models-16gb-gpu-minimum-2026/]

---

## Gaps

**Missing Information:**
1. **ROCm Version Compatibility:** Specific ROCm versions that support which AMD cards and models is not detailed in the sources found. Users need to verify ROCm compatibility for their specific use case before buying AMD for AI. [Source: Reference data]

2. **RTX 50-Series Actual Pricing:** The RTX 5060, 5060 Ti, and other Blackwell cards are marked as "PRELIMINARY" with no confirmed pricing or bandwidth specifications. Availability and actual pricing TBD for the RTX 5070 [Source: Reference data]

3. **Real-World Thermal Performance:** While XDA mentions that dual-slot blower cooler cards run especially hot on the RTX 3090, detailed thermal throttling data and cooling solutions are not fully covered.

4. **Multi-GPU SLI/CrossFire for AI:** The Tech Tactician article explicitly states it won't cover running multiple graphics cards on one system in an SLI configuration, noting this is "a topic for a whole new article."

5. **Specific Model Compatibility Lists:** While general ROCm compatibility issues are mentioned, specific models and frameworks that work or don't work with AMD cards are not listed.

6. **DDR Memory Bandwidth Data