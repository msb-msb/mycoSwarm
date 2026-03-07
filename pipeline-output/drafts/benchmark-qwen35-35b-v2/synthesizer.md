## Price Data
*Data sourced from eBay sold auctions (real hammer prices) as of 2026-03-02. New MSRP noted where available.*

**RTX 3090 (24GB)** — *CRITICAL GAP CHECK: DATA FOUND*
- **Used:** $950–$1,125 (Typical: ~$1,040)
- **Status:** High demand "budget local AI king" but significantly over budget for <$500 builds.

**RTX 3060 (12GB)** — *CRITICAL GAP CHECK: DATA FOUND*
- **Used:** $170–$380 (Typical: ~$275)
- **New:** Not explicitly listed in price range, but implied availability via MSRP context.
- **Note:** Entry-level workhorse; best value under $200 used if found at lower end of range.

**Other NVIDIA Used Prices:**
- GTX 1660 Super (6GB): $90–$120 (~$105)
- RTX 2060 12GB: $140–$180 (~$160)
- RTX 3070 (8GB): $210–$300 (~$255)
- RTX 3070 Ti (8GB): $100–$280 (~$190) — *Price anomaly noted: lower typical price than non-Ti variant.*
- RTX 3080 10GB: $325–$400 (~$365)
- RTX 3080 12GB: $230–$380 (~$305) — *Sleeper pick, high demand.*
- RTX 4060 (8GB): $230–$310 (~$270)
- RTX 4060 Ti 8GB: $240–$300 (~$270)
- RTX 4060 Ti 16GB: $380–$480 (~$430)

**AMD Used Prices:**
- RX 7600 (8GB): $170–$225 (~$200)
- RX 7700 XT (12GB): $300–$350 (~$325)
- RX 7800 XT (16GB): $380–$550 (~$465)
- RX 7900 GRE (16GB): $400–$550 (~$475)
- RX 7900 XT (20GB): $500–$700 (~$600)

**New MSRP (Preliminary):**
- RTX 5070: $549 (12GB GDDR7)
- RX 9070: $549 (16GB GDDR6)
- RX 9070 XT: $599 (16GB GDDR6)

**System RAM Used Prices:**
- DDR3 (16GB): $15–$30
- DDR4 (16GB): $50–$80; (32GB): $100–$170
- DDR5 (32GB): $200–$300; (64GB): $320–$450

## Benchmark Data
*Metrics: Tokens per second (tok/s). NVIDIA data based on CUDA; AMD data based on ROCm.*

**RTX 3090 (24GB)** — *CRITICAL GAP CHECK: DATA FOUND*
- llama3 8b Q4: 112 tok/s
- llama3 8b F16: 47 tok/s
- llama3 70b Q4: 16 tok/s
- mistral 7b Q6: 85 tok/s
- gemma3 27b: 39.9 tok/s

**RTX 3060 (12GB)** — *CRITICAL GAP CHECK: DATA FOUND*
- llama3 8b Q4: 51 tok/s
- llama2 7b Q4: 76 tok/s
- llama2 13b Q4: 35 tok/s
- qwen35 9b (think off): 47.1 tok/s
- qwen35 9b (think on): 46.6 tok/s
- deepseek r1 14b (think off): 35.6 tok/s
- deepseek r1 14b (think on): 35.1 tok/s

**Other NVIDIA Benchmarks:**
- RTX 3070: llama3 8b Q4: 71 tok/s
- RTX 3080 10GB: llama3 8b Q4: 106 tok/s
- RTX 3080 12GB: llama3 8b Q4: 107 tok/s
- RTX 4060: llama3 8b Q4: 38 tok/s (Lowest bandwidth, avoid for AI)
- RTX 4060 Ti 8GB: llama3 8b Q4: 48 tok/s | llama2 7b Q4: 64 tok/s
- RTX 4060 Ti 16GB: llama3 8b Q4: 48 tok/s | llama2 7b Q4: 64 tok/s

**AMD Benchmarks (ROCm):**
- RX 7600: llama3 8b Q4: 39 tok/s | llama2 7b Q4: 96 tok/s
- RX 7800 XT: llama3 8b Q4: 39 tok/s | llama2 7b Q4: 96 tok/s
- RX 7900 XT: llama2 7b Q4: 116 tok/s | sustained: 97 tok/s

**Performance Efficiency Notes:**
- NVIDIA efficiency: ~0.13 tok/s per GB/s bandwidth (Llama 3 8B Q4).
- AMD efficiency: ~0.06 tok/s per GB/s bandwidth (less optimized kernels).
- Rule of thumb: Double bandwidth ≈ double tok/s for models fitting in VRAM.

## Key Specs
*Architecture, VRAM, Bandwidth, and TDP.*

**NVIDIA Ampere (Best Value Tier):**
- **RTX 3060 12GB:** 12GB GDDR6, 360 GB/s, 170W. *Arch: Ampere*
- **RTX 3070:** 8GB GDDR6, 448 GB/s, 220W. *Arch: Ampere*
- **RTX 3070 Ti:** 8GB GDDR6X, 608 GB/s, 290W. *Arch: Ampere*
- **RTX 3080 10GB:** 10GB GDDR6X, 760 GB/s, 320W. *Arch: Ampere*
- **RTX 3080 12GB:** 12GB GDDR6X, 912 GB/s, 350W. *Arch: Ampere*
- **RTX 3090:** 24GB GDDR6X, 936 GB/s, 350W. *Arch: Ampere*

**NVIDIA Ada Lovelace (Newer but Bandwidth-Limited):**
- **RTX 4060:** 8GB GDDR6, 272 GB/s, 115W. *Arch: Ada*
- **RTX 4060 Ti 8GB:** 8GB GDDR6, 288 GB/s, 160W. *Arch: Ada*
- **RTX 4060 Ti 16GB:** 16GB GDDR6, 288 GB/s, 165W. *Arch: Ada*

**AMD RDNA 3 (ROCm):**
- **RX 7600:** 8GB GDDR6, 288 GB/s, 165W.
- **RX 7700 XT:** 12GB GDDR6, 432 GB/s, 245W.
- **RX 7800 XT:** 16GB GDDR6, 624 GB/s, 263W.
- **RX 7900 GRE:** 16GB GDDR6, 576 GB/s, 260W.
- **RX 7900 XT:** 20GB GDDR6, 800 GB/s, 315W.

**Future/Preliminary Specs:**
- **RTX 5070:** 12GB GDDR7, 672 GB/s, 250W. *Arch: Blackwell* (MSRP $549)
- **RX 9070:** 16GB GDDR6, 608 GB/s, 220W. *Arch: RDNA 4* (MSRP $549)

**VRAM Capacity Guide:**
- 6GB: 7B Q4 max (tight)
- 8GB: 8B Q4 comfortable
- 10GB: 14B Q3 possible
- 12GB: 14B Q4, 8B Q8/FP16
- 16GB: 30B Q3, 14B Q6
- 20GB: 30B Q4, 70B Q2
- 24GB: 70B Q2-Q3

## Competitor Coverage
*Analysis of external recommendations based on retrieved context.*

- **Tech Tactician (Tom Smigla):**
    - Recommends **NVIDIA RTX 4060 Ti 16GB** as a top budget pick for its VRAM capacity, despite lower bandwidth compared to the RTX 4070.
    - Identifies **AMD RX 7800 XT** as the "Best AMD GPU for local AI," noting it is the closest competitor to NVIDIA in terms of performance if ROCm works.
    - Emphasizes that **VRAM is the #1 priority** for smooth inference; models should fit entirely in VRAM to avoid CPU offloading slowdowns (DDR4/DDR5 is ~37x slower than GDDR6X).
    - Notes that for beginners, AMD is now a "powerful and easy-to-use option" via Ollama/LM Studio/KoboldCpp.

- **General Consensus:**
    - NVIDIA CUDA is still the gold standard for ease of use; AMD ROCm support is improving but requires verification per stack.
    - RTX 4060 series (8GB/128-bit bus) are widely flagged as "avoid" for AI due to bandwidth starvation despite newer architecture.

## Internal Context
*InsiderLLM specific data and strategy.*

- **Brand Voice:** Practical, honest, no fluff. Focus on budget-friendly local AI for hobbyists/developers.
- **Content Strategy:** Currently 57 of 100 articles published. Strong focus on "Budget AI PC Under $500" and "Best Used GPUs."
- **Meta Description Insights (A/B Tests):**
    - Updated descriptions focusing on specific VRAM requirements (e.g., "7B needs 5GB", "70B needs 40GB") improved relevance.
    - "Budget Local AI PC" page CTR improved from 0.2% to higher after clarifying the $450 build cost (Optiplex + RTX 3060).
- **Existing Guides:** Published guides on VRAM requirements for 4GB, 6GB, 8GB, 12GB, 16GB, and 24GB.
- **Hardware Database:** Manually verified used prices from eBay sold auctions (not Buy It Now) as of March 2026.

## Gaps
*Missing data points required for a complete article.*

**CRITICAL MISSING DATA:**
- **RTX 3090 (24GB):** Used price, LLM benchmarks, and TDP are **PRESENT** in the input data ($1040 typical, 112 tok/s, 350W). *No action needed for this specific GPU.*
- **RTX 3060 (12GB):** Used price, LLM benchmarks, and TDP are **PRESENT** in the input data ($275 typical, 51 tok/s, 170W). *No action needed for this specific GPU.*

**Specific Data Gaps to Address:**
1. **RTX 4060 Ti 16GB vs 3080 12GB Direct Comparison:** While individual benchmarks exist (4060 Ti 16GB: 48 tok/s vs 3080 12GB: 107 tok/s), a direct analysis of the "VRAM vs Speed" trade-off for the $400-$500 price bracket needs explicit synthesis in the article.
2. **AMD ROCm Software Friction Points:** The input notes "ROCm compatibility varies" and "test before committing," but lacks specific examples of *which* models or frameworks (e.g., specific GGUF quantization versions) fail on RDNA 3 vs Ampere.
3. **New RTX 5060/5070 Availability:** Specs are preliminary ("None GB/s", "NoneW"). The writer needs to clarify that these are unconfirmed and likely not available for immediate purchase compared to the used market data.
4. **CPU Offloading Real-World Impact:** While theoretical speeds (DDR3/4/5) are listed, specific benchmarks of a 70B model offloaded to DDR4 vs DDR5 on a budget build are missing.

## Suggested Angle
*How InsiderLLM's take should differ.*

- **The "Bandwidth Trap" Warning:** Unlike competitors who might tout the RTX 4060 Ti 16GB as a top pick due to VRAM alone, InsiderLLM must emphasize that for local AI, **VRAM without bandwidth is useless**. Use the data to show the 3080 12GB (912 GB/s) vastly outperforming the 4060 Ti 16GB (288 GB/s) in speed, even at similar prices.
- **The "Used Market King" Narrative:** Position the **RTX 3090** not as a luxury item, but as the *only* card under $1,100 that truly handles 70B models locally, making it the "budget king" for serious hobbyists despite its high power draw.
- **AMD as the "Second Choice":** Frame AMD (RX 7800 XT) not as a direct rival to NVIDIA in ease-of-use, but as a high-performance alternative *only* for users willing to troubleshoot ROCm drivers. Use the "0.06 tok/s per GB/s" metric to quantify the efficiency gap.
- **Build Specifics:** Leverage the internal context of the $450 build (Optiplex + 3060) to provide a concrete, step-by-step "Minimum Viable AI PC" guide, contrasting it with the "Prosumer" (RTX 3090/3080) builds.