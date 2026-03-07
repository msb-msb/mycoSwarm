# Research Bundle: Best Budget GPU for Local AI in 2026 (Updated)

---

## SYNTHESIZER OUTPUT
*(Merged with NEW FACTS from "RETRIEVED CONTEXT")*

## Price Data
*All prices are used market hammer prices from eBay (as of 2026-03-02) unless noted as MSRP/preliminary.*

**NVIDIA RTX 30-Series (Used)**
*   **RTX 3060 (12GB):** $170–$380 (Typical: ~$275). *Note: Price variance driven by condition, mining history, and warranty status. Units under $200 often have high fan noise or cosmetic wear.*
*   **RTX 3070 (8GB):** $210–$300 (Typical: ~$255).
*   **RTX 3070 Ti (8GB):** $100–$280 (Typical: ~$190). *Note: Significant price variance; often found cheap but power-hungry.*
*   **RTX 3080 (10GB):** $325–$400 (Typical: ~$365).
*   **RTX 3080 (12GB):** $230–$380 (Typical: ~$305). *Note: "Sleeper pick" high value.*
*   **RTX 3090 (24GB):** $950–$1,125 (Typical: ~$1,040). *Note: Blower models run significantly hotter; triple-fan variants preferred for sustained AI loads.*
*   **RTX 2060 (12GB):** $140–$180 (Typical: ~$160).
*   **GTX 1660 Super (6GB):** $90–$120 (Typical: ~$105).

**NVIDIA RTX 40-Series (Used)**
*   **RTX 4060 (8GB):** $230–$310 (Typical: ~$270). *Note: Poor value for AI due to bandwidth.*
*   **RTX 4060 Ti (8GB):** $240–$300 (Typical: ~$270).
*   **RTX 4060 Ti (16GB):** $380–$480 (Typical: ~$430).

**AMD Radeon RX 7000 Series (Used)**
*   **RX 7600 (8GB):** $170–$225 (Typical: ~$200).
*   **RX 7700 XT (12GB):** $300–$350 (Typical: ~$325).
*   **RX 7800 XT (16GB):** $380–$550 (Typical: ~$465).
*   **RX 7900 GRE (16GB):** $400–$550 (Typical: ~$475).
*   **RX 7900 XT (20GB):** $500–$700 (Typical: ~$600).

**Preliminary/MSRP (Not yet verified used market)**
*   **RTX 5070:** MSRP $549. *Specs: 12GB GDDR7, 672 GB/s.*
*   **RX 9070:** MSRP $549. *Specs: 16GB GDDR6, 608 GB/s.*
*   **RX 9070 XT:** MSRP $599. *Specs: 16GB GDDR6, 608 GB/s.*

## Benchmark Data
*Metrics in tokens per second (tok/s). Models run at Quantization levels specified.*

**RTX 3060 (12GB)**
*   llama3 8b Q4: 51 tok/s
*   llama2 7b Q4: 76 tok/s
*   llama2 13b Q4: 35 tok/s
*   qwen35 9b think off: 47.1 tok/s | on: 46.6 tok/s
*   deepseek r1 14b think off: 35.6 tok/s | on: 35.1 tok/s

**RTX 3070 (8GB)**
*   llama3 8b Q4: 71 tok/s

**RTX 3080 (10GB & 12GB)**
*   llama3 8b Q4: 106–107 tok/s

**RTX 3090 (24GB)**
*   llama3 8b Q4: 112 tok/s
*   llama3 8b FP16: 47 tok/s
*   llama3 70b Q4: 16 tok/s
*   mistral 7b Q6: 85 tok/s
*   gemma3 27b: 39.9 tok/s

**RTX 40-Series (Low Bandwidth Impact)**
*   RTX 4060 (8GB): llama3 8b Q4: 38 tok/s
*   RTX 4060 Ti 16GB: llama3 8b Q4: 48 tok/s | llama2 7b Q4: 64 tok/s

**AMD Radeon RX 7000 Series (ROCm)**
*   RX 7800 XT (16GB): llama3 8b Q4: 39 tok/s | llama2 7b Q4: 96 tok/s
*   RX 7900 XT (20GB): llama2 7b Q4: 116 tok/s (Sustained: 97 tok/s)

## Key Specs
*Architecture, VRAM, Bandwidth, and TDP.*

**NVIDIA Ampere (Best Value)**
*   **RTX 3060:** 12GB GDDR6 | 360 GB/s | 170W | Ampere.
*   **RTX 3070:** 8GB GDDR6 | 448 GB/s | 220W | Ampere.
*   **RTX 3070 Ti:** 8GB GDDR6X | 608 GB/s | 290W | Ampere.
*   **RTX 3080 (10GB):** 10GB GDDR6X | 760 GB/s | 320W | Ampere.
*   **RTX 3080 (12GB):** 12GB GDDR6X | 912 GB/s | 350W | Ampere.
*   **RTX 3090:** 24GB GDDR6X | 936 GB/s | 350W | Ampere.

**NVIDIA Ada Lovelace (Bandwidth Limited)**
*   **RTX 4060:** 8GB GDDR6 | 272 GB/s | 115W | 128-bit bus.
*   **RTX 4060 Ti (8GB):** 8GB GDDR6 | 288 GB/s | 160W | 128-bit bus.
*   **RTX 4060 Ti (16GB):** 16GB GDDR6 | 288 GB/s | 165W | 128-bit bus.

**AMD RDNA 3**
*   **RX 7600:** 8GB GDDR6 | 288 GB/s | 165W.
*   **RX 7700 XT:** 12GB GDDR6 | 432 GB/s | 245W.
*   **RX 7800 XT:** 16GB GDDR6 | 624 GB/s | 263W.
*   **RX 7900 GRE:** 16GB GDDR6 | 576 GB/s | 260W.
*   **RX 7900 XT:** 20GB GDDR6 | 800 GB/s | 315W.

**Preliminary (Blackwell/RDNA4)**
*   **RTX 5070:** 12GB GDDR7 | 672 GB/s | 250W.
*   **RX 9070 / 9070 XT:** 16GB GDDR6 | 608 GB/s | 220-250W.

## Competitor Coverage
*   **XDA Developers:** Highlights RTX 30-series as the "quietly best value" due to normalized used prices post-crypto/pandemic inflation. Emphasizes Ampere cards (3070/3080) as sensible buys compared to inflated new pricing.
*   **TechRadar:** Focuses on deal hunting for the RTX 3060, noting it is "finally affordable" but still above MSRP in their specific listings.
*   **Northflank / SitePoint:** Identify the RTX 4060 Ti 16GB and AMD RX 7800 XT as the minimum viable options for 70B models (via Q2 quantization) due to 16GB VRAM requirements, despite lower bandwidth.
*   **Tech Tactician:** Recommends NVIDIA CUDA for ease of use but acknowledges AMD's raster performance advantage; notes that Deep Learning Super Sampling is a key differentiator for NVIDIA.

## Internal Context
*   **Content Strategy:** Focus on "Budget-friendly local AI for hobbyists and developers" with a brand voice of "Practical, honest, no fluff."
*   **Meta Description Testing:** Recent A/B tests (Feb 2026) show higher CTR for descriptions specifying exact VRAM requirements (e.g., "3B models need 2GB... 70B needs 40GB") compared to vague summaries.
*   **Existing Content:** InsiderLLM has published guides on "Best GPU Under $300/$500," "VRAM Requirements," and specific model capacity guides (What can you run on 8/12/16/24 GB).
*   **Performance Rules:** NVIDIA cards average ~0.13 tok/s per GB/s of bandwidth for Llama 3 8B Q4; AMD ROCm achieves ~0.06 tok/s per GB/s. DDR4-3200 offloading is ~37x slower than GDDR6X VRAM inference.

## New Data
*The following specific data points were found to fill previously identified gaps:*

1.  **RTX 3090 Thermal Nuance:** Verified that dual-slot blower cooler models on the RTX 3090 run "especially hot" under sustained AI loads, leading to potential thermal throttling. Triple-fan models are strongly recommended for stability in local AI scenarios.
2.  **RTX 3060 Price Variance Explanation:** The wide price range ($170–$380) is attributed to unit condition (cosmetic wear vs. pristine), mining history (hash rate degradation risks), and remaining warranty status. Units under $200 often show signs of high fan noise or cosmetic damage.
3.  **ROCm Version Mapping:** While specific version numbers are not listed in the source text, it is confirmed that "ROCm support is improving but not all models/frameworks work." Users must verify ROCm compatibility for their specific stack before purchasing AMD cards.
4.  **RTX 50-Series Confirmation:** The RTX 5070 specs (12GB GDDR7, 672 GB/s, 250W) are confirmed as preliminary with an MSRP of $549. Availability and actual pricing remain TBD.
5.  **CPU Offloading Realities:** Confirmed that DDR4-3200 (25.6 GB/s) is ~37x slower per layer than GDDR6X (936 GB/s). DDR4 offloading is viable for models that fit mostly in VRAM but causes significant latency penalties for large offloaded layers.
6.  **Multi-GPU Limitation:** Confirmed that SLI/CrossFire for AI training/inference is not a supported or practical strategy for these cards; multi-card setups require distinct PCIe lanes and are "a topic for a whole new article."

## Gaps
**CRITICAL MISSING DATA (MANDATORY CHECK):**
*   **RTX 50-Series Availability:** While specs and MSRP ($549) for the RTX 5070 are now confirmed, actual market availability and real-world performance benchmarks in 2026 remain TBD. The decision to recommend waiting vs. buying Ampere is pending launch data.
*   **Specific ROCm Version Numbers:** The specific ROCm version numbers (e.g., "ROCm 6.1 supports RX 7800 XT") are still not listed. Users need exact version compatibility lists to avoid installation failures.
*   **DDR Memory Pricing Tiers:** While bandwidth ratios for DDR3/4/5 are known, specific used pricing for high-capacity RAM kits (e.g., "Price per 16GB/32GB" for budget builds) was found in the reference data but not fully integrated into the "Budget PC Build" section of this specific bundle.
*   **RTX 4070 Ti Super Comparison:** While mentioned as a competitor in some contexts, specific price/benchmark data for the RTX 4070 Ti Super relative to the RTX 3090 is not fully detailed in the current "Price Data" section (only implied via internal context).

**Other Missing Data:**
1.  **Specific Model Framework Compatibility:** Detailed lists of which AI frameworks (e.g., Ollama, LM Studio, Text Generation WebUI) have confirmed working status with specific AMD ROCm versions are not provided.
2.  **Long-Term Reliability Data:** No data on the long-term failure rates of used mining GPUs vs. gaming GPUs for the RTX 30-series in 2026.

---
*Note: All data points are derived strictly from the provided inputs and verified reference data. Contradictions regarding "Turing" vs "Ampere" architecture for the 30-series have been resolved (corrected to Ampere).*