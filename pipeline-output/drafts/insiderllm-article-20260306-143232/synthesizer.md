## Price Data
*All prices are used market hammer prices from eBay (as of 2026-03-02) unless noted as MSRP/preliminary.*

**NVIDIA RTX 30-Series (Used)**
*   **RTX 3060 (12GB):** $170–$380 (Typical: ~$275). *Note: Entry-level workhorse; prices vary significantly based on condition.*
*   **RTX 3070 (8GB):** $210–$300 (Typical: ~$255).
*   **RTX 3070 Ti (8GB):** $100–$280 (Typical: ~$190). *Note: Significant price variance; often found cheap but power-hungry.*
*   **RTX 3080 (10GB):** $325–$400 (Typical: ~$365).
*   **RTX 3080 (12GB):** $230–$380 (Typical: ~$305). *Note: "Sleeper pick" high value.*
*   **RTX 3090 (24GB):** $950–$1,125 (Typical: ~$1,040).
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
*   **RTX 3060:** 12GB GDDR6 | 360 GB/s | 170W | Turing Architecture (Wait, source says Ampere). *Correction: Ampere.*
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

## Gaps
**CRITICAL MISSING DATA (MANDATORY CHECK):**
*   **RTX 3090 (24GB):** While price ($950–$1,125) and benchmarks are present, the *specific* used market trend for this card in early 2026 relative to the RTX 4070 Ti Super needs more granular comparison on thermal performance of blower vs. triple-fan models in high-load AI scenarios.
*   **RTX 3060 (12GB):** The price range ($170–$380) is extremely wide. A specific breakdown of why some units are $170 vs $380 (e.g., mining history, physical condition, warranty status) is missing.

**Other Missing Data:**
1.  **ROCm Version Mapping:** No data on which specific ROCm versions support which AMD RDNA 3 cards for local LLM inference. Users need to know if a card will work *today* with current stacks.
2.  **Thermal Throttling Data:** Specific thermal limits and throttling points for the RTX 3090 (known to run hot) and RX 7900 series during sustained AI loads are not quantified.
3.  **RTX 50-Series Confirmation:** Bandwidth, TDP, and actual availability/pricing for Blackwell cards (5060/5070) are currently "PRELIMINARY." The writer needs to decide if they should recommend waiting or stick to Ampere.
4.  **CPU Offloading Realities:** While DDR bandwidth ratios are theoretical (37x slower), real-world latency and system stability data for "hybrid" setups (e.g., 12GB GPU + 32GB RAM) is sparse compared to pure VRAM inference.

## Suggested Angle
**InsiderLLM Take:** Move beyond "Best GPU" lists to **"VRAM vs. Bandwidth Trade-offs."**
*   **The Hook:** Most guides recommend the newest cards (40-series) or highest VRAM (3090). The InsiderLLM angle should argue that for *budget* AI, the **RTX 3080 12GB** and **RTX 3060 12GB** offer superior value because they balance cost, model capacity, and acceptable speed better than the bandwidth-starved RTX 4060 Ti or the power-hungry RTX 3090.
*   **The Differentiator:** Explicitly debunk the "Newer is Better" myth for AI by highlighting how the RTX 4060 series (272 GB/s) is slower than the older RTX 3060 (360 GB/s) despite being newer. Use the "0.13 tok/s per GB/s" internal rule to prove that bandwidth > compute power for inference.
*   **Actionable Advice:** Provide a decision matrix: "If you need 7B speed, buy 3070; if you need 13B capacity, buy 3060; if you need 70B, buy 3090." Avoid generic recommendations.