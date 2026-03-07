## Price Data
*   **RTX 3090 (24GB):** Used $950–$1,125 (Typical ~$1,040). Source: eBay sold auctions (2026-03-02).
*   **RTX 3060 (12GB):** Used $170–$380 (Typical ~$275). Source: eBay sold auctions (2026-03-02). *Note: Price variance is high; best value found under $200.*
*   **RTX 3060 (12GB) New:** No specific new price listed, but used market dominates.
*   **RTX 4060 Ti 16GB:** Used $380–$480 (Typical ~$430).
*   **RTX 3070 Ti:** Used $100–$280 (Typical ~$190). *Note: Significant price drop noted, possibly due to crypto mining end.*
*   **RTX 3080 12GB:** Used $230–$380 (Typical ~$305).
*   **RTX 4060 Ti 8GB:** Used $240–$300 (Typical ~$270).
*   **AMD RX 7900 XT:** Used $500–$700 (Typical ~$600).
*   **DDR4 RAM:** 16GB used $50–$80; 32GB used $100–$170.

## Benchmark Data
*   **RTX 3090:**
    *   Llama3 8b Q4: 112 tok/s
    *   Llama3 8b F16: 47 tok/s
    *   Llama3 70b Q4: 16 tok/s
    *   Mistral 7b Q6: 85 tok/s
    *   Gemma3 27b: 39.9 tok/s
*   **RTX 3060 (12GB):**
    *   Llama3 8b Q4: 51 tok/s
    *   Llama2 7b Q4: 76 tok/s
    *   Llama2 13b Q4: 35 tok/s
    *   Qwen3.5 9b (think off): 47.1 tok/s
    *   Qwen3.5 9b (think on): 46.6 tok/s
    *   DeepSeek R1 14b (think off): 35.6 tok/s
    *   DeepSeek R1 14b (think on): 35.1 tok/s
*   **RTX 3070:**
    *   Llama3 8b Q4: 71 tok/s
*   **RTX 3080 10GB/12GB:**
    *   Llama3 8b Q4: ~106–107 tok/s (Both models show similar performance for this model size).
*   **RTX 4060 Ti 16GB:**
    *   Llama3 8b Q4: 48 tok/s
    *   Llama2 7b Q4: 64 tok/s
*   **AMD RX 7800 XT:**
    *   Llama3 8b Q4: 39 tok/s
    *   Llama2 7b Q4: 96 tok/s
*   **AMD RX 7900 XT:**
    *   Llama2 7b Q4: 116 tok/s (Peak) / 97 tok/s (Sustained)

## Key Specs
*   **RTX 3090:** 24GB GDDR6X, 936 GB/s, TDP 350W, Ampere.
*   **RTX 3060 (12GB):** 12GB GDDR6, 360 GB/s, TDP 170W, Ampere.
*   **RTX 3070:** 8GB GDDR6, 448 GB/s, TDP 220W, Ampere.
*   **RTX 3070 Ti:** 8GB GDDR6X, 608 GB/s, TDP 290W, Ampere.
*   **RTX 3080 10GB:** 10GB GDDR6X, 760 GB/s, TDP 320W, Ampere.
*   **RTX 3080 12GB:** 12GB GDDR6X, 912 GB/s, TDP 350W, Ampere.
*   **RTX 4060 Ti 16GB:** 16GB GDDR6, 288 GB/s, TDP 165W, Ada Lovelace.
*   **AMD RX 7900 XT:** 20GB GDDR6, 800 GB/s, TDP 315W, RDNA 3.
*   **RTX 4060 Ti 8GB:** 8GB GDDR6, 288 GB/s, TDP 160W, Ada Lovelace.
*   **AMD RX 7800 XT:** 16GB GDDR6, 624 GB/s, TDP 263W, RDNA 3.

## Competitor Coverage
*   **General Consensus:** Most competitors prioritize the RTX 4060 Ti 16GB for "budget" AI due to its 16GB VRAM capacity, despite bandwidth limitations.
*   **AMD Positioning:** Competitors acknowledge AMD ROCm improvements but warn of software compatibility risks (specifically with PyTorch/LLM frameworks) compared to NVIDIA CUDA. The RX 7900 XT is noted as a strong alternative for those willing to test ROCm support.
*   **RTX 3060 vs 4060 Ti:** Competitors often highlight the RTX 3060 (12GB) as the "sweet spot" for value under $200, whereas the 4060 Ti is seen as a premium budget option for those needing >12GB VRAM.
*   **RTX 3090:** Widely recommended as the "king" of used local AI hardware despite high cost, due to 24GB capacity enabling 70B models.

## Internal Context
*   **A/B Testing Results:** Meta description changes for "VRAM Requirements" and "Budget PC" guides improved CTR significantly (from ~0% to 0.1-0.2%). Key messaging focus: "Exact VRAM requirements," "under $450 builds," and "practical setup."
*   **Content Plan:** InsiderLLM has published guides on "Best Used GPUs," "VRAM Requirements," and "Budget AI PC Under $500." Current focus is on practical, honest advice for hobbyists.
*   **Brand Voice:** Practical, no fluff, emphasizing cost-per-use over sticker price.
*   **Existing Guides:** Specific guides exist for "What Can You Run on 8GB/12GB/16GB/24GB VRAM."

## Gaps
**CRITICAL MISSING DATA:**
*   **RTX 3090 (24GB):** While price and specs are present, specific *LLM benchmark data for the RTX 3090* is required to be fully fleshed out in the final article to match the depth of the RTX 3060 section. The input provides benchmarks, but the "Gaps" logic requires explicit confirmation of the presence of this data point as a critical asset. (Data present: Llama3 70b Q4, etc.).
*   **RTX 3060 (12GB):** Input contains price, specs, and benchmarks. **CRITICAL:** Ensure the final article explicitly addresses the *wide price variance* ($170–$380) as a key buying decision factor.

**Additional Gaps for Writer:**
*   **Specific Cooling Advice:** No data on specific blower vs. triple-fan model temperature differences for the RTX 3090 beyond general warnings.
*   **Power Supply Recommendations:** No specific wattage recommendations for dual-GPU or high-TDP builds (e.g., 3090 + CPU) in the input.
*   **ROCm Version Specifics:** Input mentions "ROCm support improving" but does not list specific version numbers required for AMD cards to run LLMs reliably.
*   **Future Price Projections:** No data on expected price drops for RTX 50-series or AMD RDNA 4 in late 2026.

## Suggested Angle
**Differentiation Strategy:** Move beyond simple "best value" lists. InsiderLLM should position the article as a **"Total Cost of Ownership (TCO) vs. VRAM Reality Check."**

1.  **The "Bandwidth Trap":** Explicitly call out why newer cards (RTX 4060 Ti, RX 7800 XT) can be slower than older Ampere cards (3080/3090) for LLMs due to the 128-bit bus bottleneck, despite being "newer." This contradicts the common consumer assumption that newer = faster for AI.
2.  **The "Used Market" Warning:** Highlight the specific price volatility of the RTX 3060 (12GB) as a primary buying risk, advising readers to hunt for sub-$200 units rather than settling for $300+.
3.  **AMD Reality Check:** Instead of just saying "ROCm is improving," provide a concrete checklist for the reader to verify compatibility *before* buying an AMD card (e.g., specific quantization formats, OS requirements).
4.  **The "Sleeper Pick":** Champion the RTX 3080 12GB ($305 used) over the RTX 4060 Ti 16GB ($430 used) for pure LLM inference speed, framing it as the only card that offers 12GB VRAM *plus* high bandwidth at a lower price point.