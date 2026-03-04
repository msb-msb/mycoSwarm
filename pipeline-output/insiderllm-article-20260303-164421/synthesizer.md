## Price Data

*   **NVIDIA RTX 3090 (24GB):** Used: $950–$1125 (typical ~$1040) (2026-03-02) (Source: InsiderLLM)
*   **NVIDIA RTX 3060 (12GB):** Used: $170–$380 (typical ~$275) (2026-03-02) (Source: InsiderLLM)
*   **NVIDIA RTX 5060 Ti (16GB):** Used: $380–$480 (typical ~$430) (2026-03-02) (Source: InsiderLLM)
*   **NVIDIA RTX 4060 Ti (16GB):** Used: $380–$480 (typical ~$430) (2026-03-02) (Source: InsiderLLM)
*   **AMD RX 9070 (16GB):** MSRP: $549 (PRELIMINARY) (Source: InsiderLLM)

## Benchmark Data

*   **NVIDIA RTX 3090:**
    *   llama3 8b Q4: 112 tok/s (Source: InsiderLLM)
    *   llama3 8b F16: 47 tok/s (Source: InsiderLLM)
    *   llama3 70b Q4: 16 tok/s (Source: InsiderLLM)
*   **NVIDIA RTX 3060:**
    *   llama3 8b Q4: 51 tok/s (Source: InsiderLLM)
    *   llama2 7b Q4: 76 tok/s (Source: InsiderLLM)
    *   llama2 13b Q4: 35 tok/s (Source: InsiderLLM)
*   **NVIDIA RTX 5060 Ti (16GB):**
    *   llama3 8b Q4: 48 tok/s (Source: InsiderLLM)
    *   llama2 7b Q4: 64 tok/s (Source: InsiderLLM)
*   **NVIDIA RTX 4060 Ti (16GB):**
    *   llama3 8b Q4: 60 tok/s (Source: InsiderLLM)

## Key Specs

*   **NVIDIA RTX 3090:** 24GB GDDR6X, 936 GB/s, TDP: 350W, Arch: Ampere (Source: InsiderLLM)
*   **NVIDIA RTX 3060:** 12GB GDDR6, 360 GB/s, TDP: 170W, Arch: Ampere (Source: InsiderLLM)
*   **NVIDIA RTX 5060 Ti:** 16GB GDDR7, 448 GB/s, TDP: 165W, Arch: Blackwell (Source: InsiderLLM)
*   **NVIDIA RTX 4060 Ti:** 16GB GDDR6, 288 GB/s, TDP: 165W, Arch: Ada Lovelace (Source: InsiderLLM)
*   **AMD RX 9070:** 16GB GDDR6, 608 GB/s, TDP: 220W, Arch: RDNA 4 (Source: InsiderLLM)

## Competitor Coverage

NO DATA FOUND

## Internal Context

*   InsiderLLM has published articles covering GPU buying guides, VRAM requirements, used GPU recommendations, and budget AI PC builds (Source: D3, D4, D5)
*   Meta description A/B tests are in progress for articles related to VRAM, UI comparisons, and text-generation-webui (Source: D1, D2)
*   Brand voice is practical, honest, and avoids fluff (Source: D3)

## Gaps

*   **CRITICAL:** Benchmarks for AMD RX 9070. (Missing performance data for the AMD card)
*   **CRITICAL:** RTX 5060 Ti specs are preliminary. Need confirmed bandwidth and TDP figures.
*   Pricing for the RTX 5060 Ti is missing.
*   Comparative benchmarks between all GPUs listed, specifically focusing on price/performance for different model sizes.
*   More detailed analysis of AMD ROCm compatibility and performance compared to CUDA.
*   Power consumption benchmarks for all GPUs.
*   Specific model recommendations for each GPU (e.g., "RTX 3060 is best for running X model").

## Suggested Angle

InsiderLLM should focus on *practical* performance and value, going beyond raw specs. The article should emphasize *real-world* usability – what models can you *actually* run on each card, and how quickly? Given the existing content on budget builds (D3, D4, D5), a focus on maximizing performance *within* a specific budget (e.g., $500, $800) would be a strong differentiator. Highlight the trade-offs between VRAM and bandwidth, and explain how these impact performance for different quantization levels. A clear explanation of how to interpret benchmarks and choose the right GPU for specific use cases (e.g., experimentation vs. daily use) is crucial. Given the ROCm concerns, a section dedicated to troubleshooting and verifying AMD compatibility would be valuable.