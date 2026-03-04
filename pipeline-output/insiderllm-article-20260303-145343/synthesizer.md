## Price Data

*   **NVIDIA RTX 3090 (24GB - Used):** $950-$1040-$1125 (Source: InsiderLLM, 2026-03-02)
*   **NVIDIA RTX 3060 (12GB - Used):** $170-$275-$380 (Source: InsiderLLM, 2026-03-02)
*   **NVIDIA RTX 5060 Ti (16GB - Used):** $380-$430-$480 (Source: InsiderLLM, 2026-03-02)
*   **NVIDIA RTX 4060 Ti (16GB - Used):** $380-$430-$480 (Source: InsiderLLM, 2026-03-02)
*   **AMD Radeon RX 9070 (16GB - Used):** $400-$475-$550 (Source: InsiderLLM, 2026-03-02)

## Benchmark Data

*   **RTX 3060 (12GB):**
    *   llama2 7b Q4: 76 tok/s (Source: InsiderLLM)
    *   llama2 13b Q4: 35 tok/s (Source: InsiderLLM)
    *   llama3 8b Q4: 51 tok/s (Source: InsiderLLM)
*   **RTX 3080 (10GB):**
    *   llama3 8b Q4: 106 tok/s (Source: InsiderLLM)
*   **RTX 3080 (12GB):**
    *   llama3 8b Q4: 107 tok/s (Source: InsiderLLM)
*   **RTX 3090 (24GB):**
    *   llama3 8b Q4: 112 tok/s (Source: InsiderLLM)
    *   llama3 8b F16: 47 tok/s (Source: InsiderLLM)
    *   llama3 70b Q4: 16 tok/s (Source: InsiderLLM)
    *   mistral 7b Q6: 85 tok/s (Source: InsiderLLM)
*   **RTX 4060 Ti (16GB):**
    *   llama3 8b Q4: 48 tok/s (Source: InsiderLLM)
    *   llama2 7b Q4: 64 tok/s (Source: InsiderLLM)
*   **AMD Radeon RX 7700 XT (12GB):**
    *   llama2 7b Q4: 96 tok/s (Source: InsiderLLM)
*   **AMD Radeon RX 7800 XT (16GB):**
    *   llama3 8b Q4: 39 tok/s (Source: InsiderLLM)
    *   llama2 7b Q4: 96 tok/s (Source: InsiderLLM)
*   **RTX 3070 (8GB):**
    *   llama3 8b Q4: 71 tok/s (Source: InsiderLLM)

## Key Specs

*   **NVIDIA RTX 3060 (12GB):** 12GB GDDR6, 360 GB/s, TDP: 170W, Arch: Ampere (Source: InsiderLLM)
*   **NVIDIA RTX 3090 (24GB):** 24GB GDDR6X, 936 GB/s, TDP: 350W, Arch: Ampere (Source: InsiderLLM)
*   **NVIDIA RTX 4060 Ti (16GB):** 16GB GDDR6, 288 GB/s, TDP: 165W, Arch: Ada Lovelace (Source: InsiderLLM)
*   **NVIDIA RTX 5060 Ti (16GB):** 16GB GDDR7, (Bandwidth unknown), TDP: (Unknown), Arch: Blackwell (Source: InsiderLLM)
*   **AMD Radeon RX 9070 (16GB):** 16GB GDDR6, 608 GB/s, TDP: 220W, Arch: RDNA 4 (Source: InsiderLLM)

## Competitor Coverage

NO DATA FOUND

## Internal Context

*   InsiderLLM has published several articles on related topics: GPU Buying Guide for Local AI, Used RTX 3090 Buying Guide, Best GPU Under $300 for Local AI, Budget AI PC Under $500 (Source: D3, D4, D5).
*   A/B testing shows meta description changes have had minimal impact on CTR for related articles (Source: D1, D2).
*   Content focus is on budget-friendly local AI for hobbyists and developers, with a practical and honest brand voice. (Source: D3)

## Gaps

*   **RTX 5060 Ti bandwidth and TDP:** Crucial for comparison, especially given the focus on bandwidth being key for performance. *CRITICAL*
*   **RTX 5060/5070 Availability & Actual Pricing:** The MSRP is listed but actual street price is needed. *CRITICAL*
*   **AMD Radeon RX 9070 ROCm compatibility:** Confirmation of model/framework support is needed. *CRITICAL*
*   **AMD Radeon RX 7900 GRE & XT benchmarks:** Performance data is missing for these cards.
*   **More diverse benchmark data:** Performance data for more models and quantization levels would be valuable.
*   **Power consumption data:** Actual power draw during inference for each card would be useful for a complete picture.
*   **Specific model recommendations:** Which models work best on each GPU for different use cases (chat, image generation, etc.).

## Suggested Angle

InsiderLLM should focus on *practical performance per dollar*. Many sites list specs, but few demonstrate what you can *actually do* with each GPU. Given the focus on budget builds (evidenced by existing content - D3, D4, D5), InsiderLLM should emphasize the best GPUs for running specific models at various quantization levels *within a tight budget*. Specifically, highlight the value proposition of the RTX 3060 12GB and RTX 3080 12GB, given their bandwidth/VRAM combination. A deep dive into the trade-offs between VRAM, bandwidth, and quantization is needed. Also, provide clear guidance on CPU/RAM pairing to minimize offloading penalties. A section on the realistic expectations for AMD ROCm compatibility is also important.