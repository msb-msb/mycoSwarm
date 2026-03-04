## Price Data

*   **NVIDIA RTX 3090 (24GB):** Used: $950-$1040-$1125 (2026-03-02) (Source: InsiderLLM canonical hardware database)
*   **NVIDIA RTX 3060 (12GB):** Used: $170-$275-$380 (2026-03-02) (Source: InsiderLLM canonical hardware database)
*   **NVIDIA RTX 4060 Ti 16GB:** Used: $380-$430-$480 (2026-03-02) (Source: InsiderLLM canonical hardware database)
*   **AMD Radeon RX 9070 (16GB):** Used: $400-$475-$550 (2026-03-02) (Source: InsiderLLM canonical hardware database)
*   **NVIDIA RTX 5060 Ti (16GB):** PRELIMINARY - Price not confirmed (Source: InsiderLLM canonical hardware database)

## Benchmark Data

*   **RTX 3060 (12GB):** llama3 8b Q4: 51 tok/s, llama2 7b Q4: 76 tok/s, llama2 13b Q4: 35 tok/s, qwen35 9b think off: 47.1 tok/s, qwen35 9b think on: 46.6 tok/s, deepseek r1 14b think off: 35.6 tok/s, deepseek r1 14b think on: 35.1 tok/s (Source: InsiderLLM canonical hardware database)
*   **RTX 3090 (24GB):** llama3 8b Q4: 112 tok/s, llama3 8b F16: 47 tok/s, llama3 70b Q4: 16 tok/s, mistral 7b Q6: 85 tok/s, gemma3 27b: 39.9 tok/s (Source: InsiderLLM canonical hardware database)
*   **RTX 4060 Ti 16GB:** llama3 8b Q4: 48 tok/s, llama2 7b Q4: 64 tok/s (Source: InsiderLLM canonical hardware database)
*   **AMD Radeon RX 7800 XT (16GB):** llama3 8b Q4: 39 tok/s, llama2 7b Q4: 96 tok/s (Source: InsiderLLM canonical hardware database)
*   **RTX 3080 (10GB):** llama3 8b Q4: 106 tok/s (Source: InsiderLLM canonical hardware database)
*   **RTX 3080 (12GB):** llama3 8b Q4: 107 tok/s (Source: InsiderLLM canonical hardware database)
*   **AMD Radeon RX 7900 XT (20GB):** llama2 7b Q4: 116 tok/s, llama2 7b Q4 sustained: 97 tok/s (Source: InsiderLLM canonical hardware database)

## Key Specs

*   **NVIDIA RTX 3090 (24GB):** 24GB GDDR6X, 936 GB/s, TDP: 350W, Arch: Ampere
*   **NVIDIA RTX 3060 (12GB):** 12GB GDDR6, 360 GB/s, TDP: 170W, Arch: Ampere
*   **NVIDIA RTX 4060 Ti 16GB:** 16GB GDDR6, 288 GB/s, TDP: 165W, Arch: Ada Lovelace
*   **AMD Radeon RX 9070 (16GB):** 16GB GDDR6, 608 GB/s, TDP: 220W, Arch: RDNA 4
*   **NVIDIA RTX 5060 Ti (16GB):** 16GB GDDR7, Bandwidth unknown, TDP: NoneW, Arch: Blackwell

## Competitor Coverage

NO DATA FOUND

## Internal Context

*   InsiderLLM has published articles covering GPU buying guides, VRAM requirements, used GPU recommendations, and budget PC builds for local AI (D3, D4, D5).
*   A/B testing of meta descriptions is ongoing for related articles, with a focus on clarity around model sizes and VRAM requirements (D1, D2).
*   Focus is on budget-friendly local AI for hobbyists and developers (D3).

## Gaps

*   **CRITICAL:** RTX 5060 Ti 16GB - Bandwidth and TDP are unknown. Need confirmed specs for a proper evaluation.
*   RTX 5060 - Need confirmed specs.
*   AMD Radeon RX 7700 XT benchmarks.
*   AMD Radeon RX 7900 GRE benchmarks.
*   More detailed performance data for AMD GPUs with ROCm. Specifically, how performance compares to NVIDIA across different models and quantization levels.
*   Comparative pricing data beyond used prices (e.g., new prices where available).
*   Power consumption data for all GPUs under load running LLMs.
*   Testing data for different CPU/RAM configurations and their impact on offloading performance.
*   Performance data on the AMD Radeon RX 9070 with different LLMs/frameworks on ROCm.

## Suggested Angle

InsiderLLM should focus on *practical* advice for building a functional local AI setup on a tight budget. Unlike many sites that focus on the latest high-end hardware, we should highlight the value proposition of used GPUs (especially the RTX 3060 and 3090) and emphasize the importance of VRAM *and* bandwidth. Our coverage should go beyond simply listing specs and benchmarks to provide *real-world* performance expectations for different model sizes and quantization levels. We should also highlight the trade-offs between AMD and NVIDIA, specifically addressing the ROCm compatibility issues and providing clear guidance on when AMD is a viable option. Finally, we should lean into the data on CPU/RAM configurations, providing advice on how to optimize performance for offloading.