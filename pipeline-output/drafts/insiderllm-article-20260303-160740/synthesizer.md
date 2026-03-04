## Price Data

* **NVIDIA RTX 3060 (12GB):** Used: $170–$380 (typical ~$275) (InsiderLLM canonical hardware database)
* **NVIDIA RTX 3090 (24GB):** Used: $950–$1125 (typical ~$1040) (InsiderLLM canonical hardware database)
* **NVIDIA RTX 4060 Ti 16GB:** Used: $380–$480 (typical ~$430) (InsiderLLM canonical hardware database)
* **NVIDIA RTX 5060 Ti 16GB:** Used: $380–$480 (typical ~$430) (InsiderLLM canonical hardware database)
* **AMD Radeon RX 9070 (16GB):** Used: $400–$550 (typical ~$475) (InsiderLLM canonical hardware database)

## Benchmark Data

* **NVIDIA RTX 3060 (12GB):**
    * llama2 7b Q4: 76 tok/s (InsiderLLM canonical hardware database)
    * llama2 13b Q4: 35 tok/s (InsiderLLM canonical hardware database)
    * llama3 8b Q4: 51 tok/s (InsiderLLM canonical hardware database)
    * qwen35 9b think off: 47.1 tok/s (InsiderLLM canonical hardware database)
    * qwen35 9b think on: 46.6 tok/s (InsiderLLM canonical hardware database)
    * deepseek r1 14b think off: 35.6 tok/s (InsiderLLM canonical hardware database)
    * deepseek r1 14b think on: 35.1 tok/s (InsiderLLM canonical hardware database)
* **NVIDIA RTX 3090 (24GB):**
    * llama3 8b Q4: 112 tok/s (InsiderLLM canonical hardware database)
    * llama3 8b F16: 47 tok/s (InsiderLLM canonical hardware database)
    * llama3 70b Q4: 16 tok/s (InsiderLLM canonical hardware database)
    * mistral 7b Q6: 85 tok/s (InsiderLLM canonical hardware database)
    * gemma3 27b: 39.9 tok/s (InsiderLLM canonical hardware database)
* **NVIDIA RTX 4060 Ti 16GB:**
    * llama3 8b Q4: 48 tok/s (InsiderLLM canonical hardware database)
    * llama2 7b Q4: 64 tok/s (InsiderLLM canonical hardware database)
* **AMD Radeon RX 7800 XT (16GB):**
    * llama3 8b Q4: 39 tok/s (InsiderLLM canonical hardware database)
    * llama2 7b Q4: 96 tok/s (InsiderLLM canonical hardware database)
* **AMD Radeon RX 7900 XT (20GB):**
    * llama2 7b Q4: 116 tok/s (InsiderLLM canonical hardware database)
    * llama2 7b Q4 sustained: 97 tok/s (InsiderLLM canonical hardware database)
* **NVIDIA RTX 3080 10GB:**
    * llama3 8b Q4: 106 tok/s (InsiderLLM canonical hardware database)
* **NVIDIA RTX 3080 12GB:**
    * llama3 8b Q4: 107 tok/s (InsiderLLM canonical hardware database)
* **NVIDIA RTX 3070:**
    * llama3 8b Q4: 71 tok/s (InsiderLLM canonical hardware database)

## Key Specs

* **NVIDIA RTX 3060 (12GB):** VRAM: 12GB GDDR6, TDP: 170W, Architecture: Ampere (InsiderLLM canonical hardware database)
* **NVIDIA RTX 3090 (24GB):** VRAM: 24GB GDDR6X, TDP: 350W, Architecture: Ampere (InsiderLLM canonical hardware database)
* **NVIDIA RTX 4060 Ti 16GB:** VRAM: 16GB GDDR6, TDP: 165W, Architecture: Ada Lovelace (InsiderLLM canonical hardware database)
* **NVIDIA RTX 5060 Ti 16GB:** VRAM: 16GB GDDR7, TDP: 165W, Architecture: Blackwell (InsiderLLM canonical hardware database)
* **AMD Radeon RX 9070 (16GB):** VRAM: 16GB GDDR6, TDP: 263W, Architecture: RDNA 4 (InsiderLLM canonical hardware database)

## Competitor Coverage

NO DATA FOUND

## Internal Context

* InsiderLLM has published articles covering VRAM requirements for different model sizes, used RTX 3090 buying guide, and budget AI PC builds. (D3, D4, D5)
* Meta description A/B tests show some CTR improvement with more detailed descriptions focusing on model sizes and benchmarks. (D1, D2)
* Focus is on budget-friendly local AI for hobbyists and developers. (D3)

## Gaps

* **CRITICAL:** Benchmarks for RTX 5060 and RX 9070 are missing.  Need to find performance data for key LLMs.
* Benchmarks for RTX 4060 Ti 16GB with a wider range of models are missing.
* AMD ROCm compatibility details for specific models and frameworks are missing.
* Detailed analysis of CPU offloading performance with different RAM configurations (DDR3, DDR4, DDR5) beyond the general statements provided.
* Information on power supply requirements for each GPU.
* Availability and expected pricing for RTX 5060 and RX 9070.
* Real-world usage scenarios and example projects for each GPU.

## Suggested Angle

InsiderLLM should focus on *practical* advice for building a functional local AI setup *at each price point*.  Rather than simply listing specs, the article should guide users through the trade-offs between VRAM, bandwidth, and CPU offloading.  Highlight the "sweet spot" GPUs (like the RTX 3060 12GB and RTX 3080 12GB) that offer the best value for specific model sizes.  Emphasize *what you can actually do* with each GPU, rather than just listing benchmark numbers. Given the existing content around VRAM requirements, this article should directly tie GPU recommendations to those requirements. We should also address the AMD ROCm situation honestly—highlighting the potential benefits but also the risks and need for verification.