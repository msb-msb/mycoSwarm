## Price Data

*   **RTX 3060 (12GB):** Used: $170-200 [D1, D1]
*   **RTX 3090 (24GB):** Used price not explicitly stated, but recommended as cost-effective when used. [D5]

## Benchmark Data

*   **RTX 3090 (24GB):** 1700 tokens/second [reddit.com]
*   **RTX 3090 (24GB):** 144.19 t/s (vs RTX 4090 at 170.63 t/s) [jan.ai]
*   **RTX 3060 (12GB):** Can run 13B/14B LLMs and Stable Diffusion [D1, D1] - *Specific performance numbers not provided.*
*   **ComfyUI (using GPU):** 8 seconds for SDXL at 9.2GB VRAM [D1] - *GPU not specified, but likely a high-end card.*

## Key Specs

*   **RTX 3090 (24GB):** VRAM: 24GB, TDP not specified. Architecture not specified.
*   **RTX 3060 (12GB):** VRAM: 12GB, TDP not specified. Architecture not specified.

## Competitor Coverage

NO DATA FOUND

## Internal Context

*   InsiderLLM is focused on budget-friendly local AI for hobbyists and developers [D3].
*   Recent articles cover GPU buying guides, VRAM requirements, used GPU recommendations, and price analysis [D5].
*   A build using a used Dell Optiplex + RTX 3060 12GB can run 13B/14B LLMs and Stable Diffusion for under $450 [D1].
*   Meta description A/B tests are ongoing for several related articles [D1, D2].

## Gaps

*   **CRITICAL:** RTX 3090 TDP.
*   **CRITICAL:** RTX 3060 new price.
*   RTX 3060 benchmark data for LLM inference (tokens/second, time to first token).
*   RTX 3090 benchmark data for different model sizes (7B, 13B, 70B) and quantization levels.
*   Specific performance data for the RTX 3060 running different LLM sizes.
*   TDP for RTX 3060.
*   Architecture for both GPUs.
*   Competitor coverage - what are other sites recommending for budget local AI GPUs and why?

## Suggested Angle

InsiderLLM should focus on *practical* build guides and performance benchmarks for real-world use cases. We’ve already published content on this, but can go deeper. While other sites may focus on the newest, most powerful GPUs, we should emphasize maximizing performance *within a limited budget*. Highlight the value of used GPUs like the RTX 3090 and the RTX 3060, and provide detailed, actionable advice on getting the most out of them. Specifically, focus on the optimal settings and software (like text-generation-webui) to achieve good performance even on lower-end hardware [D1]. Our content should be "no fluff" and geared towards hobbyists and developers who want to *actually build* and *use* local AI systems [D3].