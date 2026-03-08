# Research Bundle: Best Open Source Coding Models for Local Development (2026)

## Key Facts & Data Points

**Model Performance & Rankings (Source: Onyx AI Leaderboard, Feb 2026)**
*   **Kimi K2.5 (Moonshot):** 1T parameters (32B active), 262k context. Leads with 96.1 HumanEval and 87.1 MMLU-Pro. [Source: https://onyx.app/open-llm-leaderboard]
*   **GLM-4.7 (Zhipu AI):** 355B parameters, 200k context. Top performer in SWE-bench Verified (73.8) and MATH-500 (95.7). [Source: https://onyx.app/open-llm-leaderboard]
*   **Qwen 3.5 (Alibaba):** 397B parameters, 262k context. Strong in general reasoning (87.8 MMLU-Pro) and SWE-bench (83.6). [Source: https://onyx.app/open-llm-leaderboard]
*   **DeepSeek R1:** 671B parameters. High performance in AIME 2025 (87.5) and HumanEval (90.2). [Source: https://onyx.app/open-llm-leaderboard]
*   **Llama 4 Maverick:** 400B parameters, 1M context. Lower coding scores (HumanEval 43.4) compared to specialized coding models but high general reasoning. [Source: https://onyx.app/open-llm-leaderboard]
*   **Gemma 3:** 27B parameters. Strong HumanEval (89.0) for its size class. [Source: https://onyx.app/open-llm-leaderboard]

**Deployment Tooling & Best Practices (Source: Ollama Blog, Clawdbook)**
*   **Top Recommended Models for Local Coding:** Qwen3-Coder Series (32B/72B/480B), GLM-4.7 Flash, GPT-OSS 20B, DeepSeek R1/Coder-V2. [Source: https://clawdbook.org/en/blog/openclaw-best-ollama-models-2026]
*   **Tool Calling Stability:** Qwen3-Coder is noted for "extremely stable tool calling" and rarely hallucinating parameters. [Source: https://clawdbook.org/en/blog/openclaw-best-ollama-models-2026]
*   **Optimization Strategy:** Use `q4_K_M` or `q5_K_M` quantization for best speed/accuracy balance. Set Temperature to 0–0.2 for tool calling tasks. [Source: https://clawdbook.org/en/blog/openclaw-best-ollama-models-2026]
*   **Dual-Model Strategy:** Recommended combo is `qwen3-coder:32b` (main) + `glm-4.7-flash` (backup). [Source: https://clawdbook.org/en/blog/openclaw-best-ollama-models-2026]

**Quantization & Hardware Efficiency (Source: Unsloth, Synthmetric)**
*   **Qwen3-Coder-30B:** Requires ~18GB unified memory for dynamic 4-bit quantization to achieve >6 tok/s. Full precision (UD_Q8) requires ~33GB. [Source: https://unsloth.ai/docs/models/tutorials/qwen3-coder-how-to-run-locally]
*   **Qwen3-Coder-480B:** Requires 180GB unified memory for optimal speed with Q2_K_XL (1-bit) quantization, achieving >6 tok/s. [Source: https://unsloth.ai/docs/models/tutorials/qwen3-coder-how-to-run-locally]
*   **Quantization Impact:** INT4 reduces model size by ~8x vs FP32 but introduces discretization error; per-channel quantization recommended for weights. [Source: https://synthmetric.com/quantization-in-plain-english-8‑bit-4‑bit-and-what-you-lose/]
*   **Memory Bandwidth Rule:** For models fitting in VRAM, double bandwidth ≈ double tok/s. DDR5 offloading is ~37x slower than GDDR6X per layer. [Source: InsiderLLM Canonical Database]

**Licensing Landscape (Source: HuggingFace Blog, Local-AI-Zone)**
*   **Permissive Licenses:** Qwen 3, Mixtral 8x22B, DeepSeek-V3, Grok-1, and Gemma 2 allow commercial use and fine-tuning. [Source: https://huggingface.co/blog/daya-shankar/open-source-llms]
*   **Restrictive Licenses:** Llama 4/3.3 (Community License) and Command R+ (CC-BY-NC) have usage restrictions or non-commercial defaults requiring commercial licenses for business use. [Source: https://huggingface.co/blog/daya-shankar/open-source-llms]
*   **Definition:** "Open weights" does not always equal OSI-compliant open source; check for "Acceptable Use Policies" (e.g., military restrictions, non-commercial clauses). [Source: https://local-ai-zone.github.io/guides/ai-model-licensing-complete-legal-guide-2025.html]

## Price Data

**NVIDIA GPUs (Used Market - eBay Sold Auctions)**
*   **GTX 1660 Super (6GB):** $90–$120 (Typical: $105). [Source: InsiderLLM Canonical Database]
*   **RTX 2060 12GB:** $140–$180 (Typical: $160). [Source: InsiderLLM Canonical Database]
*   **RTX 3060 12GB:** $170–$380 (Typical: $275). [Source: InsiderLLM Canonical Database]
*   **RTX 3070 (8GB):** $210–$300 (Typical: $255). [Source: InsiderLLM Canonical Database]
*   **RTX 3070 Ti (8GB GDDR6X):** $100–$280 (Typical: $190). [Source: InsiderLLM Canonical Database]
*   **RTX 3080 10GB:** $325–$400 (Typical: $365). [Source: InsiderLLM Canonical Database]
*   **RTX 3080 12GB:** $230–$380 (Typical: $305). [Source: InsiderLLM Canonical Database]
*   **RTX 3090 (24GB):** $950–$1,125 (Typical: $1,040). [Source: InsiderLLM Canonical Database]
*   **RTX 4060 (8GB):** $230–$310 (Typical: $270). [Source: InsiderLLM Canonical Database]
*   **RTX 4060 Ti 8GB:** $240–$300 (Typical: $270). [Source: InsiderLLM Canonical Database]
*   **RTX 4060 Ti 16GB:** $380–$480 (Typical: $430). [Source: InsiderLLM Canonical Database]

**AMD GPUs (Used Market - eBay Sold Auctions)**
*   **RX 7600 (8GB):** $170–$225 (Typical: $200). [Source: InsiderLLM Canonical Database]
*   **RX 7700 XT (12GB):** $300–$350 (Typical: $325). [Source: InsiderLLM Canonical Database]
*   **RX 7800 XT (16GB):** $380–$550 (Typical: $465). [Source: InsiderLLM Canonical Database]
*   **RX 7900 GRE (16GB):** $400–$550 (Typical: $475). [Source: InsiderLLM Canonical Database]
*   **RX 7900 XT (20GB):** $500–$700 (Typical: $600). [Source: InsiderLLM Canonical Database]

**Future/Unreleased Pricing (MSRP Estimates)**
*   **RTX 5070:** MSRP $549. [Source: InsiderLLM Canonical Database]
*   **RX 9070:** MSRP $549. [Source: InsiderLLM Canonical Database]
*   **RX 9070 XT:** MSRP $599. [Source: InsiderLLM Canonical Database]

## Synthesis & Analysis

### 1. Model Selection Strategy for Local Coding
The optimal choice depends on hardware constraints and the specific need for tool-calling stability versus general reasoning.
*   **For High-Stability Tool Calling:** The **Qwen3-Coder** series (specifically the 32B or 72B variants) is the consensus leader. It demonstrates exceptional reliability in function calling, rarely hallucinating parameters, which is critical for agents like OpenClaw. [Source: https://clawdbook.org/en/blog/openclaw-best-ollama-models-2026]
*   **For General Coding & Reasoning:** **GLM-4.7** and **Qwen 3.5** offer the highest scores on SWE-bench Verified and MMLU-Pro, making them superior for complex problem-solving and multi-step reasoning tasks. [Source: https://onyx.app/open-llm-leaderboard]
*   **For Small Hardware (8–16GB VRAM):** The **Gemma 3 (27B)** or **Qwen3-Coder (14B)** are viable entry points, though they may require careful prompt tuning and have higher loop rates. [Source: https://clawdbook.org/en/blog/openclaw-best-ollama-models-2026]

### 2. Hardware & Quantization Economics
Running large models locally requires balancing VRAM capacity with memory bandwidth.
*   **VRAM Requirements:** A "Sweet Spot" for high-performance local coding is **24–32GB VRAM**, which allows running `qwen3-coder:32b` or `glm-4.7` with Q4 quantization. Models like `qwen3:72b` require 48GB+. [Source: https://clawdbook.org/en/blog/openclaw-best-ollama-models-2026]
*   **Quantization Impact:** Moving from FP16 to INT4 (Q4_K_M) reduces model size by ~8x with minimal accuracy loss for most tasks, but requires careful calibration to avoid rare-token errors. [Source: https://synthmetric.com/quantization-in-plain-english-8‑bit-4‑bit-and-what-you-lose/]
*   **Bandwidth Bottleneck:** For models fitting entirely in VRAM, inference speed is linearly proportional to memory bandwidth (e.g., GDDR6X vs. DDR5). A GPU with 2x the bandwidth will yield ~2x tokens/second. [Source: InsiderLLM Canonical Database]
*   **Cost-Effective Build:** The **RTX 3090 (24GB)** remains the best value for high-end local AI, offering massive VRAM at a used price of ~$1,040. For budget builds, the **RTX 3060 (12GB)** is the minimum viable option for smaller models. [Source: InsiderLLM Canonical Database]

### 3. Licensing & Commercial Viability
*   **Safe for Commercial Use:** Models like **Qwen 3**, **DeepSeek-V3**, and **Gemma 2** are licensed under Apache-2.0 or similar permissive licenses, allowing commercial deployment and fine-tuning without additional fees. [Source: https://huggingface.co/blog/daya-shankar/open-source-llms]
*   **Caution Areas:** **Llama 4/3.3** and **Command R+** have restrictive clauses (e.g., non-commercial defaults or acceptable use policies). Users must verify the specific license file for their build before commercial deployment. [Source: https://huggingface.co/blog/daya-shankar/open-source-llms]
*   **"Open Source" Definition:** Be wary of "open weights" models that are not OSI-compliant open source due to usage restrictions (e.g., non-military clauses). True open source requires the freedom to use, modify, and distribute. [Source: https://local-ai-zone.github.io/guides/ai-model-licensing-complete-legal-guide-2025.html]

### 4. Implementation Best Practices
*   **Temperature:** Set to **0** or **0.1–0.2** for tool-calling tasks to minimize hallucinations. [Source: https://clawdbook.org/en/blog/openclaw-best-ollama-models-2026]
*   **Context:** Prioritize models with **32k+ context windows** (e.g., Qwen3, GLM-4.7) to handle large codebases and long prompts effectively. [Source: https://clawdbook.org/en/blog/openclaw-best-ollama-models-2026]
*   **Calibration:** Always run activation calibration with representative real-world data (not synthetic) to minimize quantization errors, especially for INT4/INT8 models. [Source: https://synthmetric.com/quantization-in-plain-english-8‑bit-4‑bit-and-what-you-lose/]

## Source URLs
*   **Model Leaderboards:** https://onyx.app/open-llm-leaderboard
*   **Deployment Guides (Ollama/LM Studio):** https://clawdbook.org/en/blog/openclaw-best-ollama-models-2026, https://ollama.com/blog/coding-models
*   **Quantization Technical Details:** https://synthmetric.com/quantization-in-plain-english-8‑bit-4‑bit-and-what-you-lose/, https://unsloth.ai/docs/models/tutorials/qwen3-coder-how-to-run-locally
*   **Licensing & Legal:** https://huggingface.co/blog/daya-shankar/open-source-llms, https://local-ai-zone.github.io/guides/ai-model-licensing-complete-legal-guide-2025.html
*   **Hardware Benchmarks:** InsiderLLM Canonical Database (eBay sold auctions)