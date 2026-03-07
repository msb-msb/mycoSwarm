# Research Bundle: Local LLM Inference, Quantization Security, and Fine-Tuning Strategies

## 1. Executive Summary
This bundle synthesizes findings on three critical pillars of modern local LLM deployment: **inference engine performance** (vLLM vs. llama.cpp), **security vulnerabilities in quantization** (specifically the GGUF format), and **cost-effective fine-tuning methodologies** for code generation models.

Key takeaways include:
*   **Performance:** vLLM significantly outperforms llama.cpp in high-concurrency scenarios (16+ parallel requests) due to continuous batching, though llama.cpp remains competitive or superior in single-request latency on specific hardware configurations.
*   **Security:** A novel attack vector ("Mind the Gap") demonstrates that GGUF quantization is susceptible to adversarial injection, where models appear benign in full precision but trigger malicious behavior (e.g., insecure code generation) once quantized.
*   **Fine-Tuning:** Parameter-Efficient Fine-Tuning (PEFT/LoRA) is essential for cost-effective adaptation of large code models (e.g., StarCoder), while Full Fine-Tuning requires massive multi-GPU infrastructure (FSDP).

---

## 2. Inference Engine Performance: vLLM vs. llama.cpp

### Context
Benchmarking was conducted on a single NVIDIA RTX 4090 (frequency-limited to 1350 MHz) using the **Qwen 2.5 Instruct 3B** model. The study compared `vllm` (BF16, no FlashInfer) against `llama.cpp` (FP16, `-fa` enabled).

### Methodology
*   **Workload:** Parallel requests ranging from 1 to 16 concurrent users.
*   **Variables:** Prompt tokens (2k–30k), Generation tokens (256, 512, 768, 1024).
*   **Metrics:** Runtime in seconds, fitted to a linear model separating base token costs from context-depth penalties.

### Key Findings

#### A. Single Request Latency (1 Parallel Request)
*   **Result:** `llama.cpp` and `vLLM` are nearly equivalent.
*   **Data:** `llama.cpp` required **93.6% to 100.2%** of the time taken by `vLLM`. In many cases, `llama.cpp` was slightly faster (e.g., -4.7% difference at 2k prompt tokens).
*   **Analysis:** For single-user or low-latency use cases, `llama.cpp` offers a robust, low-overhead alternative that matches `vLLM` performance on consumer hardware.

#### B. High Concurrency (16 Parallel Requests)
*   **Result:** `vLLM` demonstrates superior throughput and scalability.
*   **Data:** `vLLM` required only **79% to 98%** of the time taken by `llama.cpp`. Conversely, `llama.cpp` took **20% to 30% longer** than `vLLM` under load.
    *   *Example:* At 16 parallel requests with 2k prompt tokens and 256 gen tokens:
        *   vLLM: 215.3s
        *   llama.cpp: 265.5s (+23.3% latency)
*   **Analysis:** `vLLM`'s continuous batching mechanism allows it to handle concurrent requests efficiently, whereas `llama.cpp` (in its default server configuration) scales less effectively under high concurrency without specific tuning.

#### C. Optimization Recommendations
*   **For llama.cpp:** Performance on 16 parallel requests could be improved by:
    *   Moving samplers (top-k, top-p, min-p) into the `ggml` graph to reduce candidate filtering overhead.
    *   Increasing operation fusion.
    *   Enabling FP16/BF16 for the `ggml` graphs (currently limited support in some builds).

---

## 3. Security Vulnerability: The "Mind the Gap" Attack on GGUF

### Context
Post-training quantization (PTQ) is standard for deploying large models on resource-constrained devices. While previous research identified security risks in simple rounding-based quantization, the widely used **GGUF format** (used by `llama.cpp`, `ollama`) was assumed to be more secure due to its complexity.

### The Attack Vector
*   **Source:** ICML 2025 Poster "Mind the Gap: A Practical Attack on GGUF Quantization".
*   **Mechanism:** The attack exploits the **quantization error** (the difference between full-precision weights and quantized weights). This error provides sufficient flexibility to construct a malicious model.
    *   The attacker trains a target LLM with specific constraints based on quantization errors.
    *   The resulting model behaves benignly in full precision (passing safety checks).
    *   Once quantized to GGUF, the hidden malicious behavior is triggered.

### Impact Metrics
The attack was demonstrated across three popular LLMs and nine GGUF data types with high success rates:
1.  **Insecure Code Generation:** $\Delta = 88.7\%$ (Attack success rate).
2.  **Targeted Content Injection:** $\Delta = 85.0\%$.
3.  **Benign Instruction Refusal:** $\Delta = 30.1\%$ (Bypassing safety refusals).

### Conclusion on Security
*   Complexity of the quantization scheme is **not** a sufficient defense against adversarial interference.
*   Users cannot rely on testing a full-precision model to guarantee safety after quantization.

---

## 4. Fine-Tuning Strategies for Code LLMs

### Objective
Fine-tune code generation models (e.g., StarCoder, DeciCoder) for "Fill In The Middle" (FIM) tasks using cost-effective methods.

### Model Selection & Resources
*   **Models Tested:**
    *   `bigcode/starcoder` (15.5B params)
    *   `bigcode/starcoderbase-1b` (1B params)
    *   `Deci/DeciCoder-1b` (1B params)
*   **Hardware:**
    *   *PEFT Experiments:* Single A100 40GB GPU.
    *   *Full Fine-Tuning:* 8x A100 80GB GPUs (using 🤗 Accelerate FSDP).

### Methodology: PEFT vs. Full Fine-Tuning
*   **Why PEFT?** Full fine-tuning is prohibitively expensive for most users.
    *   *Memory Cost Calculation:* For a 15.5B model, full fine-tuning requires ~248GB of GPU memory just for weights/gradients/optimizer states (16 bytes/param), excluding activations. This necessitates at least 4x A100 80GB GPUs.
    *   *PEFT Solution:* Uses LoRA or similar adapters to update a small subset of parameters, reducing VRAM requirements significantly and enabling training on single consumer/prosumer GPUs.

### Dataset Preparation
*   **Source:** Top 10 Hugging Face repositories by star count (e.g., `transformers`, `pytorch-image-models`, `diffusers`).
*   **Extraction Strategy:**
    *   Local cloning of repos (to avoid API rate limits).
    *   Parallel downloading via Python `multiprocessing`.
    *   **Filtering:** Excluded non-code files (images, presentations), `.git`, `__pycache__`, `xcodeproj`.
    *   **Parsing:** UTF-8 for standard files; code-cell extraction for Jupyter Notebooks.
    *   **Serialization:** Chunked Feather format for memory efficiency.
*   **Note:** Deduplication was not applied in this baseline to reduce complexity, though recommended for production.

---

## 5. Recommendations & Best Practices

| Domain | Recommendation | Rationale |
| :--- | :--- | :--- |
| **Inference** | Use **vLLM** for production serving with high concurrency. | Superior throughput and latency scaling under load compared to llama.cpp. |
| **Inference** | Use **llama.cpp** for single-user, low-latency, or edge scenarios. | Competitive performance on single requests; lower dependency footprint. |
| **Security** | **Do not assume safety** of quantized models based on full-precision testing. | GGUF format is vulnerable to "Mind the Gap" attacks that hide malicious logic in quantization errors. |
| **Fine-Tuning** | Prioritize **PEFT (LoRA)** for code adaptation. | Reduces VRAM requirements from ~250GB+ to single GPU levels; sufficient for FIM tasks. |
| **Data Prep** | Implement strict file filtering and chunked serialization. | Reduces noise (non-code assets) and memory overhead during dataset creation. |

## 6. References & Sources
1.  **ICML 2025:** *Mind the Gap: A Practical Attack on GGUF Quantization* (Egashira et al.).
2.  **GitHub Discussion:** `ggml-org/llama.cpp` #15180 (Performance benchmarks by JohannesGaessler).
3.  **Hugging Face Blog:** Fine-tuning personal Co-Pilots using StarCoder and PEFT.
4.  **Technical Reports:** Comparisons of vLLM, Ollama, and llama.cpp for local LLM deployment (2025-2026 context).