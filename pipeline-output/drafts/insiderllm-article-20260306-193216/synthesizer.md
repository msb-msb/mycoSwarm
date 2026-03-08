# Research Bundle: Local Multilingual Code Generation & Inference Infrastructure (2026)

## 1. Executive Summary
This bundle synthesizes data regarding **CodeGeeX** (a multilingual code generation model), its evolution to version 4, and the current landscape of local inference engines (**Ollama**, **vLLM**) as of early 2026. The analysis highlights a critical trade-off in the developer ecosystem: while larger models (e.g., Qwen Coder 480B) offer high initial quality, they suffer from "hallucination propagation" where errors compound quickly. Conversely, smaller models or specific quantization strategies are preferred for speed and stability, though they may lack the reasoning depth of massive architectures. The technical infrastructure has shifted heavily toward **vLLM** for high-throughput serving, offering significantly higher token generation speeds compared to **Ollama**, which remains a strong contender for ease of use and lower resource overhead.

## 2. Core Asset: CodeGeeX Model Family
### Overview
*   **Developer:** ZAI Team (THUDM).
*   **Current Version:** CodeGeeX4 (Released early 2026), following the release of CodeGeeX2 in mid-2023.
*   **Architecture:** Large-scale multilingual code generation model.
*   **Original Scale:** 13 billion parameters (CodeGeeX 1).
*   **Training Data:** Pre-trained on >850 billion tokens across >20 programming languages (Python, C++, Java, JS, Go, etc.).
*   **Hardware Support:** Originally Ascend 910 AI Processors; now supports NVIDIA platforms (V100, A100) via PyTorch/Megatron-LM compatibility.

### Key Features & Capabilities
1.  **Multilingual Code Generation:** Generates executable code in mainstream languages with high accuracy.
2.  **Crosslingual Translation:** Translates code snippets between languages (e.g., Python to Java) with high fidelity.
3.  **IDE Integration:** Available as a free extension for VS Code, JetBrains IDEs (IntelliJ, PyCharm, etc.), and Cloud Studio.
4.  **Quantization & Parallelism:** Supports INT8 quantization (reducing VRAM from ~27GB to ~15GB) and model parallelism for multi-GPU inference with <8GB RAM per card.

### Benchmarking: HumanEval-X
*   **Purpose:** A standardized benchmark for multilingual code generation and translation.
*   **Composition:** 820 human-crafted coding problems in 5 languages (Python, C++, Java, JavaScript, Go).
*   **Performance:** CodeGeeX achieves the highest average performance among open-sourced multilingual baselines on this benchmark.

## 3. Inference Engine Landscape (2026)
### vLLM vs. Ollama: Performance Comparison
As of January 2026, **vLLM** is the dominant choice for high-throughput production and local serving of large coding models, significantly outperforming Ollama in raw throughput.

| Feature | vLLM | Ollama |
| :--- | :--- | :--- |
| **Throughput** | ~840 tokens/sec (DeepSeek model, 50 concurrent) | ~142 tokens/sec (DeepSeek model, 50 concurrent) |
| **Relative Speed** | Baseline (High Performance) | ~70% lower throughput than vLLM |
| **Primary Use Case** | High-concurrency serving, production, large batch inference. | Local development, single-user interactive use, ease of setup. |
| **Model Support** | Extensive (Transformers compatible, custom implementations). | Broad but limited to supported community models. |
| **Quantization** | Supports FP8, INT4, AWQ, GPTQ via `bitsandbytes`/`llm-compressor`. | Native GGUF support; easy quantization workflow. |

### User Sentiment & Model Selection (Reddit r/LocalLLaMA)
*   **The "Junk Code" Problem:** Users report that while massive models like **Qwen Coder 480B** (running at 4-bit) are excellent for initial drafts, they are prone to hallucinations. Once an error occurs in a large context, the model tends to compound the mistake, filling the codebase with "junk."
*   **Model Recommendations:**
    *   **Top Tier (Quality):** Llama 3.1 70B (45% HumanEval), DeepSeek V2 16B, Qwen Coder 32B.
    *   **Efficiency/Speed:** Smaller models (e.g., 7B-16B) are often preferred for speed and stability over "brain dead" 30B+ models if the user lacks time for extensive iteration.
    *   **Hardware Constraints:** A 512GB Mac Studio can run large models, but users prioritize models that fit comfortably to avoid swapping or latency issues.

## 4. Technical Implementation Guide

### Installation & Setup (CodeGeeX)
*   **Prerequisites:** Python 3.7+, CUDA 11+, PyTorch 1.10+, DeepSpeed 0.6+.
*   **Installation:**
    ```bash
    git clone git@github.com:THUDM/CodeGeeX.git
    cd CodeGeeX
    pip install -e .
    # Or use Docker
    docker pull codegeex/codegeex:latest
    docker run --gpus '"device=0,1"' -it --ipc=host --name=codegeex codegeex/codegeex
    ```
*   **Model Weights:** Available via Hugging Face or direct download link (requires `urls.txt` and `aria2c`). Total size ~26GB for the 13B checkpoint.

### Inference Strategies
1.  **Single GPU (Standard):** Requires >27GB VRAM.
    ```bash
    bash ./scripts/test_inference.sh <GPU_ID> ./tests/test_prompt.txt
    ```
2.  **Quantized (Optimized):** Reduces VRAM to ~15GB.
    ```bash
    bash ./scripts/test_inference_quantized.sh <GPU_ID> ./tests/test_prompt.txt
    ```
3.  **Multi-GPU (Parallelism):** Requires >6GB VRAM per card; requires checkpoint conversion (`convert_ckpt_parallel.sh`).

### vLLM Integration
*   **Compatibility:** Supports Hugging Face Transformers models with `trust_remote_code=True`.
*   **Performance:** Offers ~5% performance delta compared to dedicated implementations.
*   **Features:** Automatic Prefix Caching, Speculative Decoding (EAGLE, MTP), and OpenAI-compatible API.

## 5. Strategic Recommendations for Developers

1.  **For Local Development (IDE Plugins):** Continue using the **CodeGeeX VS Code/JetBrains extension**. It provides immediate feedback on completion, explanation, and summarization without heavy local compute requirements.
2.  **For High-Throughput Local Serving:** Deploy **vLLM** rather than Ollama if running coding models like DeepSeek or Qwen Coder. The ~6x throughput increase is critical for batch processing or multi-user scenarios.
3.  **Model Selection Strategy:**
    *   Avoid relying solely on massive models (e.g., 480B) for autonomous code generation without human-in-the-loop verification, as they propagate errors rapidly.
    *   Consider a hybrid approach: Use a smaller, faster model (e.g., 16B-32B) for initial generation and a larger model or LLM-based linter for validation/refinement.
4.  **Hardware Optimization:** Utilize **INT8 quantization** for CodeGeeX on consumer-grade GPUs (e.g., RTX 3090/4090) to achieve <15ms/token speeds while maintaining acceptable accuracy, reducing VRAM requirements from 27GB to 15GB.

## 6. References & Resources
*   **CodeGeeX Repository:** `zai-org/CodeGeeX` (GitHub)
*   **HumanEval-X Benchmark:** Official repository for multilingual code evaluation.
*   **vLLM Documentation:** `docs.vllm.ai` (Supported models, quantization guides).
*   **Community Insights:** r/LocalLLaMA discussions on 2026 model performance (Qwen Coder vs. Llama 3.1).
*   **License:** Apache-2.0 (CodeGeeX); vLLM and Ollama have respective open-source licenses.