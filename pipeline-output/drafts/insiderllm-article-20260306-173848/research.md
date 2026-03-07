# Research Bundle: Local LLM Quantization, Formats, and Code Models (2025-2026)

## 1. Executive Summary
This bundle synthesizes current best practices for running Large Language Models (LLMs) locally as of late 2025/early 2026. The primary focus is on **quantization** as the critical enabler for local deployment, balancing VRAM constraints with inference speed and model quality.

*   **Key Insight:** GGUF is the dominant universal format for local users (CPU/GPU/Mixed), while GPTQ/AWQ/EXL2 are specialized for NVIDIA GPU throughput or specific serving scenarios.
*   **Code Models:** The landscape has shifted toward open-source coding specialists, with **Qwen2.5-Coder** series leading performance benchmarks, challenging closed-source giants like o1 and Claude in specific coding tasks.
*   **Trend:** Quantization at 4-bit (Q4_K_M) is the practical "sweet spot" for most use cases, offering near-baseline quality with massive VRAM savings.

---

## 2. Quantization & Format Decision Matrix

### 2.1 The Core Question: Which Format to Use?
The choice of format depends entirely on the **hardware**, **software stack**, and **deployment goal**.

| User Scenario | Recommended Format | Primary Software/Engine | Rationale |
| :--- | :--- | :--- | : |
| **Ollama / LM Studio Users** | **GGUF** | llama.cpp, Ollama, LM Studio | The *only* supported format for these tools. Single-file convenience. |
| **CPU or Apple Silicon (M1/M2/M3)** | **GGUF** | llama.cpp, MLX | Only viable option for non-NVIDIA hardware; supports mixed precision. |
| **AMD ROCm Users** | **GGUF** | llama.cpp | Limited/no support for other formats on AMD currently. |
| **NVIDIA (Max Personal Speed)** | **EXL2** | ExLlamaV2, TabbyAPI | Best inference speed on consumer NVIDIA GPUs; supports mixed precision (2-8 bit). |
| **Multi-User Serving / Production** | **GPTQ** | vLLM (Marlin kernels) | Optimized for high throughput and concurrency via Marlin kernels. |
| **Best Quality @ 4-bit** | **AWQ** or **EXL2** | Transformers, ExLlamaV2 | AWQ often preserves higher fidelity at low bitrates; EXL2 offers flexibility. |
| **Undecided / General Purpose** | **GGUF (Q4_K_M)** | llama.cpp | "Works everywhere," competitive quality, easy to switch later. |

### 2.2 Format Deep Dive & Compatibility

#### **GGUF (`.gguf`)**
*   **Type:** File format & Container.
*   **Status:** The industry standard for local inference. Replaced GGML in Aug 2023.
*   **Pros:** Single file containing weights + metadata; works on CPU/GPU/Mixed; supported by Ollama, LM Studio, llama.cpp.
*   **Cons:** Slightly slower than optimized CUDA kernels (EXL2/GPTQ) on high-end NVIDIA cards for pure speed.
*   **Quantization Levels:** 1-bit to 8-bit mixed precision (e.g., Q4_K_M, Q5_K_S).

#### **GPTQ (`safetensors`)**
*   **Type:** Quantization Method (stored in HuggingFace `safetensors`).
*   **Status:** Standard for NVIDIA GPU inference.
*   **Pros:** Extremely fast on NVIDIA via Marlin kernels; widely supported by vLLM.
*   **Cons:** Requires NVIDIA GPUs; calibration dataset needed for conversion; less flexible than GGUF for CPU fallback.

#### **AWQ (`safetensors`)**
*   **Type:** Quantization Method (stored in HuggingFace `safetensors`).
*   **Status:** Strong contender for quality preservation at 4-bit.
*   **Pros:** Excellent quality retention; compatible with transformers and vLLM.
*   **Cons:** Primarily NVIDIA-focused; requires calibration.

#### **EXL2 / EXL3 (`safetensors`)**
*   **Type:** Quantization Method (stored in HuggingFace `safetensors`).
*   **Status:** Emerging leader for raw speed on consumer NVIDIA hardware.
*   **Pros:** Supports mixed precision (e.g., 4.65 bpw); fastest inference speeds reported for local setups; EXL3 is a QTIP variant.
*   **Cons:** Requires specific Python dependencies (ExLlamaV2); less "plug-and-play" than GGUF for non-technical users.

#### **BitsAndBytes (`safetensors`)**
*   **Type:** On-the-fly quantization (HuggingFace `load_in_4bit`).
*   **Status:** Good for testing, bad for production.
*   **Performance:** ~23 tokens/sec vs 31-64 tokens/sec for pre-quantized formats on same hardware.

#### **Legacy / Deprecated**
*   **GGML:** Deprecated Aug 2023. Use GGUF instead.
*   **`.bin` (Pickle):** Legacy PyTorch format. Security risk (arbitrary code execution). Replaced by SafeTensors.
*   **SafeTensors:** Not a quantization format itself, but the secure container for *unquantized* weights and the storage medium for GPTQ/AWQ/EXL2 weights.

---

## 3. Conversion & Implementation Guide

Most users should download pre-quantized models from HuggingFace (e.g., TheBloke, bartowski). However, conversion is possible:

1.  **FP16 SafeTensors → GGUF**
    *   *Tool:* `llama.cpp` convert script.
    *   *Command:* `python convert_hf_to_gguf.py /path/to/model --outtype f16` then `./llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M`.
2.  **FP16 SafeTensors → GPTQ**
    *   *Tool:* `AutoGPTQ`.
    *   *Requirement:* Calibration dataset required for optimal results.
3.  **FP16 SafeTensors → AWQ**
    *   *Tool:* `AutoAWQ`.
4.  **FP16 SafeTensors → EXL2**
    *   *Tool:* ExLlamaV2 `convert.py`.
    *   *Command:* `python convert.py -i /path/to/model -o /output -cf /output/4.65bpw -b 4.65`.

---

## 4. Code-Specific LLM Landscape (2025-2026)

The "Awesome Code LLM" repository and benchmarks indicate a shift where open-source models are closing the gap with closed-source giants, particularly in code generation.

### 4.1 Top Performing Code Models
Based on HumanEval Pass@1 and MBPP metrics:

| Rank | Model | Params | HumanEval | MBPP | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | o1-mini (2024-09) | - | 97.6 | 93.9 | Paper |
| **2** | o1-preview (2024-09) | - | 95.1 | 93.4 | Paper |
| **3** | **Qwen2.5-Coder-32B-Instruct** | **32B** | **92.7** | **90.2** | **GitHub** |
| **4** | Claude-3.5-Sonnet | - | 92.1 | 91.0 | Paper |
| **6** | **Qwen2.5-Coder-14B-Instruct** | **14B** | **89.6** | **86.2** | **GitHub** |
| **9** | **Qwen2.5-Coder-7B-Instruct** | **7B** | **88.4** | **83.5** | **GitHub** |

*   **Key Takeaway:** **Qwen2.5-Coder** (0.5B to 32B) is the dominant open-source family, with the 32B variant outperforming many closed models in code generation.
*   **Other Notables:** DeepSeek-Coder-V2, StarCoder2, CodeLlama-70B, and OpenHands-LM (32B).

### 4.2 Evaluation & Benchmarking
*   **HumanEval/MBPP:** Standard metrics for code generation correctness.
*   **LiveCodeBench:** Holistic, contamination-free evaluation.
*   **SWE-bench:** Evaluates ability to resolve real-world GitHub issues (complex reasoning).
*   **EvalPlus:** Rigorous testing of generated code correctness.

### 4.3 Hardware & Deployment for Code Models
*   **Local Optimization:** For coding tasks, 4-bit quantization is the practical limit, maintaining 88-95% of baseline performance. Heavier quantization (2-bit) often degrades code quality significantly.
*   **Recommended Local Stack:**
    *   **Ollama:** Best for quick deployment of Qwen2.5-Coder variants (e.g., `ollama run qwen2.5-coder:7b`).
    *   **LM Studio:** Good for GUI-based testing and comparison.
    *   **vLLM:** Ideal if serving a coding assistant API to multiple developers.

---

## 5. Practical Tips & Constraints

1.  **VRAM vs. Context Length:** Quantization allows larger context windows. A 7B model requiring 28GB VRAM at FP16 can run at 4-bit in ~4GB, enabling massive context expansion on consumer GPUs.
2.  **Diminishing Returns:** For large models, heavily quantizing (below 4-bit) yields diminishing returns on code quality. Stick to Q4_K_M or Q5_K_S for coding tasks.
3.  **Security:** Always prefer `safetensors` or `GGUF`. Avoid loading `.bin` files from untrusted sources due to pickle execution risks.
4.  **Serving Strategy:**
    *   *Single User:* EXL2 (NVIDIA) or GGUF (Universal).
    *   *Multi-User:* GPTQ with vLLM.
5.  **Future Proofing:** The ecosystem is moving toward "one file" simplicity (GGUF) for general users, while specialized high-performance setups continue to use EXL2/GPTQ.

## 6. References & Sources
*   *Hardware Corner / InsiderLLM:* Quantization formats decision trees and compatibility matrices (Oct 2025).
*   *huybery/Awesome-Code-LLM:* Comprehensive GitHub repository tracking code LLMs, papers, and leaderboards (Nov 2024 updates).
*   *Practical Web Tools:* Ollama model guide and benchmarks (Dec 2025).
*   *HuggingFace:* Technical guides on quantization theory and conversion scripts.
*   *CVPR 2021 / ICML 2023:* Foundational papers on optimal quantization algorithms and code generation evaluation.