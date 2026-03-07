# Research Bundle: GPU Compatibility & Performance for Local AI (2025–2026)

## 1. Executive Summary
The landscape for local AI inference has shifted significantly from a strict dependency on NVIDIA CUDA to a more diverse ecosystem supporting AMD ROCm, Intel oneAPI/SYCL, and cross-vendor standards like Vulkan and DirectML. While **NVIDIA remains the gold standard** for ease of use, performance consistency, and training capabilities, **AMD and Intel GPUs now offer viable, cost-effective alternatives** for inference workloads (LLMs, Stable Diffusion) provided users are willing to navigate specific software configurations.

Key trends include:
*   **VRAM is King:** For local LLMs and image generation, VRAM capacity often outweighs raw compute speed. High-end AMD cards (7900 series) offer superior price/VRAM ratios.
*   **Software Maturity:** ROCm on Windows has improved, but Linux remains the preferred OS for non-NVIDIA hardware.
*   **Inference vs. Training:** CUDA is still dominant for training/fine-tuning; however, inference performance gaps between vendors are narrowing due to quantization and optimized backends (llama.cpp, ONNX Runtime).

---

## 2. Hardware Landscape & Ecosystem Comparison

### A. NVIDIA (CUDA Ecosystem)
*   **Status:** Market leader, "Plug-and-Play."
*   **Strengths:**
    *   Mature software stack (cuBLAS, cuDNN, TensorRT-LLM).
    *   Highest single-GPU performance for both training and inference.
    *   Universal support across all local AI frameworks (Ollama, LM Studio, ComfyUI, etc.).
*   **Weaknesses:** Higher cost per GB of VRAM compared to AMD; proprietary closed-source nature limits flexibility.
*   **Best For:** Users prioritizing ease of setup, training/fine-tuning, and maximum performance stability.

### B. AMD (ROCm Ecosystem)
*   **Status:** Strong contender for budget/VRAM-heavy workloads.
*   **Strengths:**
    *   Excellent price-to-VRAM ratio (e.g., RX 7900 XT/XL series).
    *   Open-source stack (HIP, ROCm kernels).
    *   Linux support is robust; Windows support has improved significantly for consumer cards.
*   **Weaknesses:** Historically a Linux-first platform; Windows setup can require more tinkering; some specialized AI tools still lag in compatibility compared to CUDA.
*   **Best For:** Budget-conscious users needing high VRAM (16GB+), local LLM inference, and Stable Diffusion on Linux or configured Windows.

### C. Intel (oneAPI / SYCL / IPEX-LLM)
*   **Status:** Emerging budget solution.
*   **Strengths:**
    *   Cost-effective entry point for AI (Arc B580/A770).
    *   Cross-architecture standard (SYCL) allows code to run on CPU, GPU, and NPU.
    *   Integrated graphics/NPUs offer low-power inference options.
*   **Weaknesses:** Youngest ecosystem; software maturity is lower than NVIDIA/AMD; optimization for specific AI tasks is still ongoing.
*   **Best For:** Budget builds, testing cross-platform capabilities, and users invested in the Intel hardware ecosystem.

### D. Cross-Vendor Standards (Vulkan / DirectML)
*   **Status:** The "Universal" path.
*   **Mechanism:** Backends like `llama.cpp` (Vulkan), `ONNX Runtime` (DirectML), and `Metal` (Apple Silicon) allow LLMs to run on non-NVIDIA hardware without vendor-specific drivers.
*   **Performance:** Competitive for inference, especially with quantized models (GGUF/GPTQ), though often slightly slower than native CUDA/ROCm due to abstraction overhead.

---

## 3. Recommended Hardware Configurations (2025–2026)

### Tier 1: High-End / Professional (Unlimited Budget)
*   **NVIDIA:** RTX 4090 / RTX 5090 (Future). Best for training and massive models.
*   **AMD:** RX 7900 XTX (24GB VRAM). Excellent alternative for inference-heavy workloads where CUDA isn't strictly required.

### Tier 2: Best Value / Sweet Spot ($500 - $1,000)
*   **NVIDIA RTX 4060 Ti 16GB:** The most accessible NVIDIA card with sufficient VRAM for 7B-13B parameter LLMs and SDXL.
*   **AMD RX 7800 XT (16GB):** Superior value proposition; excellent for Stable Diffusion and running larger models via ROCm.
*   **NVIDIA RTX 4070 (12GB):** Good balance of speed and VRAM for mid-range tasks.

### Tier 3: Budget / Entry Level (<$500)
*   **AMD RX 6800 XT (16GB):** A budget king for VRAM capacity, suitable for local AI with some setup effort.
*   **Intel Arc B580 (12GB) / A770 (16GB):** Emerging budget options. The A770 16GB is particularly notable for its VRAM at a low price point, though it requires specific driver/software configuration.
*   **NVIDIA RTX 3060 12GB:** The "starter" card for AI. Widely supported, cheap, and sufficient for 7B models and SD1.5/SDXL with quantization.

### Tier 4: The "Used Market" Gem
*   **RTX 3090 / 3090 Ti (24GB):** Often available used for under $600. Unbeatable VRAM capacity for the price, making it a top choice for running large context windows or fine-tuning on a budget.

---

## 4. Software Compatibility & Backend Analysis

| Backend/Runtime | Primary Hardware | OS Support | Best Use Case | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **CUDA (cuBLAS/cuDNN)** | NVIDIA | Win/Linux | Training, Inference, All frameworks | Industry standard. Fastest setup. |
| **ROCm (HIP/hipBLAS)** | AMD Radeon | Linux > Windows | High VRAM Inference | Improving on Windows; requires specific versions for consumer cards. |
| **oneAPI / SYCL** | Intel Arc | Win/Linux | Emerging budget AI | IPEX-LLM library provides PyTorch optimizations. |
| **Vulkan Compute** | NVIDIA/AMD/Intel | Win/Linux/Mac | Universal Inference | Used by `llama.cpp`, `KoboldCpp`. Hardware agnostic. |
| **DirectML / ONNX** | AMD/Intel/NVIDIA | Windows | Cross-vendor Inference | Stable path for non-NVIDIA on Windows. |
| **Metal** | Apple Silicon | macOS | Local LLM/Image Gen | Unified memory architecture offers unique efficiency. |

---

## 5. Critical Insights & Recommendations

1.  **VRAM > Compute Speed:** For local LLMs, the ability to load a model into VRAM is more critical than raw FLOPS. A slower card with 16GB+ VRAM (e.g., RX 7800 XT) often outperforms a faster card with 8GB VRAM (which would force swapping to system RAM).
2.  **Quantization is Key:** The performance gap between vendors shrinks when using quantized models (4-bit/5-bit GGUF/GPTQ). This reduces memory bandwidth requirements, making cross-vendor backends (Vulkan/DirectML) highly viable.
3.  **OS Matters:** While Windows support for AMD ROCm has improved, **Linux remains the most stable environment** for non-NVIDIA AI workloads. Users should be prepared to use WSL2 or dual-boot if prioritizing AMD/Intel for heavy AI tasks.
4.  **Training vs. Inference:** If the goal is *training* or *fine-tuning*, NVIDIA CUDA is still the only practical choice for most users due to the maturity of PyTorch/TensorFlow support. For *inference* (running existing models), AMD and Intel are now fully competitive.

## 6. Sources & References
*   *Tech Tactician:* "Do You Really Need CUDA For Local LLMs?" (Oct 2025); "Top 7 Best Budget GPUs for AI" (Apr 2025).
*   *Tom's Hardware:* "Stable Diffusion Benchmarks: 45 Nvidia, AMD, and Intel GPUs Compared" (Dec 2023).
*   *UserBenchmark:* CPU/GPU hierarchy data and reputation analysis.
*   *Emergent Mind / MMLU:* Benchmarking standards for LLM capabilities.
*   *Ollama Docs & GitHub Repositories:* Backend configuration details.