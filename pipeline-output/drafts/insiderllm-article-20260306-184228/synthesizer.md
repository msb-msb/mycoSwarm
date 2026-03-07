# Research Bundle: Qwen3 Coder (480B A35B & 80B A3B) & Local LLM Quantization Strategies

## 1. Executive Summary
This bundle synthesizes technical findings regarding **Qwen3 Coder**, specifically the **480B A35B Instruct** and **Next 80B A3B** variants. Key insights highlight that despite massive parameter counts, the Mixture-of-Experts (MoE) architecture (activating only ~3B parameters per inference) makes these models surprisingly hardware-friendly for local deployment compared to dense equivalents.

*   **Primary Constraint:** VRAM capacity and memory bandwidth, not raw compute power.
*   **Best Deployment Paths:**
    *   **Cloud/API:** Novita AI (Lowest barrier to entry, high throughput).
    *   **High-End Local:** Single NVIDIA RTX PRO 6000 (96GB) for full 256k context stability.
    *   **Budget/Consumer Dual-GPU:** Dual RTX 3090/4090 using `llama-server` with `--fit` parameter to bypass standard memory limits.
    *   **Unified Memory:** Strix Halo (Beelink GTR9 Pro) offers the best price/performance balance for non-workstation setups.

---

## 2. Model Specifications & Architecture

### Qwen3 Coder 480B A35B Instruct
*   **Total Parameters:** 480 Billion.
*   **Active Parameters (MoE):** ~35 Billion per inference.
*   **Context Window:** 66K max output, 262K context support.
*   **Latency/Throughput (API):** 6.82s latency, 76.35 TPS throughput.
*   **Cost Structure (Novita AI):** $0.95 per input token, $5.00 per output token.
*   **Primary Use Case:** Complex code agents, long-context reasoning, precise instruction following.

### Qwen3 Coder Next 80B A3B
*   **Total Parameters:** 80 Billion.
*   **Active Parameters (MoE):** ~3 Billion per inference.
*   **Context Window:** Full 256k support.
*   **Architecture Benefit:** Hybrid MoE design results in minimal VRAM delta between short and long contexts.
*   **Primary Use Case:** Local coding agents, iterative development workflows, cost-efficient local inference.

---

## 3. Deployment Strategy Comparison

| Feature | Cloud API (Novita AI) | Single High-End GPU | Unified Memory (Strix Halo/Apple) | Dual Consumer GPUs |
| :--- | :--- | :--- | :--- | :--- |
| **Setup Complexity** | Instant (API Key) | Moderate (Driver/CUDA install) | Low (Pre-built or simple config) | High (Requires `--fit` tuning) |
| **Cost Profile** | Pay-per-use ($0.95/$5) | High CapEx (~$8,000 for PRO 6000) | Medium (~$1,500–$2,500) | Low CapEx (~$1,600 total) |
| **Privacy** | Data leaves local network | Full Local | Full Local | Full Local |
| **Scalability** | Automatic | Limited by single card | Limited by RAM bandwidth | Linear scaling with cards |
| **Best For** | Fast start, variable load | Max performance, 256k context | Best balance of price/perf | Budget-conscious builds |

---

## 4. Hardware Performance Benchmarks

### A. Single GPU: NVIDIA RTX PRO 6000 (96GB VRAM)
*   **Quantization:** Q4_K_XL (4-bit).
*   **Context Scaling:** Minimal VRAM overhead. Jumping from 4k to 256k context costs only ~7 GB VRAM.
    *   4k Context: ~47 GB VRAM usage.
    *   256k Context: ~54 GB VRAM usage.
*   **Speed (Tokens/sec):**
    *   **Prompt Processing:** 1,472 t/s at 256k context (Exceptional for 80B class).
    *   **Token Generation:** 61.0 t/s at 256k context.
    *   **Degradation:** Gradual; performance remains usable for interactive work even at max context.

### B. Unified Memory: Strix Halo (Beelink GTR9 Pro)
*   **Configuration:** 64 GB RAM.
*   **Performance (32k Context):**
    *   Prompt Processing: ~500 t/s.
    *   Token Generation: ~37 t/s.
*   **Verdict:** Slower than workstation GPUs but highly cost-effective for development and agent workflows.

### C. Budget Dual-GPU: RTX 3090 / 4090 (2x)
*   **Challenge:** Standard `llama-bench` fails to load Q4_K_XL due to memory fragmentation.
*   **Solution:** Use `llama-server` with the `--fit` parameter.
    *   Command: `./llama-server --fit on --fit-ctx 131072 ...`
*   **Performance (Q4_K_XL with --fit):**
    *   **131k Context:** ~987 t/s prompt processing, ~24.8 t/s generation.
    *   **Verdict:** Flawless execution for roughly $1,600 hardware cost.

---

## 5. Quantization & Software Ecosystem

### Recommended Quantization Formats
| Format | Type | Best Use Case | Compatibility | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **GGUF** | File Format | CPU/GPU hybrid, Local CLI | llama.cpp, LM Studio, Ollama | "One file to run"; supports K-quants (Q4_K_M best balance). |
| **EXL2** | Method | High quality at low VRAM | ExLlamaV2, TGWUI | Mixed precision (2-8 bit); often outperforms GPTQ/AWQ. |
| **GPTQ** | Method | Pure GPU inference | vLLM, AutoGPTQ | Higher accuracy than GGUF; requires sufficient VRAM for full load. |
| **AWQ** | Method | Instruction-tuned models | transformers, vLLM | Activation-aware; excellent for consumer GPUs (12GB+). |
| **MLX/DWQ**| Method | Apple Silicon | MLX Framework | DWQ uses distillation to match 6-8bit quality at 4-bit size. |

### Key Technical Findings on Quantization
1.  **VRAM vs. Compute:** For MoE models like Qwen3 Coder, memory bandwidth is the bottleneck, not raw FLOPS. Unified memory systems (Apple/Strix Halo) perform well because they activate only ~3B parameters per step.
2.  **Context Length Impact:** VRAM usage grows linearly but slowly with context length for MoE models. The "KV cache" is not a runaway problem as seen in dense 70B+ models.
3.  **Unsloth Dynamic Quantization:** Models like `UD-Q4_K_XL` apply layer-wise quantization, preserving critical layers at higher precision while compressing others, often yielding better accuracy than standard Q4_K_M.

---

## 6. Implementation Guide: Novita AI API (Python)

For users opting for the cloud route to avoid hardware constraints, the following Python implementation is optimized for Qwen3 Coder via Novita AI.

**Prerequisites:**
```bash
pip install 'openai>=1.0.0'
```

**Code Snippet:**
```python
from openai import OpenAI

# Initialize Client with Novita AI Endpoint
client = OpenAI(
    base_url="https://api.novita.ai/v3/openai",
    api_key="YOUR_API_KEY_HERE" 
)

model = "qwen/qwen3-coder-480b-a35b-instruct"
stream = True  # Set to False for non-streaming response
max_tokens = 131072

# Configuration for high-quality code generation
params = {
    "temperature": 1,
    "top_p": 1,
    "min_p": 0,
    "top_k": 50,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "repetition_penalty": 1,
    "response_format": {"type": "text"}
}

# System Prompt to enforce coding persona
system_content = "Be a helpful assistant specialized in complex code generation and debugging."

messages = [
    {"role": "system", "content": system_content},
    {"role": "user", "content": "Write a Python script to implement a fast API for processing large JSON datasets using async/await."}
]

# Execute Request
completion = client.chat.completions.create(
    model=model,
    messages=messages,
    stream=stream,
    max_tokens=max_tokens,
    **params,
    extra_body=params # Novita specific parameter passing
)

if stream:
    for chunk in completion:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")
else:
    print(completion.choices[0].message.content)
```

---

## 7. Strategic Recommendations

1.  **For Enterprises with Advanced Infrastructure:** Deploy locally using **NVIDIA RTX PRO 6000** or similar 96GB+ cards. The stability of single-GPU inference for 256k contexts eliminates inter-GPU latency and split-KV cache complexities.
2.  **For Budget-Conscious Developers/Researchers:** Utilize a **Dual RTX 3090/4090** setup with `llama-server` and the `--fit` flag. This provides ~25 t/s generation at 131k context for under $2,000 hardware cost.
3.  **For Rapid Prototyping & Agents:** Use **Novita AI API**. It removes all setup friction (drivers, CUDA, quantization selection) and offers immediate access to the full 480B parameter capability with high throughput.
4.  **For Mac/Apple Users:** Leverage **MLX/DWQ** quantized models on M-series chips or Strix Halo laptops for a seamless "unified memory" experience that balances portability with coding agent performance.

## 8. References & Sources
*   *Novita AI Documentation:* Qwen3 Coder API specs and pricing.
*   *Hardware Corner:* "Qwen3 Coder Next 80B A3B: what it takes to run it locally" (Chavy Levi, March 2026).
*   *Hardware Corner:* "Quantization for Local LLMs" (Allan Witt, Oct 2025).
*   *Hugging Face:* DavidAU/Maximizing-Model-Performance-All-Quants-Types.
*   *llama.cpp:* Official benchmarks and `--fit` parameter documentation.