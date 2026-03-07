# Research Bundle: AMD ROCm vs CUDA for Local AI in 2026

## Key Facts & Data Points

| Fact | Source URL |
|------|------------|
| CUDA typically outperforms ROCm by 10% to 30% in compute-intensive workloads | https://www.thundercompute.com/blog/rocm-vs-cuda-gpu-computing |
| NVIDIA cards average ~0.13 tok/s per GB/s of bandwidth for Llama 3 8B Q4; AMD ROCm achieves ~0.06 tok/s per GB/s | Verified Reference Data (InsiderLLM canonical database) |
| ROCm support is improving but not all models/frameworks work — always verify compatibility before buying AMD | Verified Reference Data |
| llamacpp can leverage Vulkan to run LLM inference on any modern GPU (NVIDIA, AMD, or Intel) that supports Vulkan standard | https://techtactician.com/cuda-vs-alternatives-for-local-llms/ |
| ROCm 7.2.0 adds support for RDNA4 GPUs | https://rocm.docs.amd.com/en/latest/about/release-notes.html |
| AMD emphasizes openness, memory capacity, and sustainability with ROCm 7 and HIP for CUDA compatibility | https://www.bestgpusforai.com/blog/best-amd-gpus-for-ai |
| Some recent testing shows CUDA outperforms ROCm by 10% to 30% in compute-intensive workloads; AMD's MI325X represents a turning point | https://www.thundercompute.com/blog/rocm-vs-cuda-gpu-computing |
| For local LLMs, backend choice matters more than GPU brand when using frameworks like llama.cpp that support Vulkan/Metal | https://techtactician.com/cuda-vs-alternatives-for-local-llms/ |

---

## Price Data (Used Market - eBay Sold Auctions)

| GPU | VRAM | Used Price Range | Typical Price | Source URL |
|-----|------|------------------|---------------|------------|
| NVIDIA GTX 1660 Super | 6GB GDDR6 | $90–$120 | ~$105 | Verified Reference |
| NVIDIA RTX 2060 12GB | 12GB GDDR6 | $140–$180 | ~$160 | Verified Reference |
| NVIDIA RTX 3060 12GB | 12GB GDDR6 | $170–$380 | ~$275 | Verified Reference |
| NVIDIA RTX 3070 | 8GB GDDR6 | $210–$300 | ~$255 | Verified Reference |
| NVIDIA RTX 3070 Ti | 8GB GDDR6X | $100–$280 | ~$190 | Verified Reference |
| NVIDIA RTX 3080 10GB | 10GB GDDR6X | $325–$400 | ~$365 | Verified Reference |
| NVIDIA RTX 3080 12GB | 12GB GDDR6X | $230–$380 | ~$305 | Verified Reference |
| NVIDIA RTX 3090 | 24GB GDDR6X | $950–$1125 | ~$1040 | Verified Reference |
| NVIDIA RTX 4060 Ti 8GB | 8GB GDDR6 | $240–$300 | ~$270 | Verified Reference |
| NVIDIA RTX 4060 Ti 16GB | 16GB GDDR6 | $380–$480 | ~$430 | Verified Reference |
| AMD RX 7600 | 8GB GDDR6 | $170–$225 | ~$200 | Verified Reference |
| AMD RX 7700 XT | 12GB GDDR6 | $300–$350 | ~$325 | Verified Reference |
| AMD RX 7800 XT | 16GB GDDR6 | $380–$550 | ~$465 | Verified Reference |
| AMD RX 7900 GRE | 16GB GDDR6 | $400–$550 | ~$475 | Verified Reference |
| AMD RX 7900 XT | 20GB GDDR6 | $500–$700 | ~$600 | Verified Reference |

---

## Benchmark Data (Local LLM Inference)

### NVIDIA Cards
| GPU | Model | Quantization | Tokens/sec | Source URL |
|-----|-------|--------------|------------|------------|
| RTX 3060 12GB | llama3 8B | Q4 | 51 tok/s | Verified Reference |
| RTX 3070 | llama3 8B | Q4 | 71 tok/s | Verified Reference |
| RTX 3080 10GB | llama3 8B | Q4 | 106 tok/s | Verified Reference |
| RTX 3080 12GB | llama3 8B | Q4 | 107 tok/s | Verified Reference |
| RTX 3090 | llama3 8B | Q4 | 112 tok/s | Verified Reference |
| RTX 3090 | llama3 8B | F16 | 47 tok/s | Verified Reference |
| RTX 3090 | llama3 70B | Q4 | 16 tok/s | Verified Reference |

### AMD Cards (ROCm)
| GPU | Model | Quantization | Tokens/sec | Source URL |
|-----|-------|--------------|------------|------------|
| RX 7800 XT | llama3 8B | Q4 | 39 tok/s | Verified Reference |
| RX 7800 XT | llama2 7B | Q4 | 96 tok/s | Verified Reference |
| RX 7900 XT | llama2 7B | Q4 | 116 tok/s | Verified Reference |
| RX 7900 XT | llama2 7B sustained | Q4 | 97 tok/s | Verified Reference |

**Bandwidth Performance Notes:**
- For memory-bound inference, double the bandwidth ≈ double the tok/s
- DDR4 (25.6 GB/s) vs GDDR6X (936 GB/s) = ~37x slower per offloaded layer

---

## Key Specs (VRAM, Bandwidth, TDP, Architecture)

| GPU | VRAM | Memory Bandwidth | TDP | Architecture | Source URL |
|-----|------|------------------|-----|--------------|------------|
| GTX 1660 Super | 6GB GDDR6 | 336 GB/s | 125W | Turing | Verified Reference |
| RTX 2060 12GB | 12GB GDDR6 | 336 GB/s | 185W | Turing | Verified Reference |
| RTX 3060 12GB | 12GB GDDR6 | 360 GB/s | 170W | Ampere | Verified Reference |
| RTX 3070 | 8GB GDDR6 | 448 GB/s | 220W | Ampere | Verified Reference |
| RTX 3070 Ti | 8GB GDDR6X | 608 GB/s | 290W | Ampere | Verified Reference |
| RTX 3080 10GB | 10GB GDDR6X | 760 GB/s | 320W | Ampere | Verified Reference |
| RTX 3080 12GB | 12GB GDDR6X | 912 GB/s | 350W | Ampere | Verified Reference |
| RTX 3090 | 24GB GDDR6X | 936 GB/s | 350W | Ampere | Verified Reference |
| RX 7600 | 8GB GDDR6 | 288 GB/s | 165W | RDNA 3 | Verified Reference |
| RX 7700 XT | 12GB GDDR6 | 432 GB/s | 245W | RDNA 3 | Verified Reference |
| RX 7800 XT | 16GB GDDR6 | 624 GB/s | 263W | RDNA 3 | Verified Reference |
| RX 7900 GRE | 16GB GDDR6 | 576 GB/s | 260W | RDNA 3 | Verified Reference |
| RX 7900 XT | 20GB GDDR6 | 800 GB/s | 315W | RDNA 3 | Verified Reference |

**Model Size Guide (VRAM → Model Capacity):**
- 6GB: 7B Q4 max, tight
- 8GB: 8B Q4 comfortable, 14B Q2 possible
- 10GB: 8B Q6, 14B Q3 possible
- 12GB: 14B Q4, 8B Q8 or FP16
- 16GB: 30B Q3, 14B Q6
- 20GB: 30B Q4, some 70B Q2
- 24GB: 30B Q5, 70B Q2-Q3

---

## Expert Opinions & Analysis

### TechnoLynx (Jan 20, 2026)
> "A practical comparison of CUDA vs ROCm for GPU compute in modern AI, covering performance, developer experience, software stack maturity, cost..."

**Key Points:**
- Practical considerations beyond raw performance
- Developer experience is a major factor
- Software stack maturity heavily favors NVIDIA currently

### ThunderCompute (Oct 27, 2025)
> "Some recent testing shows that CUDA typically outperforms ROCm by 10% to 30% in compute-intensive workloads. AMD's MI325X represents a turning..."

**Key Points:**
- 10-30% CUDA performance advantage in typical workloads
- Enterprise-grade MI325X shown as potential alternative (not consumer GPUs)
- Focus on enterprise use cases primarily

### TechTactician (2026)
> "For easily accessible local LLM model text generation and chatting with AI models privately have similar best-case scenarios when it comes to the top..."

**Key Points:**
- AMD RX 7900 XTX is recommended as best AMD option for serious users
- ROCm compatibility must be verified for specific use case
- Practical advice: test before committing to AMD

### Medium Article (Dec 11, 2025)
> "NVIDIA vs AMD : Best GPU for AI? comparing AMD and NVIDIA GPUs By 2025, the GPU world has split into two very distinct personalities."

**Key Points:**
- Market segmentation between NVIDIA and AMD ecosystems
- Tradeoffs in openness vs optimization

### ROCm Documentation (Official)
> "ROCm provides a prebuilt optimized Docker image for validating the performance of LLM inference with vLLM on MI300X Series GPUs."

**Key Points:**
- Official ROCm 7.2.0 adds RDNA4 support
- Focus on enterprise MI-series GPUs, not consumer Radeon cards

---

## Gaps

1. **Specific ROCm vs CUDA benchmark comparisons for specific models in 2026** - Need more granular data comparing same models on both platforms
2. **Framework compatibility details** - Which exact versions of PyTorch, llama.cpp, vLLM work best with each platform
3. **Power efficiency comparisons** - TDP vs actual power draw for local AI workloads
4. **Community support and troubleshooting resources** - How much harder is it to find help with ROCm vs CUDA?
5. **Future roadmap certainty** - Will AMD continue supporting consumer Radeon cards with ROCm?

---

## New Data

*   **ROCm 7.2.0 support for RDNA4 GPUs:** Official documentation confirms ROCm 7.2.0 now supports AMD's latest RDNA4 architecture, opening the door for potential local AI applications on the RX 9000 series.
*   **Preliminary RX 9070/9070 XT specs:** Initial reports indicate the RX 9070 and 9070 XT will feature 16GB GDDR6 VRAM and 608 GB/s memory bandwidth.



---

## Suggested Angle for InsiderLLM

**"ROCm or CUDA: The Local AI Buyer's Dilemma in 2026"**

**Core Narrative:**
InsiderLLM should position this not as a simple "which is better" story, but as a **practical decision framework** for local AI users. The key insight: **CUDA wins on ease-of-use and performance; AMD ROCm wins on raw VRAM capacity per dollar IF it works for your stack.**

**Recommended Structure:**

1. **The Hard Truth**: CUDA is ~2x faster per GB/s, but NVIDIA's pricing has created an affordability crisis
2. **The AMD Compromise**: RX 7900 XT (20GB) can run models that fit on RTX 3090, but ROCm compatibility varies wildly by framework/version
3. **The Sweet Spot Cards**: 
   - For CUDA: RTX 3080 12GB ($305 used) = best bandwidth/price ratio
   - For AMD: RX 7900 XT ($600 used) = only if you verify ROCm works for your stack
4. **The Framework Factor**: llamacpp's Vulkan support changes the equation — can you run on AMD even without full ROCm?
5. **Decision Matrix**: Include a practical checklist: "Test these 3 things before buying AMD"

**Unique InsiderLLM Angle:**
Focus on **bandwidth vs VRAM tradeoffs**. The RTX 3080 12GB (912 GB/s) outperforms most ROCm-compatible AMD cards despite lower VRAM because memory-bound inference is the primary bottleneck. But if you need 24GB+ for 70B models, AMD's 20-24GB options become necessary.

**Forward-Looking Element:**
Monitor RDNA 4 (RX 9000 series) — ROCm support status unknown at launch. This creates uncertainty that InsiderLLM can track for readers deciding between "wait and see" vs buy now.

## DOCUMENT CONTEXT
[D1] (insiderllm-content-plan.md)
### Priority 2
- [ ] Anthropic vs OpenAI vs Local — decision framework
- [ ] When to Use Cloud vs Local
- [ ] Privacy: Local vs Cloud AI
- [ ] Cost Calculator: Local AI vs API

[D2] (insiderllm-content-plan.md)
Running LLMs on Mac M-Series 50. Best GPU Under $300 for Local AI 51. RTX 3090 vs RTX 4070 Ti Super for Local LLMs 52. Best GPU Under $500 for Local AI 53. Best Used GPUs for Local AI 2026 54. How Much Does It Cost to Run LLMs Locally 55. OpenClaw Token Optimization 56. Tiered AI Model Strategy / Stop Using Frontier AI for Everything

[D3] (insiderllm-content-plan.md)
### Published (21)
- [x] GPU Buying Guide for Local AI
- [x] RTX 5060 Ti 16GB News
- [x] VRAM Requirements Guide
- [x] Used RTX 3090 Buying Guide
- [x] NVIDIA Price Hikes Analysis
- [x] AMD vs NVIDIA for Local AI
- [x] Budget AI PC Under $500
- [x] What Can You Run on 8GB VRAM
- [x] What Can You Run on 12GB VRAM
- [x] What Can You Run on 16GB VRAM
- [x] What Can You Run on 24GB VRAM
- [x] What Can You Run on 4GB VRAM
- [x] CPU-Only LLMs
- [x] Mac vs PC for Local AI
- [x] Best Models Under 3B Parameters
- [x] Running LLMs on Mac M-Series
- [x] Best GPU Under $300 for Local AI
- [x] RTX 3090 vs RTX 4070 Ti Super for Local LLMs
- [x] Best GPU Under $500 for Local AI
- [x] Best Used GPUs for Local AI 2026
- [x] Laptop vs Desktop for Local AI

[D4] (insiderllm-content-plan.md)
### Published (4)
- [x] Local LLMs vs ChatGPT: Honest Comparison
- [x] Local LLMs vs Claude: When Each Wins
- [x] OpenClaw vs Commercial AI Agents
- [x] How Much Does It Cost to Run LLMs Locally

[D5] (INSIDERLLM-PROJECT.md)
### Recurring Tasks
- **Daily:** 2-3 Reddit comments on r/LocalLLaMA
- **Daily:** 2-4 new articles via CC
- **Weekly:** Check GSC data, update content plan
- **Weekly:** Review Reddit engagement, adjust strategy

## SESSION CONTEXT
[S1]


[S2]


[S3]


[S4]


[S5]