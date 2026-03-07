```yaml
title: 'Why Your Local LLM is Slow: The Hidden VRAM Overflow Trap'
meta_description: 'Stop your local LLM from slowing down! Learn how num_ctx causes VRAM overflow and slow inference. We compare NVIDIA RTX 3060 vs 3080 bandwidth and AMD ROCm speeds.'
slug: 'local-llm-slow-num-ctx-vram-trap'
keywords: ['local LLM', 'VRAM overflow', 'num_ctx', 'memory bandwidth', 'AI inference speed', 'NVIDIA GPUs', 'system RAM', 'GPU benchmarks']
category: 'Hardware Guides'
estimated_read_time: '6 min'
---

![A frustrated person staring at a computer screen with error messages.](placeholder-hero.jpg)

**TL;DR:** Your local LLM isn't slow because the model is bad—it's likely spilling into system RAM due to context window overflow. Fixing `num_ctx` and choosing high-bandwidth GPUs (like the **RTX 3080 12GB**) can boost speeds by 3-4x.

Let's be real: you dropped a few hundred bucks on a GPU, wrestled with Ollama or LM Studio, and expected to be churning out AI-powered text at lightning speed. Instead, you're watching a snail generate a sentence. What gives? It's rarely the *GPU itself* that's the problem. More often, it's a silent killer: your context window (`num_ctx`) overflowing your VRAM.

I learned this the hard way. We were debugging our own distributed AI pipeline, mycoSwarm, and saw a **deepseek-r1:14b** model crawl along at 4.8 tokens per second on a perfectly capable **RTX 3060 12GB**. We chased phantom bugs for *hours* – thinking mode, code paths, parameters – everything looked right. Then we tweaked `num_ctx` from 4096 to 16384. Boom. The model timed out. Back down to 4096? A blistering **35.3 tok/s**.

The same model, same GPU, same prompt… a *seven-fold* speed difference. This isn't about magic; it's about memory management and bandwidth architecture.

This article isn't a GPU spec-dump. It's a breakdown of how to avoid this common pitfall, get the most out of your hardware, and actually enjoy running AI locally. Forget chasing the latest and greatest blindly; focus on understanding the fundamentals.

## The VRAM Overflow: A Silent Performance Killer

Here's the brutal truth: your GPU's VRAM is a finite resource. A 14 billion parameter model quantized to Q4 (a common compression level) needs roughly **8GB** to store its weights. That sounds manageable on a 12GB card, right? Not quite. The **KV cache** – the memory needed to store the attention weights for each token in your prompt and generated text – scales with context length.

At 16,000 tokens (`num_ctx=16384`), that cache easily exceeds 4GB. 
*   **8GB (Model) + 4GB (Cache) = 12GB+**
Suddenly, your GPU is overflowing, spilling data into your system RAM.

And system RAM is *slow*. We're talking **37x slower** per offloaded layer compared to VRAM. A model that fits entirely within VRAM will sing. One that's constantly swapping data with system RAM will choke. The impact is dramatic: our pipeline's total time dropped from 742 seconds to 643 seconds just by optimizing `num_ctx`. The extractor step alone went from 173 seconds to 64 seconds.

Don't assume a bigger number is always better. Experiment with `num_ctx` and **monitor your VRAM usage**. Tools like `nvidia-smi` (for NVIDIA) or your system monitor can show you exactly how much VRAM is being used. If it's pegged at 100%, you're in overflow territory.

## VRAM, Bandwidth, and Your Budget: What Matters Most

Okay, so you know to keep `num_ctx` in check. But what about choosing the right GPU? Everyone talks about teraflops, but for local LLM inference, two things matter more: **VRAM capacity** and **memory bandwidth**.

### VRAM Capacity Guide
This dictates the largest model you can run *without* offloading. Here is a quick guide based on current market data:

| VRAM | Max Model (Q4) | Notes |
| :--- | :--- | :--- |
| **6GB** | 7B Q4 | Tight fit, not for daily use. |
| **8GB** | 8B Q4 / 14B Q2 | 14B pushes limits; consider overflow risks. |
| **10GB** | 8B Q6 / 14B Q3 | Good middle ground. |
| **12GB** | **14B Q4** | The sweet spot for most users. |
| **16GB** | 30B Q3 / 14B Q6 | Best for larger context needs. |
| **24GB** | 70B Q2-Q3 | Budget king for large models. |

### Memory Bandwidth: The Hidden Bottleneck
Once your model *fits* in VRAM, bandwidth becomes the primary bottleneck. 

The **RTX 3080 12GB** ($305 used) is a prime example. It has the same VRAM capacity as the **RTX 3060 12GB** ($275 used), but nearly 2.5x the bandwidth (**912 GB/s** vs 360 GB/s). This translates to significantly faster token generation. The data backs it up: the RTX 3060 12GB hits **35 tok/s** with deepseek r1 14b, while the RTX 3080 12GB can achieve similar speeds with more headroom.

Don't fall for the trap of chasing the newest architecture if it doesn't deliver bandwidth. The **RTX 4060** ($270 used) boasts the latest Ada Lovelace architecture, but its paltry **272 GB/s** bandwidth makes it slower than the RTX 3060 12GB for LLM inference due to the narrow 128-bit bus.

## AMD and ROCm: Proceed with Caution

AMD cards offer compelling price-to-performance, but ROCm (AMD's equivalent of CUDA) is still a wildcard. While ROCm support is improving, it's not universal. 

The **RX 7800 XT** ($465 used) offers 16GB VRAM and 624 GB/s bandwidth, rivaling the RTX 3080 12GB. However, you *must* verify ROCm compatibility with your chosen model and framework before committing. The data shows AMD ROCm cards achieve roughly **0.06 tok/s per GB/s** of bandwidth compared to NVIDIA's **0.13 tok/s**.

If you stick with NVIDIA, the **RTX 3090** ($1040 used) remains the budget local AI king with 24GB VRAM and 936 GB/s bandwidth. It runs 70B models (Q2) comfortably, though it runs hot—look for triple-fan models to manage thermal throttling.

## System RAM: The Forgotten Component

Don't neglect your system RAM. If you're offloading layers due to VRAM limits, you need fast RAM to minimize the performance hit. 

*   **DDR3:** Ancient history. 14.9 GB/s dual-channel max. Painful for offloaded layers.
*   **DDR4:** Sweet spot for budget builds (e.g., Optiplex 5060/7060). Offers a significant boost over DDR3. 32GB kits cost $100-170 used.
*   **DDR5:** Top tier. 48 GB/s at 6000MHz dual-channel. Best for CPU-heavy workloads or large models needing lots of offloaded layers.

Each layer offloaded to CPU RAM runs at system RAM bandwidth instead of VRAM bandwidth. **DDR4-3200 (25.6 GB/s)** vs GDDR6X (**936 GB/s**) = ~37x slower per offloaded layer.

## Recommendation: The RTX 3080 12GB is the King

For pure value, the **NVIDIA RTX 3080 12GB** ($305 used) is the clear winner. It combines 12GB VRAM with a massive 912 GB/s bandwidth, giving you the capacity to run 14B Q4 models comfortably and the speed to generate text quickly. It's a sleeper pick that delivers performance comparable to much more expensive cards.

If you're on a tighter budget, the **RTX 3060 12GB** ($275 used) is a solid choice, but be mindful of your `num_ctx` setting. Avoid the RTX 4060 and 4060 Ti unless you absolutely need the latest features and are willing to sacrifice performance—their narrow bus width hurts inference speed significantly.

Don't just buy a GPU based on benchmarks. Understand your workload, your VRAM needs, and the importance of bandwidth. Optimize your `num_ctx` setting, monitor your VRAM usage, and you'll be well on your way to enjoying the power of local LLMs.