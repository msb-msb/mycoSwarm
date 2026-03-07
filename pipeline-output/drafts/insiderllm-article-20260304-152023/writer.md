## Why Your Local LLM is Slower Than It Should Be — The Hidden `num_ctx` VRAM Overflow Trap

![A frustrated person staring at a computer screen with error messages.](placeholder-hero.jpg)

Let’s be real: you dropped a few hundred bucks on a GPU, wrestled with Ollama or LM Studio, and expected to be churning out AI-powered text at lightning speed. Instead, you’re watching a snail generate a sentence. What gives? It’s rarely the *GPU itself* that’s the problem. More often, it’s a silent killer: your context window (`num_ctx`) overflowing your VRAM. 

I learned this the hard way. We were debugging our own distributed AI pipeline, mycoSwarm, and saw a deepseek-r1:14b model crawl along at 4.8 tokens per second on a perfectly capable RTX 3060 12GB. We chased phantom bugs for *hours* – thinking mode, code paths, parameters – everything looked right. Then we tweaked `num_ctx` from 4096 to 16384. Boom. The model timed out. Back down to 4096? A blistering 35.3 tok/s.  The same model, same GPU, same prompt… a *seven-fold* speed difference. This isn’t about magic; it’s about memory.

This article isn’t a GPU spec-dump. It’s a breakdown of how to avoid this common pitfall, get the most out of your hardware, and actually enjoy running AI locally. Forget chasing the latest and greatest; focus on understanding the fundamentals.

## The VRAM Overflow: A Silent Performance Killer

Here's the brutal truth: your GPU's VRAM is a finite resource.  A 14 billion parameter model quantized to Q4 (a common compression level) needs roughly 8GB to store its weights.  That sounds manageable on a 12GB card, right? Not quite. The *KV cache* – the memory needed to store the attention weights for each token in your prompt and generated text – scales with context length.  At 16,000 tokens (`num_ctx=16384`), that cache easily exceeds 4GB. 8GB + 4GB+ = 12GB+.  Suddenly, your GPU is overflowing, spilling data into your system RAM.

And system RAM is *slow*.  We’re talking 37x slower per offloaded layer compared to VRAM.  A model that fits entirely within VRAM will sing. One that's constantly swapping data with system RAM will choke.  The impact is dramatic: our pipeline’s total time dropped from 742 seconds to 643 seconds just by optimizing `num_ctx`. The extractor step alone went from 173 seconds to 64 seconds.

Don’t assume a bigger number is always better.  Experiment with `num_ctx` and *monitor your VRAM usage*. Tools like `nvidia-smi` (for NVIDIA) or your system monitor can show you exactly how much VRAM is being used.  If it's pegged at 100%, you're in overflow territory.

## VRAM, Bandwidth, and Your Budget: What Matters Most

Okay, so you know to keep `num_ctx` in check. But what about choosing the right GPU? Everyone talks about teraflops, but for local LLM inference, two things matter more: **VRAM capacity** and **memory bandwidth**.

**VRAM Capacity:** This dictates the largest model you can run *without* offloading. Here’s a quick guide (based on the data):

*   **6GB:** 7B Q4 models, but it will be tight.
*   **8GB:** 8B Q4 comfortable, 14B Q2 *possible* but pushing it.
*   **10GB:** 8B Q6, 14B Q3 possible.
*   **12GB:** 14B Q4 is the sweet spot. 8B Q8 or FP16 for even better quality.
*   **16GB:** 30B Q3, 14B Q6.
*   **20GB:** 30B Q4, some 70B Q2.
*   **24GB:** 30B Q5, 70B Q2-Q3.

**Memory Bandwidth:** Once your model *fits* in VRAM, bandwidth becomes the primary bottleneck.  The RTX 3080 12GB ($305 used) is a prime example. It has the same VRAM capacity as the RTX 3060 12GB ($275 used), but nearly 2.5x the bandwidth (912 GB/s vs 360 GB/s). This translates to significantly faster token generation. The data backs it up: the RTX 3060 12GB hits 35 tok/s with deepseek r1 14b, while the RTX 3080 12GB can achieve similar speeds with more headroom.

Don't fall for the trap of chasing the newest architecture if it doesn't deliver bandwidth. The RTX 4060 ($270 used) boasts the latest Ada Lovelace architecture, but its paltry 272 GB/s bandwidth makes it slower than the RTX 3060 12GB for LLM inference.

## AMD and ROCm: Proceed with Caution

AMD cards offer compelling price-to-performance, but ROCm (AMD’s equivalent of CUDA) is still a wildcard.  While ROCm support is improving, it's not universal. The RX 7800 XT ($465 used) offers 16GB VRAM and 624 GB/s bandwidth, rivaling the RTX 3080 12GB.  However, you *must* verify ROCm compatibility with your chosen model and framework before committing.  The data shows AMD ROCm cards achieve roughly 0.06 tok/s per GB/s of bandwidth compared to NVIDIA's 0.13 tok/s.

## System RAM: The Forgotten Component

Don’t neglect your system RAM. If you’re offloading layers, you need fast RAM to minimize the performance hit. DDR3 is ancient history. DDR4 is a sweet spot for budget builds, offering a significant performance boost over DDR3. DDR5 is the top tier, providing the fastest speeds, but at a premium price.  A 16GB kit of DDR4-3200 can be found for $50-80 used, while 32GB will run you $100-170.

## Recommendation: The RTX 3080 12GB is the King

For pure value, the **NVIDIA RTX 3080 12GB** ($305 used) is the clear winner. It combines 12GB VRAM with a massive 912 GB/s bandwidth, giving you the capacity to run 14B Q4 models comfortably and the speed to generate text quickly. It's a sleeper pick that delivers performance comparable to much more expensive cards.

If you're on a tighter budget, the **RTX 3060 12GB** ($275 used) is a solid choice, but be mindful of your `num_ctx` setting. Avoid the RTX 4060 and 4060 Ti unless you absolutely need the latest features and are willing to sacrifice performance.



Don’t just buy a GPU based on benchmarks. Understand your workload, your VRAM needs, and the importance of bandwidth. Optimize your `num_ctx` setting, monitor your VRAM usage, and you’ll be well on your way to enjoying the power of local LLMs.