# RLM Research Debug Log

- Topic: Best open source coding models for local development 2026
- Model: qwen3.5:9b
- Root model: qwen3.5:35b-a3b
- Decompose: qwen3.5:9b @ http://192.168.50.17:11434
- Synthesis: qwen3.5:35b-a3b @ http://127.0.0.1:11434
- Subtopics: 5
- Total searches: 15
- Total pages fetched: 28
- Phase 1 (parallel search): 9s
- Phase 2 (serial inference): 103s
- Duration: 195s
- Output words: 824

## Subtopic Findings

### model_candidates
- Queries: ['latest open source coding models released late 2025 for local deployment', 'comparison of Qwen-Coder-2 and StarCoder2-7B performance metrics', 'top rated code LLMs on HuggingFace trending page 2026']
- Searches: 3
- Pages fetched: 6
- Sources: ['https://docs.mistral.ai/getting-started/models', 'https://www.scrile.com/blog/best-llm-for-coding', 'https://arxiv.org/html/2409.12186v1', 'https://slashdot.org/software/comparison/Qwen-7B-vs-StarCoder/', 'https://huggingface.co/papers/trending', 'https://onyx.app/llm-leaderboard']
- Content words: 3001

### hardware_requirements
- Queries: ['VRAM requirements for running CodeLlama-34B quantized locally', 'CPU inference speed benchmarks for open source coding models', 'best GGUF quantization formats for limited RAM local setups']
- Searches: 3
- Pages fetched: 6
- Sources: ['https://www.reddit.com/r/LocalLLaMA/comments/18o5u0k/helpful_vram_requirement_table_for_qlora_lora_and/', 'https://huggingface.co/TheBloke/CodeLlama-34B-GGUF', 'https://www.analyticsvidhya.com/blog/2025/01/deep-learning-cpu-benchmarks/', 'https://ahelpme.com/ai/llm-inference-benchmarks-with-llamacpp-and-amd-epyc-7282-cpu/', 'https://apidog.com/blog/small-local-llm/', 'https://cosmo-edge.com/comfyui-format-bf16-fp16-fp8-gguf/']
- Content words: 3001

### benchmark_performance
- Queries: ['SWE-bench leaderboard results for open source models 2026', 'HumanEval pass rate comparison latest coding LLMs', 'LiveCodeBench scores for local deployment capable models']
- Searches: 3
- Pages fetched: 5
- Sources: ['https://www.marc0.dev/en/leaderboard', 'https://www.swebench.com/', 'https://llm-stats.com/benchmarks/humaneval', 'https://github.com/LiveCodeBench/LiveCodeBench', 'https://livecodebench.github.io/']
- Content words: 3001

### context_window_limits
- Queries: ['maximum context window size supported by CodeLlama-34B', 'handling full repository context in local coding models', 'context truncation behavior for code generation tasks']
- Searches: 3
- Pages fetched: 5
- Sources: ['https://github.com/mlc-ai/mlc-llm/issues/923', 'https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf/discussions/16', 'https://www.devasking.com/issue/should-we-saveupdate-models-in-repository-pattern', 'https://arxiv.org/html/2602.11988v1', 'https://www.lmdconsulting.com/blogs/context-overload-and-truncation-implementation-issues-in-retrieval-augmented-generation-rag-framework-part-2-of-5']
- Content words: 3001

### inference_ecosystem
- Queries: ['Ollama model compatibility list for latest coding architectures', 'LM Studio plugin support for quantized code models', 'vLLM configuration for high-throughput local code inference']
- Searches: 3
- Pages fetched: 6
- Sources: ['https://github.com/ollama/ollama', 'https://collabnix.com/choosing-ollama-models-the-complete-2025-guide-for-developers-and-enterprises/', 'https://deepwiki.com/lmstudio-ai/lmstudio-js/5.3-model-quantization', 'https://lmstudio.ai/docs/typescript/plugins', 'https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html', 'https://www.aleksagordic.com/blog/vllm']
- Content words: 3001

## Agent Log

- decomposed: 5 subtopics, 15 queries
-   model_candidates (depth_hint=2): ['latest open source coding models released late 2025 for local deployment', 'comparison of Qwen-Coder-2 and StarCoder2-7B performance metrics', 'top rated code LLMs on HuggingFace trending page 2026']
-   hardware_requirements (depth_hint=1): ['VRAM requirements for running CodeLlama-34B quantized locally', 'CPU inference speed benchmarks for open source coding models', 'best GGUF quantization formats for limited RAM local setups']
-   benchmark_performance (depth_hint=2): ['SWE-bench leaderboard results for open source models 2026', 'HumanEval pass rate comparison latest coding LLMs', 'LiveCodeBench scores for local deployment capable models']
-   context_window_limits (depth_hint=1): ['maximum context window size supported by CodeLlama-34B', 'handling full repository context in local coding models', 'context truncation behavior for code generation tasks']
-   inference_ecosystem (depth_hint=1): ['Ollama model compatibility list for latest coding architectures', 'LM Studio plugin support for quantized code models', 'vLLM configuration for high-throughput local code inference']
- parallel search: 5 subtopics dispatched
-   ✓ benchmark_performance — 3 searches, 5 pages
-   ✓ model_candidates — 3 searches, 6 pages
-   ✓ context_window_limits — 3 searches, 5 pages
-   ✓ inference_ecosystem — 3 searches, 6 pages
-   ✓ hardware_requirements — 3 searches, 6 pages
- all subtopics complete: 15 searches, 28 pages fetched
- phase 1 (parallel search): 9s
- compiling final bundle (qwen3.5:35b-a3b synthesis)
- phase 2 (serial inference): 103s
- done: 824 words, 5 subtopics, 195s
