# RLM Research Debug Log

- Topic: Best open source coding models for local development 2026
- Model: qwen3.5:9b
- Root model: qwen3.5:35b-a3b
- Decompose: qwen3.5:9b @ http://192.168.50.17:11434
- Synthesis: qwen3.5:35b-a3b @ http://127.0.0.1:11434
- Subtopics: 5
- Total searches: 10
- Total pages fetched: 16
- Phase 1 (parallel search): 9s
- Phase 2 (serial inference): 124s
- Duration: 221s
- Output words: 930

## Subtopic Findings

### coding-benchmarks
- Queries: ['HumanEval pass rate comparison Qwen2.5-Coder vs Llama-3.1-8B', 'LiveCodeBench leaderboard open source models 2024 2025']
- Searches: 2
- Pages fetched: 4
- Sources: ['https://rankllms.com/compare/llama-3-1-8b-vs-qwen-2-5-7b/', 'https://blogs.novita.ai/qwen-2-5-72b-vs-llama-3-3-70b-which-model-suits-your-needs/', 'https://github.com/LiveCodeBench/LiveCodeBench', 'https://artificialanalysis.ai/evaluations/livecodebench']
- Content words: 3001

### hardware-compatibility
- Queries: ['VRAM requirements for running CodeLlama-70B locally on consumer GPU', 'context window limits local coding models 2026']
- Searches: 2
- Pages fetched: 4
- Sources: ['https://www.hardware-corner.net/llm-database/CodeLlama/', 'https://markaicode.com/ollama-multi-gpu-setup/', 'https://www.morphllm.com/llm-token-limit', 'https://www.elvex.com/blog/context-length-comparison-ai-models-2026']
- Content words: 3001

### quantization-formats
- Queries: ['best GGUF quantization level for code generation tasks Q4_K_M vs Q8_0', 'EXL2 support status for CodeGeeX open source models']
- Searches: 2
- Pages fetched: 3
- Sources: ['https://ai-manual.ru/article/qwen3-coder-next-v-ollama-hvatit-li-128-gb-ram-i-zachem-nuzhen-q80/', 'https://github.com/zai-org/CodeGeeX', 'https://huggingface.co/zai-org/codegeex4-all-9b']
- Content words: 3001

### inference-engines
- Queries: ['Ollama speed comparison local coding models 2026', 'vLLM compatibility list open source coding models']
- Searches: 2
- Pages fetched: 2
- Sources: ['https://www.reddit.com/r/LocalLLaMA/comments/1q82ae8/start_of_2026_whats_the_best_open_coding_model/', 'https://docs.vllm.ai/en/latest/models/supported_models/']
- Content words: 2952

### license-maintenance
- Queries: ['open source coding model license change history StarCoder 2024-2026', 'active development status CodeGen vs Llama-Coder repositories']
- Searches: 2
- Pages fetched: 3
- Sources: ['https://ollama.com/library/starcoder2/blobs/4ec42cd966c9', 'https://codewiki.ai/', 'https://news.smol.ai/issues/25-08-08-not-much']
- Content words: 3001

## Agent Log

- decomposed: 5 subtopics, 10 queries
-   coding-benchmarks (depth_hint=2): ['HumanEval pass rate comparison Qwen2.5-Coder vs Llama-3.1-8B', 'LiveCodeBench leaderboard open source models 2024 2025']
-   hardware-compatibility (depth_hint=1): ['VRAM requirements for running CodeLlama-70B locally on consumer GPU', 'context window limits local coding models 2026']
-   quantization-formats (depth_hint=2): ['best GGUF quantization level for code generation tasks Q4_K_M vs Q8_0', 'EXL2 support status for CodeGeeX open source models']
-   inference-engines (depth_hint=1): ['Ollama speed comparison local coding models 2026', 'vLLM compatibility list open source coding models']
-   license-maintenance (depth_hint=2): ['open source coding model license change history StarCoder 2024-2026', 'active development status CodeGen vs Llama-Coder repositories']
- parallel search: 5 subtopics dispatched
-   ✓ inference-engines — 2 searches, 2 pages
-   ✓ hardware-compatibility — 2 searches, 4 pages
-   ✓ license-maintenance — 2 searches, 3 pages
-   ✓ quantization-formats — 2 searches, 3 pages
-   ✓ coding-benchmarks — 2 searches, 4 pages
- all subtopics complete: 10 searches, 16 pages fetched
- phase 1 (parallel search): 9s
- compiling final bundle (qwen3.5:35b-a3b synthesis)
- phase 2 (serial inference): 124s
- done: 930 words, 5 subtopics, 221s
