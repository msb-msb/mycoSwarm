# RLM Research Debug Log

- Topic: Best open source coding models for local development 2026
- Model: qwen3.5:9b
- Root model: qwen3.5:35b-a3b
- Decompose: qwen3.5:9b @ http://192.168.50.17:11434
- Synthesis: qwen3.5:35b-a3b @ http://127.0.0.1:11434
- Subtopics: 5
- Total searches: 15
- Total pages fetched: 25
- Duration: 245s
- Output words: 991

## Subtopic Findings

### model_selection
- Queries: ['open source coding model releases Q4 2024', 'Qwen2.5 Coder vs Llama-3.1-Coder specs comparison', 'StarCoder2 latest open weights availability']
- Searches: 3
- Pages fetched: 5
- Sources: ['https://www.llama.com/', 'https://github.com/nomic-ai/gpt4all', 'https://www.ankursnewsletter.com/p/comparing-open-source-ai-models-llama', 'https://huggingface.co/docs/transformers/en/model_doc/starcoder2', 'https://clarifai.com/bigcode/code/models/starCoder2-15b']
- Content words: 3001

### performance_benchmarks
- Queries: ['human eval pass rate coding models 2024', 'livebench coding leaderboard open source', 'mbpp benchmark results local models']
- Searches: 3
- Pages fetched: 6
- Sources: ['https://arxiv.org/abs/2402.14852', 'https://evalplus.github.io/leaderboard.html', 'https://livebench.ai/', 'https://livecodebench.github.io/leaderboard.html', 'https://mbrenndoerfer.com/writing/mbpp-mostly-basic-python-programming-benchmark', 'https://llm-stats.com/benchmarks/mbpp']
- Content words: 3001

### hardware_requirements
- Queries: ['minimum VRAM for running Qwen coder locally', 'GGUF quantization levels code model performance', 'RAM requirements for 70B coding model inference']
- Searches: 3
- Pages fetched: 5
- Sources: ['https://blogs.novita.ai/qwen3-coder-480b-a35b-vram-how-much-memory-do-you-need/', 'https://www.localai.computer/models/qwen-qwen3-coder-next', 'https://github.com/omarbasha19/GGUF-Quantization-Benchmark', 'https://www.shepbryan.com/blog/what-is-gguf', 'https://www.reddit.com/r/LocalLLaMA/comments/1dzrh7a/minimal_requirements_for_running_a_70b_model/']
- Content words: 3001

### inference_speed
- Queries: ['tokens per second local inference coding models', 'context window limits for code generation local deployment', 'latency comparison vLLM vs llama.cpp coding']
- Searches: 3
- Pages fetched: 4
- Sources: ['https://techtactician.com/llm-tokens-per-second-inference-speed-simulator-tool/', 'https://localaimaster.com/models/context-windows-coding-explained', 'https://forums.developer.nvidia.com/t/vllm-on-gb10-gpt-oss-120b-mxfp4-slower-than-sglang-llama-cpp-what-s-missing/356651', 'https://stable-learn.com/en/ai-model-tools-comparison/']
- Content words: 3001

### future_roadmaps
- Queries: ['open source coding model roadmap 2026 predictions', 'upcoming quantized releases for local deployment', 'community expectations for next gen code models']
- Searches: 3
- Pages fetched: 5
- Sources: ['https://www.index.dev/blog/qwen-ai-coding-review', 'https://betterstack.com/community/guides/ai/open-source-ai-coding-tools/', 'https://localaimaster.com/blog/best-local-ai-models-2025-complete-guide', 'https://chat-deep.ai/guide/deepseek-quantization/', 'https://huggingface.co/spaces']
- Content words: 3001

## Agent Log

- decomposed: 5 subtopics, 15 queries
-   model_selection (depth_hint=1): ['open source coding model releases Q4 2024', 'Qwen2.5 Coder vs Llama-3.1-Coder specs comparison', 'StarCoder2 latest open weights availability']
-   performance_benchmarks (depth_hint=2): ['human eval pass rate coding models 2024', 'livebench coding leaderboard open source', 'mbpp benchmark results local models']
-   hardware_requirements (depth_hint=1): ['minimum VRAM for running Qwen coder locally', 'GGUF quantization levels code model performance', 'RAM requirements for 70B coding model inference']
-   inference_speed (depth_hint=2): ['tokens per second local inference coding models', 'context window limits for code generation local deployment', 'latency comparison vLLM vs llama.cpp coding']
-   future_roadmaps (depth_hint=3): ['open source coding model roadmap 2026 predictions', 'upcoming quantized releases for local deployment', 'community expectations for next gen code models']
- subtopic 1/5: model_selection — 3 queries, depth_hint=1
-   done — 3 searches, 5 pages fetched, 5 sources
- subtopic 2/5: performance_benchmarks — 3 queries, depth_hint=2
-   done — 3 searches, 6 pages fetched, 6 sources
- subtopic 3/5: hardware_requirements — 3 queries, depth_hint=1
-   done — 3 searches, 5 pages fetched, 5 sources
- subtopic 4/5: inference_speed — 3 queries, depth_hint=2
-   done — 3 searches, 4 pages fetched, 4 sources
- subtopic 5/5: future_roadmaps — 3 queries, depth_hint=3
-   done — 3 searches, 5 pages fetched, 5 sources
- all subtopics complete: 15 searches, 25 pages fetched
- compiling final bundle (qwen3.5:35b-a3b synthesis)
- done: 991 words, 5 subtopics, 245s
