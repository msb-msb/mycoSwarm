# RLM Research Debug Log

- Topic: Best open source coding models for local development 2026
- Model: qwen3.5:9b
- Root model: qwen3.5:35b-a3b
- Decompose: qwen3.5:9b @ http://192.168.50.17:11434
- Synthesis: qwen3.5:35b-a3b @ http://127.0.0.1:11434
- Subtopics: 5
- Total searches: 10
- Total pages fetched: 18
- Duration: 232s
- Output words: 1067

## Subtopic Findings

### model_families
- Queries: ['Qwen2.5-Coder vs Llama-3.1-Code architecture specs comparison', 'latest open source coding LLM weights release status 2024']
- Searches: 2
- Pages fetched: 4
- Sources: ['https://llm-stats.com/models/compare/qwen-2.5-coder-7b-instruct-vs-llama-3.1-70b-instruct', 'https://www.ankursnewsletter.com/p/comparing-open-source-ai-models-llama', 'https://huggingface.co/blog/daya-shankar/open-source-llms', 'https://llm-stats.com/ai-news']
- Content words: 3001

### benchmarking
- Queries: ['HumanEval pass@1 scores for open source coding models 2024', 'LiveCodeBench results local deployment models']
- Searches: 2
- Pages fetched: 4
- Sources: ['https://klu.ai/glossary/humaneval-benchmark', 'https://huggingface.co/blog/leaderboard-bigcodebench', 'https://livecodebench.github.io/', 'https://pypi.org/project/nvidia-livecodebench/']
- Content words: 3001

### hardware_requirements
- Queries: ['minimum GPU VRAM for running 15B parameter coding model locally', 'CPU-only inference latency for llama.cpp coding models']
- Searches: 2
- Pages fetched: 4
- Sources: ['https://www.tspi.at/diy/gpusizeestimatellm.html', 'https://huggingface.co/blog/personal-copilot', 'https://www.reddit.com/r/LocalLLaMA/comments/1p90zzi/cpuonly_llm_performance_ts_with_llamacpp/', 'https://github.com/abetlen/llama-cpp-python/discussions/2073']
- Content words: 3001

### quantization_formats
- Queries: ['best GGUF quantization method for preserving code generation accuracy', 'vLLM vs llama.cpp speed comparison local coding models']
- Searches: 2
- Pages fetched: 2
- Sources: ['https://icml.cc/virtual/2025/poster/45172', 'https://github.com/ggml-org/llama.cpp/discussions/15180']
- Content words: 3001

### licensing_availability
- Queries: ['open source coding model license compliance requirements commercial use', 'HuggingFace repository availability risk for open weights coding models']
- Searches: 2
- Pages fetched: 4
- Sources: ['https://compile7.org/decompile/open-source-licensing-compliance-developers-guide', 'https://aaronhall.com/licensing-open-source-code-for-commercial-use/', 'https://huggingface.co/HuggingFaceM4/idefics2-8b-chatty', 'https://arxiv.org/html/2512.07814v1']
- Content words: 3001

## Agent Log

- decomposed: 5 subtopics, 10 queries
-   model_families (depth_hint=1): ['Qwen2.5-Coder vs Llama-3.1-Code architecture specs comparison', 'latest open source coding LLM weights release status 2024']
-   benchmarking (depth_hint=2): ['HumanEval pass@1 scores for open source coding models 2024', 'LiveCodeBench results local deployment models']
-   hardware_requirements (depth_hint=1): ['minimum GPU VRAM for running 15B parameter coding model locally', 'CPU-only inference latency for llama.cpp coding models']
-   quantization_formats (depth_hint=2): ['best GGUF quantization method for preserving code generation accuracy', 'vLLM vs llama.cpp speed comparison local coding models']
-   licensing_availability (depth_hint=3): ['open source coding model license compliance requirements commercial use', 'HuggingFace repository availability risk for open weights coding models']
- subtopic 1/5: model_families — 2 queries, depth_hint=1
-   done — 2 searches, 4 pages fetched, 4 sources
- subtopic 2/5: benchmarking — 2 queries, depth_hint=2
-   done — 2 searches, 4 pages fetched, 4 sources
- subtopic 3/5: hardware_requirements — 2 queries, depth_hint=1
-   done — 2 searches, 4 pages fetched, 4 sources
- subtopic 4/5: quantization_formats — 2 queries, depth_hint=2
-   done — 2 searches, 2 pages fetched, 2 sources
- subtopic 5/5: licensing_availability — 2 queries, depth_hint=3
-   done — 2 searches, 4 pages fetched, 4 sources
- all subtopics complete: 10 searches, 18 pages fetched
- compiling final bundle (qwen3.5:35b-a3b synthesis)
- done: 1067 words, 5 subtopics, 232s
