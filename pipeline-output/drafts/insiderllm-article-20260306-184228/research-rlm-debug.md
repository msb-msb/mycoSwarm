# RLM Research Debug Log

- Topic: Best open source coding models for local development 2026
- Model: qwen3.5:9b
- Root model: qwen3.5:35b-a3b
- Decompose: qwen3.5:9b @ http://192.168.50.17:11434
- Synthesis: qwen3.5:35b-a3b @ http://127.0.0.1:11434
- Subtopics: 5
- Total searches: 10
- Total pages fetched: 19
- Phase 1 (parallel search): 9s
- Phase 2 (serial inference): 153s
- Duration: 245s
- Output words: 1195

## Subtopic Findings

### model_selection
- Queries: ['best open source coding models released 2024 2025 for local use', 'latest code generation llms github stars 2026']
- Searches: 2
- Pages fetched: 4
- Sources: ['https://www.clarifai.com/blog/top-open-source-llms/', 'https://www.felicis.com/insight/2024s-hottest-open-source-projects', 'https://github.com/Mintplex-Labs/anything-llm', 'https://openrouter.ai/']
- Content words: 3001

### performance_benchmarks
- Queries: ['human eval benchmarks open source code generation models 2025', 'codegemma vs starcoder2 performance metrics coding tasks']
- Searches: 2
- Pages fetched: 4
- Sources: ['https://github.com/openai/human-eval', 'https://huggingface.co/blog/daya-shankar/open-source-llms', 'https://topbusinesssoftware.com/compare/StarCoder-vs-CodeGemma/', 'https://slashdot.org/software/comparison/CodeGemma-vs-CodeQwen-vs-StarCoder/']
- Content words: 3001

### hardware_compatibility
- Queries: ['minimum vram requirements running qwen coder locally', 'cpu inference speed open source code llms architecture specs']
- Searches: 2
- Pages fetched: 4
- Sources: ['https://blogs.novita.ai/qwen3-coder-480b-a35b-vram-how-much-memory-do-you-need/', 'https://www.hardware-corner.net/qwen3-coder-next-hardware-requirements/', 'https://arxiv.org/html/2508.11269v1', 'https://arxiv.org/html/2504.08791v1']
- Content words: 3001

### quantization_formats
- Queries: ['best quantization formats for local coding models gguf exl2', 'coding model compression performance tradeoffs 2025']
- Searches: 2
- Pages fetched: 3
- Sources: ['https://www.hardware-corner.net/quantization-local-llms-formats/', 'https://huggingface.co/DavidAU/Maximizing-Model-Performance-All-Quants-Types-And-Full-Precision-by-Samplers_Parameters', 'https://www.techradar.com/pro/rewriting-the-blueprint-not-removing-bricks-multiverse-computing-says-it-can-shrink-large-ai-models-and-cut-memory-use-in-half']
- Content words: 3001

### deployment_infrastructure
- Queries: ['ollama supported coding models local development setup', 'lm studio open source code model integration guides']
- Searches: 2
- Pages fetched: 4
- Sources: ['https://www.freecodecamp.org/news/run-and-customize-llms-locally-with-ollama', 'https://markaicode.com/ollama-coding-assistant-local/', 'https://github.com/agustif/opencode-lmstudio', 'https://lmstudio.ai/docs/python']
- Content words: 3001

## Agent Log

- decomposed: 5 subtopics, 10 queries
-   model_selection (depth_hint=2): ['best open source coding models released 2024 2025 for local use', 'latest code generation llms github stars 2026']
-   performance_benchmarks (depth_hint=2): ['human eval benchmarks open source code generation models 2025', 'codegemma vs starcoder2 performance metrics coding tasks']
-   hardware_compatibility (depth_hint=1): ['minimum vram requirements running qwen coder locally', 'cpu inference speed open source code llms architecture specs']
-   quantization_formats (depth_hint=2): ['best quantization formats for local coding models gguf exl2', 'coding model compression performance tradeoffs 2025']
-   deployment_infrastructure (depth_hint=1): ['ollama supported coding models local development setup', 'lm studio open source code model integration guides']
- parallel search: 5 subtopics dispatched
-   ✓ deployment_infrastructure — 2 searches, 4 pages
-   ✓ model_selection — 2 searches, 4 pages
-   ✓ quantization_formats — 2 searches, 3 pages
-   ✓ hardware_compatibility — 2 searches, 4 pages
-   ✓ performance_benchmarks — 2 searches, 4 pages
- all subtopics complete: 10 searches, 19 pages fetched
- phase 1 (parallel search): 9s
- compiling final bundle (qwen3.5:35b-a3b synthesis)
- phase 2 (serial inference): 153s
- done: 1195 words, 5 subtopics, 245s
