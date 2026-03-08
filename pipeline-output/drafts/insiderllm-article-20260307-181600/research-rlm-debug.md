# RLM Research Debug Log

- Topic: Best open source coding models for local development 2026
- Model: qwen3.5:9b
- Root model: qwen3.5:35b-a3b
- Decompose: qwen3.5:9b @ http://192.168.50.17:11434
- Synthesis: qwen3.5:35b-a3b @ http://127.0.0.1:11434
- Subtopics: 5
- Total searches: 10
- Total pages fetched: 16
- Phase 1 (parallel search): 16s
- Phase 2 (serial summarize): 294s
- Phase 3 (synthesis): 203s
- Duration: 572s
- Output words: 1110

## Subtopic Findings

### model_selection
- Queries: ['best open source coding LLMs released 2024-2026 comparison', 'top code generation models local inference specs Qwen-Coder StarCoder']
- Searches: 2
- Pages fetched: 4
- Sources: ['https://onyx.app/open-llm-leaderboard', 'https://www.bentoml.com/blog/navigating-the-world-of-open-source-large-language-models', 'https://www.labellerr.com/blog/best-coding-llms/', 'https://arxiv.org/html/2410.14766v1']
- Content words: 843

### performance_benchmarks
- Queries: ['HumanEval scores open source coding models 2026', 'LiveCodeBench results Qwen-Coder vs StarCoder local']
- Searches: 2
- Pages fetched: 3
- Sources: ['https://vertu.com/lifestyle/open-source-llm-leaderboard-2026-rankings-benchmarks-the-best-models-right-now/?srsltid=AfmBOopmLcRCBcgAOkYJCF0de9mJkp2G04_FkZ3iEk5xt0Qt4ivCEBGv', 'https://arxiv.org/html/2510.05788v1', 'https://arxiv.org/html/2512.18456v1']
- Content words: 442

### hardware_efficiency
- Queries: ['quantization methods for coding models 4-bit 8-bit local', 'VRAM requirements for running CodeLlama 35B locally']
- Searches: 2
- Pages fetched: 3
- Sources: ['https://synthmetric.com/quantization-in-plain-english-8‑bit-4‑bit-and-what-you-lose/', 'https://unsloth.ai/docs/models/tutorials/qwen3-coder-how-to-run-locally', 'https://ollama.com/blog/run-code-llama-locally']
- Content words: 618

### deployment_tooling
- Queries: ['Ollama supported coding models list 2026', 'LM Studio open source code model compatibility']
- Searches: 2
- Pages fetched: 4
- Sources: ['https://clawdbook.org/en/blog/openclaw-best-ollama-models-2026', 'https://ollama.com/blog/coding-models', 'https://lmstudio.ai/', 'https://lmstudio.ai/models']
- Content words: 710

### licensing_availability
- Queries: ['open source coding model licenses MIT Apache 2026', 'commercial use restrictions local coding models availability']
- Searches: 2
- Pages fetched: 2
- Sources: ['https://huggingface.co/blog/daya-shankar/open-source-llms', 'https://local-ai-zone.github.io/guides/ai-model-licensing-complete-legal-guide-2025.html']
- Content words: 862

## Agent Log

- decomposed: 5 subtopics, 10 queries
-   model_selection (depth_hint=1): ['best open source coding LLMs released 2024-2026 comparison', 'top code generation models local inference specs Qwen-Coder StarCoder']
-   performance_benchmarks (depth_hint=2): ['HumanEval scores open source coding models 2026', 'LiveCodeBench results Qwen-Coder vs StarCoder local']
-   hardware_efficiency (depth_hint=2): ['quantization methods for coding models 4-bit 8-bit local', 'VRAM requirements for running CodeLlama 35B locally']
-   deployment_tooling (depth_hint=1): ['Ollama supported coding models list 2026', 'LM Studio open source code model compatibility']
-   licensing_availability (depth_hint=3): ['open source coding model licenses MIT Apache 2026', 'commercial use restrictions local coding models availability']
- parallel search: 5 subtopics dispatched
-   ✓ licensing_availability — 2 searches, 2 pages
-   ✓ model_selection — 2 searches, 4 pages
-   ✓ deployment_tooling — 2 searches, 4 pages
-   ✓ performance_benchmarks — 2 searches, 3 pages
-   ✓ hardware_efficiency — 2 searches, 3 pages
- all subtopics complete: 10 searches, 16 pages fetched
- phase 1 (parallel search): 16s
- serial summarize: 5 subtopics (qwen3.5:9b)
-   ✓ model_selection → 843 words
-   ✓ performance_benchmarks → 442 words
-   ✓ hardware_efficiency → 618 words
-   ✓ deployment_tooling → 710 words
-   ✓ licensing_availability → 862 words
- phase 2 (serial summarize): 294s
- compiling final bundle (qwen3.5:35b-a3b synthesis)
- phase 3 (synthesis): 203s
- done: 1110 words, 5 subtopics, 572s
