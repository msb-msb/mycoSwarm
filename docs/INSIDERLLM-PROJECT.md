# InsiderLLM Project Memory

## Reference: Sky Memory System
- GitHub: https://github.com/jbbottoms/sky-memory-system
- Pattern: NOW.md (current state) + MEMORY.md (long-term) + ChromaDB (semantic search) + SQLite (relationships)
- Solves: Context compaction amnesia in persistent AI sessions

---

## PROJECT: InsiderLLM

### Current State (NOW)

**Site:** https://insiderllm.com
**Hosting:** DreamHost (SSH deploy)
**Tech:** Hugo static site
**Repo:** ~/Desktop/InsiderLLM/insiderllm-hugo

**Articles Published:** 57
**Article Goal:** 100 (30 by end of Week 1)

**Reddit Account:** LocalLLMHobbyist
**Reddit Status:** Building karma (Week 1), no promo links yet
**Comments Posted:** ~8

**Google Search Console:** Set up, awaiting indexing (1-3 days)
**Email:** Buttondown integration on /subscribed/

---

### Key Files & Locations

| What | Where |
|------|-------|
| Hugo site | ~/Desktop/InsiderLLM/insiderllm-hugo |
| Content | ~/Desktop/InsiderLLM/insiderllm-hugo/content/guides/ |
| Deploy script | ./deploy.sh |
| CC Skill | /mnt/skills/user/insiderllm-writer |
| Content plan | insiderllm-content-plan.md (100 articles) |
| Transcripts | /mnt/transcripts/ |

---

### Brand Guidelines

**Voice:** Practical, honest, budget-focused, no fluff
**Audience:** Hobbyists, developers, privacy-focused users with modest hardware
**Sweet spot:** "What can I run on X VRAM?" questions
**Differentiator:** Real experience, not marketing copy

**Avoid:** Enterprise, cloud-first, overly theoretical

---

### Content Categories (from master plan)

1. Hardware & VRAM (25 articles) — 21 done
2. Software & Tools (20 articles) — 13 done
3. Models & Use Cases (25 articles) — 10 done
4. Image & Creative AI (12 articles) — 3 done
5. Advanced Topics (10 articles) — 6 done
6. Comparisons & Alternatives (8 articles) — 4 done

---

### Reddit Strategy

**Week 1 (current):**
- Build karma with helpful comments
- Target "Question | Help" posts
- Focus: budget hardware, model selection, beginner setup
- NO promotional links

**Week 2:**
- First article link (VRAM guide or 3090 guide)
- 80/20 rule: 80% helpful, 20% promo
- Continue engaging authentically

---

### Recent Progress (Last 3 Sessions)

**Jan 28:**
- GSC setup, About page, email capture
- Articles 5-10 (Ollama vs LM Studio, First LLM, Budget PC, Quantization, AMD vs NVIDIA)

**Jan 29:**
- Internal linking across all articles
- Homepage redesign (8 need-based cards)
- Reddit strategy planning
- Article 11 (8GB VRAM guide)

**Jan 30:**
- Reddit account created (LocalLLMHobbyist)
- 5+ comments posted, building karma
- Articles 12-20 (12GB, Coding, CPU-only, SD, 24GB, 3B models, 16GB, Mac vs PC, Context Length)
- GSC indexing requested
- Created 100-article master plan

---

### Next Session Priorities

1. Deploy latest articles: `./deploy.sh`
2. Knock out 3-4 more articles toward 30
3. Check Reddit for replies
4. Monitor GSC for indexing progress

---

### Recurring Tasks

- **Daily:** 2-3 Reddit comments on r/LocalLLaMA
- **Daily:** 2-4 new articles via CC
- **Weekly:** Check GSC data, update content plan
- **Weekly:** Review Reddit engagement, adjust strategy

---

### Key Decisions Made

- Hugo over WordPress (speed, simplicity)
- DreamHost shared hosting (cheap, works)
- Buttondown for email (free tier)
- Reddit over Twitter for initial traffic
- Budget/hobbyist focus over enterprise
- Used 3090 as flagship recommendation

---

### Traffic & Revenue Strategy

**3-Engine Traffic System:**
- Engine 1: SEO + LLM-Search (long-term, biggest driver)
- Engine 2: Social Discovery (Reddit, X, Bluesky — fastest traffic)
- Engine 3: Authority Backlinks (GitHub repo, HN, newsletters)

**Revenue Targets ($1000/month):**
- Amazon Affiliate: $300-600/month (GPU guides)
- Digital Product: $200-400/month ("Local LLM Handbook", "Used GPU Buyer's Checklist" — $10-20)
- Newsletter Sponsorships: $200-400/month (once 1,500+ subscribers)

**Traffic Projections:**
- March-April: 50-150 daily visitors
- May-June: 150-300 daily visitors
- July-September: 300-600 daily visitors
- October-December: 600-1,200 daily visitors

**Missing Infrastructure (TODO):**
- [ ] Schema markup (FAQ, HowTo, Product) — real SEO boost
- [ ] Breadcrumbs on site
- [ ] Digital product (weekend project)
- [ ] Glossary page

---

### Monetization Setup (Feb 4)

- Amazon Associates: approved, store ID = insiderllm-20
- Affiliate links live in: gpu-buying-guide, vram-requirements, budget-local-ai-pc-500, used-rtx-3090-buying-guide
- Footer disclosure added
- TODO: Sign up for Newegg affiliate, eBay Partner Network

---

### Accounts

- **Reddit:** LocalLLMHobbyist (karma issues — possibly bot downvotes, keep posting helpful content)
- **Hacker News:** insiderllm
- **Bluesky:** active
- **Buttondown:** connected for newsletter signups

---

### GSC Insights (Feb 4)

- Top page: llama.cpp vs Ollama vs vLLM (370 impressions, 2 clicks)
- Comparison articles ("X vs Y") outperform single-topic guides
- Specific download queries appearing (text-generation-webui cuda versions)

---

### Meta Description A/B Tests

Tracking CTR changes after meta description rewrites. Baseline period: Jan 27 - Feb 17, 2026.

| Page | Old Description | New Description | Date Changed | Baseline CTR | Baseline Impressions |
|------|----------------|-----------------|-------------|-------------|---------------------|
| vram-requirements-local-llms | Llama 3 70B needs 40GB at Q4, Mixtral 8x7B fits in 24GB, 7B models run on 6GB. Complete VRAM charts for every model size and quantization—plus which GPU to buy. | 3B models need 2GB. 7B needs 5GB. 70B needs 40GB. Exact VRAM requirements for every model size at Q4 through FP16, plus which GPU to buy at every budget. | Feb 18, 2026 | 0% (1,339 imp) | 1,339 |
| comfyui-vs-automatic1111-vs-fooocus | ComfyUI vs Automatic1111 vs Fooocus compared: VRAM usage, speed benchmarks, and Flux support in 2026. ComfyUI hits 8 seconds for SDXL at 9.2GB. Which UI fits your workflow? | Fooocus if you want results in 5 minutes. ComfyUI if you want total control. A1111 if you're already using it. Honest comparison with speed and VRAM benchmarks. | Feb 18, 2026 | 0% (396 imp) | 396 |
| text-generation-webui-oobabooga-guide | Supports GGUF, GPTQ, EXL2, and AWQ across 4 backends with extensions for voice, RAG, and LoRA training. When text-generation-webui beats Ollama and LM Studio. | Install text-generation-webui in 10 minutes. GPU offloading, GGUF/GPTQ/EXL2 model loading, extensions, and the settings most guides skip. Practical setup. | Feb 18, 2026 | 0.1% (750 imp) | 750 |
| budget-local-ai-pc-500 | Dell Optiplex ($100-150) + used RTX 3060 12GB ($170-200) = a machine that runs 13B LLMs and Stable Diffusion for under $450. Sample builds and what each budget level gets you. | A used Dell Optiplex + RTX 3060 12GB runs 14B LLMs and Stable Diffusion for under $450. Full parts list, real speed benchmarks, and what to skip to save money. | Feb 18, 2026 | 0.2% (584 imp) | 584 |

Check results: Feb 25, 2026 (1 week) and March 4, 2026 (2 weeks)

---

### GoatCounter (Feb 4)

- First real traffic: 30+ visitors
- Top pages: Mac M-series (8), OpenClaw models (5), llama.cpp comparison (4)

---

### Content Sources

- Matt Ganzac (OpenClaw token optimization) — ~/Desktop/InsiderLLM/openclaw-token-optimization-transcript.txt

---

### Open Questions / Future Ideas

- [ ] Add search to site?
- [ ] YouTube channel eventually?
- [x] Affiliate links (Amazon, eBay)? — Done, integrated
- [ ] Newsletter content strategy?
- [ ] Guest posts / collaborations?

---

## Related Project: Wise Advisor AI

**Status:** Active development (separate project)
**Tech:** Python, Ollama, ChromaDB, sentence-transformers
**Domains:** tai_chi_chuan, IFS, Taoism, Buddhism, etc.
**Memory reference:** Sky Memory System (see top of file)

---

## Site Health & SEO Auditing

### Audit Document
- INSIDERLLM-SEO-AUDIT.md is the single source of truth for all SEO checks
- Section 8 contains the CC-ready audit command

### When to Run Audits
- After every batch of new articles (before deploy)
- After any slug, title, or description changes
- Weekly Sunday maintenance (Section 7 checklist)
- After any deploy.sh changes or Hugo config changes

### Post-Deploy Verification
- After deploy, curl all changed URLs to confirm HTTP 200
- Verify IndexNow ping returned HTTP 202
- Run Section 4a critical files check

### Broken Link Prevention
- Never change a published slug
- Always include trailing slash on internal links (/guides/slug-name/)
- Run Section 8 audit #5 (internal link check) before every deploy

---

*Last updated: Feb 6, 2026*
