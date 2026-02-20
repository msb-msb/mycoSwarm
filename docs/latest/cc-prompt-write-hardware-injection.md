# Hardware Self-Injection in /write Article Mode

## What This Is

Monica runs on a 5-node swarm with a 3090 and real Ollama models. When
writing articles about hardware, models, or VRAM — she should reference
her own actual data instead of guessing. This auto-injects hardware context
into article mode.

## What Gets Injected

When /write activates, automatically gather and inject:

1. **Ollama model list** — what models are actually installed, their sizes
2. **Node hardware specs** — from the /status API endpoint
3. **Recent session tok/s** — actual inference speeds from session history

## Implementation

### Step 1: Gather hardware context on /write activation

```python
import subprocess
import json

def _gather_hardware_context() -> str:
    """Gather local hardware data for article context."""
    context_parts = []

    # 1. Ollama model list
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            context_parts.append("## Installed Ollama Models (this node)\n")
            context_parts.append(f"```\n{result.stdout.strip()}\n```\n")
    except Exception:
        pass

    # 2. Node hardware from daemon /status
    try:
        import httpx
        headers = _swarm_headers()  # reuse existing auth helper
        resp = httpx.get("http://localhost:7890/status", headers=headers, timeout=5)
        if resp.status_code == 200:
            status = resp.json()
            context_parts.append("## This Node Hardware\n")
            context_parts.append(f"- Hostname: {status.get('hostname', 'unknown')}")
            context_parts.append(f"- GPU: {status.get('gpu_name', 'unknown')}")
            context_parts.append(f"- VRAM: {status.get('vram_used', '?')}/{status.get('vram_total', '?')} MB")
            context_parts.append(f"- CPU: {status.get('cpu_model', 'unknown')}")
            context_parts.append(f"- RAM: {status.get('ram_used', '?')}/{status.get('ram_total', '?')} GB")
            context_parts.append("")
    except Exception:
        pass

    # 3. Swarm peer overview
    try:
        resp = httpx.get("http://localhost:7890/peers", headers=headers, timeout=5)
        if resp.status_code == 200:
            peers = resp.json()
            if peers:
                context_parts.append(f"## Swarm: {len(peers) + 1} nodes total\n")
                for p in peers:
                    name = p.get("hostname", "unknown")
                    device = p.get("device_type", "unknown")
                    context_parts.append(f"- {name}: {device}")
                context_parts.append("")
    except Exception:
        pass

    if not context_parts:
        return ""

    header = ("## Hardware Context (from this mycoSwarm node)\n"
              "Use this real data when writing about hardware, models, or VRAM.\n"
              "These are actual specs from the author's setup.\n\n")
    return header + "\n".join(context_parts)
```

**NOTE TO CC:** Adapt the /status field names to match what your API
actually returns. Check the status endpoint response format and use the
correct keys.

### Step 2: Gather recent tok/s from session history

Check the last few sessions for actual inference speed data:

```python
def _gather_recent_performance() -> str:
    """Pull recent tok/s data from session files."""
    import glob
    import os

    sessions_dir = os.path.expanduser("~/.config/mycoswarm/sessions")
    if not os.path.exists(sessions_dir):
        return ""

    # Find recent session files
    session_files = sorted(glob.glob(os.path.join(sessions_dir, "*.jsonl")),
                          key=os.path.getmtime, reverse=True)[:5]

    tok_s_data = []
    for sf in session_files:
        try:
            with open(sf) as f:
                for line in f:
                    data = json.loads(line)
                    if "tok_s" in data and "model" in data:
                        tok_s_data.append({
                            "model": data["model"],
                            "tok_s": data["tok_s"],
                        })
        except Exception:
            continue

    if not tok_s_data:
        return ""

    # Deduplicate and average by model
    from collections import defaultdict
    model_speeds = defaultdict(list)
    for d in tok_s_data:
        model_speeds[d["model"]].append(d["tok_s"])

    context = "## Actual Inference Speeds (from recent sessions)\n\n"
    context += "| Model | Avg tok/s | Samples |\n"
    context += "|-------|-----------|----------|\n"
    for model, speeds in sorted(model_speeds.items()):
        avg = sum(speeds) / len(speeds)
        context += f"| {model} | {avg:.1f} | {len(speeds)} |\n"
    context += "\n"

    return context
```

**NOTE TO CC:** Adapt this to match the actual session file format. The
key is finding where tok/s and model name are stored per message. Check
the session JSONL structure.

### Step 3: Inject into /write activation

In the /write handler, right after activating article mode, gather and
inject the hardware context:

```python
# In /write handler, after setting up article mode:
hardware_context = _gather_hardware_context()
performance_context = _gather_recent_performance()

if hardware_context or performance_context:
    hw_msg = f"""## Your Hardware (Real Data)

You are writing from firsthand experience on this hardware. Use these
real numbers instead of guessing. When you cite specs or tok/s, note
they come from actual testing on your setup.

{hardware_context}
{performance_context}
"""
    messages.append({"role": "system", "content": hw_msg})
    print("   📊 Loaded hardware context from swarm")
```

### Step 4: Print what was loaded

Show the Guardian what Monica can see:

```python
if hardware_context:
    print("   📊 Hardware: GPU, VRAM, CPU, RAM from /status")
if performance_context:
    print("   📊 Performance: tok/s from recent sessions")

model_list = subprocess.run(["ollama", "list"], capture_output=True, text=True)
if model_list.returncode == 0:
    model_count = len(model_list.stdout.strip().split("\n")) - 1  # minus header
    print(f"   📊 Models: {model_count} Ollama models available")
```

## Why This Matters

The InsiderLLM voice is "I figured this out so you don't have to." Monica
literally runs these models on her own hardware. If she reports "R1:14b
runs at 33 tok/s on a 3090" — that's not a benchmark she read somewhere,
it's her lived experience. That authenticity is what differentiates
InsiderLLM from every other AI blog.

## Test

```bash
mycoswarm chat
/write "DeepSeek Models Guide"
```

Should see:
```
✍️  Article mode activated: "DeepSeek Models Guide"
   Drafts will be saved to: /home/minotaur/insiderllm-drafts/
   📊 Hardware: GPU, VRAM, CPU, RAM from /status
   📊 Performance: tok/s from recent sessions
   📊 Models: 20 Ollama models available
   Searching for style guide and content plan...
```

And the draft should include real tok/s numbers and actual model sizes
from Ollama, not hallucinated specs.
