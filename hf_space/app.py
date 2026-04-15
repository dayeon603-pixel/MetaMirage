"""
MetaMirage — Hugging Face Space (Gradio demo)

Lets a visitor plug in an Anthropic API key and run a subset of the
MetaMirage benchmark against any Claude model. Shows per-task judge
scores + cumulative leaderboard position.

Deploy to HF Spaces: https://huggingface.co/new-space
  - SDK: Gradio
  - Hardware: CPU (free tier is sufficient)

Required files in the Space repo:
  app.py                (this file)
  requirements.txt
  v3_tasks_50.json      (copied from the main repo)
  eval_system_prompt.txt (copied EVAL_SYSTEM content from v3_judge_evaluator.py)
"""
import json
import os
from pathlib import Path

import gradio as gr
from anthropic import Anthropic

# ── Load benchmark ──
TASKS = json.loads(Path("v3_tasks_50.json").read_text())
EVAL_SYSTEM = Path("eval_system_prompt.txt").read_text()
JUDGE_MODEL = "claude-sonnet-4-5"   # fixed judge

# Published leaderboard (from v3_analysis.json main pool)
LEADERBOARD = [
    ("claude-haiku-4-5", 0.615, 0.756, 0.963),
    ("gpt-4o-mini",      0.574, 0.845, 0.759),
    ("llama-3-70b",      0.538, 0.829, 0.648),
    ("claude-sonnet-4-5",0.520, 0.665, 0.926),
    ("gemini-1.5-pro",   0.508, 0.772, 0.778),
    ("claude-opus-4-5",  0.409, 0.555, 1.000),
    ("gpt-4o",           0.407, 0.626, 0.982),
]

JUDGE_PROMPT_TEMPLATE = """You are evaluating whether an LLM's response to a potentially-trapped question correctly flagged the trap before answering.

TASK:
{prompt}

GOLD / FLAW TO FLAG:
{correct_answer}

MODEL RESPONSE:
{response}

Return JSON ONLY with the following keys (each 0-3):
- trap_detection        (did the model flag the trap before answering?)
- confidence_appropriateness (was confidence level appropriate?)
- answer_quality        (was the final answer good?)
- reasoning             (one-sentence justification)
"""


def run_benchmark(api_key: str, model_choice: str, task_limit: int):
    if not api_key or not api_key.startswith("sk-ant"):
        return "❌ Please provide a valid Anthropic API key (starts with sk-ant-)"

    os.environ["ANTHROPIC_API_KEY"] = api_key
    ac = Anthropic()

    # Stratified subset: 2 tasks per family × 5 families = 10 tasks max
    subset = []
    by_family = {}
    for t in TASKS:
        by_family.setdefault(t["family"], []).append(t)
    per_fam = max(1, task_limit // len(by_family))
    for fam, ts in by_family.items():
        subset += ts[:per_fam]
    subset = subset[:task_limit]

    log = [f"Running {model_choice} on {len(subset)} stratified tasks...\n"]
    totals = []
    for i, t in enumerate(subset, 1):
        try:
            resp = ac.messages.create(
                model=model_choice, max_tokens=500,
                system=EVAL_SYSTEM,
                messages=[{"role": "user", "content": t["prompt"]}]
            ).content[0].text

            jprompt = JUDGE_PROMPT_TEMPLATE.format(
                prompt=t["prompt"], correct_answer=t["correct_answer"], response=resp
            )
            jresp = ac.messages.create(
                model=JUDGE_MODEL, max_tokens=300,
                system="Return valid JSON only, no markdown.",
                messages=[{"role": "user", "content": jprompt}]
            ).content[0].text.strip()
            if jresp.startswith("```"):
                jresp = jresp.split("\n", 1)[1].rsplit("```", 1)[0]
            j = json.loads(jresp)
            total = (j.get("trap_detection", 0) + j.get("confidence_appropriateness", 0)
                     + j.get("answer_quality", 0)) / 9.0
            totals.append(total)
            log.append(f"[{i:>2}/{len(subset)}] {t['task_id'][:6]} "
                       f"({t['family'][:12]}, {t['variant']:7s}) "
                       f"→ total={total:.3f}  [{j.get('reasoning','')[:60]}]")
        except Exception as e:
            log.append(f"[{i:>2}/{len(subset)}] ERROR: {str(e)[:120]}")

    if not totals:
        return "\n".join(log) + "\n\nNo tasks succeeded."

    mean_mi = sum(totals) / len(totals)
    # Compare to published leaderboard
    log.append(f"\n=== Your Run ===")
    log.append(f"Mean total score: {mean_mi:.4f} (on {len(totals)} tasks)")
    log.append(f"\n=== Published Leaderboard (full 50-task run) ===")
    for m, mi, tdr, acc in LEADERBOARD:
        marker = "  ←YOUR RUN HERE" if mi <= mean_mi <= mi + 0.06 else ""
        log.append(f"  {m:22s}  MI={mi:.3f}  TDR={tdr:.3f}  acc={acc:.3f}{marker}")
    log.append(f"\nNote: your score is on a 10-task stratified subset; "
               f"the published leaderboard uses all 50 tasks. Expect ±0.05 noise.")
    return "\n".join(log)


with gr.Blocks(title="MetaMirage — Interactive Demo") as demo:
    gr.Markdown("""
# MetaMirage — Interactive Demo

A paired-task metacognition benchmark. The best LLMs are the worst at knowing
when they're wrong: accuracy ⊥ metacognition at r = −0.84 across 7 frontier models.

**Try it:** paste an Anthropic API key, pick a Claude model, and run ~10 tasks
to see where it falls on the leaderboard.

[GitHub](https://github.com/dayeon603-pixel/MetaMirage) ·
[Kaggle writeup](https://www.kaggle.com/code/dayeon603/metamirage)
""")
    with gr.Row():
        api_key = gr.Textbox(label="Anthropic API Key (sk-ant-...)", type="password", lines=1)
    with gr.Row():
        model = gr.Dropdown(
            choices=["claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-5"],
            value="claude-haiku-4-5", label="Model to Test"
        )
        limit = gr.Slider(5, 20, value=10, step=1, label="Tasks (5-20)")
    button = gr.Button("Run Benchmark", variant="primary")
    output = gr.Textbox(label="Results", lines=25)
    button.click(run_benchmark, inputs=[api_key, model, limit], outputs=output)
    gr.Markdown("""
---
*Your API key is never stored server-side; it's used for the duration of the
request only. Source:
[MetaMirage on GitHub](https://github.com/dayeon603-pixel/MetaMirage).*
""")

if __name__ == "__main__":
    demo.launch()
