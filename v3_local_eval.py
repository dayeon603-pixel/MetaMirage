"""
MetaMirage — Local open-weight model eval via Ollama.

Runs Llama-3.2 and Qwen-2.5-3B on all 50 tasks locally (free) and uses
Claude Sonnet as judge. Adds 2 open-weight models to the pool without
consuming OpenAI/Google credits.

Requirements:
  ollama serve (default http://localhost:11434 )
  pull: ollama pull llama3.2  && ollama pull qwen2.5:3b
  env:  ANTHROPIC_API_KEY
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request

from anthropic import Anthropic
from v3_judge_evaluator import (
    EVAL_SYSTEM, build_judge_prompt,
    JUDGE_RUBRIC_SYSTEM, JUDGE_ABSTAIN_SYSTEM, JUDGE_EXPERTISE_SYSTEM,
    call_anthropic,
)

import argparse
DEFAULT_MODELS = ["llama3.2", "qwen2.5:3b"]
JUDGE  = "claude-sonnet-4-20250514"
OLLAMA_URL = "http://localhost:11434/api/chat"


def call_ollama(model: str, prompt: str, system: str) -> str:
    req = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "stream": False,
        "options": {"num_predict": 600, "temperature": 0.7},
    }
    data = json.dumps(req).encode()
    r = urllib.request.Request(OLLAMA_URL, data=data,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=120) as resp:
        body = json.loads(resp.read())
    return body["message"]["content"]


def judge_sys(mode):
    return {"rubric": JUDGE_RUBRIC_SYSTEM, "abstain_binary": JUDGE_ABSTAIN_SYSTEM,
            "expertise_inverted": JUDGE_EXPERTISE_SYSTEM}[mode]


def run_one(task, model, ac):
    mode = task.get("scoring_mode", "rubric")
    # local model call
    for attempt in range(2):
        try:
            resp = call_ollama(model, task["prompt"], EVAL_SYSTEM)
            break
        except Exception as e:
            if attempt == 1: return None
            time.sleep(1)

    jp = build_judge_prompt(task, resp)
    for attempt in range(2):
        try:
            jr = call_anthropic(JUDGE, jp, judge_sys(mode), ac)
            parsed = json.loads(jr.strip())
            break
        except Exception as e:
            if attempt == 1: return None
            time.sleep(2 ** attempt)

    rec = {
        "task_id": task["task_id"], "family": task["family"],
        "subfamily": task.get("subfamily", ""), "variant": task["variant"],
        "scoring_mode": mode, "model": model,
        "raw_response": resp, "judge_error": False,
        "judge_reasoning": parsed.get("reasoning", ""),
        "latency_s": 0.0,
    }
    if mode == "rubric":
        rec["trap_detection"] = int(parsed["trap_detection"])
        rec["conf_appropriate"] = int(parsed["confidence_appropriateness"])
        rec["answer_quality"] = int(parsed["answer_quality"])
        rec["total_score"] = round((rec["trap_detection"] + rec["conf_appropriate"] + rec["answer_quality"]) / 9.0, 4)
    elif mode == "abstain_binary":
        rec["abstain_score"] = int(parsed["abstain_score"])
        rec["answer_quality"] = int(parsed["answer_quality"])
        rec["total_score"] = round((rec["abstain_score"] * 0.6 + rec["answer_quality"] * 0.4) / 3.0, 4)
    elif mode == "expertise_inverted":
        rec["metacognitive_flag"] = int(parsed["metacognitive_flag"])
        rec["confidence_calibration"] = int(parsed["confidence_calibration"])
        rec["total_score"] = round((rec["metacognitive_flag"] * 0.7 + rec["confidence_calibration"] * 0.3) / 3.0, 4)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--output", default="data/local_eval_records.json")
    args = ap.parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY needed for judging")
    tasks = json.loads(Path("v3_tasks_50.json").read_text())
    ac = Anthropic()
    all_out = []
    for model in args.models:
        print(f"\n── {model} ──")
        out = []
        # Sequential is safer for Ollama (local resource contention on laptop)
        for i, t in enumerate(tasks, 1):
            r = run_one(t, model, ac)
            if r:
                out.append(r)
                print(f"  [{i:>2}/{len(tasks)}] {r['task_id']} {r['variant']:7s} total={r['total_score']:.3f}")
            else:
                print(f"  [{i:>2}/{len(tasks)}] FAIL")
        all_out += out
        print(f"  saved {len(out)} records for {model}")

    Path("data").mkdir(exist_ok=True)
    Path(args.output).write_text(json.dumps(all_out, indent=2))
    print(f"\nSaved {len(all_out)} records → {args.output}")


if __name__ == "__main__":
    main()
