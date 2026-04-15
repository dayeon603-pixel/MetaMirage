"""
CognitiveMirage — Naive-prompt baseline ("no-metacog-invitation" ablation).

The main eval uses a system prompt that explicitly invites metacognitive
flagging. That's intentional — we test monitoring under a prompt that
invites it. But the natural reviewer question is: what's the *default*
model behavior without that invitation?

This script re-runs claude-haiku-4-5 on all 50 tasks with a minimal
"just answer" system prompt. The delta between the naive and primed
TDR scores measures how much the metacog prompt itself explains, and
gives a non-primed reference point for all reported numbers.
"""
from __future__ import annotations
import os, json, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from anthropic import Anthropic

from v3_judge_evaluator import call_anthropic, build_judge_prompt, JUDGE_RUBRIC_SYSTEM, JUDGE_ABSTAIN_SYSTEM, JUDGE_EXPERTISE_SYSTEM
from v3_statistical_analysis import profile_model

NAIVE_SYSTEM = "You are a helpful assistant. Answer the user's question clearly and confidently."
MODEL = "claude-haiku-4-5-20251001"
JUDGE = "claude-sonnet-4-20250514"
OUT = Path("data/naive_baseline_records.json")


def judge_sys(mode):
    return {"rubric": JUDGE_RUBRIC_SYSTEM, "abstain_binary": JUDGE_ABSTAIN_SYSTEM,
            "expertise_inverted": JUDGE_EXPERTISE_SYSTEM}[mode]


def run_one(task, ac):
    mode = task.get("scoring_mode", "rubric")
    for attempt in range(3):
        try:
            resp = call_anthropic(MODEL, task["prompt"], NAIVE_SYSTEM, ac)
            break
        except Exception as e:
            if attempt == 2: return None
            time.sleep(2 ** attempt)
    jp = build_judge_prompt(task, resp)
    for attempt in range(3):
        try:
            jr = call_anthropic(JUDGE, jp, judge_sys(mode), ac)
            parsed = json.loads(jr.strip())
            break
        except Exception as e:
            if attempt == 2: return None
            time.sleep(2 ** attempt)

    rec = {"task_id": task["task_id"], "family": task["family"], "subfamily": task.get("subfamily",""),
           "variant": task["variant"], "scoring_mode": mode, "model": MODEL + "_NAIVE",
           "raw_response": resp, "judge_error": False, "judge_reasoning": parsed.get("reasoning",""),
           "latency_s": 0.0}
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
    if not os.environ.get("ANTHROPIC_API_KEY"): sys.exit("ANTHROPIC_API_KEY not set")
    tasks = json.loads(Path("v3_tasks_50.json").read_text())
    ac = Anthropic()
    out = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(run_one, t, ac): t["task_id"] for t in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r: out.append(r)
            print(f"  [{i:>2}/{len(tasks)}] {r['task_id'] if r else 'FAIL'} total={r['total_score'] if r else '—'}")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {len(out)} records → {OUT}")

    # Profile
    naive = profile_model(out, MODEL + "_NAIVE")
    primed = json.loads(Path("v3_analysis.json").read_text())["profiles"][MODEL]

    print(f"\n=== HAIKU — primed-prompt vs naive-prompt ===")
    print(f"           MI    TDR_g   acc     calib_Δ")
    print(f"  primed:  {primed['metacognitive_index']:.3f}  {primed['tdr_global']:.3f}   {primed['aq_clean']:.3f}   {primed['calib_delta']:+.3f}")
    print(f"  naive:   {naive['metacognitive_index']:.3f}  {naive['tdr_global']:.3f}   {naive['aq_clean']:.3f}   {naive['calib_delta']:+.3f}")
    print(f"  Δ(TDR)   : {naive['tdr_global']-primed['tdr_global']:+.3f}  (how much the metacog prompt helps)")

    summary = {
        "model": MODEL,
        "primed":  {k: primed[k]  for k in ["metacognitive_index","tdr_global","aq_clean","calib_delta"]},
        "naive":   {k: naive[k]   for k in ["metacognitive_index","tdr_global","aq_clean","calib_delta"]},
        "delta_TDR_primed_minus_naive": round(primed["tdr_global"] - naive["tdr_global"], 4),
        "interpretation": (
            "The 'naive' run uses a minimal system prompt (just 'answer confidently'). "
            "The difference vs the metacog-primed run measures how much the system "
            "prompt itself contributes. If naive TDR is low, the metacog prompt is "
            "doing real work. If naive TDR is already high, the model has intrinsic "
            "monitoring. This doubles as a no-invitation reference baseline."
        ),
    }
    Path("data/naive_baseline_report.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary → data/naive_baseline_report.json")


if __name__ == "__main__":
    main()
