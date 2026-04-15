"""
MetaMirage — Absolute calibration analysis (Brier / ECE / reliability).

The main writeup reports *Calibration Delta* (how confidence drops on
mirage vs clean). That's relative calibration. The metacognition literature
also demands *absolute* calibration metrics:

  Brier score:  mean squared error between stated confidence and correctness.
                Lower is better. Breaks into reliability + resolution + uncertainty.
  ECE:          Expected Calibration Error — weighted gap between confidence
                and accuracy across confidence bins.

These are computed on the 3-model Anthropic subset (haiku, opus-4-5, sonnet-4-5)
for which we have per-task rubric scores including confidence-appropriateness.

We derive a model's "stated confidence" from conf_appropriate (0-3 rubric):
  0 → 0.1, 1 → 0.33, 2 → 0.67, 3 → 0.9
and correctness from answer_quality >= 2 (binarized).

Output: data/calibration_report.json
"""
from __future__ import annotations
import json
from pathlib import Path
from statistics import mean

CONF_MAP = {0: 0.10, 1: 0.33, 2: 0.67, 3: 0.90}


def records_for_model(model):
    out = []
    haiku = json.loads(Path("data/haiku_records.json").read_text())
    for r in haiku:
        if r["model"] == model: out.append(r)
    cj = json.loads(Path("data/cross_judge_records.json").read_text())
    for r in cj:
        if r["model"] != model: continue
        s = r["judged"].get("sonnet")
        if not s: continue
        out.append({
            "task_id": r["task_id"], "variant": r["variant"],
            "scoring_mode": r["scoring_mode"],
            "conf_appropriate": s.get("conf_appropriate"),
            "answer_quality":   s.get("answer_quality"),
            "trap_detection":   s.get("trap_detection"),
        })
    return out


def brier(confidences, outcomes):
    return round(mean((c - o) ** 2 for c, o in zip(confidences, outcomes)), 4)


def ece(confidences, outcomes, n_bins=5):
    """Expected calibration error with equal-width bins."""
    total = len(confidences)
    if total == 0: return None
    edges = [i / n_bins for i in range(n_bins + 1)]
    weighted_gap = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        if i == n_bins - 1:
            mask = [(lo <= c <= hi) for c in confidences]
        else:
            mask = [(lo <= c < hi) for c in confidences]
        n_in = sum(mask)
        if n_in == 0: continue
        mean_conf = mean(c for c, m in zip(confidences, mask) if m)
        mean_acc  = mean(o for o, m in zip(outcomes, mask) if m)
        weighted_gap += (n_in / total) * abs(mean_conf - mean_acc)
    return round(weighted_gap, 4)


def main():
    models = ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5-20251001"]
    report = {"method": (
        "Brier score and Expected Calibration Error (ECE, 5 equal-width bins) on "
        "rubric-mode tasks. Confidence derived from conf_appropriate rubric "
        "(0→0.1, 1→0.33, 2→0.67, 3→0.9). Correctness binarized from "
        "answer_quality >= 2. Lower Brier and ECE = better calibrated."
    )}

    for m in models:
        recs = [r for r in records_for_model(m)
                if r["scoring_mode"] == "rubric"
                and r.get("conf_appropriate") is not None
                and r.get("answer_quality") is not None]
        if not recs:
            report[m] = {"note": "no rubric records"}; continue

        confs   = [CONF_MAP.get(r["conf_appropriate"], 0.5) for r in recs]
        outcomes = [1 if r["answer_quality"] >= 2 else 0 for r in recs]

        b = brier(confs, outcomes)
        e = ece(confs, outcomes)

        # Split clean vs mirage for resolution
        clean_pairs   = [(c, o) for c, o, r in zip(confs, outcomes, recs) if r["variant"] == "clean"]
        mirage_pairs  = [(c, o) for c, o, r in zip(confs, outcomes, recs) if r["variant"] != "clean"]

        b_clean  = brier([p[0] for p in clean_pairs], [p[1] for p in clean_pairs]) if clean_pairs else None
        b_mirage = brier([p[0] for p in mirage_pairs], [p[1] for p in mirage_pairs]) if mirage_pairs else None

        report[m] = {
            "n_rubric_tasks":  len(recs),
            "brier_score":     b,
            "ECE_5_bins":      e,
            "brier_clean":     b_clean,
            "brier_mirage":    b_mirage,
            "mean_confidence": round(mean(confs), 4),
            "mean_accuracy":   round(mean(outcomes), 4),
        }
        print(f"{m}:")
        print(f"  n      = {len(recs)}")
        print(f"  Brier  = {b}  (clean {b_clean} / mirage {b_mirage})")
        print(f"  ECE    = {e}")
        print(f"  mean conf = {mean(confs):.4f}  mean acc = {mean(outcomes):.4f}")

    Path("data/calibration_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved → data/calibration_report.json")


if __name__ == "__main__":
    main()
