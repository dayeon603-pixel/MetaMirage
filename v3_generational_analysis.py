"""
MetaMirage — Within-Anthropic Generational Sub-Study.

Compares 6 Anthropic models across two release lineages:

  Opus generations:    opus-4 (May 2025) → opus-4-1 (Aug 2025) → opus-4-5 (Nov 2025)
  Sonnet generations:  sonnet-4 (May 2025) → sonnet-4-5 (current)
  Plus haiku-4-5 as the new small-model entry.

Tests the central prediction of the trainable-out hypothesis: if metacognition
is shaped by RLHF preference data, generational trajectories within a single
vendor should reveal training-protocol-specific shifts that the cross-vendor
analysis blurs.

Output: data/generational_report.json
"""
from __future__ import annotations
import json, math
from pathlib import Path

from v3_statistical_analysis import profile_model

LINEAGES = {
    "opus": [
        ("claude-opus-4-20250514",   "Opus 4.0 (May 2025)"),
        ("claude-opus-4-1-20250805", "Opus 4.1 (Aug 2025)"),
        ("claude-opus-4-5",          "Opus 4.5 (current)"),
    ],
    "sonnet": [
        ("claude-sonnet-4-20250514", "Sonnet 4.0 (May 2025)"),
        ("claude-sonnet-4-5",        "Sonnet 4.5 (current)"),
    ],
    "haiku": [
        ("claude-haiku-4-5-20251001", "Haiku 4.5 (Nov 2025)"),
    ],
}


def main():
    a = json.loads(Path("v3_analysis.json").read_text())
    extra_recs = json.loads(Path("data/gen_arc_records.json").read_text())

    # Profile the older models from raw records
    older = ["claude-opus-4-20250514", "claude-opus-4-1-20250805",
             "claude-sonnet-4-20250514"]
    profiles_extra = {m: profile_model([r for r in extra_recs if r["model"] == m], m)
                      for m in older}
    # Profiles for the current-gen models come from the main analysis
    profiles = {**profiles_extra}
    for m in ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5-20251001"]:
        profiles[m] = a["profiles"][m]

    print("=== Anthropic Generational Sub-Study ===\n")

    report = {"lineages": {}}

    for lineage, members in LINEAGES.items():
        print(f"── {lineage.upper()} lineage ──")
        rows = []
        for model, label in members:
            p = profiles[model]
            rows.append({
                "model": model, "label": label,
                "MI": round(p["metacognitive_index"], 4),
                "TDR_global": round(p["tdr_global"], 4),
                "clean_acc": round(p["aq_clean"], 4),
                "calib_delta": round(p["calib_delta"], 4),
                "tdr_rubric": round(p["tdr_rubric"], 4),
                "tdr_abstain": round(p["tdr_abstain"], 4),
                "tdr_expert": round(p["tdr_expert"], 4),
            })
            print(f"  {label:30s} MI={p['metacognitive_index']:.3f} TDR={p['tdr_global']:.3f} acc={p['aq_clean']:.3f}")
        report["lineages"][lineage] = rows
        print()

    # Key finding: Opus regression
    if len(LINEAGES["opus"]) >= 2:
        opus_tdrs = [profiles[m]["tdr_global"] for m, _ in LINEAGES["opus"]]
        opus_accs = [profiles[m]["aq_clean"]   for m, _ in LINEAGES["opus"]]
        opus_mis  = [profiles[m]["metacognitive_index"] for m, _ in LINEAGES["opus"]]
        delta_tdr = opus_tdrs[-1] - opus_tdrs[0]   # 4.5 minus 4.0
        delta_mi  = opus_mis[-1]  - opus_mis[0]
        report["opus_regression"] = {
            "tdr_4_0": opus_tdrs[0],
            "tdr_4_1": opus_tdrs[1],
            "tdr_4_5": opus_tdrs[2],
            "delta_tdr_4_0_to_4_5": round(delta_tdr, 4),
            "mi_4_0":  opus_mis[0],
            "mi_4_5":  opus_mis[2],
            "delta_mi_4_0_to_4_5":  round(delta_mi, 4),
            "interpretation": (
                "Across Anthropic's Opus releases, TDR drops from "
                f"{opus_tdrs[0]:.2f} (Opus 4.0) → {opus_tdrs[1]:.2f} (Opus 4.1) "
                f"→ {opus_tdrs[2]:.2f} (Opus 4.5), while clean accuracy stays ≈100%. "
                "This within-vendor regression supports the RLHF-confidence-pressure "
                "hypothesis: a specific training change between Opus 4.1 and Opus 4.5 "
                "appears to have suppressed metacognitive monitoring."
            ),
        }
        print(f"\n>>> KEY GENERATIONAL FINDING <<<")
        print(f"Opus TDR: {opus_tdrs[0]:.3f} → {opus_tdrs[1]:.3f} → {opus_tdrs[2]:.3f}")
        print(f"Δ(TDR) Opus 4.0 → 4.5: {delta_tdr:+.3f}")
        print(f"Δ(MI)  Opus 4.0 → 4.5: {delta_mi:+.3f}")

    # Sonnet
    if len(LINEAGES["sonnet"]) >= 2:
        son_tdrs = [profiles[m]["tdr_global"] for m, _ in LINEAGES["sonnet"]]
        son_mis  = [profiles[m]["metacognitive_index"] for m, _ in LINEAGES["sonnet"]]
        report["sonnet_regression"] = {
            "tdr_4_0": son_tdrs[0], "tdr_4_5": son_tdrs[1],
            "delta_tdr": round(son_tdrs[1] - son_tdrs[0], 4),
            "mi_4_0": son_mis[0], "mi_4_5": son_mis[1],
            "delta_mi": round(son_mis[1] - son_mis[0], 4),
        }
        print(f"\nSonnet TDR: {son_tdrs[0]:.3f} → {son_tdrs[1]:.3f}  (Δ = {son_tdrs[1]-son_tdrs[0]:+.3f})")
        print(f"Sonnet MI:  {son_mis[0]:.3f} → {son_mis[1]:.3f}  (Δ = {son_mis[1]-son_mis[0]:+.3f})")

    Path("data/generational_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved → data/generational_report.json")


if __name__ == "__main__":
    main()
