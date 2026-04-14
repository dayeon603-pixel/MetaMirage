"""
CognitiveMirage — add a new model's records into v3_analysis.json.

Reads records from a v3_judge_evaluator.py output file, profiles the model
using the canonical profile_model() from v3_statistical_analysis, merges the
profile into v3_analysis.json['profiles'], and regenerates all derived
statistics (global correlations, family correlations, LOO, leaderboard).

Usage:
    python v3_add_model.py data/haiku_records.json
"""
from __future__ import annotations
import sys, json, math
from pathlib import Path

from v3_statistical_analysis import profile_model, pearson, cohen_d, bootstrap_ci, fisher_ci

ANALYSIS = Path("v3_analysis.json")


def _loo(models, xfn, yfn):
    rs = []
    for held in models:
        rest = [m for m in models if m != held]
        if len(rest) < 3: continue
        xr = [xfn(m) for m in rest]; yr = [yfn(m) for m in rest]
        if len({round(v,6) for v in xr}) <= 1: continue
        r, _ = pearson(xr, yr)
        rs.append(round(r, 4))
    if not rs: return None
    return {
        "loo_r_values": rs,
        "min_abs_r": round(min(abs(v) for v in rs), 4),
        "sign_stable": len({1 if v > 0 else -1 for v in rs}) == 1,
        "min_r": round(min(rs), 4),
        "max_r": round(max(rs), 4),
    }


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python v3_add_model.py <records_file>")
    recs_file = Path(sys.argv[1])
    new_records = json.loads(recs_file.read_text())
    if not new_records:
        sys.exit("no records in file")

    new_model = new_records[0]["model"]
    print(f"Adding model: {new_model} ({len(new_records)} records)")

    # Profile the new model using the canonical function
    new_profile = profile_model(new_records, new_model)

    # Load existing analysis; merge
    a = json.loads(ANALYSIS.read_text())
    a["profiles"][new_model] = new_profile
    profiles = a["profiles"]
    models = sorted(profiles.keys())
    print(f"Now have {len(models)} models: {', '.join(models)}")

    # Recompute global correlations
    tdrs = [profiles[m]["tdr_global"] for m in models]
    accs = [profiles[m]["aq_clean"] for m in models]
    mis  = [profiles[m]["metacognitive_index"] for m in models]
    r_g, p_g = pearson(tdrs, accs)
    r_mi, p_mi = pearson(mis, accs)
    a["global_correlation"]["tdr_vs_accuracy"] = {"r": r_g, "p": p_g}
    a["global_correlation"]["mi_vs_accuracy"]  = {"r": r_mi, "p": p_mi}
    # global CI
    lo, hi = fisher_ci(r_g, len(models))
    a["global_correlation"]["tdr_vs_accuracy"]["ci95"] = [lo, hi]
    print(f"\nGlobal TDR vs acc: r={r_g} p={p_g} CI=[{lo},{hi}]")
    print(f"Global MI vs acc:  r={r_mi} p={p_mi}")

    # Recompute per-family correlations
    families = sorted({f for p in profiles.values() for f in p["family"]})
    fam_new = {}
    for fam in families:
        fam_tdrs = [profiles[m]["family"][fam]["tdr"] for m in models]
        if len({round(t,6) for t in fam_tdrs}) <= 1:
            fam_new[fam] = {"r": None, "p": None, "n": len(models),
                            "note": "degenerate (TDR constant across models)"}
            continue
        r, p = pearson(fam_tdrs, accs)
        ci = fisher_ci(r, len(models))
        fam_new[fam] = {"r": r, "p": p, "n": len(models), "ci95": ci}
    a["family_correlations"] = fam_new
    print(f"\nPer-family (n={len(models)}):")
    for f, v in sorted(fam_new.items(), key=lambda x: (x[1].get("r") or 0)):
        if v.get("r") is None:
            print(f"  {f:22s} r=n/a")
        else:
            print(f"  {f:22s} r={v['r']:+.4f} CI=[{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] p={v['p']:.4f}")

    # LOO
    loo = {}
    loo["global_tdr_vs_accuracy"] = _loo(
        models,
        lambda m: profiles[m]["tdr_global"],
        lambda m: profiles[m]["aq_clean"],
    )
    for fam in families:
        res = _loo(
            models,
            lambda m, f=fam: profiles[m]["family"][f]["tdr"],
            lambda m: profiles[m]["aq_clean"],
        )
        loo[f"family_{fam}_tdr_vs_accuracy"] = res or {"note": "degenerate"}
    a["loo_stability"] = loo
    print(f"\nLOO:")
    for k, v in loo.items():
        if v and "min_abs_r" in v:
            print(f"  {k}: min|r|={v['min_abs_r']} range=[{v['min_r']},{v['max_r']}] stable={v['sign_stable']}")

    # Leaderboard
    a["leaderboard"] = sorted(
        [(m, profiles[m]["metacognitive_index"]) for m in models],
        key=lambda x: -x[1]
    )
    a["mi_spread"] = round(
        max(p["metacognitive_index"] for p in profiles.values()) -
        min(p["metacognitive_index"] for p in profiles.values()), 4)

    a["methodology_note"] = (a.get("methodology_note", "") +
        f" Model pool expanded to n={len(models)}: {', '.join(models)}.")

    ANALYSIS.write_text(json.dumps(a, indent=2))
    print(f"\nMI spread: {a['mi_spread']}")
    print(f"Leaderboard:")
    for r, (m, mi) in enumerate(a["leaderboard"], 1):
        p = profiles[m]
        print(f"  {r}. {m:26s} MI={mi:.4f} TDR={p['tdr_global']:.4f} acc={p['aq_clean']:.4f}")
    print(f"\nWrote → {ANALYSIS}")


if __name__ == "__main__":
    main()
