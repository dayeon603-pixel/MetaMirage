"""
MetaMirage — Robustness analysis for the main r = −0.84 claim.

Adds two non-parametric tests alongside the Fisher-z CI reported in the
main analysis:

  1. Bootstrap CI (10 000 resamples with replacement). Makes no
     distributional assumption; valid at small n when Fisher-z's
     normality-of-z approximation is weak.

  2. Permutation p-value. Shuffles TDR labels against clean-accuracy
     10 000 times; reports p = Pr(|r_shuffled| ≥ |r_observed|).
     The gold-standard answer to "can a small-n correlation be trusted?".

Writes results to v3_analysis.json['robustness'].
"""
from __future__ import annotations
import json, math, random
from pathlib import Path

random.seed(42)
ANALYSIS = Path("v3_analysis.json")


def pearson_r(xs, ys):
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys))
    return num/den if den else 0.0


def bootstrap_ci(xs, ys, n_boot=10000, alpha=0.05):
    """Percentile bootstrap CI on Pearson r."""
    n = len(xs)
    rs = []
    for _ in range(n_boot):
        idx = [random.randrange(n) for _ in range(n)]
        xb = [xs[i] for i in idx]; yb = [ys[i] for i in idx]
        # Skip degenerate resamples (constant vector)
        if len(set(xb)) < 2 or len(set(yb)) < 2:
            continue
        rs.append(pearson_r(xb, yb))
    rs.sort()
    lo = rs[int(alpha/2 * len(rs))]
    hi = rs[int((1 - alpha/2) * len(rs))]
    return round(lo, 4), round(hi, 4), len(rs)


def permutation_p(xs, ys, n_perm=10000):
    """Two-sided permutation p-value for H0: r = 0."""
    r_obs = pearson_r(xs, ys)
    n = len(xs)
    count = 0
    for _ in range(n_perm):
        perm_ys = ys[:]
        random.shuffle(perm_ys)
        r_perm = pearson_r(xs, perm_ys)
        if abs(r_perm) >= abs(r_obs) - 1e-12:
            count += 1
    return round(count / n_perm, 4), round(r_obs, 4)


def main():
    a = json.loads(ANALYSIS.read_text())
    profiles = a["profiles"]
    models = list(profiles.keys())
    tdrs = [profiles[m]["tdr_global"] for m in models]
    accs = [profiles[m]["aq_clean"]   for m in models]

    print(f"n = {len(models)} models")
    r_obs = pearson_r(tdrs, accs)
    print(f"observed r = {r_obs:.4f}")

    # Bootstrap
    lo, hi, n_valid = bootstrap_ci(tdrs, accs, n_boot=10000)
    print(f"Bootstrap 95% CI (n_valid={n_valid}): [{lo}, {hi}]")

    # Permutation
    p_perm, r_val = permutation_p(tdrs, accs, n_perm=10000)
    print(f"Permutation p (two-sided, 10 000 shuffles): {p_perm}")

    # Per-family too
    fam_robust = {}
    for fam in sorted({f for p in profiles.values() for f in p["family"]}):
        fam_tdrs = [profiles[m]["family"][fam]["tdr"] for m in models]
        if len(set(fam_tdrs)) <= 1:
            fam_robust[fam] = {"note": "degenerate"}
            continue
        flo, fhi, _ = bootstrap_ci(fam_tdrs, accs, n_boot=10000)
        fp, fr = permutation_p(fam_tdrs, accs, n_perm=10000)
        fam_robust[fam] = {
            "r": round(fr, 4),
            "bootstrap_ci95": [flo, fhi],
            "permutation_p": fp,
        }
        print(f"  {fam:22s} r={fr:+.4f}  bootCI=[{flo:+.3f},{fhi:+.3f}]  permP={fp}")

    a["robustness"] = {
        "method_note": (
            "Two non-parametric checks alongside the Fisher-z CI reported "
            "in the main analysis. Bootstrap: 10 000 percentile resamples "
            "with replacement; valid without the normality-of-Fisher-z "
            "assumption at small n. Permutation: 10 000 label shuffles; "
            "p = fraction of shuffles with |r| ≥ |r_observed|."
        ),
        "global": {
            "r": round(r_obs, 4),
            "bootstrap_ci95": [lo, hi],
            "bootstrap_n_valid_resamples": n_valid,
            "permutation_p_two_sided": p_perm,
            "n_permutations": 10000,
        },
        "per_family": fam_robust,
    }
    ANALYSIS.write_text(json.dumps(a, indent=2))
    print(f"\nWrote robustness block → {ANALYSIS}")


if __name__ == "__main__":
    main()
