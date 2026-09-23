"""Significance for the causal override-rate matrix (Step 1).

Two jobs, both pure-numpy (run after judging, no model needed):
  - bootstrap_ci: a free +/- band on any override rate, by resampling the judged 0/1 labels we
    already have. Captures WHICH-PROMPTS-WE-TESTED noise only, NOT run-to-run generation noise
    (that needs actual reruns / multiple seeds).
  - specificity_report: for each behavior, compares each signal-direction effect against the
    MEANINGFUL control band (sentiment/formality/topic) -- the bar that separates a resistance-
    specific effect from general 'meshing'. The random band is shown too as the noise floor.

The two-proportion z here is UNPAIRED (treats the control band as an independent baseline). The
prompts are actually shared across ablations, so a paired test (McNemar) would be a touch more
powerful; unpaired is the conservative first pass and matches the field-standard rate-vs-rate read.
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci(labels, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """95% percentile bootstrap CI for the mean of 0/1 `labels`."""
    x = np.asarray(list(labels), dtype=float)
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return (float(lo), float(hi))


def two_prop_z(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-proportion z for p1 (condition) vs p2 (baseline). +z => condition's rate is higher."""
    if n1 == 0 or n2 == 0:
        return 0.0
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = (p * (1.0 - p) * (1.0 / n1 + 1.0 / n2)) ** 0.5
    return float((p1 - p2) / se) if se > 0 else 0.0


def _rate(labels, axis, b):
    L = labels[axis][b]
    return float(np.mean(L)) if len(L) else float("nan")


def specificity_report(labels, control_axes, signal_axes, behaviors,
                       baseline_axis: str = "none", random_axes=(), n_boot: int = 2000) -> dict:
    """labels[axis][behavior] -> list of 0/1 (per-prompt judge labels).

    Per behavior, prints: baseline, random band (noise floor), the CONTROL band (meaningful floor,
    with per-dir rates + a pooled CI), then each signal axis with its bootstrap CI and z vs the
    pooled control band. z>2 => the signal dir beats the meaningful null (evidence of specificity);
    z<1 => it sits inside the band (evidence of general meshing). Returns the numbers as a dict.
    """
    out = {}
    for b in behaviors:
        print(f"\n=== {b}: does a signal dir beat the MEANINGFUL control band? ===")
        blo, bhi = bootstrap_ci(labels[baseline_axis][b], n_boot=n_boot)
        print(f"  baseline (none)      {_rate(labels, baseline_axis, b):.2f}  [{blo:.2f}, {bhi:.2f}]")
        if random_axes:
            rr = [_rate(labels, a, b) for a in random_axes]
            print(f"  random band          mean {np.mean(rr):.2f}   max {np.max(rr):.2f}   (noise floor)")
        # pooled meaningful control band
        cpool = [x for a in control_axes for x in labels[a][b]]
        ck, cn = int(np.sum(cpool)), len(cpool)
        clo, chi = bootstrap_ci(cpool, n_boot=n_boot)
        per_dir = " ".join(f"{a.replace('ctrl_', '')}={_rate(labels, a, b):.2f}" for a in control_axes)
        print(f"  CONTROL band         mean {np.mean([_rate(labels, a, b) for a in control_axes]):.2f}"
              f"  [{clo:.2f}, {chi:.2f}]   (per-dir: {per_dir})")
        print(f"  {'signal dir':<18}{'rate':>6}{'  95% CI':>14}   z vs control band")
        for a in signal_axes:
            k, n = int(np.sum(labels[a][b])), len(labels[a][b])
            lo, hi = bootstrap_ci(labels[a][b], n_boot=n_boot)
            z = two_prop_z(k, n, ck, cn)
            flag = "  <-- beats band (specific)" if z > 2 else ("  (inside band -> meshing)" if z < 1 else "")
            print(f"  {a:<18}{_rate(labels, a, b):>6.2f}   [{lo:.2f}, {hi:.2f}]   z={z:+.1f}{flag}")
            out[(b, a)] = {"rate": _rate(labels, a, b), "ci": (lo, hi), "z_vs_control": z}
    print("\nRead: signal dir with z>2 AND a CI clear of the control band = resistance-SPECIFIC on that"
          "\nbehavior. Signal dirs sitting inside the band = general meshing (any rich direction erodes it).")
    return out
