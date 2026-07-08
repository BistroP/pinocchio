"""Truth-is-Universal-style test for a SHARED suppression axis (arXiv:2407.12831).

Bürger et al. show a single affirmative-only truth direction fails to generalise to negated
statements, but a general direction t_G separates truth across *both* polarities. Here the analog of
"polarity" is "behaviour": we ask whether a single general suppression axis t_G generalises across
sycophancy / safety / knowledge-conflict, the way t_G generalises across statement types.

The pairwise cosine is a weak proxy. The decisive tests operate on the cached overrode/resisted
ACTIVATIONS (no GPU pass):
  - transfer_matrix : does behaviour A's suppression direction separate behaviour B's overrode vs
                      resisted? (AUROC). High off-diagonal = a direction that generalises = t_G-like.
  - shared_axis     : t_G = normalised mean of the per-behaviour unit directions, + alignment /
                      uncentered-dominance metrics (NOT centered — centering removes the shared axis).
  - tG_generalization: does that single t_G separate EVERY behaviour's overrode/resisted?
  - random_baseline : AUROC a random direction achieves per behaviour (chance ~0.5) — the bar
                      off-diagonal transfer and t_G must clear to be real.
"""
from __future__ import annotations

import numpy as np
import torch

from . import probing as Pr
from .directions import Direction


def _xy(acts: dict, masks: dict):
    """(manip activations, overrode labels) — the suppression axis' supervised problem for a behaviour."""
    return acts["manip"], masks["overrode"].int().cpu().tolist()


def _both_classes(y) -> bool:
    s = int(sum(y))
    return s >= 2 and (len(y) - s) >= 2


def transfer_matrix(sup_dirs: dict, acts_by: dict, masks_by: dict, behaviors: list[str]) -> np.ndarray:
    """T[i, j] = AUROC of behaviour i's suppression direction separating behaviour j's overrode vs
    resisted. Diagonal is IN-SAMPLE (optimistic); the OFF-diagonal is the generalisation signal."""
    n = len(behaviors)
    T = np.full((n, n), np.nan)
    for j, b in enumerate(behaviors):
        X, y = _xy(acts_by[b], masks_by[b])
        if not _both_classes(y):
            continue
        for i, a in enumerate(behaviors):
            T[i, j] = Pr.probe_auc(sup_dirs[a], X, y)
    return T


def shared_axis(sup_dirs: dict, behaviors: list[str]) -> tuple[Direction, dict]:
    """Candidate general suppression axis t_G = normalised mean of the unit directions, WITHOUT
    centering (centering would subtract the very shared component). Metrics:
      alignment            = ||mean of unit dirs||, in [1/sqrt(n) (orthogonal) .. 1 (identical)]
      uncentered_pc1_frac  = sigma_1^2 / sum sigma^2 of the stacked dirs, [1/n (orth) .. 1 (identical)]
      per_dir_cos_to_tG    = each behaviour's cosine with t_G."""
    V = torch.stack([d.vec.float() / (d.vec.float().norm() + 1e-8) for d in
                     (sup_dirs[b] for b in behaviors)], dim=0)                 # [n, d]
    mean = V.mean(0)
    tG = mean / (mean.norm() + 1e-8)
    S = torch.linalg.svdvals(V)
    meta = {
        "alignment": float(mean.norm()),
        "orthogonal_floor": 1.0 / np.sqrt(len(behaviors)),
        "uncentered_pc1_frac": float(S[0] ** 2 / (S ** 2).sum()),
        "per_dir_cos_to_tG": {b: float(V[i] @ tG) for i, b in enumerate(behaviors)},
    }
    ref = sup_dirs[behaviors[0]]
    return Direction(tG, ref.layer, ref.position, "shared_tG", None, "t_G^supp"), meta


def tG_generalization(tG: Direction, acts_by: dict, masks_by: dict, behaviors: list[str]) -> dict:
    """AUROC of the in-sample shared axis t_G separating each behaviour's overrode/resisted, next to
    the OWN-direction AUROC. CIRCULAR: t_G contains 1/n of each direction, so tG_auc ~ own_auc holds
    BY CONSTRUCTION even for fully distinct directions. Use tG_loo for the honest test."""
    out = {}
    for b in behaviors:
        X, y = _xy(acts_by[b], masks_by[b])
        out[b] = (Pr.probe_auc(tG, X, y) if _both_classes(y) else float("nan"))
    return out


def tG_loo(sup_dirs: dict, acts_by: dict, masks_by: dict, behaviors: list[str]) -> dict:
    """Leave-one-out shared axis (the honest test): for each behaviour, build t_G from the OTHER
    behaviours' directions only (no self-contribution) and test on the held-out behaviour. LOO AUROC
    above the random baseline = a genuine shared axis that GENERALISES; ~chance = distinct directions."""
    out = {}
    for b in behaviors:
        others = [o for o in behaviors if o != b]
        V = torch.stack([sup_dirs[o].vec.float() / (sup_dirs[o].vec.float().norm() + 1e-8) for o in others], 0)
        m = V.mean(0)
        tG = Direction(m / (m.norm() + 1e-8), sup_dirs[b].layer, sup_dirs[b].position, "shared_tG_loo", None, "t_G_loo")
        X, y = _xy(acts_by[b], masks_by[b])
        out[b] = (Pr.probe_auc(tG, X, y) if _both_classes(y) else float("nan"))
    return out


def random_baseline(acts_by: dict, masks_by: dict, behaviors: list[str], n: int = 200, seed: int = 0) -> dict:
    """Per-behaviour AUROC distribution for random unit directions (chance ~0.5). Returns
    (mean, 95th percentile) — the bar off-diagonal transfer / t_G must clear."""
    d = acts_by[behaviors[0]]["manip"].shape[1]
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(n, d, generator=g)
    R = R / R.norm(dim=1, keepdim=True)
    out = {}
    for b in behaviors:
        X, y = _xy(acts_by[b], masks_by[b])
        if not _both_classes(y):
            out[b] = (float("nan"), float("nan")); continue
        aucs = [Pr.probe_auc(R[k], X, y) for k in range(n)]
        out[b] = (float(np.mean(aucs)), float(np.quantile(aucs, 0.95)))
    return out


def cosine_significance(sup_dirs: dict, behaviors: list[str]) -> list[tuple]:
    """(pair, cosine, z) with z = cos * sqrt(d) (random unit-vector cosine has std ~ 1/sqrt(d))."""
    d = sup_dirs[behaviors[0]].vec.shape[0]
    U = {b: sup_dirs[b].vec.float() / (sup_dirs[b].vec.float().norm() + 1e-8) for b in behaviors}
    out = []
    for i in range(len(behaviors)):
        for j in range(i + 1, len(behaviors)):
            c = float(U[behaviors[i]] @ U[behaviors[j]])
            out.append((f"{behaviors[i]}–{behaviors[j]}", c, c * np.sqrt(d)))
    return out


def report(sup_dirs: dict, acts_by: dict, masks_by: dict, behaviors: list[str]) -> dict:
    """Run every test and print a readable report. Returns the raw results too."""
    T = transfer_matrix(sup_dirs, acts_by, masks_by, behaviors)
    rand = random_baseline(acts_by, masks_by, behaviors)
    tG, meta = shared_axis(sup_dirs, behaviors)
    gen = tG_generalization(tG, acts_by, masks_by, behaviors)
    own = {b: Pr.probe_auc(sup_dirs[b], acts_by[b]["manip"], _xy(acts_by[b], masks_by[b])[1])
           if _both_classes(_xy(acts_by[b], masks_by[b])[1]) else float("nan") for b in behaviors}

    w = max(len(b) for b in behaviors) + 1
    print("TRANSFER  T[A→B] = AUROC of A's suppression dir on B's overrode/resisted (diag = in-sample)")
    print(" " * w + "".join(f"{b[:10]:>12}" for b in behaviors))
    for i, a in enumerate(behaviors):
        print(f"{a:<{w}}" + "".join(f"{T[i, j]:>12.2f}" for j in range(len(behaviors))))

    print("\nrandom-direction baseline AUROC (chance) — off-diagonal transfer must beat this:")
    for b in behaviors:
        print(f"  {b:<{w}} mean={rand[b][0]:.2f}  95th%={rand[b][1]:.2f}")

    print(f"\nshared axis t_G:  alignment={meta['alignment']:.3f} "
          f"(orthogonal floor={meta['orthogonal_floor']:.3f}, identical=1.0)  |  "
          f"uncentered PC1 frac={meta['uncentered_pc1_frac']:.3f} (orth={1/len(behaviors):.3f})")
    print("t_G IN-SAMPLE (CIRCULAR — t_G contains 1/n of each dir, so tG_auc≈own by construction):")
    for b in behaviors:
        print(f"  {b:<{w}} tG_auc={gen[b]:.2f}   own_dir_auc={own[b]:.2f}   cos(dir,tG)={meta['per_dir_cos_to_tG'][b]:+.2f}")
    loo = tG_loo(sup_dirs, acts_by, masks_by, behaviors)
    print("t_G LEAVE-ONE-OUT (HONEST — shared axis from the OTHER behaviours; beat random95% = real):")
    for b in behaviors:
        print(f"  {b:<{w}} loo_tG_auc={loo[b]:.2f}   random95%={rand[b][1]:.2f}   own={own[b]:.2f}")

    print(f"\ncosine significance (random cosine std ~ {1/np.sqrt(sup_dirs[behaviors[0]].vec.shape[0]):.3f}; |z|>3 = real):")
    for pair, c, z in cosine_significance(sup_dirs, behaviors):
        print(f"  {pair:<28} cos={c:+.3f}  z={z:+.1f}")

    return {"transfer": T, "random": rand, "tG": tG, "tG_meta": meta,
            "tG_gen": gen, "tG_loo": loo, "own": own}
