"""Cross-behavior subspace characterization (Step 5) — the decisive analysis.

Given the per-behavior suppression directions (raw and residual), decide between the three
pre-registered outcomes:
  * one shared direction        -> high pairwise cosines, PC1 explains ~everything, PR ~ 1
  * distinct directions         -> low pairwise cosines, PR ~ N, no shared axis
  * distinct-but-shared-knob    -> low raw cosines BUT a single axis (PC1) aligns/controls all
"""
from __future__ import annotations

import torch

from .directions import Direction


def _stack(dirs: list[Direction]) -> tuple[torch.Tensor, list[str]]:
    M = torch.stack([d.vec.float() / (d.vec.float().norm() + 1e-8) for d in dirs], dim=0)  # [n, d]
    labels = [d.behavior or d.name or f"dir{i}" for i, d in enumerate(dirs)]
    return M, labels


def _finite(dirs: list[Direction]) -> list[Direction]:
    """Directions whose unit vector is all-finite and non-zero (drops degenerate/nan directions,
    e.g. a behavior whose override/resisted split was empty under the chosen contrast)."""
    keep = []
    for d in dirs:
        v = d.vec.float()
        if torch.isfinite(v).all() and float(v.norm()) > 1e-8:
            keep.append(d)
    return keep


def cosine_matrix(dirs: list[Direction]) -> tuple[torch.Tensor, list[str]]:
    """Pairwise cosine similarity matrix among (unit) directions — THE MVP output.
    Degenerate directions surface as nan rows/cols on purpose (honest about which behavior failed)."""
    M, labels = _stack(dirs)
    return M @ M.T, labels


def pca_spectrum(dirs: list[Direction], center: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Singular values and explained-variance ratio of the stacked directions.
    Degenerate directions are dropped first; returns nan if <2 finite directions remain."""
    dirs = _finite(dirs)
    if len(dirs) < 2:
        nan = torch.tensor([float("nan")])
        return nan, nan
    M, _ = _stack(dirs)
    if center:
        M = M - M.mean(0, keepdim=True)
    try:
        S = torch.linalg.svdvals(M)
    except Exception:                                   # ill-conditioned / repeated singular values
        return torch.tensor([float("nan")]), torch.tensor([float("nan")])
    var = S ** 2
    return S, var / (var.sum() + 1e-12)


def participation_ratio(singular_values: torch.Tensor) -> float:
    """Effective dimensionality: ~1 => one shared axis; ~N => fully distinct."""
    s2 = singular_values.float() ** 2
    return float((s2.sum() ** 2) / ((s2 ** 2).sum() + 1e-12))


def shared_knob_test(dirs: list[Direction]) -> dict:
    """Distinct-but-shared-knob probe: how much does a single shared axis explain the set?

    Reports PC1's variance share, the participation ratio, and the pairwise cosines of the
    *residual* directions after removing PC1 (if these collapse toward 0, the directions are
    distinct once the shared knob is removed -> supports 'distinct-but-shared-knob').
    """
    all_labels = [d.behavior or d.name or f"dir{i}" for i, d in enumerate(dirs)]
    finite = _finite(dirs)
    n = len(dirs)
    if len(finite) < 2:                                 # not enough valid directions to find an axis
        return {"labels": all_labels, "pc1_var_explained": float("nan"),
                "explained_variance": torch.tensor([float("nan")]),
                "participation_ratio": float("nan"),
                "residual_cosines_after_pc1": torch.full((n, n), float("nan")),
                "n_degenerate": n - len(finite)}
    M, labels = _stack(finite)
    Mc = M - M.mean(0, keepdim=True)
    try:
        U, S, Vh = torch.linalg.svd(Mc, full_matrices=False)
    except Exception:                                   # ill-conditioned / repeated singular values
        return {"labels": all_labels, "pc1_var_explained": float("nan"),
                "explained_variance": torch.tensor([float("nan")]),
                "participation_ratio": float("nan"),
                "residual_cosines_after_pc1": torch.full((n, n), float("nan")),
                "n_degenerate": n - len(finite)}
    pc1 = Vh[0]
    recon = (M @ pc1[:, None]) * pc1[None, :]          # rank-1 reconstruction along PC1
    resid = M - recon
    resid = resid / (resid.norm(dim=1, keepdim=True) + 1e-8)
    ev = S ** 2 / ((S ** 2).sum() + 1e-12)
    return {
        "labels": labels,
        "pc1_var_explained": float(ev[0]),
        "explained_variance": ev,
        "participation_ratio": participation_ratio(S),
        "residual_cosines_after_pc1": resid @ resid.T,
        "n_degenerate": n - len(finite),
    }


def principal_angles(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Principal angles (radians) between subspaces spanned by rows of A and B.
    Used by the phase-2 subspace-removal consistency check."""
    Qa, _ = torch.linalg.qr(A.float().T)
    Qb, _ = torch.linalg.qr(B.float().T)
    s = torch.linalg.svdvals(Qa.T @ Qb).clamp(-1.0, 1.0)
    return torch.arccos(s)


def summarize(dirs: list[Direction]) -> dict:
    """One call -> cosine matrix + spectrum + knob test, for logging to results/."""
    C, labels = cosine_matrix(dirs)
    S, ev = pca_spectrum(dirs)
    knob = shared_knob_test(dirs)
    return {"labels": labels, "cosine_matrix": C, "singular_values": S,
            "explained_variance": ev, "participation_ratio": participation_ratio(S),
            "shared_knob": knob}
