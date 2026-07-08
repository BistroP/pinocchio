"""Project signal directions out of suppression directions -> residual directions (Step 5).

The residual suppression direction is what enters the subspace analysis: it strips the part of
the suppression direction that is trivially the signal it gates (truth / harm / fact), so any
remaining cross-behavior overlap is about the *suppressor*, not the signal.
"""
from __future__ import annotations

import torch

from .directions import Direction


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-8)


def project_out(vec: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Remove the component of `vec` lying in span(rows of `basis`)."""
    if basis.ndim == 1:
        basis = basis[None, :]
    Q, _ = torch.linalg.qr(basis.float().T)      # Q: [d, r], orthonormal columns
    v = vec.float()
    return v - Q @ (Q.T @ v)


def residual_direction(sup_dir: Direction, signal_dirs: list[Direction],
                       renormalize: bool = True) -> Direction:
    basis = torch.stack([_unit(s.vec.float()) for s in signal_dirs], dim=0)
    r = project_out(sup_dir.vec, basis)
    if renormalize:
        r = _unit(r)
    return Direction(r, sup_dir.layer, sup_dir.position, "suppression_residual",
                     behavior=sup_dir.behavior, name=(sup_dir.name or "") + "|resid",
                     meta={**sup_dir.meta, "projected_out": [s.name for s in signal_dirs]})


def residualize_all(sup_dirs: list[Direction], signal_dirs: list[Direction],
                    own_only: bool = True) -> list[Direction]:
    """For each suppression direction, project out either its own behavior's signal direction
    (own_only=True — the pre-registered control) or the entire signal basis (own_only=False)."""
    by_behavior = {s.behavior: s for s in signal_dirs}
    out = []
    for sd in sup_dirs:
        sig = [by_behavior[sd.behavior]] if own_only else signal_dirs
        out.append(residual_direction(sd, sig))
    return out
