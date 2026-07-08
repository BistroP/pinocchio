"""Steering: activation addition + directional / subspace ablation, plus the phase-2 causal
experiments (Step 6) — steered generation, the subspace-removal consistency check, and the
random-direction baseline.

The linear-algebra core (hook builders, subspace ablation, random sampling, the consistency-matrix
assembly and its summary) is pure and unit-tested on synthetic tensors; the orchestration wrappers
that generate + score need a GPU model (transformer_lens).
"""
from __future__ import annotations

import numpy as np
import torch

from .directions import Direction


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-8)


def _vec(d: Direction | torch.Tensor) -> torch.Tensor:
    v = (d.vec if isinstance(d, Direction) else d).float()
    if not torch.isfinite(v).all() or float(v.norm()) < 1e-8:
        return torch.zeros_like(v)      # degenerate direction -> no-op hook (never poison generation)
    return _unit(v)


def add_direction_hook(direction, coeff: float):
    """Activation addition: resid <- resid + coeff * unit(direction). Add at one layer."""
    v = _vec(direction)

    def hook(resid, hook=None):
        return resid + coeff * v.to(resid.dtype).to(resid.device)
    return hook


def ablate_direction_hook(direction):
    """Directional ablation: project `direction` out of resid. Apply at EVERY layer."""
    v = _vec(direction)

    def hook(resid, hook=None):
        vv = v.to(resid.dtype).to(resid.device)
        return resid - (resid @ vv).unsqueeze(-1) * vv
    return hook


def ablate_subspace_hook(basis: torch.Tensor):
    """Subspace ablation: project the span of `basis` (rows) out of resid. Apply at every layer.
    Degenerate (nan/zero) rows are dropped; an empty basis yields a no-op hook."""
    B = basis.float()
    B = B[torch.isfinite(B).all(dim=1) & (B.norm(dim=1) > 1e-8)]   # drop degenerate rows
    if B.numel() == 0:
        return lambda resid, hook=None: resid
    Q, _ = torch.linalg.qr(B.T)                  # [d, r]

    def hook(resid, hook=None):
        Qd = Q.to(resid.dtype).to(resid.device)
        return resid - (resid @ Qd) @ Qd.T
    return hook


# ---------------------------------------------------------------- hook-set builders (all layers)
def _resid_hook_name(layer: int, site: str) -> str:
    return f"blocks.{layer}.hook_{site}"


def ablation_hooks(bundle, direction, sites=("resid_pre", "resid_post")):
    """fwd_hooks that ablate `direction` from every layer/site — directional ablation, i.e. remove
    the direction from the model's computation entirely (cf. Arditi et al.)."""
    fn = ablate_direction_hook(direction)
    return [(_resid_hook_name(l, s), fn) for l in range(bundle.n_layers) for s in sites]


def subspace_ablation_hooks(bundle, basis, sites=("resid_pre", "resid_post")):
    """fwd_hooks that project the span of `basis` (rows) out of every layer/site."""
    fn = ablate_subspace_hook(basis)
    return [(_resid_hook_name(l, s), fn) for l in range(bundle.n_layers) for s in sites]


def sample_random_directions(d_model: int, n: int, seed: int = 0, device="cpu") -> torch.Tensor:
    """n unit-norm directions, deterministic given seed (matched-norm random baseline)."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(n, d_model, generator=g, device=device)
    return x / (x.norm(dim=1, keepdim=True) + 1e-8)


# ---------------------------------------------------------------- steered generation (GPU)
@torch.no_grad()
def run_with_hooks(bundle, prompts, hooks, max_new_tokens: int = 64, batch_size: int = 8,
                   prepend_bos: bool = False) -> list[str]:
    """Greedy generation with `hooks` (list of (hook_name, fn)) attached; returns new text only.
    Pass hooks=[] for an un-intervened baseline run."""
    model = bundle.model
    outs: list[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = list(prompts[i:i + batch_size])
        tokens = model.to_tokens(batch, prepend_bos=prepend_bos)
        with model.hooks(fwd_hooks=hooks):
            gen = model.generate(tokens, max_new_tokens=max_new_tokens, do_sample=False, verbose=False)
        for j in range(gen.shape[0]):
            outs.append(model.to_string(gen[j, tokens.shape[1]:]))
    return outs


def behavior_rate(bundle, behavior, prompts, pairs, hooks=(), max_new_tokens: int = 64) -> float:
    """Mean behavioral score (1 = signal overridden) over `prompts` under `hooks`."""
    from . import eval as _ev
    outs = run_with_hooks(bundle, prompts, list(hooks), max_new_tokens=max_new_tokens)
    return float(np.mean([_ev.score(behavior, o, p) or 0 for o, p in zip(outs, pairs)]))


# ---------------------------------------------------------------- consistency matrix (pure)
def consistency_matrix(behaviors, score_fn) -> dict:
    """score_fn(removed, measured) -> behavior rate of `measured` when `removed`'s subspace is
    ablated. Returns {removed: {measured: rate}}. Pure: inject a fake score_fn to test."""
    return {rem: {meas: float(score_fn(rem, meas)) for meas in behaviors} for rem in behaviors}


def consistency_summary(matrix: dict, baseline: dict) -> dict:
    """For each removed behavior B: own_drop = baseline[B]-matrix[B][B]; other_drop = mean drop of
    the rest. `selective` (causal separability) when own_drop clearly exceeds other_drop."""
    out = {}
    for rem, row in matrix.items():
        own = baseline[rem] - row[rem]
        others = [baseline[a] - row[a] for a in row if a != rem]
        other = float(np.mean(others)) if others else 0.0
        out[rem] = {"own_drop": own, "other_drop": other,
                    "selective": bool(own > 2 * other + 1e-9)}
    return out


# ---------------------------------------------------------------- phase-2 experiments (GPU)
def subspace_removal_consistency_check(bundle, behaviors, subspace_by_behavior, prompts_by_behavior,
                                       pairs_by_behavior, sites=("resid_pre", "resid_post"),
                                       max_new_tokens: int = 64) -> dict:
    """Remove one behavior's suppression subspace and test whether the OTHERS persist
    (arXiv:2509.21305). Separable structure => ablating B collapses B's suppression selectively."""
    baseline = {b: behavior_rate(bundle, b, prompts_by_behavior[b], pairs_by_behavior[b],
                                  max_new_tokens=max_new_tokens) for b in behaviors}

    def score_fn(removed, measured):
        hooks = subspace_ablation_hooks(bundle, subspace_by_behavior[removed], sites=sites)
        return behavior_rate(bundle, measured, prompts_by_behavior[measured],
                             pairs_by_behavior[measured], hooks=hooks, max_new_tokens=max_new_tokens)

    matrix = consistency_matrix(behaviors, score_fn)
    return {"baseline": baseline, "matrix": matrix, "summary": consistency_summary(matrix, baseline)}


def random_direction_baseline(bundle, behavior, real_direction, prompts, pairs, n_random: int = 10,
                              seed: int = 0, sites=("resid_pre", "resid_post"),
                              max_new_tokens: int = 64) -> dict:
    """Necessity/sufficiency control: does ablating a RANDOM matched-norm direction reproduce the
    effect of ablating the real suppression direction? Returns baseline, real, and random effects."""
    base = behavior_rate(bundle, behavior, prompts, pairs, max_new_tokens=max_new_tokens)
    real = behavior_rate(bundle, behavior, prompts, pairs,
                         hooks=ablation_hooks(bundle, real_direction, sites=sites),
                         max_new_tokens=max_new_tokens)
    rand = sample_random_directions(bundle.d_model, n_random, seed=seed)
    rand_effects = [behavior_rate(bundle, behavior, prompts, pairs,
                                  hooks=ablation_hooks(bundle, rand[i], sites=sites),
                                  max_new_tokens=max_new_tokens) for i in range(n_random)]
    return {"baseline": base, "real_ablated": real, "random_ablated": rand_effects}
