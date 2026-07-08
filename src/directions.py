"""Difference-of-means signal directions (Step 2) and per-behavior suppression directions
(Step 4), extracted with a COMMON contrast across the family.

The two design decisions (see README) are applied here IDENTICALLY across behaviors, read from
configs/behaviors.yaml. This module implements EVERY mode; the config selects one, so the human
call is a switch, not a rewrite.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-8)


def diff_of_means(acts_pos: torch.Tensor, acts_neg: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    d = acts_pos.float().mean(0) - acts_neg.float().mean(0)
    return _unit(d) if normalize else d


@dataclass
class Direction:
    vec: torch.Tensor                       # [d_model], unit norm unless noted
    layer: int
    position: int
    kind: str                               # signal | suppression | suppression_residual | commitment
    behavior: Optional[str] = None
    name: Optional[str] = None
    meta: dict = field(default_factory=dict)

    def cosine(self, other: "Direction | torch.Tensor") -> float:
        o = other.vec if isinstance(other, Direction) else other
        return float(torch.dot(_unit(self.vec), _unit(o)))


# --------------------------------------------------------------------------------------
# Step 2 — signal direction (truth / harm / fact)
# --------------------------------------------------------------------------------------
def signal_direction(acts_pos: torch.Tensor, acts_neg: torch.Tensor, *, layer: int, position: int,
                     name: str, behavior: str | None = None) -> Direction:
    return Direction(diff_of_means(acts_pos, acts_neg), layer, position, "signal",
                     behavior=behavior, name=name)


# --------------------------------------------------------------------------------------
# Step 4 — suppression direction (DESIGN DECISION 1: contrast.mode)
# --------------------------------------------------------------------------------------
# See docs/contrast-decision.md for the literature-grounded comparison + recommendation.
# Family-general (work on every behavior): manip_vs_clean, overrode_vs_resisted,
#   paired_manip_minus_clean, did_overrode_vs_resisted.
# Within-behavior validation only (needs a legitimate same-output twin, absent for safety):
#   matched_output.
CONTRAST_MODES = ("manip_vs_clean", "overrode_vs_resisted", "paired_manip_minus_clean",
                  "did_overrode_vs_resisted", "matched_output")


def suppression_direction(acts: dict[str, torch.Tensor], masks: dict[str, torch.Tensor], *,
                          mode: str, layer: int, position: int, behavior: str) -> Direction:
    """Extract one behavior's suppression direction under the common contrast `mode`.

    acts
        "manip": activations on the manipulated arm  [n, d]
        "clean": activations on the matched clean arm [n, d]  (index-aligned with "manip")
        "legit":  (matched_output only) activations on legitimate SAME-OUTPUT cases —
                  e.g. genuine agreement vs sycophantic agreement (see genuine-update control)
    masks (boolean over the n items)
        "overrode":   manipulated arm gated the signal (output flipped / complied / followed context)
        "resisted":   manipulated arm kept the signal despite the manipulation
        "clean_kept": clean arm did NOT override (the matched baseline)
    """
    name = f"{behavior}:suppression:{mode}"
    if mode == "manip_vs_clean":
        # field-standard stimulus contrast (cf. Arditi refusal dir, 2602.02132). Confounds the
        # manipulation stimulus with the suppression act.
        v = diff_of_means(acts["manip"][masks["overrode"]], acts["clean"][masks["clean_kept"]])
    elif mode == "overrode_vs_resisted":
        # holds the manipulation FIXED in both arms -> isolates the gating outcome.
        v = diff_of_means(acts["manip"][masks["overrode"]], acts["manip"][masks["resisted"]])
    elif mode == "paired_manip_minus_clean":
        idx = masks["overrode"]
        v = _unit((acts["manip"][idx].float() - acts["clean"][idx].float()).mean(0))
    elif mode == "did_overrode_vs_resisted":
        # difference-in-differences (recommended): [manip-clean | overrode] - [manip-clean | resisted].
        # Cancels item baseline (paired clean subtraction) AND common manipulation response
        # (resisted control) -> the most confound-robust family-general contrast.
        over, res = masks["overrode"], masks["resisted"]
        d_over = (acts["manip"][over].float() - acts["clean"][over].float()).mean(0)
        d_res = ((acts["manip"][res].float() - acts["clean"][res].float()).mean(0)
                 if int(res.sum()) > 0 else torch.zeros_like(d_over))
        v = _unit(d_over - d_res)
    elif mode == "matched_output":
        # within-behavior validation (2509.21305): same output, different cause. NOT family-general
        # (safety has no legitimate same-output twin). Requires acts["legit"]; if masks carries
        # "legit_followed", restrict to legit cases that actually produced the same surface output.
        legit = acts["legit"]
        if "legit_followed" in masks:
            legit = legit[masks["legit_followed"]]
        v = diff_of_means(acts["manip"][masks["overrode"]], legit)
    else:
        raise ValueError(f"unknown contrast mode {mode!r}; choose from {CONTRAST_MODES}")
    return Direction(v, layer, position, "suppression", behavior=behavior, name=name,
                     meta={"contrast": mode, "n_overrode": int(masks["overrode"].sum())})


# --------------------------------------------------------------------------------------
# Behavioral object (DESIGN DECISION 2: behavioral_object.mode)
# --------------------------------------------------------------------------------------
BEHAVIORAL_MODES = ("correctness_label", "commitment_direction", "natural_per_behavior")


def behavioral_direction(*, mode: str, layer: int, position: int, behavior: str,
                         signal_dir: Direction | None = None,
                         acts: dict[str, torch.Tensor] | None = None) -> Direction:
    """The per-behavior behavioral object.

    correctness_label     -> reuse the signal direction as the behavioral axis; the object is
                             output correctness, read as projection onto signal_dir.
    commitment_direction  -> diff-of-means mu(committed) - mu(abstained/refused) from `acts`
                             (keys "committed", "abstained").
    natural_per_behavior  -> commitment_direction where a native object exists (e.g. refusal),
                             else falls back to the signal direction.
    """
    if mode == "correctness_label":
        if signal_dir is None:
            raise ValueError("correctness_label requires signal_dir")
        return Direction(signal_dir.vec.clone(), layer, position, "commitment",
                         behavior=behavior, name=f"{behavior}:behavioral:correctness")
    if mode in ("commitment_direction", "natural_per_behavior"):
        if acts and "committed" in acts and "abstained" in acts:
            v = diff_of_means(acts["committed"], acts["abstained"])
            return Direction(v, layer, position, "commitment", behavior=behavior,
                             name=f"{behavior}:behavioral:{mode}")
        if mode == "natural_per_behavior" and signal_dir is not None:
            return behavioral_direction(mode="correctness_label", layer=layer, position=position,
                                        behavior=behavior, signal_dir=signal_dir)
        raise ValueError(f"{mode} needs acts with 'committed'/'abstained'")
    raise ValueError(f"unknown behavioral mode {mode!r}; choose from {BEHAVIORAL_MODES}")
