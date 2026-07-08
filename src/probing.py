"""Probes, the override-vs-ignorance admission check (Step 3), and transfer checks.

The admission check is the gate for family membership: a behavior is admitted only if, on the
manipulated cases whose OUTPUT overrode the signal, the signal is STILL linearly decodable from
the residual stream — i.e. the model gated a signal it represents (override), rather than losing
it (ignorance / genuine update).
"""
from __future__ import annotations

import numpy as np
import torch

from .directions import Direction

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    _HAS_SK = True
except Exception:
    _HAS_SK = False


def project(acts: torch.Tensor, direction: Direction | torch.Tensor) -> torch.Tensor:
    v = direction.vec if isinstance(direction, Direction) else direction
    v = v.float() / (v.float().norm() + 1e-8)
    return acts.float() @ v


def fit_probe(acts: torch.Tensor, labels) -> "LogisticRegression":
    if not _HAS_SK:
        raise RuntimeError("scikit-learn required for fit_probe")
    X = acts.float().numpy()
    y = np.asarray(labels).astype(int)
    return LogisticRegression(max_iter=1000).fit(X, y)


def probe_auc(direction: Direction, acts: torch.Tensor, labels) -> float:
    """AUC of the 1-D projection onto `direction` as a classifier of `labels`."""
    if not _HAS_SK:
        raise RuntimeError("scikit-learn required for probe_auc")
    scores = project(acts, direction).numpy()
    return float(roc_auc_score(np.asarray(labels).astype(int), scores))


def override_check(signal_dir: Direction, acts_manip_overrode: torch.Tensor,
                   signal_labels_overrode, output_flip_rate: float,
                   auc_threshold: float = 0.70, flip_threshold: float = 0.50) -> dict:
    """THE admission check.

    acts_manip_overrode / signal_labels_overrode : residual activations and the *true* signal
        labels, restricted to manipulated cases whose OUTPUT overrode the signal.
    output_flip_rate : fraction of manipulated cases whose output flipped vs. clean.

    Admits iff the signal probe still reads the signal (auc > auc_threshold) AND the output
    actually flipped often enough (flip_rate > flip_threshold). Thresholds are tunable.
    """
    auc = probe_auc(signal_dir, acts_manip_overrode, signal_labels_overrode)
    return {
        "signal_probe_auc": auc,                 # high => signal still represented (override)
        "output_flip_rate": float(output_flip_rate),
        "admits": bool(auc > auc_threshold and output_flip_rate > flip_threshold),
        "thresholds": {"auc": auc_threshold, "flip": flip_threshold},
    }


def transfer_check(direction: Direction, acts: torch.Tensor, labels) -> dict:
    """Does a direction extracted on one behavior/layer classify another's activations?"""
    return {"auc": probe_auc(direction, acts, labels)}
