"""Per-behavior extraction pipeline shared by the MVP driver and the phase-2 steering notebook.

One GPU pass per behavior produces: the signal direction, the suppression direction (under the
configured contrast), the override-check verdict, and the cached activations / masks / prompts /
pairs needed downstream (residualization, subspace analysis, steering, subspace-removal).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from . import data as D
from . import directions as Dir
from . import eval as Ev
from . import model as M
from . import probing as Pr


@dataclass
class Extraction:
    behavior: str
    signal_dir: Dir.Direction
    suppression_dir: Dir.Direction
    override_check: dict
    acts: dict                       # {"manip","clean"[,"legit"]}  -> [n, d] cpu float
    masks: dict                      # {"overrode","resisted","clean_kept"[,"legit_followed"]}
    prompts: dict                    # {"clean","manip"[,"legit"]}  -> chat-formatted strings
    pairs: list = field(default_factory=list)


def extract_behavior(bundle: M.ModelBundle, behavior: str, cfg: dict, *,
                     layer: int, position: int, contrast_mode: str,
                     max_items: int | None = None, max_new_tokens: int = 64) -> Extraction:
    bcfg = cfg["behaviors"][behavior]
    pairs = D.load_pairs(behavior, cfg)
    if max_items:                                # cap for a fast local/validation run
        pairs = pairs[:max_items]

    # ---- signal direction: diff-of-means on clean true/false | harmful/harmless ----
    pos_txt, neg_txt = D.load_signal_statements(behavior, cfg)
    if max_items:
        pos_txt, neg_txt = pos_txt[:max_items], neg_txt[:max_items]
    acts_pos = M.get_activations(bundle, M.format_chat(bundle, pos_txt), [layer], [position])[(layer, position)]
    acts_neg = M.get_activations(bundle, M.format_chat(bundle, neg_txt), [layer], [position])[(layer, position)]
    signal_dir = Dir.signal_direction(acts_pos, acts_neg, layer=layer, position=position,
                                      name=bcfg["signal"], behavior=behavior)

    # ---- clean vs manipulated activations (index-aligned per pair) ----
    clean_prompts = M.format_chat(bundle, [p.prompt_clean for p in pairs])
    manip_prompts = M.format_chat(bundle, [p.prompt_manip for p in pairs])
    acts_clean = M.get_activations(bundle, clean_prompts, [layer], [position])[(layer, position)]
    acts_manip = M.get_activations(bundle, manip_prompts, [layer], [position])[(layer, position)]

    # ---- behavioral outcome -> masks ----
    gen = lambda pr: M.generate(bundle, pr, max_new_tokens=max_new_tokens)
    s_manip = [Ev.score(behavior, o, p) or 0 for o, p in zip(gen(manip_prompts), pairs)]
    s_clean = [Ev.score(behavior, o, p) or 0 for o, p in zip(gen(clean_prompts), pairs)]
    overrode = torch.tensor([s == 1 for s in s_manip])
    masks = {"overrode": overrode, "resisted": ~overrode,
             "clean_kept": torch.tensor([s == 0 for s in s_clean])}
    acts = {"manip": acts_manip, "clean": acts_clean}
    prompts = {"clean": clean_prompts, "manip": manip_prompts}

    # ---- legit (same-output) arm for matched_output validation, where constructible ----
    if D.supports_matched_output(behavior) and pairs and pairs[0].prompt_legit:
        legit_prompts = M.format_chat(bundle, [p.prompt_legit for p in pairs])
        acts["legit"] = M.get_activations(bundle, legit_prompts, [layer], [position])[(layer, position)]
        masks["legit_followed"] = torch.tensor(
            [Ev.followed_legit(behavior, o, p) for o, p in zip(gen(legit_prompts), pairs)])
        prompts["legit"] = legit_prompts

    # ---- suppression direction (common contrast) + admission check ----
    sup_dir = Dir.suppression_direction(acts, masks, mode=contrast_mode,
                                        layer=layer, position=position, behavior=behavior)
    bal_acts = torch.cat([acts_pos, acts_neg], 0)
    bal_labels = [1] * len(acts_pos) + [0] * len(acts_neg)
    try:
        chk = Pr.override_check(signal_dir, bal_acts, bal_labels, float(overrode.float().mean()))
    except Exception as e:                       # e.g. sklearn missing or single-class batch
        chk = {"error": repr(e)}

    return Extraction(behavior, signal_dir, sup_dir, chk, acts, masks, prompts, pairs)
