"""Clean-baseline run: judged labels + HarmBench safety, end to end.

Three phases (models loaded sequentially so a 7B target + 13B judge fit an A100 one at a time):
  A) target model  -> cache activations + generate completions (transformer_lens)
  B) judges        -> HarmBench-cls labels safety; a local LLM labels sycophancy/knowledge-conflict
  C) directions    -> suppression directions from cached acts + judged masks, then the shared-axis report

Run (Colab, after unzip + pip install + `huggingface_hub.login()` for the Llama-2-based cls):
    CD_MODEL=qwen2.5-7b-instruct CD_LAYER=16 python scripts/clean_baseline.py
Env knobs: CD_MODEL, CD_JUDGE_LLM, CD_LAYER, CD_CONTRAST, CD_MAX_PAIRS, CD_MAX_NEW_TOKENS, CD_SUBSET.
"""
import gc
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from src import data as D, directions as Dir, judge as J, model as M, projection as P, universal as U

MODEL     = os.environ.get("CD_MODEL", "qwen2.5-7b-instruct")
JUDGE_LLM = os.environ.get("CD_JUDGE_LLM", "Qwen/Qwen2.5-7B-Instruct")   # sycophancy / kc judge
CONTRAST  = os.environ.get("CD_CONTRAST", "overrode_vs_resisted")
MAXB      = int(os.environ.get("CD_MAX_PAIRS", "0")) or None
MAXNT     = int(os.environ.get("CD_MAX_NEW_TOKENS", "64"))
POS       = -1

cfg        = D.load_behaviors_config("configs/behaviors.yaml")
models_cfg = D.load_yaml("configs/models.yaml")
SUBSET     = os.environ["CD_SUBSET"].split(",") if os.environ.get("CD_SUBSET") else cfg["mvp_subset"]


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----- Phase A: target model — cache activations + generate completions -----
mc = M.resolve_model_cfg(models_cfg, MODEL)
bundle = M.load_model(mc["tl_name"], dtype=mc.get("dtype", "bfloat16"))
LAYER = int(os.environ["CD_LAYER"]) if os.environ.get("CD_LAYER") else M.default_layer_for(mc, bundle)
print(f"target {bundle.name} | layer {LAYER} | behaviors {SUBSET}")

store = {}
for b in SUBSET:
    pairs = D.load_pairs(b, cfg)
    if MAXB:
        pairs = pairs[:MAXB]
    clean = M.format_chat(bundle, [p.prompt_clean for p in pairs])
    manip = M.format_chat(bundle, [p.prompt_manip for p in pairs])
    pos_txt, neg_txt = D.load_signal_statements(b, cfg)
    if MAXB:
        pos_txt, neg_txt = pos_txt[:MAXB], neg_txt[:MAXB]
    store[b] = dict(
        pairs=pairs,
        acts_manip=M.get_activations(bundle, manip, [LAYER], [POS])[(LAYER, POS)],
        acts_clean=M.get_activations(bundle, clean, [LAYER], [POS])[(LAYER, POS)],
        acts_pos=M.get_activations(bundle, M.format_chat(bundle, pos_txt), [LAYER], [POS])[(LAYER, POS)],
        acts_neg=M.get_activations(bundle, M.format_chat(bundle, neg_txt), [LAYER], [POS])[(LAYER, POS)],
        gen_manip=M.generate(bundle, manip, max_new_tokens=MAXNT),
        gen_clean=M.generate(bundle, clean, max_new_tokens=MAXNT),
    )
    print(f"  cached {b}: {len(pairs)} pairs")
del bundle
_free()

# ----- Phase B: judges -> masks (safety = HarmBench-cls; syc/kc = local LLM) -----
cls = J.load_hf(J.HARMBENCH_CLS)
if "safety" in SUBSET:
    s = store["safety"]
    s["overrode"]   = J.score("safety", s["pairs"], s["gen_manip"], cls=cls)
    s["clean_over"] = J.score("safety", s["pairs"], s["gen_clean"], cls=cls)   # should be ~all 0 (refused)
del cls
_free()

llm = J.load_hf(JUDGE_LLM)
for b in [x for x in SUBSET if x in ("sycophancy", "knowledge_conflict")]:
    s = store[b]
    s["overrode"]   = J.score(b, s["pairs"], s["gen_manip"], llm=llm)
    s["clean_over"] = J.score(b, s["pairs"], s["gen_clean"], llm=llm)
del llm
_free()

# ----- Phase C: directions + shared-axis report -----
sup_dirs, sig_dirs, acts_by, masks_by, rates = {}, {}, {}, {}, {}
for b in SUBSET:
    s = store[b]
    ov = torch.tensor([x == 1 for x in s["overrode"]])
    ck = torch.tensor([x == 0 for x in s["clean_over"]])          # clean_kept = clean arm NOT gated
    acts_by[b] = {"manip": s["acts_manip"], "clean": s["acts_clean"]}
    masks_by[b] = {"overrode": ov, "resisted": ~ov, "clean_kept": ck}
    sig_dirs[b] = Dir.signal_direction(s["acts_pos"], s["acts_neg"], layer=LAYER, position=POS, name="s", behavior=b)
    sup_dirs[b] = Dir.suppression_direction(acts_by[b], masks_by[b], mode=CONTRAST, layer=LAYER, position=POS, behavior=b)
    rates[b] = f"{int(ov.sum())}/{len(ov)} = {ov.float().mean():.2f}"
    print(f"  {b:<20} judged override rate: {rates[b]}")

sup_res = dict(zip(SUBSET, P.residualize_all([sup_dirs[b] for b in SUBSET],
                                             [sig_dirs[b] for b in SUBSET], own_only=True)))
print(f"\n===== shared-suppression report (JUDGED labels, layer {LAYER}, contrast {CONTRAST}) =====")
res = U.report(sup_res, acts_by, masks_by, SUBSET)

os.makedirs("results", exist_ok=True)
with open("results/clean_baseline.json", "w") as f:
    json.dump({"model": mc["tl_name"], "layer": LAYER, "contrast": CONTRAST,
               "override_rates": rates,
               "transfer": np.asarray(res["transfer"]).tolist(),
               "tG_loo": res["tG_loo"], "tG_meta": {k: v for k, v in res["tG_meta"].items()
                                                    if k != "per_dir_cos_to_tG"}}, f, indent=2)
print("\nsaved -> results/clean_baseline.json")
