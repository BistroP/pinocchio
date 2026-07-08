"""Materialize the real datasets into data/ as the JSONL that data.py's loaders expect.

Uses ONLY the standard library (urllib / csv / json) via the Hugging Face dataset-viewer HTTP API
and the AdvBench CSV — no `datasets` dependency — so it runs anywhere with a network connection,
including before transformer_lens is installed. Re-run to refresh or expand (--limit).

    python -m src.prepare --limit 256
    python -m src.prepare --behaviors knowledge_conflict --limit 500

Sources (verified schemas):
  sycophancy          truthfulqa/truthful_qa (generation/validation)   question, best_answer, incorrect_answers[]
  safety              llm-attacks AdvBench CSV (goal) + tatsu-lab/alpaca (instruction, input=="")
  knowledge_conflict  pminervini/NQ-Swap (default/dev)                 question, org_answer[], sub_answer[], sub_context
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import urllib.parse
import urllib.request

from . import data as D

VIEWER = "https://datasets-server.huggingface.co"
ADVBENCH_CSV = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"  # legacy
HARMBENCH_CSV = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
FALSEQA_CSV = "https://raw.githubusercontent.com/thunlp/FalseQA/main/dataset/{split}.csv"
_UA = {"User-Agent": "context-deference/prepare"}


def _http_json(url: str) -> dict:
    import time
    last = None
    for attempt in range(4):                       # HF dataset-viewer 502s/timeouts are transient
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=60) as r:
                return json.load(r)
        except Exception as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise last


def _viewer_rows(dataset: str, config: str, split: str, limit: int) -> list[dict]:
    """Page through the dataset-viewer /rows endpoint (max 100/request)."""
    out, offset = [], 0
    while len(out) < limit:
        n = min(100, limit - len(out))
        url = f"{VIEWER}/rows?" + urllib.parse.urlencode(
            dict(dataset=dataset, config=config, split=split, offset=offset, length=n))
        batch = _http_json(url).get("rows", [])
        if not batch:
            break
        out.extend(row["row"] for row in batch)
        offset += len(batch)
        if len(batch) < n:
            break
    return out[:limit]


def _first(v):
    return v[0] if isinstance(v, (list, tuple)) and v else v


def _clean(text: str) -> str:
    text = re.sub(r"</?[A-Za-z][^>]*>", " ", text or "")   # strip simple tags e.g. <P> ... </P>
    return re.sub(r"\s+", " ", text).strip()


# ------------------------------------------------------------------------------ per behavior
def prepare_sycophancy(cfg: dict, limit: int):
    bcfg = cfg["behaviors"]["sycophancy"]
    pairs, signal = [], []
    for r in _viewer_rows("truthfulqa/truthful_qa", "generation", "validation", limit * 2):
        q, best = r.get("question"), r.get("best_answer")
        wrongs = r.get("incorrect_answers") or []
        if not (q and best and wrongs):
            continue
        wrong = wrongs[0]
        pairs.append({"question": q, "correct": best, "wrong": wrong})
        signal.append({"text": f"Q: {q} A: {best}", "label": 1})
        signal.append({"text": f"Q: {q} A: {wrong}", "label": 0})
        if len(pairs) >= limit:
            break
    D.write_jsonl(bcfg["dataset"], pairs)
    D.write_jsonl(bcfg["signal_data"], signal)
    return len(pairs), len(signal)


def prepare_safety(cfg: dict, limit: int):
    bcfg = cfg["behaviors"]["safety"]
    # HarmBench 'standard' behaviors (self-contained harmful requests, 7 semantic categories).
    beh = [r for r in _github_csv(HARMBENCH_CSV) if r.get("FunctionalCategory") == "standard"][:limit]
    harmful = [{"request": r["Behavior"], "behavior_id": r["BehaviorID"],
                "category": r.get("SemanticCategory", "")} for r in beh]
    D.write_jsonl(bcfg["dataset"], harmful)                      # write pairs FIRST (don't lose on a signal hiccup)
    alp = _viewer_rows("tatsu-lab/alpaca", "default", "train", limit * 3)
    harmless = [r["instruction"] for r in alp if not (r.get("input") or "").strip()][:limit]
    signal = ([{"text": h["request"], "label": 1} for h in harmful] +   # harm direction: harmful vs harmless
              [{"text": h, "label": 0} for h in harmless])
    D.write_jsonl(bcfg["signal_data"], signal)
    return len(harmful), len(signal)


def prepare_knowledge_conflict(cfg: dict, limit: int):
    bcfg = cfg["behaviors"]["knowledge_conflict"]
    pairs, signal, seen = [], [], set()
    for r in _viewer_rows("pminervini/NQ-Swap", "default", "dev", limit * 3):
        q = r.get("question")
        org, sub = _first(r.get("org_answer")), _first(r.get("sub_answer"))
        passage = _clean(r.get("sub_context"))            # conflicting (substituted) context
        org_passage = _clean(r.get("org_context"))        # matching (parametric) context -> legit twin
        if not (q and org and sub and passage and org_passage) or str(org).strip().lower() == str(sub).strip().lower():
            continue
        if q in seen:
            continue
        seen.add(q)
        pairs.append({"question": q, "parametric": org, "context": sub,
                      "context_passage": passage, "parametric_passage": org_passage})
        signal.append({"text": f'The answer to "{q}" is {org}.', "label": 1})
        signal.append({"text": f'The answer to "{q}" is {sub}.', "label": 0})
        if len(pairs) >= limit:
            break
    D.write_jsonl(bcfg["dataset"], pairs)
    D.write_jsonl(bcfg["signal_data"], signal)
    return len(pairs), len(signal)


def _github_csv(url: str) -> list[dict]:
    with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=60) as resp:
        return list(csv.DictReader(io.StringIO(resp.read().decode())))


def prepare_false_premise(cfg: dict, limit: int):
    bcfg = cfg["behaviors"]["false_premise"]
    # behavior pairs from the TEST split; FalseQA is grouped label=1 (FPQ) then label=0 (revised TPQ),
    # paired by within-group index (FPQ[i] <-> its revised TPQ[i]).
    test = _github_csv(FALSEQA_CSV.format(split="test"))
    fpq = [r for r in test if r["label"] == "1"]              # false-premise question -> manip
    tpq = [r for r in test if r["label"] == "0"]              # revised true-premise question -> clean
    pairs = []
    for f, t in zip(fpq, tpq):
        if not (f.get("question") and t.get("question")):
            continue
        pairs.append({"neutral": t["question"], "premise": f["question"],
                      "correct": (f.get("answer") or "").strip(), "wrong": None})  # answer = factual rebuttal
        if len(pairs) >= limit:
            break
    D.write_jsonl(bcfg["dataset"], pairs)

    # fact (premise-validity) signal from the TRAIN split (disjoint items): true-premise questions are
    # fact-respecting (label 1), false-premise questions are fact-violating (label 0).
    train = _github_csv(FALSEQA_CSV.format(split="train"))
    tp = [r["question"] for r in train if r["label"] == "0"][:limit]
    fp = [r["question"] for r in train if r["label"] == "1"][:limit]
    signal = [{"text": q, "label": 1} for q in tp] + [{"text": q, "label": 0} for q in fp]
    D.write_jsonl(bcfg["signal_data"], signal)
    return len(pairs), len(signal)


PREPARERS = {
    "sycophancy": prepare_sycophancy,
    "safety": prepare_safety,
    "knowledge_conflict": prepare_knowledge_conflict,
    "false_premise": prepare_false_premise,
}


def main():
    ap = argparse.ArgumentParser(description="Materialize real datasets into data/ (stdlib HTTP).")
    ap.add_argument("--behaviors", nargs="*", default=list(PREPARERS), choices=list(PREPARERS))
    ap.add_argument("--limit", type=int, default=128, help="max items per behavior")
    ap.add_argument("--config", default="configs/behaviors.yaml")
    args = ap.parse_args()
    cfg = D.load_behaviors_config(args.config)
    for b in args.behaviors:
        n_pairs, n_sig = PREPARERS[b](cfg, args.limit)
        print(f"{b:<20} pairs={n_pairs:<5} signal={n_sig}")


if __name__ == "__main__":
    main()
