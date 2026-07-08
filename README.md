# Context-Deference

**The structure of suppression across a family of behaviors.**

## The question (one line)

What is the internal **structure of suppression** — the gating of a signal the model
*still represents* — across a *family* of behaviors spanning the factual↔safety boundary?
One shared direction, a set of distinct directions, or distinct directions sharing a
one-dimensional control knob? The result comes from characterizing the **subspace** spanned
by per-behavior suppression directions, after the signal directions they trivially overlap
are projected out, licensed by causal steering and subspace-removal.

This is a **characterization of suppression structure across a family** — the *Truth-is-Universal*
playbook (difference-of-means directions, subspace geometry) applied to suppression across
domains. The toolkit is borrowed; the contribution is the structure of the subspace. The unit
of analysis is the **family**, so everything is built for **N behaviors**, not a pair.

**Pre-registered:** all three outcomes are reportable results — one shared direction,
distinct directions, or distinct-but-shared-knob. **A null is a finding, not a failure.**

## The family of behaviors

Membership criterion: a behavior where a signal the model **still represents** is gated out of
the output (**override, not ignorance**). Each candidate must pass the override check
(`probing.override_check`, Step 3) to be admitted.

| Behavior | Signal (still represented) | Gate / manipulation |
|---|---|---|
| **Sycophancy** | truth | confident false user belief ("are you sure") |
| **Safety / refusal** | harm | jailbreak / persona |
| **Knowledge-conflict** | fact (parametric) | conflicting injected context (NQSwap) |
| **False-premise** *(family expansion)* | fact | false presupposition |

**MVP subset:** sycophancy + safety + knowledge-conflict.

## ⚠️ Open design decisions (resolve on paper before trusting Step 4–5 numbers)

These are applied **identically across every behavior** and live in
[`configs/behaviors.yaml`](configs/behaviors.yaml). `directions.py` implements *every* mode;
the config just selects one.

**DESIGN DECISION 1 — common contrast for the per-behavior suppression direction.**
*Investigated against the literature — see [`docs/contrast-decision.md`](docs/contrast-decision.md)
for the full writeup and citations.* Every suppression direction must be extracted with the *same*
contrast, or the subspace spanned by them is an artifact and pairwise cosines mean nothing. The
contrast must isolate the *suppressor*, not the manipulation stimulus (whose content differs by
behavior and would inflate spurious distinctness). Options (`contrast.mode`):
- `did_overrode_vs_resisted` — **recommended primary.** Difference-in-differences:
  [manip−clean │ overrode] − [manip−clean │ resisted]. Cancels *both* item baseline and
  manipulation-stimulus content; lowest-bias family-general contrast.
- `overrode_vs_resisted` — μ(manip & overrode) − μ(manip & resisted). Holds the manipulation fixed;
  isolates the gating outcome. **← current config value** (simplest of the recommended pair).
- `manip_vs_clean` — μ(manip & overrode) − μ(matched clean). The spec's example / field-standard
  (cf. Arditi, 2602.02132), but carries manipulation content → **robustness check only.**
- `paired_manip_minus_clean` — per-pair (manip − clean) over overridden pairs. Robustness check.
- `matched_output` — μ(manip & overrode) − μ(legitimate same-output). Cleanest isolation
  (sycophantic-vs-genuine, 2509.21305) but **not family-general** (safety has no legitimate
  same-output twin) → within-behavior validation only. **Wired & runnable:** the driver's last cell
  reports `cos(primary, matched_output)` per behavior (high ⇒ the primary contrast is capturing
  genuine suppression, not manipulation-stimulus content).

**Recommendation:** `did_overrode_vs_resisted` primary; `{overrode_vs_resisted, manip_vs_clean,
paired_manip_minus_clean}` as a pre-registered robustness set; `matched_output` + probe transfer
(`probing.transfer_check`) as validation. **Your call pending** — change `contrast.mode` to adopt.

**DESIGN DECISION 2 — per-behavior behavioral object. ✅ RESOLVED (2026-07-05): `correctness_label`.**
Use output correctness as the label, read via the signal direction; no separate behavioral
direction. Simplest and symmetric across the family, and the MVP suppression-cosine result is only
weakly sensitive to this choice. (Alternatives kept in code for phase 2: `commitment_direction` — a
separate answer-commitment direction per behavior; `natural_per_behavior` — native object where it
exists, e.g. refusal for safety.)

> Decision 1 changes the suppression directions directly; Decision 2 mainly affects the override
> check and phase-2 steering. **Do not silently re-pick these; change the config deliberately and
> note it here.**

## Repo layout

```
context-deference/
  docs/contrast-decision.md # DESIGN DECISION 1 investigation: contrast forms, literature, recommendation
  configs/behaviors.yaml   # the N behaviors, datasets, manipulations, + the two design decisions
  configs/models.yaml      # model set, chosen layers, token positions
  data/signal/             # clean true/false statements; harmful/harmless requests
  data/<behavior>/         # matched clean vs manipulated pairs, one folder per behavior
  src/model.py             # load model; hook residual stream; cache activations (transformer_lens)
  src/data.py              # build/load pairs; manipulations; filters; genuine-update control
  src/prepare.py           # materialize real datasets into data/ (stdlib HTTP; python -m src.prepare)
  src/directions.py        # diff-means signal dirs + per-behavior suppression dirs (common contrast)
  src/probing.py           # probe projections; override-vs-ignorance check; transfer check
  src/projection.py        # project signal dirs out; residual per-behavior dirs
  src/subspace.py          # cosines, PCA on stacked dirs, shared-knob test, principal angles
  src/pipeline.py          # shared per-behavior extraction (used by both notebooks)
  src/steering.py          # activation-add + directional/subspace ablation; subspace-removal + random baseline
  src/eval.py              # binary behavioral scoring; selectivity; random-baseline p-value
  src/sweep.py             # layer / token-position sweep utilities
  notebooks/driver.ipynb          # MVP: raw+residual cosines, robustness set, matched_output validation
  notebooks/phase2_steering.ipynb # causal: ablation necessity, random baseline, subspace-removal consistency
  results/
```

### Module → methodology-step map

| Step | What | Module |
|---|---|---|
| 1 | matched pairs, filters, genuine-update control | `data.py` (+ `prepare.py` for real data) |
| 2 | signal directions | `directions.py` (activations via `model.py`) |
| 3 | verify override per behavior (**admission check**) | `probing.py` |
| 4 | per-behavior suppression directions (common contrast) | `directions.py` |
| 5 | **subspace characterization + projection control** *(decisive)* | `projection.py` + `subspace.py` |
| 6 | causal: steering, ablation, subspace-removal, random baseline | `steering.py` + `eval.py` |
| — | shared per-behavior extraction (both notebooks) | `pipeline.py` |
| — | layer / token sweeps | `sweep.py` |

## MVP (build first)

**One model** (Llama-3.1-8B-Instruct or Qwen2.5-7B-Instruct), **three behaviors**, and one output:
the **residual pairwise-cosine matrix** across the three suppression directions — raw, then after
projecting out each behavior's signal direction (truth / harm / fact). Do they cluster into a
shared subspace, sit orthogonal, or something in between? That matrix answers "shared vs distinct"
at the correlational level, across a real family rather than a pair.

**Phase 2 (now implemented, `notebooks/phase2_steering.ipynb`):** causal steering, the
subspace-removal consistency check, and the random-baseline sufficiency test — turning the
correlational cosines into a causal shared-vs-distinct read.

**The family is now four** (`cfg["family"]`): sycophancy, safety, knowledge-conflict, and
**false_premise** (← FalseQA — deliberately an *independent* source, so no shared items inflate the
cross-behavior suppression cosine). `mvp_subset` stays the fast three; point a notebook's `SUBSET`
at `cfg["family"]` (or `CD_SUBSET=...`) to run all four.

## How to run

The repo is **structure, not compute**. Run the GPU cells from the notebook on Colab Pro (A100)
or a cloud box; it imports `src/`.

```bash
pip install -r requirements.txt

# 1. Fetch the real datasets into data/ (stdlib HTTP only — no GPU, no `datasets` lib needed).
python -m src.prepare --limit 256           # writes data/<behavior>/pairs.jsonl + data/signal/*.jsonl

# 2. Run the MVP (needs a GPU for the activation caching).
jupyter notebook notebooks/driver.ipynb            # raw+residual cosines, robustness, matched_output

# 3. Run the causal layer (phase 2).
jupyter notebook notebooks/phase2_steering.ipynb   # ablation necessity, random baseline, subspace removal
```

**Science run vs local check.** Both notebooks default to the spec'd model + full data (a GPU job).
Env vars let the *same* notebooks do a fast local validation on a small open model (no HF token, fits
an MPS/CPU box) — e.g. Colab-free sanity check:

```bash
CD_MODEL=qwen2.5-1.5b-instruct CD_MAX_PAIRS=16 CD_MAX_NEW_TOKENS=24 jupyter nbconvert \
  --to notebook --execute notebooks/driver.ipynb
```

This proves the pipeline end-to-end on real activations; the *scientific* numbers need the 7–8B model
on a real GPU (small models exhibit the behaviors too weakly to interpret).

`src/` modules are import-only (no GPU needed to import). **`src/prepare.py` materializes the real
datasets** (sycophancy ← `truthfulqa/truthful_qa`; safety ← llm-attacks AdvBench + `tatsu-lab/alpaca`;
knowledge-conflict ← `pminervini/NQ-Swap`) into the JSONL that `data.py` reads. If those files are
absent, `data.py` falls back to tiny **inline example pairs** so the pipeline stays smoke-testable
offline. Only the activation caching (step 2) needs a GPU.

## Compute & environment

- **GPU required.** 7–8B in bf16 fits a single 24–40GB card (Colab Pro A100). 70B is out of scope.
- **Hooking:** `transformer_lens` for the MVP (clean resid caching + add/ablate hooks, ergonomic
  ≤9B). Switch to `nnsight` past ~9B.
- Keep cached activations out of git (see `.gitignore`).

## Datasets & prior work

- Refusal direction + steering/ablation: `github.com/andyrdt/refusal_direction` (Arditi et al.).
- Persona / trait vectors: `github.com/safety-research/persona_vectors`.
- Subspace-removal consistency check: *Sycophancy Is Not One Thing* (arXiv:2509.21305).
- Within-refusal precedent we extend across domains: *There Is More to Refusal than a Single
  Direction* (arXiv:2602.02132).
- Harmful/harmless prompts: AdvBench / HarmBench-style splits.
- Context-memory conflict: NQSwap (Longpre et al.). Sycophancy QA: TruthfulQA / SycophancyEval
  "are you sure".

## Positioning (internal, not for the paper body)

Contribution is a **characterization of suppression structure across the factual↔safety family** —
a synthesis, not a discovered mechanism. Neighbors: 2602.02132 (within refusal), 2604.19117
(shared substrate within the epistemic domain), 2603.24543 (factual↔safety crossing as a single
cosine inside a steering-safety audit). Our object is the cross-domain structure itself, treated as
the finding. Keep the signals distinct in every sentence: the family is unified (if at all) by a
shared **suppressor**, never by a shared signal and never by "these failures are the same thing."
# pinocchio
