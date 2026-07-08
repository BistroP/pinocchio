# Design Decision 1 — the common suppression-direction contrast

**Status:** investigated, recommendation below, **awaiting your call after reading**.
**TL;DR recommendation:** use **`did_overrode_vs_resisted`** (a difference-in-differences on the
overrode-vs-resisted split) as the *primary* contrast; report **`overrode_vs_resisted`**,
**`manip_vs_clean`**, and **`paired_manip_minus_clean`** as a pre-registered **robustness set**;
use **`matched_output`** (sycophantic-vs-genuine style) as a **within-behavior validation** where a
legitimate same-output twin exists; and run **probe transfer** alongside the cosines. Rationale below.

---

## 1. What we must isolate

The object is **suppression = the gating of a signal the model still represents**, *not* the signal
and *not* the behavior. The pre-registered claim is about a shared **suppressor**, so each
per-behavior direction must capture the *act of gating*, with as little behavior-specific content
(the manipulation stimulus, the item, the signal itself) baked in as possible. Contamination is not
neutral: **behavior-specific stimulus content inflates apparent distinctness**, biasing the cosine
matrix toward a spurious "distinct directions" result — the exact failure mode the build spec warns
about ("if the contrasts differ in kind across behaviors, the subspace spanned by them is an
artifact"). Note the manipulation *content* differs in kind across the family (jailbreak text vs
injected context vs "are you sure") **even when the contrast form is identical**, so the contrast
form has to actively cancel that content.

## 2. The candidate contrasts

Notation: `manip`/`clean` are the two arms per item; `overrode`/`resisted` split manipulated items
by whether the output gated the signal; `legit` is a legitimate *same-output* case (e.g. genuine
agreement).

| mode | direction | cancels item? | cancels manip-stimulus? | family-general? |
|---|---|---|---|---|
| `manip_vs_clean` | μ(manip·overrode) − μ(clean·kept) | ✗ | ✗ | ✓ |
| `paired_manip_minus_clean` | mean(manip − clean) over overrode | ✓ | ✗ | ✓ |
| `overrode_vs_resisted` | μ(manip·overrode) − μ(manip·resisted) | partial | ✓ | ✓ (needs resisted cases) |
| **`did_overrode_vs_resisted`** | [manip−clean │ overrode] − [manip−clean │ resisted] | ✓ | ✓ | ✓ (needs resisted cases) |
| `matched_output` | μ(manip·overrode) − μ(legit) | n/a | ✓ (output fixed) | ✗ (safety has no twin) |

"Cancels manip-stimulus" = both arms contain the manipulation, so the diff removes its
representation. `did_*` is the only family-general contrast that cancels **both** item baseline and
manipulation stimulus.

## 3. What the literature does — and what it implies

- **Arditi et al., *Refusal is mediated by a single direction*** (arXiv:2406.11717).
  Contrast = μ(harmful) − μ(harmless), last token, layer ≈14; validated by directional ablation +
  activation addition. This is a **stimulus contrast** and it yields a **behavior (refusal)**
  direction — the stimulus *is* the signal of interest there. For us the signal is a confound, so
  we cannot just copy it. → analog of `manip_vs_clean`.

- **Joad et al., *There Is More to Refusal than a Single Direction*** (arXiv:2602.02132) — our
  closest precedent, one level up.
  Contrast = μ(refusal-warranting) − μ(benign) per category, token **−2** ("assistant decision
  state"), mid-layer (15–16 Llama). Distinctness shown by a **pairwise cosine matrix** (typical
  0.4–0.6, some near-orthogonal) — *exactly our MVP output*. Shared knob shown by a **steering
  trade-off** (refusal-rate vs over-refusal-rate curves near-identical across directions). This is
  the **distinct-but-shared-knob** outcome demonstrated *within* refusal; we extend it across the
  factual↔safety boundary. Two lessons: (i) the cosine matrix is the accepted geometry test;
  (ii) their contrast is still a stimulus contrast, so their directions carry category-specific
  content — which is fine when the object is refusal *style*, but is the confound we must remove.

- **Vennemeyer et al., *Sycophancy Is Not One Thing*** (arXiv:2509.21305).
  Contrast = μ(sycophantic behavior) − μ(**genuine agreement**): **same output, different cause**.
  This is the cleanest "isolate the deference, not the agreement" contrast, and it is the direct
  motivation for `matched_output`. Validation = **subspace-removal consistency**: a behavior
  collapses only when *its own* subspace is removed; steering survives removal of the others. →
  motivates `matched_output` and the phase-2 subspace-removal check (already stubbed in
  `steering.py`).

- **Anon., *LLMs Know They're Wrong and Agree Anyway: the Shared Sycophancy-Lying Circuit***
  (arXiv:2604.19117).
  Shows a **shared substrate** *within the epistemic domain* via **probe transfer** (sycophancy→lying
  AUROC 0.83–0.85) + projection-ablation, with the **opinion/genuine arm orthogonal**. Two lessons:
  (i) pair the cosine geometry with **probe transfer** (`probing.transfer_check`) as a second, more
  behavioral test of shared structure; (ii) the genuine/legitimate arm being orthogonal validates
  the genuine-update control and the `matched_output` baseline.

- **Rimsky et al., *Contrastive Activation Addition*** (arXiv:2312.06681).
  Steering vectors = mean of activation differences over contrastive pairs **matched for length,
  topic, and syntax to minimize confounds beyond the target trait**. The governing principle:
  *match everything except the thing you want*. This is the general argument for holding the
  manipulation fixed (→ `overrode_vs_resisted` / `did_*`) over a raw condition contrast.

## 4. The deciding factor: the cross-family constraint

`matched_output` is the *cleanest* isolation of the suppressor and is what the nearest precedent
(2509.21305) actually uses — but it **does not generalize to safety**: there is no "legitimate
compliance" twin for a harmful request (complying with a *different, benign* request changes the
content entirely). Demanding one contrast applied identically across the factual↔safety family
therefore **rules `matched_output` out as the primary**. That leaves the family-general options, and
among them only `did_overrode_vs_resisted` cancels both confounds. This asymmetry — that the clean
within-domain contrast is unavailable once safety is in the family — is itself a finding worth a
sentence in the writeup.

## 5. What each contrast biases the *result* toward

- `manip_vs_clean`, `paired_manip_minus_clean`: carry behavior-specific **manipulation content** →
  bias toward **spurious distinctness** (directions look different because the stimuli differ).
  Projecting out the *signal* (Step 5) does **not** remove this — the signal ≠ the manipulation
  stimulus. So this confound survives the projection control.
- `overrode_vs_resisted`: cancels the stimulus, but overrode vs resisted **items may differ**
  (resisted items may have stronger signal / be intrinsically harder to flip) → **selection bias**;
  can bias either way.
- `did_overrode_vs_resisted`: the paired clean subtraction removes item baseline and the resisted
  arm removes the common manipulation response → residual selection effects only. Lowest-bias
  family-general option.

## 6. Recommendation

1. **Primary:** `did_overrode_vs_resisted`. Report the raw + residual cosine matrix and the
   participation ratio / shared-knob test on this.
2. **Robustness (pre-register, report all):** `overrode_vs_resisted`, `manip_vs_clean`,
   `paired_manip_minus_clean`. If the cosine *structure* (not the absolute values) is stable across
   contrasts, the finding is robust; if it flips between stimulus contrasts (1,3) and
   held-fixed contrasts (2,4), that is diagnostic of stimulus contamination and is reportable.
3. **Within-behavior validation:** `matched_output` for sycophancy and knowledge-conflict (a
   context-that-*matches*-parametric twin gives a legitimate same-output case). Check that the
   primary direction aligns (high cosine) with this cleaner one; low alignment flags contamination.
4. **Second geometry test:** probe **transfer** (`probing.transfer_check`) in addition to cosines,
   per 2604.19117 — a direction trained on behavior A that classifies behavior B is stronger
   evidence of shared structure than cosine alone.

## 7. Known confounds & mitigations

- **Selection bias in overrode/resisted** — mitigate by matching overrode vs resisted items on
  clean-arm signal-probe magnitude, or (strongest) by **manipulation-strength titration**: fix the
  item, ramp the jailbreak / context strength, contrast just-below vs just-above the flip threshold.
  This holds item identity fixed and is the gold-standard version of `did_*`.
- **Too few resisted cases** for strong manipulations → `did_*`/`overrode_vs_resisted` degrade.
  Mitigate by weakening the manipulation to sit near the 50% flip point when collecting activations.
- **Scorer noise** defining overrode/resisted (currently keyword heuristics in `eval.py`) →
  replace with an LLM judge before trusting the split.

## 8. Layer / token position

Precedents cluster at **mid-layer** (Arditi ≈14; 2602.02132 15–16 Llama) and either the **last
token** (Arditi) or **−2 "decision state"** (2602.02132). `configs/models.yaml` currently uses
layer 14 / position −1. **Add position −2 to the sweep** (`sweep.py`) and pick per the override
check — the decision-state token may separate the suppressor more cleanly than the final token.

## 9. Reading list (fastest path)

1. arXiv:2602.02132 — the distinct-but-shared-knob result you are extending (skim §method + Fig/Table 1–2).
2. arXiv:2509.21305 — the sycophantic-vs-genuine contrast + subspace removal.
3. arXiv:2604.19117 — probe transfer as the shared-substrate test; orthogonal genuine arm.
4. arXiv:2312.06681 (CAA) — the "match everything except the target" principle.
5. arXiv:2406.11717 (Arditi) — the diff-of-means + ablation baseline.

## 10. Mapping to code

- All five modes are implemented in `src/directions.py::suppression_direction` (`CONTRAST_MODES`).
  `did_overrode_vs_resisted` runs in `notebooks/driver.ipynb` unchanged (it already builds
  clean+manip activations and the overrode/resisted masks).
- **`matched_output` is now wired and runnable.** The legit same-output twin is built per behavior in
  `data.py` (sycophancy: assert the *correct* answer; knowledge-conflict: inject the *matching*
  NQSwap `parametric_passage`), materialized by `src/prepare.py`, gated by
  `data.supports_matched_output` (safety excluded — no legitimate same-output twin), and scored onto
  the `legit_followed` mask by `eval.followed_legit`. The driver's final cell reports
  `cos(primary, matched_output)` per behavior as the contamination check: **high cosine ⇒ the primary
  contrast is capturing genuine suppression, not manipulation-stimulus content.**
- `configs/behaviors.yaml::contrast.mode` currently holds `overrode_vs_resisted` (the pre-existing
  default and the simplest of the recommended pair). **To adopt the recommendation, change it to
  `did_overrode_vs_resisted`.** Left unchanged pending your read of this memo.
