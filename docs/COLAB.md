# Running the real thing on a GPU (Colab / any CUDA box)

The local Mac (18 GB, MPS) can only run a tiny model as a plumbing check. Real results need a CUDA
GPU with ≥ ~20 GB (Colab A100, a lab/Savio node, or a cloud box). The notebooks self-bootstrap on
Colab; on any other box do the same steps by hand (unzip → `pip install -r requirements.txt` → run).

## Colab, step by step

1. **Runtime → Change runtime type → A100 GPU** (or any ≥20 GB CUDA GPU).
2. **File → Upload notebook →** `driver.ipynb` (from your Downloads).
3. **Gated model only (Llama-3.1-8B):** first accept the license at
   <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct> and make a token at
   <https://huggingface.co/settings/tokens>.
4. **Run the first cell** (the bootstrap). It asks you to upload `context-deference.zip`, then
   unzips it, `pip install`s, and prompts `huggingface_hub.login()` — paste your token.
5. **Run the rest top-to-bottom.** You get the raw + residual cosine matrices, the robustness set,
   the matched_output validation, and saved artifacts in `results/`.
6. **Repeat with** `phase2_steering.ipynb` for the causal layer (necessity, random baseline,
   subspace-removal matrix).

Open models (no token) work the same minus step 3 — just set `CD_MODEL` (below).

## Choosing / swapping models — you'll do this a lot

The model is picked by `os.environ["CD_MODEL"]`, falling back to `models.yaml: default_model`.
To run a different **already-configured** model, set an env var in a cell *before* the config cell:

```python
import os; os.environ["CD_MODEL"] = "qwen2.5-7b-instruct"   # or leave unset for the default
```

Configured today (`configs/models.yaml`): `llama-3.1-8b-instruct` (default, gated),
`qwen2.5-7b-instruct` (open), `qwen2.5-1.5b-instruct` (open, local-only sanity).

To try a **new** model, just point `CD_MODEL` at any
[transformer_lens-supported](https://github.com/TransformerLensOrg/TransformerLens) name — **no yaml edit**:

```python
import os; os.environ["CD_MODEL"] = "google/gemma-2-9b-it"
```

It **auto-configures** (`model.resolve_model_cfg`): `n_layers`/`d_model` are read off the loaded model
and the layer defaults to **mid-stack** (`n_layers // 2`). Add a `models.yaml` entry only when you want
a **hand-tuned** layer (e.g. one you found by sweeping) or a non-default dtype:

```yaml
  gemma-2-9b-it:
    tl_name: google/gemma-2-9b-it
    default_layer: 24     # overrides the mid-stack default (from a layer sweep)
    default_position: -1
```

Caveats: mid-stack is a *starting* layer — sweep (`sweep.py`) for a final number; and the name must be
one transformer_lens knows (most Llama/Qwen/Gemma/Mistral variants are).

## Knobs (env vars, same on both notebooks)

| var | default | use |
|---|---|---|
| `CD_MODEL` | `default_model` | which model |
| `CD_CONTRAST` | `contrast.mode` | A/B a suppression contrast without editing config |
| `CD_SUBSET` | `mvp_subset` | comma-sep behaviors; set to the 4-way `family` to include false_premise |
| `CD_MAX_PAIRS` | full | cap items/behavior (speed) |
| `CD_MAX_NEW_TOKENS` | 64 | generation length for scoring |
| `CD_N_RANDOM` | 8 | random directions in the phase-2 baseline |

For the real run leave the caps unset (full 128-item data). Rough A100 time: driver a few minutes,
phase-2 ~15–30 min depending on `CD_N_RANDOM` and subset size.

## Still open: Design Decision 1

`contrast.mode` is still `overrode_vs_resisted` pending your call (see `docs/contrast-decision.md`,
which recommends `did_overrode_vs_resisted`). You don't have to decide before running — the driver's
robustness cell reports **every** contrast side by side, which is exactly the evidence for the call.
