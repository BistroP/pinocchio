"""Load a model and cache residual-stream activations.

Thin wrapper over transformer_lens.HookedTransformer for the MVP (<=9B). Switch to nnsight
if you scale past ~9B (see README). Chat models are formatted with their chat template; we
LEFT-pad so token position -1 is the final real token for every sequence in a batch.
"""
from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Sequence

import torch

try:
    from transformer_lens import HookedTransformer
except Exception:  # allow importing this module without transformer_lens installed
    HookedTransformer = None  # type: ignore

DEFAULT_DTYPE = "bfloat16"
_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass
class ModelBundle:
    model: "HookedTransformer"
    name: str
    device: str

    @property
    def n_layers(self) -> int:
        return self.model.cfg.n_layers

    @property
    def d_model(self) -> int:
        return self.model.cfg.d_model

    @property
    def tokenizer(self):
        return self.model.tokenizer


def pick_device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_model_cfg(models_cfg: dict, name: str) -> dict:
    """Return the models.yaml entry for `name`, or a minimal auto-config for any transformer_lens
    name that isn't listed. For auto-configured models `default_layer` is None: the caller fills it
    from the loaded model (mid-stack, `bundle.n_layers // 2`) so `CD_MODEL=<any tl name>` just runs.
    Mid-stack is a starting default — sweep (sweep.py) to pick the best layer for a final number."""
    listed = models_cfg.get("models", {})
    if name in listed:
        return listed[name]
    return {"tl_name": name, "dtype": "bfloat16", "default_layer": None,
            "default_position": -1, "auto": True}


def default_layer_for(mc: dict, bundle: "ModelBundle") -> int:
    """mc['default_layer'] if set, else mid-stack of the loaded model."""
    return mc.get("default_layer") if mc.get("default_layer") is not None else bundle.n_layers // 2


def load_model(tl_name: str, device: str | None = None, dtype: str = DEFAULT_DTYPE,
               n_ctx: int | None = None) -> ModelBundle:
    if HookedTransformer is None:
        raise RuntimeError("transformer_lens is not installed (pip install -r requirements.txt).")
    device = pick_device(device)
    # Some models get an artificially low n_ctx from transformer_lens ("capped due to memory
    # issues"): Llama-3.1-8B is pinned to 2048. Long prompts (e.g. kc passages) then push a
    # position index past the rotary table -> CUDA device-side assert in apply_rotary. Setting
    # CD_N_CTX (or the n_ctx arg) raises the cap; TL only warns, and every load_model call site
    # picks it up. Unset => unchanged behaviour (so Qwen/other runs are untouched).
    if n_ctx is None and os.environ.get("CD_N_CTX"):
        n_ctx = int(os.environ["CD_N_CTX"])
    kw = dict(dtype=_DTYPES.get(dtype, torch.bfloat16), device=device,
              default_padding_side="left")  # left-pad so position -1 is the final real token
    if n_ctx is not None:
        kw["n_ctx"] = n_ctx  # resizes rotary_cos/sin so long prompts don't overflow the table
    model = HookedTransformer.from_pretrained(tl_name, **kw)
    model.eval()
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    return ModelBundle(model=model, name=tl_name, device=device)


# --------------------------------------------------------------------------------------
# Prompt formatting (instruct/chat models)
# --------------------------------------------------------------------------------------
def format_chat(bundle: ModelBundle, user_messages: Sequence[str], system: str | None = None,
                add_generation_prompt: bool = True) -> list[str]:
    """Apply the tokenizer's chat template. Returns strings (BOS included by the template),
    so tokenize downstream with prepend_bos=False."""
    tok = bundle.tokenizer
    out = []
    for u in user_messages:
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": u}]
        out.append(tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=add_generation_prompt))
    return out


# --------------------------------------------------------------------------------------
# Activation caching
# --------------------------------------------------------------------------------------
def resid_hook_name(layer: int, site: str = "resid_post") -> str:
    return f"blocks.{layer}.hook_{site}"


@torch.no_grad()
def get_activations(bundle: ModelBundle, prompts: Sequence[str], layers: Sequence[int],
                    positions: Sequence[int] = (-1,), site: str = "resid_post",
                    batch_size: int = 16, prepend_bos: bool = False) -> dict[tuple[int, int], torch.Tensor]:
    """Cache residual-stream activations.

    Returns {(layer, position): tensor[n_prompts, d_model]} on CPU in float32.
    `positions` are indices into the (left-padded) sequence; -1 = final token.
    """
    model = bundle.model
    want = {resid_hook_name(l, site) for l in layers}
    out: dict[tuple[int, int], list[torch.Tensor]] = {(l, p): [] for l in layers for p in positions}

    for i in range(0, len(prompts), batch_size):
        batch = list(prompts[i:i + batch_size])
        tokens = model.to_tokens(batch, prepend_bos=prepend_bos)
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in want)
        for l in layers:
            act = cache[resid_hook_name(l, site)]  # [batch, seq, d_model]
            for p in positions:
                out[(l, p)].append(act[:, p, :].float().cpu())
        del cache
        free()

    return {k: torch.cat(v, dim=0) for k, v in out.items()}


# --------------------------------------------------------------------------------------
# Generation (for the override check + eval scoring)
# --------------------------------------------------------------------------------------
def _prompt_lengths(model, prompts: Sequence[str]) -> list[int]:
    """Token length per prompt in ONE tokenizer call (chat-templated text already carries BOS)."""
    enc = model.tokenizer(list(prompts), add_special_tokens=False)["input_ids"]
    return [len(e) for e in enc]


def length_batches(lengths: Sequence[int], max_new_tokens: int, max_batch: int = 32,
                   token_budget: int = 8192) -> list[list[int]]:
    """Index batches sorted by prompt length, so left-padding is minimal and short prompts batch
    wide. A batch's padded footprint (max_len + max_new_tokens) * width stays under `token_budget`:
    ~150-token syco/safety prompts batch ~32 wide, 2k-token kc passages ~3 wide (prefill memory)."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for i in order:
        new_max = max(cur_max, lengths[i])
        if cur and (len(cur) >= max_batch or (new_max + max_new_tokens) * (len(cur) + 1) > token_budget):
            batches.append(cur)
            cur, new_max = [], lengths[i]
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


@torch.no_grad()
def generate(bundle: ModelBundle, prompts: Sequence[str], max_new_tokens: int = 64,
             batch_size: int | None = None, prepend_bos: bool = False, fwd_hooks=None,
             max_batch: int = 32, token_budget: int = 8192) -> list[str]:
    """Greedy decode; returns only the newly generated text per prompt, in input order.

    Batching: prompts are sorted by token length and packed under `token_budget` (padded tokens
    per batch, generation included) up to `max_batch` wide. TransformerLens decoding is
    overhead-bound at batch 8 (~155 ms/step vs a ~10 ms bandwidth floor on an A100), so wide
    batches of short prompts are near-free, while long kc passages get narrow batches that fit
    prefill memory. Pass `batch_size` for the legacy fixed-width, unsorted path.
    `fwd_hooks` (list of (hook_name, fn)) are attached for every batch: this is the single
    generation path for baselines, directional ablation, and activation steering."""
    model = bundle.model
    prompts = list(prompts)
    if batch_size:
        batches = [list(range(i, min(i + batch_size, len(prompts))))
                   for i in range(0, len(prompts), batch_size)]
    else:
        batches = length_batches(_prompt_lengths(model, prompts), max_new_tokens, max_batch, token_budget)
    outs: list = [None] * len(prompts)
    for idx in batches:
        tokens = model.to_tokens([prompts[i] for i in idx], prepend_bos=prepend_bos)
        with model.hooks(fwd_hooks=list(fwd_hooks or [])):
            gen = model.generate(tokens, max_new_tokens=max_new_tokens, do_sample=False, verbose=False)
        for j, i in enumerate(idx):
            outs[i] = model.to_string(gen[j, tokens.shape[1]:])
        free()
    return outs

def free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
