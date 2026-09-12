# Fetal-fraction conditioning: training brief

This public repository is **inference-only**. The trainer, data pipeline, and
labeled library/fragment corpora live in a private tree. Nothing in this repo
can repair published weights.

This note is the brief for a **proper continued train** of FF conditioning.
It is not a request to fine-tune the tiny `FFEmbedding` head, and it is not
a from-scratch rewrite unless a continued train from the current checkpoint
fails.

## Problem

Published v15 (`eabhaseq/cfdna-gen`) exposes a dual FF path:

1. **Bin token** — 17 non-uniform tokens, `tokens.py` `FF_BIN_BOUNDARIES`
2. **Continuous head** — `FFEmbedding` added to every hidden state

In practice the path is unused:

| Evidence | What the audit found |
|---|---|
| Continuous `FFEmbedding` | Near-constant: `\|\|FFEmbedding(0.01) − FFEmbedding(0.10)\|\|` ≈ 0 |
| FF bin token embeddings | Pairwise cosine ≈ 0.98 (nearly identical) |
| Generation | `target_ff` alone does **not** change fragment realism |
| Tests (before this work) | Never asserted that FF changes sequences |

Length and GC conditioning still do real work. The model’s actual product
value is **length/GC-conditioned sequence realism** (motifs, composition,
nucleosome-associated patterns). Maternal/fetal mix and the low-FF “look”
for detectability live in the caller / `eabhaseq-synthetic` mixer
(length mix, origin, coverage), not in `target_ff`.

Default API value is `target_ff=0.10`. The left tail is coarse: **0–2%
share a single bin** (0.5% and 1.9% are the same token; 2.0% is the next).

## Not sufficient

Do **not** ship any of the following as the fix:

- Fine-tuning **only** the tiny `FFEmbedding` head (or only the 17 FF
  token rows) while freezing the transformer
- Claiming low-FF / bimodal low-FF libraries via `target_ff=` with
  length and GC pinned
- Changing this inference repo’s docs or mixer defaults and calling
  the checkpoint “fixed”
- Inventing clinical AUCs or patient-level claims

A head-only fine-tune cannot teach the 14-block stack to move motifs
and composition with FF. The collapsed continuous path is a symptom;
the missing training signal is the cause.

## Required for a proper fix

### 1. Private trainer and data pipeline

This repo cannot run the train. You need access to the private trainer,
the library-level (and, if available, fragment-level) FF labels, and
the data pipeline that produced v15.

### 2. Continued train from the current checkpoint

Prefer **continued training from v15** over from-scratch. Re-initialize
or un-collapse `FFEmbedding` and the FF token rows if needed, but keep
the length/GC-conditioned sequence prior. From-scratch only if continued
train fails to make FF move the sequence distribution.

Supervise with **library-level FF labels** on fragments (every fragment
from a library inherits that library’s FF unless you have a better
origin/FF annotation).

### 3. Oversample the early-gestation left tail

Oversample **0.5–4% FF**. The current 0–2% bin is one coarse bucket and
the published weights do not use even that. Without oversampling, any
retrain will keep ignoring the clinically important low-FF regime.

### 4. Finer left-tail bins

Remap bins before the continued train (requires a vocab / embedding
row plan — do not silently reuse v15 row 47 for a new 0.5% edge).

Suggested direction: **0.5% steps below 4%**, then keep ~2% steps through
the typical 4–20% clinical range. Example sketch (not frozen):

```
0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040,
0.06, 0.08, 0.10, … existing mid/high edges …
```

If the vocab must stay at 64 tokens, steal resolution from the high-FF
tail (already coarse) rather than from 0.5–4%. Update
`FF_BIN_BOUNDARIES`, `decode_ff_bin_token`, and unit tests in this repo
in the same change that ships the new checkpoint.

### 5. Training signal so FF moves motifs / composition

The loss must make FF a **used** condition:

- Continuous head must **not** decay to a constant or a GC-cloned bias
- Regularize or monitor `\|\|FFEmbedding(0.01) − FFEmbedding(0.10)\|\|`
  during train (keep it in the same ballpark as a comparable GC delta)
- Do not allow the FF bin token rows to collapse onto each other
  (monitor mean off-diagonal cosine)
- Hold length and GC fixed in the FF term so the model cannot satisfy
  the FF label by cheating through length or GC

### 6. Eval gates before shipping a new checkpoint

Hold **length and GC fixed**. Do not ship if any gate fails.

1. **Embed dynamic range.**
   `\|\|FFEmbedding(0.01) − FFEmbedding(0.10)\|\|` in the GC-delta
   ballpark (compare to `\|\|GCEmbedding(0.40) − GCEmbedding(0.50)\|\|`).
   Near-zero is an automatic reject.  
   Offline check (not CI — downloads ~500MB weights):

   ```bash
   python scripts/check_ff_embedding_collapse.py --model eabhaseq/cfdna-gen
   python scripts/check_ff_embedding_collapse.py --model /path/to/new-ckpt
   ```

2. **Logit / motif / composition tables** differ for
   `FF ∈ {0.005, 0.01, 0.04, 0.10}` at pinned length and GC.

3. **Ablation.** `target_ff=None` vs `0.10` vs `0.01` must be
   distinguishable on those tables (not just on the embed L2).

4. **Optional later:** TSTR (train-on-synthetic, test-on-real)
   stratified by FF band on real data. Do not invent AUCs here.

This inference repo’s unit tests still must **not** pretend v15
satisfies these gates. After a new checkpoint, add a gated integration
test or document the offline suite next to the HF card.

### 7. API follow-up after retrain

`target_ff` is **library-level** style conditioning. If callers need
fetal-like vs maternal-like sequences **at the same length**, the API
needs a per-fragment FF and/or an origin flag. That is a follow-up in
this repo after the weights actually move. Do not add the flag now and
claim v15 implements it.

### 8. Workstream checklist

No dates or cost estimates — this repo has none.

- [ ] Access to the private trainer and data pipeline
- [ ] Confirm library-level (and any fragment-level) FF labels
- [ ] Oversample 0.5–4% FF in the train mix
- [ ] Remap left-tail bins (0.5% steps below 4%); plan vocab rows
- [ ] Continued train from v15 (from-scratch only if that fails)
- [ ] Monitor continuous-head L2 and FF-token cosine during train
- [ ] Eval gates 1–3 (length & GC pinned); optional TSTR later
- [ ] Update `FF_BIN_BOUNDARIES` / decode helpers / tests in this repo
- [ ] HF release of the new checkpoint + package version bump
- [ ] HF model card: drop the v15 collapse warning only after gates pass
- [ ] API follow-up: per-fragment FF / origin flag if still required

## What this inference repo already does

Shipped without retraining (this PR):

- Honest README / docstrings: v15 `target_ff` is not a fetal-fraction
  simulator
- `decode_ff_bin_token`, range warnings, `TOKEN_PAD` for condition padding
- Non-vacuous left-tail bin tests (`0.005` vs `0.019` vs `0.02`)
- Serve-time warning when a loaded checkpoint’s continuous FF delta is
  near zero; `scripts/check_ff_embedding_collapse.py` for real weights

Out of scope here: GPU training, downloading v15 to “edit” tensors,
`eabhaseq-synthetic` mixer changes, clinical performance claims.
