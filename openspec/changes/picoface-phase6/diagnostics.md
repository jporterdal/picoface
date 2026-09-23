# picoface-phase6 diagnostics record

A tuning investigation against the real Dataset Forge export (design Decision: "tuning first, architecture only if tuning doesn't get there"), extended with a targeted decoder-capacity change once tuning alone left two classes behind (see "Decoder capacity escalation," below — pulled into this change at the user's explicit request rather than deferred to a follow-up one). Continues `openspec/changes/archive/2026-09-23-picoface-phase5/diagnostics.md`'s numbers as one story.

## Setup

- **Machine:** same class of CPU-only environment as Phase 5's measurements; PyTorch 2.13.0+cpu, numpy, Pillow. 8 threads visible (PyTorch default).
- **Data:** unless stated otherwise, `configs/default.json`'s taxonomy/resolution (7 classes, 28×28), single or multi-seed real Dataset Forge exports rendered on demand (never committed). Sections below note which train-count default (1,000 or 2,000/class) and which `latent_dim` were in effect, since both changed mid-investigation.
- **Models:** `build_classifier()` (CNN) and `build_vae()` (the student's model), trained with `train()`'s actual defaults (10 epochs, batch size 16, learning rate 1e-3) unless a section is explicitly sweeping a different value. `classify_generated()` uses `n=20` per class.
- **Method note:** hyperparameters not exposed as `build_*()`/`train()` arguments (`LOG_VAR_FLOOR`, `LOG_VAR_LR_MULTIPLIER`, `KL_ANNEAL_FRACTION`, `LATENT_DIM`, `_HEAD_HIDDEN_SIZE`, `CLUSTER_SAMPLE_SPREAD`, `ASCENT_BLUR_EVERY`/`ASCENT_BLUR_SIGMA`) were swept by monkeypatching the relevant `_internals` module's global before each build+train call. These are read as globals at call time inside the model classes (not captured as bound defaults at class-definition time), so this reproduces exactly what editing the source constant would do — the same values shipped are the ones measured.
- **A methodological catch worth recording:** the first attempt at the `classify_generated()` sampling-spread sweep (task 7.2) patched `picoface._internals.linkage_internals._sample_near_clusters`, which had *no effect* — `picoface/linkage.py` imports that name directly (`from ... import _sample_near_clusters`), so it holds its own binding to the original function untouched by patching the source module's attribute. Every "scale" produced identical numbers, silently. The fix was patching `picoface.linkage._sample_near_clusters` (the name actually called). A second attempt, correctly targeted, still wasn't valid: without reseeding `torch` between scales, `torch.randn()`'s state carries over from one `classify_generated()` call to the next, so successive calls draw different random samples regardless of scale, and the "baseline" and "scale=1.0" runs — identical code — disagreed simply from call order. The result reported in this file (task 7.2) reseeds identically before each scale and averages over several sampling seeds.

## Baseline reproduction (task 2.1)

Fresh real-dataset export (seed 0, then-current 1,000/class default), unmodified `src/picoface/`:

| Model | Time (s) | Held-out accuracy |
|---|---|---|
| CNN | 5.2 | 0.974 |
| VAE | 9.6 | 0.954 |

`classify_generated()` overall 0.35, reconstruction agreement overall 0.26 — matching Phase 5's recorded seed-0 numbers exactly (deterministic given seed, export, and machine). Confirms the starting point before any tuning.

## Loss-balance tuning (tasks 2.2–2.4)

All runs: 1,000/class real export, seed 0 unless noted, `latent_dim=8` (pre-retune).

**Log-variance floor / learning-rate multiplier**, `KL_ANNEAL_FRACTION=0.5`:

| Setting | Floor | Multiplier | VAE acc | `classify_generated()` | Reconstruction |
|---|---|---|---|---|---|
| baseline | -6.0 | 10 | 0.918 | 0.343 | 0.293 |
| floor-2 | -2.0 | 10 | 0.930 | 0.357 | 0.264 |
| floor-3 | -3.0 | 10 | 0.914 | 0.350 | 0.279 |
| floor-1 | -1.0 | 10 | 0.934 | 0.300 | 0.236 |
| mult2-floor6 | -6.0 | 2 | 0.918 | 0.314 | 0.293 |
| floor3-mult3 | -3.0 | 3 | 0.923 | 0.279 | 0.257 |
| floor-15-mult10 | -15.0 | 10 | 0.918 | 0.343 | 0.293 |
| floor-15-mult30 | -15.0 | 30 | 0.924 | 0.314 | 0.279 |

**Key finding: the floor never binds at the shipped default.** `reconstruction_log_var`'s per-epoch trajectory at floor=-6.0 is `[-2.03, -4.07, -4.22, -4.29, -4.35, -4.38, -4.41, -4.43, -4.44, -4.45]` — settling at -4.45 by epoch 10. At floor=-15.0 (effectively unconstrained), the trajectory is **identical to 12 decimal places printed**: same settling point, -4.45. The floor was never the bottleneck Phase 3c's risk note anticipated; reconstruction's learned log-variance tracks its own natural optimum (≈ ln(mse)) and simply hadn't reached anywhere near -6 within a 10-epoch run, floor or no floor. `classification_log_var` shows the same pattern in reverse: it drifts slowly toward equilibrium (ending around -0.85 to -1.1 across settings) and never approaches any tested floor either.

**KL annealing schedule length** (`KL_ANNEAL_FRACTION`, floor=-6.0, multiplier=10):

| Fraction | VAE acc | `classify_generated()` | Reconstruction |
|---|---|---|---|
| 0.1 | 0.920 | 0.357 | 0.293 |
| 0.25 | 0.916 | 0.343 | 0.279 |
| 0.5 (baseline) | 0.918 | 0.343 | 0.293 |
| 0.8 | 0.910 | 0.329 | 0.293 |
| 1.0 | 0.899 | 0.336 | 0.293 |

No measurable effect on reconstruction agreement (0.279–0.293 throughout) or on which classes reconstruct — `reconstruction_log_var`'s trajectory is nearly identical across every KL fraction tried, confirming the KL schedule and the reconstruction/classification balance are largely decoupled in this regime.

**Per-class pattern, every setting above:** `triangle`, `star`, `smiley`, `negative_smiley` sit at 0.0 reconstruction agreement in every one of the 13 settings tried (8 floor/multiplier combinations + 5 KL fractions). Only `square`, `ring`, `circle` ever reconstruct. This is the same structural failure Phase 5 found untouched by 2× data or 2× epochs — loss-balance tuning doesn't move it either.

**Decision (2.4): no change.** None of the 13 settings beats the shipped defaults on both `classify_generated()` and reconstruction agreement simultaneously; differences are within single-seed/n=20 noise (~0.04 stderr on a 140-image sample). `LOG_VAR_FLOOR=-6.0`, `LOG_VAR_LR_MULTIPLIER=10`, `KL_ANNEAL_FRACTION=0.5` are unchanged.

## `latent_dim` retuning (task 3)

Same export, floor/multiplier/KL held at defaults. Single-seed sweep (seed 0):

| `latent_dim` | VAE acc | `classify_generated()` | Reconstruction |
|---|---|---|---|
| 4 | 0.933 | 0.264 | 0.229 |
| 8 (baseline) | 0.918 | 0.343 | 0.293 |
| 16 | 0.962 | 0.343 | 0.371 |
| 32 | 0.939 | 0.386 | 0.421 |
| 64 | 0.966 | 0.493 | 0.507 |
| 96 | 0.976 | 0.457 | 0.500 |
| 128 | 0.977 | 0.521 | 0.529 |
| 192 | 0.973 | 0.457 | 0.550 |

Unlike every loss-balance knob, this one moves the needle, and by a lot: reconstruction and `classify_generated()` roughly double between 8 and 64–128, with diminishing (and noisier) returns above 128. Confirmed across 3 seeds (8 vs. 64 vs. 128):

| `latent_dim` | mean `classify_generated()` | mean reconstruction |
|---|---|---|
| 8 | 0.357 | 0.314 |
| 64 | 0.441 | 0.476 |
| 128 | 0.474 | 0.503 |

128 is consistently, if modestly, ahead of 64 in every seed pair tried, at essentially no extra time cost (10.3s vs. 9.9s per VAE run). Held-out VAE accuracy also improves at every `latent_dim` above 8 — there is no classification/reconstruction trade-off from the extra capacity.

**Interpretation:** `latent_dim=8` was an information bottleneck. With only 8 dimensions carrying both class-discriminating structure (which the classification head reads and which is "easy," per Phase 3c's risk note) and image-reconstruction detail, the encoder was squeezing out the latter. Widening the bottleneck relieves that trade-off directly, which loss-reweighting (a different lever entirely) cannot.

**Decision (3.2): `LATENT_DIM` set to 128** in `generator_internals.py`. Full stub-based test suite re-run after the change: 200/200 passed.

## Decision checkpoint (task 4)

`generated.png`/`reconstructed.png` regenerated (via `dataset_forge.smoke.write_figures`, not committed) at `latent_dim=128`, still on the 1,000/class export used throughout tasks 2–3. Overall: CNN 0.974, VAE 0.989 held-out; `classify_generated()` 0.49; reconstruction 0.51.

**Partial resolution.** `square`, `triangle`, `ring`, `circle` are now clearly recognizable in both generated and reconstructed samples — triangle in particular goes from 0.0 reconstruction agreement (Phase 5, and every loss-balance setting above) to a visibly pointed triangle. `negative_smiley`'s agreement numbers also rose substantially (0.4–0.5), but visually the CNN appears to key on its ring-plus-dark-fill polarity rather than its facial dots, which stay lost in the reconstruction. `star` and `smiley` remain the hardest: mostly round blobs, with `star` showing faint multi-point structure in some samples but rarely reproducing enough of it for the CNN to agree, and `smiley`'s facial dots essentially never surviving.

**Open question for a follow-up change (task 4.2), as it stood at this point in the investigation:** `star` and `smiley` (its dot features specifically) are not resolved by loss-balance tuning or by widening `latent_dim` up to 192. Both levers tried in this phase change *how much* and *how* information is weighted or how much capacity the latent holds; neither changes the decoder's own capacity to render fine, spatially precise detail (a five-pointed outline, two small eye-dots plus a mouth arc) from that latent. The evidence for that being a decoder-capacity question, not a tuning question: (1) `star`/`smiley` stayed at ~0.0 reconstruction agreement across all 13 loss-balance settings and only crept up to 0.05–0.25 with 8–24× more latent capacity, well below the 0.65–1.0 the easier classes reach; (2) the decoder itself is unchanged throughout this phase — two small transpose-conv layers (Decision, `generator_internals.py`) — while every class shares the same decoder and the same training budget, so a class-specific failure that doesn't move with more latent capacity or data (see task 5.3) implicates the decoder's ability to place fine, spatially precise detail, not the information available to it. At this point the plan was to write this up as a follow-up-change candidate (decoder capacity and/or label-conditioning) rather than build it here. **That changed after task 8** — see "Decoder capacity escalation," below, where the user asked to push on this before archiving.

## Secondary, independent knobs (task 5)

All at `latent_dim=128`.

**Decoupled weight decay** (`AdamW` on the non-log-variance parameter group; identical to `Adam` at `weight_decay=0`), 1,000/class, seed 0:

| `weight_decay` | VAE acc | `classify_generated()` | Reconstruction |
|---|---|---|---|
| 0 | 0.980 | 0.543 | 0.536 |
| 1e-5 | 0.980 | 0.543 | 0.536 |
| 1e-4 | 0.981 | 0.536 | 0.550 |
| 1e-3 | 0.979 | 0.493 | 0.521 |

1e-5/1e-4 are noise-level vs. baseline; 1e-3 measurably regresses `classify_generated()`. **Not adopted.**

**Classification head width** (`_HEAD_HIDDEN_SIZE`), 1,000/class:

Single-seed screen (seed 0): 16 → 0.464 gen / 0.536 recon; 32 (baseline) → 0.521 / 0.529; 64 → 0.536 / 0.493; 128 → 0.571 / 0.600. 3-seed confirmation (32 vs. 128, retraining the CNN per seed):

| `_HEAD_HIDDEN_SIZE` | seed 1 gen/recon | seed 2 gen/recon |
|---|---|---|
| 32 | 0.479 / 0.557 | 0.443 / 0.421 |
| 128 | 0.521 / 0.543 | 0.421 / 0.521 |

128's mean is ahead on both metrics, but the per-seed signal is inconsistent (seed 2 shows `classify_generated()` regress while reconstruction improves) — the same noise-level, mixed pattern as the loss-balance knobs above, unlike `latent_dim`'s unambiguous cross-seed win. **Not adopted**; `_HEAD_HIDDEN_SIZE` stays at 32.

**Training-set size, 1,000 vs. 2,000 images/class**, `latent_dim=128`:

| Seed | Count | CNN acc | VAE acc | `classify_generated()` | Reconstruction | Time (CNN+VAE) |
|---|---|---|---|---|---|---|
| 0 | 1,000 | 0.974 | 0.989 | 0.49 | 0.51 | ~15s |
| 0 | 2,000 | 0.991 | 0.991 | 0.464 | 0.60 | ~32s |
| 1 | 1,000 | 0.961 | 0.953 | 0.45 | 0.464 | ~15s |
| 1 | 2,000 | 0.991 | 0.991 | 0.457 | 0.657 | ~32s |

**This updates Phase 5's finding, not just confirms it.** Under the old `latent_dim=8` bottleneck, Phase 5 found "more data does not help generation at all." Post-tuning, at `latent_dim=128`, 2,000/class measurably improves reconstruction agreement too (0.46–0.51 → 0.60–0.66), on top of the classification gain Phase 5 already found — the earlier finding was conditional on the latent bottleneck that was itself masking the benefit. Training time stays trivial either way (~32s combined, far under the 5-minute budget). **Adopted:** `dataset_forge/configs/default.json`'s `train_per_class` changed from 1000 to 2000.

## `train()` defaults and CPU time-budget verification (task 6)

Called `picoface.classifier.train`/`picoface.generator.train` directly (the actual public API, not a separate script — also exercised by `tests/test_real_dataset.py`) at true defaults (`epochs=10, batch_size=16, learning_rate=1e-3`) against the final 2,000/class real export, 3 seeds:

| Seed | CNN time | CNN acc | VAE time | VAE acc |
|---|---|---|---|---|
| 0 | 9.5s | 0.991 | 20.9s | 0.953 |
| 1 | 9.5s | 0.991 | 20.8s | 0.997 |
| 2 | 9.6s | 0.979 | 20.7s | 0.999 |

Accuracy (0.979–0.999) is comfortably above Phase 5's 0.94–0.97 range — a believable capstone judge. Wall-clock time (~9.5s CNN, ~20.8s VAE) is well within the 5-minute CPU budget and consistent with Phase 5's linear-in-optimizer-steps scaling. **No default change warranted**; `model-interface`/`shape-classifier` specs are left as-is (no `opsx:update` needed).

## Capstone linkage retuning (task 7)

**`classify_generated()` overall, at the tuned model:** lands around 0.45–0.59 across seeds (final numbers below) — a clear improvement on Phase 5's 0.32–0.35.

**Sampling-spread scale factor** (`CLUSTER_SAMPLE_SPREAD` on the per-dimension `std` in `_sample_near_clusters`), reseeded identically per scale (see the Setup section's methodological note):

| Scale | Mean overall (3 seeds) |
|---|---|
| 0.25 | 0.338 |
| 0.5 | 0.479 |
| 0.75 | 0.526 |
| 1.0 (then-baseline) | 0.493 |
| 1.5 | 0.338 |
| 2.0 | 0.288 |

Finer sweep, 6 sampling seeds:

| Scale | Mean | Std |
|---|---|---|
| 0.6 | 0.518 | 0.034 |
| 0.7 | 0.519 | 0.026 |
| 0.75 | 0.515 | 0.034 |
| 0.8 | 0.521 | 0.033 |
| 0.9 | 0.501 | 0.047 |
| 1.0 | 0.473 | 0.037 |

0.6–0.8 consistently beat 1.0; sampling at the class cluster's full observed spread draws from its edges more often than a slightly tighter draw, which costs agreement. **Adopted `CLUSTER_SAMPLE_SPREAD=0.75`** (the middle of the good range) in `linkage_internals.py`.

**`activation_maximize()`:** unregularized ascent (the Phase 4 default) is high-frequency adversarial noise for most classes — only `square`/`circle` show a rough blob shape; the rest are dominated by diagonal-stripe/checkerboard artifacts the CNN scores very highly but that carry no visible class structure. Tried total-variation regularization at several weights (0.001–0.1): visibly insufficient to suppress the pattern. Tried periodic Gaussian blur (every `ASCENT_BLUR_EVERY` steps, `ASCENT_BLUR_SIGMA`): at `blur_every=10, sigma=1.0`, images are visibly smoother and more shape-like — `star` in particular becomes a recognizable multi-pointed shape. The target-class logit still rises substantially from its random start (e.g. `star`: -4.71 → 106.24 with blur, vs. → 210.30 unregularized — smaller because the unregularized run's inflated score is itself mostly adversarial, not a loss of real signal). **Adopted `ASCENT_BLUR_EVERY=10`, `ASCENT_BLUR_SIGMA=1.0`** in `linkage_internals.py`. Full `capstone-linkage` test suite still passes.

## Final validation, end of task 8 (before the decoder-capacity escalation)

3 fresh real exports (2,000/class, seeds 0–2), shipped code at that point (`latent_dim=128`, defaults unchanged for floor/multiplier/KL/weight-decay/head-width, `CLUSTER_SAMPLE_SPREAD=0.75`, blurred `activation_maximize()`), via `dataset_forge.smoke.smoke()`:

| Seed | CNN acc | VAE acc | `classify_generated()` overall | Reconstruction overall |
|---|---|---|---|---|
| 0 | 0.991 | 0.991 | 0.46 | 0.60 |
| 1 | 0.991 | 0.991 | 0.46 | 0.66 |
| 2 | 0.989 | 0.994 | 0.59 | 0.66 |

Per-class reconstruction agreement (seed 2, the strongest run): `square` 0.90, `ring` 1.00, `circle` 0.95, `triangle` 0.65, `star` 0.10, `smiley` 0.25, `negative_smiley` 0.80. `star` and `smiley` remain the two weak classes; every other class is now solidly recognizable, a qualitative change from Phase 5's "only circle/ring" floor. **This is where the phase originally stood when the user reviewed it and asked to push further — see below.**

## Decoder capacity escalation (task 10, at explicit user request before archiving)

The user reviewed the diagnostics above and, rather than leave `star`/`smiley` as a follow-up-change open question, asked to push on decoder capacity before archiving — reopening the "does not change decoder architecture" non-goal (design.md) for this same change.

**A methodological trap, caught before drawing conclusions:** the first pass swept a decoder-only intermediate channel width (`mid_channels`, independent of the encoder trunk — a temporary `_WideDecoder` swapped in via monkeypatching `generator_internals._build_decoder`, same mechanism as every other sweep in this file) at a single seed. The "baseline" (`mid_channels=8`, architecturally identical to the shipped decoder) swung from the 0.10–0.25 star-reconstruction numbers recorded above to 0.70 on a different run of the *exact same code*. Cause: decoder variants with different parameter counts consume different amounts of RNG state during weight initialization before `train()`'s batch-shuffling even starts, so "seed 0" doesn't mean "the same random draws" once the architecture itself changes size — a confound scalar hyperparameters (floor, multiplier, `latent_dim`, weight decay, head width) mostly don't have, at least not obviously enough to have shown up as a problem earlier in this file. Fixed by re-running as a proper 3-seed comparison (fresh CNN per seed, `torch.manual_seed(seed)` reset immediately before each variant's construction so every variant at a given seed starts from a matched point) with `n=50` (not 20) evaluation samples to also cut per-class standard error.

**3-seed comparison, mid-channels 8 (shipped) / 64 / 128**, `latent_dim=128`, 2,000/class:

| `mid_channels` | overall gen | overall recon | `star` recon | `star` gen | `smiley` recon | `smiley` gen |
|---|---|---|---|---|---|---|
| 8 | 0.484 | 0.699 | 0.373 | 0.040 | 0.247 | 0.273 |
| 64 | 0.602 | 0.814 | 0.573 | 0.020 | 0.540 | 0.333 |
| 128 | 0.599 | 0.850 | 0.660 | 0.147 | 0.633 | 0.420 |

Consistent, large, 3-seed-robust improvement on both `star` and `smiley` reconstruction, with no regression on the other five classes. Widening further (192, 256) plateaus at essentially the same numbers as 128 — the same diminishing-returns pattern `latent_dim` showed in task 3:

| `mid_channels` | overall gen | overall recon | `star` recon | `star` gen | `smiley` recon | `smiley` gen |
|---|---|---|---|---|---|---|
| 128 | 0.599 | 0.850 | 0.660 | 0.147 | 0.633 | 0.420 |
| 192 | 0.565 | 0.855 | 0.640 | 0.093 | 0.680 | 0.447 |
| 256 | 0.608 | 0.865 | 0.647 | 0.060 | 0.707 | 0.447 |

**Decision: `_DECODER_MID_CHANNELS = 128`** (new, decoder-only constant in `generator_internals.py`; the encoder trunk's own channel counts are untouched). Full test suite re-run: 200/200 passed.

**Final combined validation** (3 fresh real exports, 2,000/class, seeds 0–2, via `dataset_forge.smoke.smoke()`, every Phase 6 change including this one):

| Seed | CNN acc | VAE acc | `classify_generated()` overall | Reconstruction overall |
|---|---|---|---|---|
| 0 | 0.991 | 0.994 | 0.643 | 0.850 |
| 1 | 0.991 | 0.983 | 0.650 | 0.814 |
| 2 | 0.989 | 0.991 | 0.686 | 0.829 |

Per-class (seed 0): `square` 0.85/0.95, `ring` 0.90/1.00, `circle` 0.75/1.00, `triangle` 0.80/1.00, `star` 0.25/0.60, `smiley` 0.20/0.45, `negative_smiley` 0.75/0.95 (`classify_generated()`/reconstruction). Across all 3 seeds, `smiley` reconstruction is now 0.45–0.70 and `classify_generated()` 0.20–0.85 — essentially resolved, no longer a standout weak class. `star` reconstruction is now solid (0.50–0.60, up from ~0.0–0.25 before this escalation) but its `classify_generated()` stays weak (0.00–0.25) even with the wider decoder.

**Narrowed open question: `star`'s `classify_generated()` specifically, not decoder capacity.** Reconstruction (decoding the exact latent mean of a real held-out image) is now solid for `star`; `classify_generated()` (decoding samples drawn from the class's *empirical cluster*, mean plus spread) is not. That gap — a class whose direct reconstruction works but whose cluster-sampled generation doesn't — points at the cluster itself being more diffuse or less coherent than the other six classes', not at the decoder. A plausible mechanism, untested here: `star`'s images are rendered at continuous random rotation (`dataset_forge/shapes.py`), and a regular 5-pointed star has partial rotational near-symmetry, so its latent encoding across many training angles may spread out (or fold onto itself) more than an asymmetric shape's would, making a "typical" sample from its cluster's mean-plus-spread land in a less decodable region even though any single real star's own latent decodes fine. Left as a named open question for a follow-up change (e.g., a smaller or asymmetric sampling scale specifically for `star`, or investigating whether its cluster's spread is measurably larger than other classes') rather than pursued further here, since it is a different kind of question than the decoder-capacity one this escalation resolved, and the user's ask was scoped to decoder capacity.

## Phase 7 inputs from this record

- The concrete accuracy/quality bar for `classify_generated()`/reconstruction agreement is still a framing decision, per this session's scoping — this record gives Phase 7 real numbers to frame against (0.64–0.69 / 0.81–0.85 overall, up from Phase 5's 0.32–0.35 / 0.26–0.31) rather than Phase 5's much lower starting point.
- `star`'s `classify_generated()` (not its reconstruction, which is now solid) is the one remaining named open question, for a follow-up change — Phase 7 can frame the capstone honestly around this ("six of seven classes generate well; one has a specific, understood sampling gap") rather than waiting on it.
- `dataset_forge/configs/default.json` now exports 2,000 images/class; anything in Phase 7's docs or notebooks that assumes 1,000 (none currently do, per this phase's grep) should check against the shipped config, not a remembered number.

## Figures

Not committed. Regenerate with:

```bash
python -m dataset_forge.smoke dataset_forge/output/default-seed0 --figures dataset_forge/output/figures/seed0
```

`generated.png`/`reconstructed.png` show, per class, real images followed by the VAE's generated/reconstructed ones. At the final shipped defaults (including the widened decoder), `square`/`ring`/`circle`/`triangle` reconstructions are sharp and clearly shaped; `negative_smiley` shows both its ring-and-fill and its dots; `smiley` shows a clear ring with visible eye/mouth detail in most samples; `star` reconstructions show a recognizable pointed outline in most samples, though its class-cluster-sampled (`generated.png`) versions are noisier. Generated (class-cluster-sampled) images remain noisier than reconstructions across every class, as expected, but no class is a featureless blob anymore.
