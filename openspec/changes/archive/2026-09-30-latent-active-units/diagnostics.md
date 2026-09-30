# latent-active-units diagnostics

Measurements of how many of the VAE's latent dimensions the trained model uses, how that depends on the nominal latent size, and which of design.md Decision 5's outcomes they support (tasks 2.1–3.1).

## Setup

- **Machine:** AMD Ryzen 7 7800X3D (8 cores), WSL2 on Linux 6.6.87.2. CPU only. Python 3.13.5, PyTorch 2.13.0+cpu, numpy 2.5.2, Pillow 12.3.0. Runs went one after another on an otherwise idle machine.
- **Code:** commit `2d1d2bf` plus this change's uncommitted tasks 1.1–1.3, for every run.
- **Exports:** `dataset_forge/output/default-seed{0,1,2}`, the flipped-polarity exports `mediacomp-bridge` made (`openspec/changes/archive/2026-09-26-mediacomp-bridge/diagnostics.md`, Setup). All use `configs/default.json`: 7 classes, 28×28 grayscale, 2,000 train and 200 test images per class. The training seed equals the export seed.
- **Models:** `build_classifier()` and `build_vae()`, trained with `train()`'s defaults (10 epochs, batch size 16, learning rate 1e-3) unless a section says otherwise.
- **Measurements:**
  - The active-units report runs over the whole test split (1,400 images). A dimension u is active when Var(μ_u) across those images exceeds the threshold (Burda et al., 2016).
  - Agreement numbers are CNN reconstruction agreement over the first 50 test images per class (`--n 50`), overall across the 7 classes.
  - "Inactive masked" decodes μ with the dimensions at or below Var(μ) = 0.01 set to 0. "Active masked" decodes μ with the dimensions above it set to 0 instead.

## Baselines

- `mediacomp-bridge` recorded overall reconstruction agreement of 0.82, 0.84 and 0.84 on these exports and seeds, at n = 20.
- Phase 6's `latent_dim` sweep (`openspec/changes/archive/2026-09-23-picoface-phase6/diagnostics.md`) found reconstruction and `classify_generated()` agreement roughly doubling from 8 to 64–128, then flattening. That was before Phase 6 widened the decoder, and on the old-polarity data.
- **A correction to design.md's context.** Design.md cites Phase 6's learned noise scale, log σ² ≈ −4.45 (σ² ≈ 0.0117). That value was measured before Phase 6 widened the decoder. With today's decoder, the learned log σ² at 128 dimensions is −5.4 to −5.9 (σ² ≈ 0.003–0.0045; see the sweep tables). The linear-VAE collapse threshold is therefore lower than design.md assumed, which makes collapse less likely, not more.

## Results

### The shipped model (task 2.1)

Command: `python -m dataset_forge.smoke dataset_forge/output/default-seed<N> --seed <N> --n 50`.

| Seed | CNN accuracy | VAE accuracy | Active > 0.001 | Active > 0.01 | Active > 0.1 | Unmasked | Inactive masked | Active masked |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.986 | 0.970 | 128 | 128 | 128 | 0.83 | 0.83 | 0.14 |
| 1 | 0.997 | 0.984 | 128 | 128 | 128 | 0.87 | 0.87 | 0.14 |
| 2 | 0.999 | 0.979 | 128 | 128 | 128 | 0.85 | 0.85 | 0.14 |

Unmasked agreement (0.83–0.87) matches `mediacomp-bridge`'s 0.82–0.84 within seed and sample-size noise.

Per-dimension summary, from the smoke report's sorted table:

| Seed | Var(μ): min / median / max | Posterior variance: min / median / max | KL (nats): min / median / max | Total KL (nats) |
|---|---|---|---|---|
| 0 | 0.43 / 1.62 / 4.64 | 0.003 / 0.028 / 0.39 | 0.92 / 2.46 / 5.26 | 329 |
| 1 | 0.43 / 1.37 / 3.25 | 0.003 / 0.025 / 0.56 | 0.49 / 2.19 / 4.26 | 284 |
| 2 | 0.38 / 1.49 / 4.11 | 0.004 / 0.034 / 0.55 | 0.72 / 2.17 / 5.68 | 283 |

Every dimension is active, at every threshold, on every seed. The smallest Var(μ) (0.38) is nearly four times the highest threshold. No dimension looks passive: a passive dimension would have Var(μ) ≈ 0 and posterior variance ≈ 1, and the largest posterior variance seen is 0.56. So the inactive-masked column is trivially the unmasked one, because there is nothing to mask. The active-masked column is 1/7, chance for 7 classes.

### Nominal-size sweep (task 2.2)

**Reproducing it.** A throwaway script, not committed, did the following for each seed s in {0, 1, 2}:

1. Loaded `dataset_forge/output/default-seed<s>`.
2. Ran `torch.manual_seed(s)`, then built and trained a CNN with `train()` defaults.
3. For each d in 8, 16, 32, 64, 128, 192: set `picoface._internals.generator_internals.LATENT_DIM = d`, ran `torch.manual_seed(s)` immediately before `build_vae(train_data)`, then trained with `train()` defaults.
4. Ran `active_units_report` and the masked `reconstruction_report`s from `dataset_forge/smoke.py` (n = 50), reusing that seed's CNN.

`_build_vae` reads `LATENT_DIM` at call time, so this is equivalent to editing the constant.

The d = 128 rows differ from the smoke-check runs above (0.91 against 0.83 on seed 0, for example). That's because `smoke()` seeds once before building the CNN, so the VAE starts from a different RNG state. Both are the same configuration.

**Seed 0**

| Nominal size | Active > 0.001 | Active > 0.01 | Active > 0.1 | Smallest Var(μ) | Median posterior variance | Total KL (nats) | Unmasked | Inactive masked | Active masked | Learned log σ² | VAE accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 8 | 8 | 8 | 4.37 | 0.0012 | 71 | 0.45 | 0.45 | 0.14 | −4.75 | 0.972 |
| 16 | 16 | 16 | 16 | 2.94 | 0.0021 | 100 | 0.59 | 0.59 | 0.14 | −5.03 | 0.979 |
| 32 | 32 | 32 | 32 | 1.31 | 0.0080 | 136 | 0.73 | 0.73 | 0.14 | −5.14 | 0.986 |
| 64 | 64 | 64 | 64 | 1.02 | 0.0133 | 222 | 0.91 | 0.91 | 0.14 | −5.57 | 0.991 |
| 128 | 128 | 128 | 128 | 0.34 | 0.0485 | 265 | 0.91 | 0.91 | 0.14 | −5.42 | 0.977 |
| 192 | 192 | 192 | 192 | 0.28 | 0.0517 | 367 | 0.91 | 0.91 | 0.14 | −5.65 | 0.994 |

**Seed 1**

| Nominal size | Active > 0.001 | Active > 0.01 | Active > 0.1 | Smallest Var(μ) | Median posterior variance | Total KL (nats) | Unmasked | Inactive masked | Active masked | Learned log σ² | VAE accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 8 | 8 | 8 | 3.61 | 0.0029 | 59 | 0.37 | 0.37 | 0.14 | −4.72 | 0.966 |
| 16 | 16 | 16 | 16 | 1.89 | 0.0025 | 94 | 0.52 | 0.52 | 0.14 | −5.01 | 0.987 |
| 32 | 32 | 32 | 32 | 1.18 | 0.0035 | 138 | 0.79 | 0.79 | 0.14 | −5.41 | 0.991 |
| 64 | 64 | 64 | 64 | 1.00 | 0.0177 | 187 | 0.88 | 0.88 | 0.14 | −5.66 | 0.982 |
| 128 | 128 | 128 | 128 | 0.35 | 0.0342 | 264 | 0.89 | 0.89 | 0.14 | −5.63 | 0.991 |
| 192 | 192 | 192 | 192 | 0.37 | 0.0618 | 358 | 0.94 | 0.94 | 0.14 | −5.66 | 0.973 |

**Seed 2**

| Nominal size | Active > 0.001 | Active > 0.01 | Active > 0.1 | Smallest Var(μ) | Median posterior variance | Total KL (nats) | Unmasked | Inactive masked | Active masked | Learned log σ² | VAE accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 8 | 8 | 8 | 5.35 | 0.0019 | 66 | 0.38 | 0.38 | 0.14 | −4.68 | 0.985 |
| 16 | 16 | 16 | 16 | 2.55 | 0.0062 | 97 | 0.58 | 0.58 | 0.14 | −4.95 | 0.966 |
| 32 | 32 | 32 | 32 | 1.79 | 0.0107 | 142 | 0.69 | 0.69 | 0.14 | −5.39 | 0.989 |
| 64 | 64 | 64 | 64 | 0.75 | 0.0152 | 197 | 0.85 | 0.85 | 0.14 | −5.47 | 0.991 |
| 128 | 128 | 128 | 128 | 0.54 | 0.0254 | 295 | 0.91 | 0.91 | 0.14 | −5.88 | 0.980 |
| 192 | 192 | 192 | 192 | 0.37 | 0.0503 | 352 | 0.86 | 0.86 | 0.14 | −5.74 | 0.991 |

The active count equals the nominal size in all 18 runs, at all three thresholds. The smallest Var(μ) falls as the nominal size grows (from about 4 at 8 dimensions to about 0.3 at 192), but it stays above the 0.1 threshold everywhere. Total KL grows steadily with the nominal size, so each added dimension carries information, about 1 to 1.5 nats per dimension at the top of the range. Reconstruction agreement reproduces Phase 6's shape: it rises steeply to 64 and is flat from 64 to 192 within seed noise.

### Supplementary: where μ's variance lies

The active-unit count looks at one axis at a time. It can't tell whether the 128 axes carry 128 independent directions or a smaller subspace spread across all of them. The same throwaway script also:

- ran PCA on μ across the test split;
- decoded each image's μ after projecting it onto the top k principal directions, around the mean μ;
- measured reconstruction agreement on those decodes the same way (n = 50).

| Seed | Nominal size | PCs for 90% of Var(μ) | 95% | 99% | Participation ratio | Top 4 | Top 8 | Top 16 | Top 32 | Top 64 | Unmasked |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 8 | 5 | 6 | 8 | 3.0 | 0.53 | — | — | — | — | 0.45 |
| 0 | 16 | 6 | 7 | 12 | 3.3 | 0.44 | 0.44 | — | — | — | 0.59 |
| 0 | 32 | 8 | 12 | 22 | 3.8 | 0.39 | 0.46 | 0.56 | — | — | 0.73 |
| 0 | 64 | 11 | 17 | 32 | 4.8 | 0.33 | 0.40 | 0.49 | 0.84 | — | 0.91 |
| 0 | 128 | 10 | 17 | 35 | 4.3 | 0.35 | 0.38 | 0.48 | 0.83 | 0.89 | 0.91 |
| 0 | 192 | 12 | 21 | 39 | 5.1 | 0.29 | 0.35 | 0.44 | 0.82 | 0.91 | 0.91 |
| 1 | 8 | 5 | 6 | 8 | 3.4 | 0.32 | — | — | — | — | 0.37 |
| 1 | 16 | 6 | 8 | 13 | 3.1 | 0.27 | 0.38 | — | — | — | 0.52 |
| 1 | 32 | 8 | 13 | 24 | 3.7 | 0.25 | 0.30 | 0.51 | — | — | 0.79 |
| 1 | 64 | 12 | 19 | 36 | 5.0 | 0.23 | 0.26 | 0.43 | 0.76 | — | 0.88 |
| 1 | 128 | 12 | 21 | 40 | 4.4 | 0.22 | 0.31 | 0.46 | 0.76 | 0.88 | 0.89 |
| 1 | 192 | 14 | 24 | 42 | 5.0 | 0.23 | 0.31 | 0.44 | 0.81 | 0.94 | 0.94 |
| 2 | 8 | 5 | 6 | 8 | 3.2 | 0.43 | — | — | — | — | 0.38 |
| 2 | 16 | 8 | 10 | 14 | 4.3 | 0.41 | 0.43 | — | — | — | 0.58 |
| 2 | 32 | 10 | 14 | 25 | 4.5 | 0.31 | 0.33 | 0.44 | — | — | 0.69 |
| 2 | 64 | 13 | 21 | 38 | 5.8 | 0.24 | 0.34 | 0.39 | 0.70 | — | 0.85 |
| 2 | 128 | 15 | 26 | 45 | 5.4 | 0.25 | 0.25 | 0.38 | 0.69 | 0.90 | 0.91 |
| 2 | 192 | 17 | 27 | 46 | 6.1 | 0.25 | 0.30 | 0.39 | 0.66 | 0.85 | 0.86 |

(Participation ratio: (Σλ)² / Σλ² over μ's covariance eigenvalues λ, a soft count of how many directions dominate.)

μ's variance is concentrated even though every axis is active. At 128 dimensions, 10–15 directions hold 90% of it and 35–45 hold 99%. Those counts barely grow from 64 to 192. What the decoder needs is less concentrated than the variance, though. The top 16 directions give only 0.38–0.48 agreement, the top 32 give 0.69–0.83, and it takes the top 64 to match the unmasked agreement. The low-variance directions carry detail the CNN needs.

### Supplementary: the linear-VAE prediction

Design.md took from Lucas et al. (2019) the prediction that a latent direction survives only if the data variance it explains exceeds σ². To test it, the script counted the principal directions of each training split's pixel data (784 values in [0, 1]) whose variance exceeds the learned σ² of each sweep run:

| Nominal size | Seed 0: log σ² | Directions above σ² | Seed 1: log σ² | Directions above σ² | Seed 2: log σ² | Directions above σ² |
|---|---|---|---|---|---|---|
| 8 | −4.75 | 147 | −4.72 | 144 | −4.68 | 142 |
| 16 | −5.03 | 168 | −5.01 | 166 | −4.95 | 163 |
| 32 | −5.14 | 179 | −5.41 | 205 | −5.39 | 203 |
| 64 | −5.57 | 221 | −5.66 | 234 | −5.47 | 212 |
| 128 | −5.42 | 205 | −5.63 | 230 | −5.88 | 265 |
| 192 | −5.65 | 232 | −5.66 | 235 | −5.74 | 246 |

At every nominal size up to 128, more data directions clear σ² than the latent has dimensions. The linear theory therefore predicts every dimension active, which is what happened. Even with 192 dimensions, 232–265 directions clear σ². σ² falls as the nominal size grows, because a bigger latent reconstructs better. That lowers the bar, so the linear prediction never catches up with the nominal size in this range.

### Training length (task 2.3)

Seed 0, 128 dimensions, the sweep's protocol. The only difference is `train(vae, train_data, epochs=30)`:

| Epochs | Active > 0.001 | Active > 0.01 | Active > 0.1 | Smallest Var(μ) | Median Var(μ) | Median posterior variance | Total KL (nats) | Mean ‖μ‖ | Unmasked | Learned log σ² | VAE accuracy | Train time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 (default) | 128 | 128 | 128 | 0.34 | 1.41 | 0.0485 | 265 | 14.1 | 0.91 | −5.42 | 0.977 | 29 |
| 30 (3×) | 128 | 128 | 128 | 0.50 | 2.20 | 0.0217 | 387 | 17.7 | 0.94 | −5.80 | 0.998 | 143 |

The count does not differ: 128 at both lengths and every threshold. Longer training moves away from collapse, not towards it. The smallest Var(μ) rises, posterior variances shrink, and total KL grows by 46%. Ten epochs is enough for the count to settle. The count at the default length is what students get, and it is the same as the count at 3×.

## Interpretation

- **No dimension collapses, at any nominal size from 8 to 192.** The "roughly ten generating factors" in proposal.md don't show up as a count of about ten. The count is flat against the nominal size at every threshold. It is not polarized into active and passive groups in Rolínek et al.'s sense, but that is because there is no passive group, not because of a smear of in-between dimensions.
- **This is what the linear theory predicts for this model.** The collapse threshold is the learned σ². With today's decoder that is about 0.003–0.0045, well below design.md's assumed 0.0117. Hundreds of pixel-space directions clear it. Dai & Wipf's (2019) prediction of exactly r active dimensions for data on an r-dimensional manifold applies as σ² → 0 with a decoder expressive enough to reach it. Here σ² settles well above zero with a small decoder, so that regime isn't the relevant one.
- **The latent is concentrated, and it is wider than the prior.** μ's variance lies mostly in 10–45 directions, spread across all 128 axes. The median per-axis Var(μ) is 1.2–1.4 at 128 dimensions (and higher at smaller sizes), while posterior variances are about 0.02–0.05. Mean ‖μ‖ is 12.8–14.1, against √128 ≈ 11.3 for draws from N(0, I). So the region real images encode to is a thin, anisotropic set that is wider than N(0, I) along its main directions and narrower along the rest. N(0, I) spreads its mass evenly over all 128. `generate()` samples N(0, I), so it spends most of its samples outside that region. `classify_generated()` is not affected, because it resamples real encoded points (design.md, Context).
- **What the masking check shows.** There are no inactive dimensions to mask, so it can't separate "inactive" from "unused". It does confirm that the decoder depends on the active set as a whole: removing it gives chance agreement. The PCA projection is the more useful check here. About 64 directions are needed to keep agreement at 128 or 192 dimensions, which is consistent with Phase 6's finding that 64 and 128 are close.

## Decision (task 3.1)

**Outcome: B, the count grows with the nominal size.** The active count equals the nominal size at every size (8–192), every threshold (0.001, 0.01, 0.1), every seed (0–2), and both training lengths (sweep tables; training-length table). Outcome A is ruled out: the count never levels off below the nominal size. Outcome C is ruled out as well. The count doesn't swing across thresholds (it is the same at all three), masking at 0.01 doesn't change agreement (nothing falls below it), and there is no band of in-between dimensions (the smallest Var(μ) is 0.28 at 192 dimensions and 0.34 or more at 128).

B's premise holds with one refinement. The objective is not pruning at the learned σ², and the linear-VAE analysis shows why: σ² sits far below the variance of hundreds of data directions. But the latent is not using its dimensions evenly. μ's variance is concentrated in a few tens of directions (supplementary tables).

**Recommendation: a follow-up change on prior mismatch in `generate()`.** Its first task should be the measurement design.md deferred: sample N(0, I) through `generate()` and record the CNN's class distribution and confidence on those samples, against the same numbers for decoded real latents. If the samples are poor, the follow-up can then evaluate a fix. Dai & Wipf (2019) propose a second-stage VAE fitted to the encoded latents, and a simpler alternative is sampling from a Gaussian fitted to the aggregate posterior. Changing the loss weighting itself is not recommended on this evidence. The weighting is already a proper ELBO with a learned σ², and the behaviour matches what that objective predicts.

**`LATENT_DIM` stays at 128.** This change's evidence doesn't argue for cutting it. Every dimension is active, total KL keeps rising with the nominal size, and the PCA projection needs about 64 directions to hold agreement. 64 and 128 are within seed noise of each other on reconstruction agreement (0.85–0.91 against 0.89–0.91 in the sweep). Phase 6 found 128 modestly ahead on `classify_generated()`. Revisiting that trade for parameter savings would be a separate question, not one this measurement answers.
