## Context

See proposal.md (Why) for the question. The state and earlier findings that shape how to answer it:

- **The loss is a proper Gaussian ELBO with a learned noise scale.** `_VAE.training_step` computes MSE / (2σ²) + ½·log σ² + KL / D, where D = 784 pixel values and σ² = exp(`reconstruction_log_var`). Multiplying through by D gives the per-image negative ELBO under a Gaussian decoder with variance σ², plus the classification term. So the trade-off between reconstruction and KL is set entirely by where σ² settles. There is no separate β.
- **σ² settles at about e^−4.45 ≈ 0.0117 (σ ≈ 0.11 in [0, 1] pixel units), and the floor never binds.** Phase 6 measured this (`picoface-phase6/diagnostics.md`, "Key finding: the floor never binds"): the trajectory is identical whether `LOG_VAR_FLOOR` is −6 or effectively unconstrained. So `LOG_VAR_FLOOR` is not a candidate explanation for dimensions staying active. The learned σ² is. σ is also well above the dataset's own pixel noise (6/255 ≈ 0.024), so the model is not trying to encode that noise.
- **In the linear case, σ² is the collapse threshold.** Lucas et al. (2019) show that a linear VAE is equivalent to probabilistic PCA: a latent direction survives only if the data variance it explains exceeds σ². The deep, nonlinear VAE here won't follow this exactly, but it gives a concrete prediction: dimensions whose contribution falls below about 0.0117 per pixel should collapse.
- **`classify_generated()` no longer samples from the prior.** The `capstone-linkage-sampling-fix` change made it resample real encoded points plus small jitter, which resolved the `star` gap. The only path that samples N(0, I) is now `generate()`. So any prior-mismatch effect this investigation finds bears on `generate()`, not on the capstone's agreement numbers.
- **Phase 6's `latent_dim` sweep is the baseline.** Reconstruction and `classify_generated()` agreement roughly doubled from 8 to 64–128 and flattened above 128. That sweep measured downstream agreement only, never the latent itself.
- **The smoke check is where diagnostics like this live.** `dataset_forge/smoke.py` already reaches into `latent_access` (`reconstruct()`), is described as "a measurement, not a test," and emits Markdown tables for diagnostics records.

## Goals / Non-Goals

**Goals:**
- A re-runnable measurement of how many of the VAE's latent dimensions are in use, and how that number depends on the nominal size.
- Evidence that the measurement's "inactive" dimensions are truly unused by the decoder, not just a threshold artifact.
- An interpretation, against outcomes written down before the numbers are in, that tells a follow-up change what (if anything) to do.

**Non-Goals:**
- Changing `LATENT_DIM`, the loss, or any other model constant. Recommendations only.
- Measuring the quality of `generate()`'s unconditional samples. See Open Questions.
- Disentanglement metrics, or attributing active dimensions to specific generating factors.
- Changing the objective (IWAE bounds, free bits, KL warm-up) to raise or lower the active count.

## Decisions

### 1. Metric: Burda et al.'s active units, computed from the encoder mean

A dimension u is active when A_u = Var over x of μ_u(x) exceeds a threshold, where μ is the encoder mean and x ranges over the test split. This is Burda et al.'s definition (Cov over x of E_q[z_u]) exactly, because E_q[z_u] is μ_u. It needs only `encode_mu`, which the `latent_access` capability already exposes.

Computed over the **whole** test split (1,400 images at the default 200 per class), in eval mode with no gradients. It is not computed on the first n per class that the agreement reports use, because a variance across a handful of images per class would mostly measure which classes were drawn.

*Alternatives:* counting dimensions by mean KL alone. Rejected as the headline number because it has no established threshold, but reported alongside (Decision 2). Mutual-information estimates. Rejected as heavier and less standard for this purpose.

### 2. Report the full per-dimension picture, not just a count

For each dimension the report gives:
- Var(μ_u) across the test split;
- the mean posterior variance, the mean over x of exp(logvar_u(x));
- the mean per-dimension KL against N(0, I).

Dimensions are sorted by Var(μ_u). The active count is reported at 0.01 (Burda's convention), and also at 0.001 and 0.1 to show how sensitive it is to the threshold.

The pairing matters because Rolínek et al. (2019) predict a **polarized** latent. Active dimensions have large Var(μ) and small posterior variance. Passive ones have Var(μ) ≈ 0 and posterior variance ≈ 1. A clean split into those two groups makes the count unambiguous. Many dimensions in between is itself a finding (Decision 5, outcome C).

The posterior variance and KL need `logvar`, which `latent_access` does not expose. The report calls `_VAE.encode` directly. That is the same instructor-tooling reach-in `reconstruct()` already makes, and the smoke check only ever builds VAEs through `build_vae()`.

*Alternative:* adding `logvar` to the `_Model` interface. Rejected: it widens an internal contract that `linkage` also depends on, for the benefit of tooling alone.

### 3. Validate "inactive" by masking

Take each test image's μ, set every dimension below the 0.01 threshold to 0 (the prior mean), decode, and measure CNN reconstruction agreement the same way `reconstruction_report` does. Compare with the unmasked agreement.

If "inactive" means "unused," masked and unmasked agreement should match within seed noise. As a sanity check in the other direction, masking the **active** dimensions instead should collapse agreement.

This lives in `smoke.py` next to the report, so anyone re-running the smoke check gets both.

*Alternative:* trusting the threshold. Rejected, since the threshold comes from a different model and dataset (binarized MNIST), and the check costs one extra decode.

### 4. Sweep protocol follows Phase 6's, including its RNG lesson

- Nominal sizes: 8, 16, 32, 64, 128, 192, matching Phase 6's range.
- Default 2,000-per-class export, `train()` defaults, 3 seeds, n = 50 for the agreement numbers.
- The sweep overrides `generator_internals.LATENT_DIM` from a throwaway script that is not committed. That is the same mechanism as every Phase 6 sweep. `_build_vae` reads the constant at call time.
- `torch.manual_seed(seed)` is reset **immediately before each model is constructed**. Phase 6 found that architectures of different sizes consume different amounts of RNG state during initialization, which confounded its single-seed decoder sweep (`picoface-phase6/diagnostics.md`, "A methodological trap").
- One additional run of the shipped 128-dimension model at 3× the default epochs checks whether ten epochs is enough for the count to settle.

### 5. Outcomes to interpret against, written down before measuring

These are fixed now so the numbers get read against them, not the other way round.

- **A. The count levels off.** The active count stays at some k well below the nominal size as the nominal size grows, and masking leaves agreement unchanged. Then 128 is harmless headroom, and the recommendation is a follow-up that evaluates a smaller `LATENT_DIM` (k plus a margin) against Phase 6's metrics for its parameter and time savings. A sub-question to record: if k exceeds 8 but is below 64, why did Phase 6's agreement keep improving up to 64–128? Candidates include easier optimization with more dimensions, or information spread across dimensions below the threshold.
- **B. The count grows with the nominal size.** Most dimensions are active at every nominal size. Then the objective is not pruning at the learned σ², and the recommendation is a follow-up on the loss balance and on prior mismatch in `generate()`, citing Dai & Wipf (2019).
- **C. The latent is not polarized.** There is no clean split, the count swings substantially across thresholds, or masking at 0.01 changes agreement. Then the count is not a reliable summary for this model, and the diagnostics record should say what the spectrum and masking results show instead.

The rough count of ten generating factors (proposal.md) is context for reading k, not a target. The classification head reads z during training and may keep dimensions active beyond what reconstruction alone would need.

## Risks / Trade-offs

- **Ten epochs may not reach equilibrium.** If the count is still falling at epoch 10, it reflects the training budget rather than the model. → Decision 4's 3×-epoch run. If the count differs materially, record both, and treat the default-length number as what students actually get.
- **Threshold arbitrariness.** → The multi-threshold counts and the sorted spectrum (Decision 2), plus the masking check (Decision 3).
- **Seed and initialization confounds across nominal sizes.** → Phase 6's reseed-before-construction protocol and 3 seeds (Decision 4).
- **The 7-class test split's variance is dominated by class identity.** Class-driven dimensions will show large Var(μ) partly because of how many classes the split contains. → Accepted. It is what the model encodes. Per-class breakdowns are out of scope unless outcome C calls for them.
- **The sweep script is not committed**, so only the smoke report is re-runnable. → Same trade-off Phase 6 accepted. The diagnostics record states the exact override and seeds so the sweep can be reproduced.

## Open Questions

- Should the Forge also measure `generate()`'s unconditional samples, for example by the class distribution and CNN confidence of samples from N(0, I)? This would directly test the prior-mismatch effect outcome B would point to. It is deferrable: it doesn't change this change's tasks, and a follow-up under outcome B would own it.
