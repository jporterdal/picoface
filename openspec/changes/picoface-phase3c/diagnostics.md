# picoface-phase3c diagnostics record

**These are stub-dataset numbers, not predictions about real content.** They exist so Phase 6 tunes from evidence (design Decision 8). All runs are CPU-only, seeded with `torch.manual_seed`.

## Pre-change baseline (task 1.3)

Phase 3b objective: `MSE (per-pixel mean) + 0.01 * KL (per image) + 1.0 * CE`, head on conv trunk, `LATENT_DIM = 2`. Brightness stub, 8 images per class, 2 classes, seed 0.

| Image | D = H·W·C | Epochs | MSE per pixel (first → last) | KL per image (first → last) | CE (last) |
|---|---|---|---|---|---|
| 16×16×3 | 768 | 10 | 0.0309 → 0.0305 | 0.0011 → 0.0004 | 0.675 |
| 16×16×3 | 768 | 50 | 0.0309 → 0.0299 | 0.0011 → 0.0023 | 0.048 |
| 32×32×3 | 3072 | 10 | 0.0309 → 0.0304 | 0.0017 → 0.0013 | 0.629 |
| 32×32×3 | 3072 | 50 | 0.0309 → 0.0301 | 0.0017 → 0.0019 | 0.003 |

- KL at about 0.001–0.002 nats per image means the latent had **collapsed** under the old objective: the posterior matched the prior and carried almost no information.
- On the per-pixel ELBO scale (design Decision 5), `BETA = 0.01` corresponds to a KL weight of `0.01 · D` = **7.68** at 16×16×3 (30.7 at 32×32×3), against the ELBO weight of 1.

## Post-change record (task 1.3)

The same brightness-stub runs under the implemented phase3c objective: per-pixel ELBO, KL annealed to 1 over the first half of training, learned log-variances at the 10× learning-rate multiplier, `LATENT_DIM = 8`.

| Image | D = H·W·C | Epochs | MSE per pixel (first → last) | KL per image (first → last) | CE (last) | log σr² (last) |
|---|---|---|---|---|---|---|
| 16×16×3 | 768 | 10 | 0.0311 → 0.0307 | 0.010 → 0.030 | 0.699 | −0.09 |
| 16×16×3 | 768 | 50 | 0.0311 → 0.0290 | 0.010 → 0.93 | 0.598 | −0.49 |
| 32×32×3 | 3072 | 10 | 0.0309 → 0.0306 | 0.005 → 0.046 | 0.697 | −0.09 |
| 32×32×3 | 3072 | 50 | 0.0309 → 0.0259 | 0.005 → 43.5 | 0.051 | −0.49 |

- This stub is 16 images at batch size 16, so **one optimizer step per epoch**. Ten steps is too few for any model to learn, and the default-size stub is only good for plumbing tests. Every accuracy test uses 64 images per class (`tests/_splits.py`). A VAE at 8 per class stayed at chance (held-out accuracy 0.50) for every seed at 10 epochs, and for half the seeds at 50. From about 80 steps (64 per class, 10 epochs) it reached 1.00 held-out in 10 of 10 seeds.
- The latent is used rather than collapsed: 43.5 nats per image at 32×32 after 50 steps, against 0.002 under the old objective.

## CPU time budget (task 7.11)

Full combined objective, CPU only:
- The default stub at default `train()` parameters takes **0.02 s**, against the 300 s ceiling in `tests/test_generator.py`.
- The 64-per-class, 3-class shapes stub takes 0.2 s for 10 epochs, 0.7 s for 30, and 1.4–1.8 s for 60 (16×16 and 32×32).

## Resolution invariance (task 1.5)

Shapes stub, 3 classes, 64 per class, torch seeds 0–2, **60 epochs (720 optimizer steps)**. That is long enough for log σr² to reach equilibrium at the 10× multiplier, which 10- and 30-epoch runs do not. Figures cover the same fraction of the image at both resolutions.

| Image | D | Held-out accuracy | MSE per pixel (last) | log σr² (vs log MSE) | log σc² (vs log 2CE) | KL per image (last) |
|---|---|---|---|---|---|---|
| 16×16×3 | 768 | 1.00, 1.00, 1.00 | 0.0136–0.0143 | −4.25, −4.20, −4.23 (−4.25 to −4.30) | −6.00 floor (≈ −11.1) | 37–39 |
| 32×32×3 | 3072 | 1.00, 1.00, 1.00 | 0.0119–0.0128 | −4.29, −4.34, −4.36 (−4.36 to −4.43) | −6.00 floor (≈ −11.7) | 72–74 |

- **Passes.** Quadrupling D moves the learned reconstruction log-variance by about 0.1. Under the rejected summed reduction it would move by log 4 ≈ 1.4 (design Decision 5). The classification weight is identical, both at the floor. So the reconstruction:classification balance does not scale with D.
- KL per image roughly doubles at 4× the pixels. This is the prior's weight falling as 1/D relative to both tasks, the intended ELBO behavior (design Decision 5).

## Phase 6 baseline (tasks 8.1–8.3)

Shapes stub (square, cross, ring), 64 images per class for training (seed 0) and held out (seed 1), 12 optimizer steps per epoch, torch seeds 0–2. Implemented objective with the 10× multiplier and floor on the stored value. Learned log-variances are listed per epoch at epochs 1, 3, 5, 7, 9 (10-epoch runs) or 1, 7, 13, 19, 25 (30-epoch runs), then the final epoch.

| `LATENT_DIM` | Epochs | Held-out accuracy | MSE per pixel (last) | KL per image (last) | CE (last) | log σr² per epoch → last (log MSE) | log σc² per epoch → last | Wall-clock per run |
|---|---|---|---|---|---|---|---|---|
| 4 | 10 | 0.85, 0.97, 0.91 | 0.035–0.041 | 5–11 | 0.53–0.96 | −0.05 −0.29 −0.53 −0.77 −1.01 → −1.13 (−3.27) | 0.05 0.28 0.45 0.57 0.62 → 0.61 | 0.2 s |
| 4 | 30 | 1.00, 1.00, 1.00 | 0.023–0.026 | 26–40 | 0.001–0.002 | −0.05 −0.77 −1.47 −2.15 −2.74 → −3.14 (−3.69) | 0.05 0.57 0.37 −0.58 −1.50 → −2.22 | 0.7 s |
| **8** (current) | 10 | 0.92, 0.96, 0.95 | 0.034–0.036 | 21–26 | 0.43–0.55 | −0.05 −0.29 −0.53 −0.77 −1.01 → −1.13 (−3.36) | 0.05 0.28 0.45 0.57 0.60 → 0.56 | 0.2 s |
| **8** (current) | 30 | 1.00, 1.00, 1.00 | 0.022–0.025 | 35–36 | 0.001 | −0.05 −0.77 −1.48 −2.14 −2.73 → −3.14 (−3.75) | 0.05 0.57 0.22 −0.72 −1.63 → −2.35 | 0.7 s |
| 16 | 10 | 0.93, 0.97, 0.95 | 0.033 | 31–43 | 0.14–0.46 | −0.05 −0.29 −0.53 −0.77 −1.01 → −1.13 (−3.41) | 0.05 0.28 0.45 0.56 0.55 → 0.49 | 0.2 s |
| 16 | 30 | 1.00, 1.00, 1.00 | 0.019–0.025 | 38–44 | 0.001 | −0.05 −0.77 −1.48 −2.15 −2.74 → −3.16 (−3.82) | 0.05 0.56 0.10 −0.81 −1.70 → −2.41 | 0.7 s |

Notes for Phase 6 (task 8.3):
- **The learned weights had not reached equilibrium in any 10- or 30-epoch run.** log σr² falls at a constant rate, the same in every seed, because Adam moves a parameter with a steady gradient sign by about its learning rate per step. At 10 epochs it ends about 2.2 above log MSE, and at 30 epochs about 0.6 above. It reached equilibrium only in the 60-epoch runs above. So the KL:reconstruction balance in these rows is heavier on KL than the calibrated ELBO (design Decision 4), and more so at 10 epochs. On real data, with more steps per epoch, the multiplier needs retuning; that is where this lag should be re-measured.
- **The floor was reached only in the 60-epoch runs**, and only by the classification log-variance. Training cross-entropy fell to about 1e-5, so log 2CE ≈ −11, well below −6. Its weight was then capped at e⁶ ≈ 400, against about 35 for reconstruction.
- **No starvation of reconstruction was observed.** Reconstruction kept improving while classification sat at the floor: MSE 0.022–0.025 at 30 epochs, 0.012–0.014 at 60. On the stub, the lower-loss-earns-higher-weight dynamic (design Decision 4) was bounded by the floor as intended. Whether that holds on real data is Phase 6's question.
- **Latent dimensionality:** a wider latent uses more KL and reconstructs slightly better; accuracy saturates by 30 epochs at every width. `LATENT_DIM = 4` was the least stable at 10 epochs (held-out 0.85–0.97). 8 is kept as the provisional value.

## Learned σ versus closed-form σ

Shapes stub (`kind="shapes"`: square, cross, ring), 64 images per class for training (seed 0), 64 per class held out (seed 1); 12 optimizer steps per epoch; torch seeds 0, 1, 2; `LATENT_DIM = 8`; `LOG_VAR_FLOOR = -6`.

| σ handling | Epochs | Held-out accuracy | MSE per pixel (last) | KL per image (last) | log σr² / log σc² (last) |
|---|---|---|---|---|---|
| `nn.Parameter`, initialized at 0 (as specified) | 10 | 0.92, 0.96, 0.94 | 0.034–0.036 | 26–43 | −0.11 / 0.10 |
| Closed-form σ (per batch, detached) | 10 | 0.93, 0.95, 0.89 | 0.033 | 18–31 | −3.42 / 0.46 |
| ×100 rescaled `nn.Parameter` | 10 | 0.93, 0.95, 0.94 | 0.033 | 14–28 | −3.41 / 0.51 |
| `nn.Parameter`, initialized at 0 (as specified) | 30 | 0.99, 1.00, 1.00 | 0.027–0.030 | 23 | −0.35 / −0.13 |
| Closed-form σ (per batch, detached) | 30 | 1.00, 1.00, 0.99 | 0.025–0.026 | 83–113 | −3.64 / −6.00 (floor) |
| ×100 rescaled `nn.Parameter` | 30 | 1.00, 1.00, 0.99 | 0.026–0.027 | 72–104 | −3.61 / −5.05 |

What this shows:
- **The specified `nn.Parameter` σs barely move.** Adam moves a scalar parameter by roughly the learning rate (1e-3) per step, so reaching log σr² ≈ log(MSE) ≈ −3.5 takes thousands of steps. At stub scale the "learned" weighting stays at its initial value, which under-weights reconstruction.
- **Both variants that reach equilibrium reconstruct better and use the latent more.** KL is 3–4× higher, and held-out accuracy is unchanged.
- **The classification floor binds at 30 epochs** under closed-form σ, because cross-entropy on the training batches approaches 0. This is the starvation dynamic from design Decision 4, and the floor is capping it as intended.

## Learning-rate multiplier on the log-variance parameters

Same setup as above. The Adam optimizer has two parameter groups: every other parameter at `lr = 1e-3`, and the two log-variance parameters at `lr × multiplier`. No weight decay is applied to any group; `torch.optim.Adam`'s default is 0. "Floor on stored value" clamps each log-variance parameter to `LOG_VAR_FLOOR` after every optimizer step. "Floor in loss" clamps only inside the loss, as `_VAE.training_step` does. Equilibrium epoch is the first epoch at which log σr² is within 0.25 of log(MSE).

| Multiplier | Floor | Epochs | Held-out accuracy | MSE per pixel (last) | KL per image (last) | log σr² (vs log MSE) | Equilibrium epoch | log σc² (vs log 2CE) | Lowest raw log σc² |
|---|---|---|---|---|---|---|---|---|---|
| 1× | stored value | 10 | 0.92, 0.96, 0.94 | 0.034–0.036 | 26–43 | −0.11 (−3.35) | not reached | 0.09 (−0.24) | 0.0 |
| 10× | stored value | 10 | 0.92, 0.96, 0.95 | 0.034–0.036 | 21–26 | −1.13 (−3.36) | not reached | 0.56 (0.0) | 0.0 |
| 30× | stored value | 10 | 0.92, 0.96, 0.97 | 0.033–0.034 | 20–24 | −3.00 (−3.40) | not reached | 0.44 (0.15) | 0.0 |
| 100× | stored value | 10 | 0.93, 0.95, 0.94 | 0.033 | 14–28 | −3.41 (−3.42) | 3 | 0.40 (0.37) | −0.13 |
| 1× | stored value | 30 | 0.99, 1.00, 1.00 | 0.027–0.030 | 23 | −0.36 (−3.57) | not reached | −0.14 (−3.9) | −0.15 |
| 10× | stored value | 30 | 1.00, 1.00, 1.00 | 0.022–0.025 | 34–36 | −3.14 (−3.75) | not reached | −2.35 (−6.4) | −2.48 |
| 30× | stored value | 30 | 1.00, 1.00, 1.00 | 0.023–0.025 | 62–90 | −3.72 (−3.73) | 11–12 | −6.00 (floor) | −6.00 |
| 100× | stored value | 30 | 1.00, 1.00, 0.99 | 0.026–0.027 | 72–104 | −3.64 (−3.64) | 3 | −6.00, −6.00, −5.05 | −6.00 |
| 100× | in loss | 30 | 1.00, 1.00, 0.99 | 0.026–0.027 | 72–104 | −3.64 (−3.64) | 3 | −6.00, −6.00, −5.05 | **−7.01**, −6.64, −5.18 |

What this shows:
- **The ×100 multiplier gives exactly the same numbers as the ×100-rescaled parameter above.** Adam normalizes away gradient scale, so rescaling the parameter and raising its learning rate are the same intervention.
- **Time to equilibrium scales as about |log MSE| / (multiplier · lr) steps.** At 12 steps per epoch, 10× does not reach equilibrium in 30 epochs, 30× takes about 140 steps, and 100× about 36 steps. At the default `epochs=10`, only 100× reaches equilibrium, and it does so before KL annealing finishes (epoch 5).
- **With the floor applied only in the loss, the classification log-variance overshoots it to −7.0** under Adam's momentum and then gets no gradient, so it could not recover if cross-entropy rose again. Clamping the stored value holds it at −6.0. Metrics are identical in this run because cross-entropy never rose.

## Freezing the log-variances during KL warmup (task 4.6)

Same setup, with the implemented 10× multiplier and floor on the stored value. "Frozen" stops gradients reaching both log-variances until the KL weight reaches 1, which is the first half of training.

| Log-variances in warmup | Epochs | Held-out accuracy | MSE per pixel (last) | KL per image (last) | log σr² (vs log MSE) | log σc² (last) |
|---|---|---|---|---|---|---|
| Free | 10 | 0.92, 0.96, 0.95 | 0.034–0.036 | 21–26 | −1.13 (−3.36) | 0.56 |
| Frozen | 10 | 0.92, 0.96, 0.95 | 0.034–0.036 | 24–37 | −0.53 (−3.36) | 0.32 |
| Free | 30 | 1.00, 1.00, 1.00 | 0.022–0.025 | 34–36 | −3.14 (−3.75) | −2.35 |
| Frozen | 30 | 1.00, 1.00, 1.00 | 0.026–0.028 | 33–36 | −1.70 (−3.61) | −1.76 |

Decision: **free**. At 10×, the log-variances already lag their equilibrium on the stub (design Decision 4). Freezing them halves the steps they get to catch up, so the model trains with the uncalibrated initial balance for longer. Reconstruction ends worse, and accuracy is unchanged.

Note that 10-epoch stub runs at the 10× multiplier end well short of equilibrium for both log-variances (log σr² −1.13 against a target of −3.36), so their KL/MSE balance is not the calibrated one.
