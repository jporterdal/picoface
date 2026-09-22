## Context

Phase 3 built the generator arm and archived it. Its VAE was unsupervised: `_train_loop` received `data.images` only and optimized `mse_loss(recon, x) + BETA * kl`, with labels used only as scatter colors in `show_latent_space()`.

Phase 3b (archived 2026-09-21) then did two things. It refactored every model onto an abstract `_Model` (`_internals/model_api.py`) with per-model `training_step`, declared capabilities, and one shared `train`/`evaluate`/`predict`/`generate`. And it added a classification head on the shared conv-trunk features of both the autoencoder and the VAE, trained with a fixed `CLASSIFICATION_WEIGHT = 1.0`. This design was drafted against Phase 3's code but supersedes 3b: **3b's refactor is kept as the substrate; 3b's model decisions (head placement, fixed weight, autoencoder classifies) are overwritten** by Decisions 1, 4, and 7. Where this document describes "today's" loss or training loop, the arithmetic still holds for 3b's `_VAE.training_step` — 3b did not change the reduction or `BETA`.

A `/opsx:explore` session established that this does not match the project's intent. The intended deliverable is one supervised model that a student trains once and then uses as a classifier and as a generator. Phase 2's standalone CNN and Phase 3's plain autoencoder both remain in the codebase but neither is the student's path any longer.

That reframe invalidates several Phase 3 and Phase 3b decisions and their spec requirements, and it pulls forward one decision Phase 3 explicitly deferred (KL annealing). The "arms share no model code" constraint Phases 2 and 3 defended was already retired by 3b's refactor; this change relies on that.

This design covers only the model and its training; it is still proven against the stub dataset, and every numeric value remains deferred to Phase 6 against real data.

## Goals / Non-Goals

**Goals:**
- Make the VAE supervised: labeled data in, a model that classifies and generates out, from one `train()` call.
- Remove the hand-tuned loss weight, replacing it with a principled mechanism that does not require a student or an instructor to pick a number per dataset.
- Prevent posterior collapse by construction (annealing) rather than by a fortunate constant.
- Free the latent dimensionality from a constraint that only ever served a visualization.
- Leave Phase 6 an evidence base — recorded per-epoch diagnostics and held-out accuracy — rather than a guess.
- Keep every change invisible to students: no new parameters, no changed call shapes, no new vocabulary.

**Non-Goals:**
- Picking final values for latent dimensionality, annealing schedule, uncertainty clamp, or an accuracy bar — Phase 6 owns all of these (Decision 8).
- Conditional generation (`generate(..., class_name=...)`). The supervised latent makes it cheap to add later by sampling a class's empirical cluster, but it is a student-visible API change and is not proposed here.
- Any change to the plain autoencoder's behavior (Decision 7).
- Any change to the CNN classifier's behavior (Decision 7).
- Real shape taxonomy, resolution, or color depth — still Phase 5.
- Renaming `build_vae()` to something architecture-neutral. Worth doing before students see it; deferred to Phase 7 with the rest of the naming pass.

## Decisions

### 1. The classifier head reads the reparameterized sample `z`, not `mu`, and not the conv trunk

The encoder trunk produces `mu` and `logvar`. `z = mu + eps * exp(0.5 * logvar)` is drawn once per forward pass and fed to **both** the decoder and a new classification head. The head is a small linear stack from `latent_dim` to `num_classes`.

Three variants were on the table and the choice is load-bearing:

- **Head on the conv trunk (bypassing the bottleneck)** — rejected. It is the higher-accuracy option and it defeats the purpose: the latent receives no supervisory pressure at all, so the class structure that makes generated samples class-coherent never forms. The model would classify well and generate exactly as it does today.
- **Head on `mu` (deterministic)** — rejected. Cleaner gradients and better accuracy, but it only separates the *means*. Nothing penalizes the per-class variances growing until the sampled distributions overlap heavily, so `generate()` — which samples, and does not read `mu` — would draw from regions the classifier never learned to separate. The supervision would not reach the thing being sampled.
- **Head on `[mu, logvar]` concatenated** — rejected. Classification gradients would flow into `logvar`, which is already contested by the KL term and the annealing schedule. Adding a third force on that parameter makes the training dynamics much harder to reason about for no clear gain.

Sampling `z` means the classification loss is computed on a stochastic input, which acts as latent-space noise augmentation and forces the *distribution*, not just its center, to be class-identifiable. That is precisely the property `generate()` needs. The cost is noisier gradients, accepted.

At inference time (`evaluate()`, `predict()`) the head reads `mu` rather than a sample, so predictions are deterministic and reproducible — the standard convention, and the same reasoning Phase 3 Decision 8 used for plotting `mu`.

### 2. `latent_dim` is unpinned and `show_latent_space()` is removed

Phase 3 Decision 2 fixed `latent_dim = 2` by this chain: avoid a PCA/t-SNE dependency, therefore `show_latent_space()` must be a direct (x, y) scatter, therefore the latent must be 2-dimensional. The visualization was the only justification.

Decision 1 routes classification through that bottleneck, which promotes `latent_dim` from a plotting convenience to the single parameter capping both classification accuracy and sample quality — two numbers would have to linearly separate every shape class *and* carry enough information to reconstruct an image. Holding it at 2 to preserve an optional plot is the wrong trade.

So the plot goes and the constraint goes with it. `latent_dim` remains a non-public module constant with no student-facing parameter; the spec no longer mandates a value. The dependency that 2 existed to avoid is still not added — removing the plot removes the need for it entirely.

**Alternatives considered:**
- **Keep `show_latent_space()`, plotting the first two dimensions of a wider latent** — rejected: a 2-of-N projection is misleading in a way a true 2-D latent is not, and the plot is not wanted enough to justify explaining that caveat.
- **Keep `show_latent_space()` and accept `latent_dim = 2`** — rejected by the same reasoning that motivates this change; the plot was explicitly ranked as nice-to-have.

Whether some latent visualization returns later is a Phase 7 question, not a blocker.

### 3. KL annealing governs the KL term; uncertainty weighting is deliberately not applied to it

The KL weight ramps from zero to **1** over an internal schedule, where 1 is the weight at which the reconstruction and KL terms together form the ELBO (Decision 5 puts both on a per-pixel scale so that "1" has this meaning at every resolution). Warming up without the prior lets reconstruction and classification establish a useful latent before the prior starts pulling it toward `N(0, I)`; this is the standard mitigation for posterior collapse and is exactly what Phase 3's design deferred ("a real technique for exactly this failure mode ... worth trying during Phase 6"). Only the schedule's *length and shape* are tuning questions; its end point is not.

Uncertainty weighting is **not** extended to the KL term, though it is superficially tempting to weight all three the same way. Kendall et al.'s derivation applies to task losses with a likelihood interpretation and an observation-noise parameter to learn. The KL divergence is a regularizer against the prior; it has no such parameter. Giving it a learned weight also creates an obvious degenerate pull — the model can make the regularizer cheap by inflating its uncertainty — which is the failure annealing exists to prevent, reintroduced through the mechanism meant to remove hand-tuning.

The two mechanisms have distinct jobs — **annealing schedules the regularizer, uncertainty weighting balances the tasks** — but they are not independent, and this should not be mistaken for a bug. The learned reconstruction noise scale `sigma_r` (Decision 4) is part of the decoder's likelihood, so it sets how strongly reconstruction pulls against the KL term: as reconstruction improves, `sigma_r` shrinks and the likelihood term strengthens relative to the prior. That is the calibrated-decoder behavior Decision 5 adopts deliberately (Rybkin, Daniilidis & Levine, 2021), and it is what lets the KL weight end at the principled value 1 instead of a tuned constant.

Whether the learned uncertainty parameters are frozen during the annealing warmup or allowed to re-equilibrate as the KL weight rises is left to implementation-time experiment and recorded in the diagnostics (Decision 8). Freezing is the more conservative default.

### 4. Uncertainty weighting (Kendall et al., 2018) balances reconstruction against classification

Each task loss carries a learned log-variance, in the form Kendall et al. derive for its likelihood:

- **Reconstruction** (Gaussian likelihood, one noise scale shared by all pixels): `MSE / (2 * sigma_r^2) + (1/2) * log(sigma_r^2)`, where `MSE` is the per-pixel mean squared error. This is exactly the per-pixel Gaussian negative log-likelihood (Decision 5), so `sigma_r` is the decoder's per-pixel noise scale.
- **Classification** (softmax likelihood): `CE / sigma_c^2 + (1/2) * log(sigma_c^2)` — i.e. `CE / sigma_c^2 + log(sigma_c)`. Note the classification form has no factor of 2 in the denominator; the reconstruction form does.

`log(sigma_r^2)` and `log(sigma_c^2)` are learned parameters of the model. This removes the need for any hand-picked reconstruction-vs-classification weight, which matters more here than in the paper's setting: the balance depends on the dataset, and neither an instructor nor a student will tune it per course.

**What the learned weights actually do.** Minimizing each term over its `sigma` gives `sigma_r^2 = MSE` and `sigma_c^2 = 2 * CE`, and substituting back leaves each task contributing `(1/2) * log(L_i)` plus a constant. Two consequences follow, and both shape the rest of this design:

- **Constant multipliers on a task loss are absorbed.** Scaling `L_i` by any constant `c` just shifts `log(sigma_i^2)` by `log(c)` and leaves the gradient, `(1/2) * grad(L_i) / L_i`, unchanged. So what sets the balance between tasks is not how each loss is reduced or scaled but the **coefficient on each `log(sigma^2)` term** — here `1/2` for both, which makes reconstruction and classification pull with equal strength at every resolution. Equal strength is a *choice* — in effect, the label counts as much as the whole image — not something the likelihood derives; it is recorded as such, and it is resolution-invariant, which the likelihood-derived alternative is not (Decision 5).
- **Lower loss earns higher weight.** The effective gradient weight on each task is proportional to `1 / L_i`.

**A known failure mode is accepted and documented rather than solved now.** Because lower loss earns higher weight, on easily-classified data cross-entropy falls fast while reconstruction MSE plateaus, so the weighting can pour capacity into the already-solved task and starve generation. This runs against the roadmap's stated goal of a generative arm that is "reliable to train, not merely impressive when it happens to converge."

The mitigation lever is a floor on each `log(sigma^2)`, which caps how large a task's weight can grow; it reintroduces exactly one hand-tuned constant — a much weaker one than the per-dataset weight it replaces, since it only bounds a pathology rather than setting the balance. Because both `sigma`s live on per-pixel / per-label scales (Decision 5), the floor means the same thing at every image resolution. Its value is a Phase 6 tuning question; the diagnostics in Decision 8 are what make it visible. Recording the learned weights per epoch is not optional polish here, it is how this risk gets caught before a student's laptop finds it.

The analysis above is at equilibrium. The `sigma`s track their losses quickly under Adam but not instantly; the recorded per-epoch weights are how the real trajectory is checked against it.

**Alternatives considered:**
- **A fixed hand-tuned classification weight alongside the existing `BETA`** — rejected: two hand-tuned constants, both invalid the moment Phase 5 changes the data.
- **GradNorm or a similar gradient-magnitude balancer** — rejected: more machinery and another schedule to tune, for a two-task problem where the simpler method is well-understood and cheap.

### 5. Reconstruction and KL form a per-pixel ELBO

Today reconstruction uses `mse_loss` with mean reduction over `N*C*H*W` (per-pixel) while KL sums over latent dimensions and means over the batch (per-image). With `D = H*W*C`, the objective `MSE + BETA * KL` is `(1/D)` times `sum_SE + (BETA * D) * KL`, so relative to the ELBO the effective KL weight is `BETA * D`: `BETA = 0.01` at 16x16x3 is a KL weight of about **7.68** — strong regularization, not the light touch the current code comment reasons its way to, and an explanation for why larger trial values collapsed the KL so hard. Worse, it is **resolution-dependent**: Phase 5 choosing a different image size would silently change it.

The fix keeps the per-pixel mean reduction and makes the rest of the objective consistent with it:

- **Reconstruction** is the per-pixel Gaussian negative log-likelihood with the learned noise scale `sigma_r` (Decision 4): `MSE / (2 * sigma_r^2) + (1/2) * log(sigma_r^2)`.
- **KL** is the existing per-image KL (sum over latent dimensions, mean over the batch) **divided by `D`**.
- **KL weight** is annealed from 0 to 1 (Decision 3).

Together these are exactly the negative ELBO divided by `D` (the "per-dimension" convention), up to the Gaussian likelihood's additive constant, so the reconstruction:KL balance is the one the likelihood prescribes at every resolution, with no tuned `BETA`. In the old units, the ELBO-consistent weight is `1/D` (about 0.0013 at 16x16x3); the old `BETA = 0.01` was 7.68 times that.

At equilibrium (Decision 4) the three terms contribute:

| Term | Contribution | Depends on `D`? |
|---|---|---|
| Reconstruction | `(1/2) * log(MSE)` | No |
| Classification | `(1/2) * log(CE)` | No |
| KL | `kl_weight * KL / D` | Yes — as the ELBO prescribes |

**The one remaining resolution dependence is deliberate.** Three terms that scale naturally with `D` (reconstruction), `latent_dim` (KL), and 1 (classification) cannot all be held in fixed ratio. This design holds the two *task* terms fixed relative to each other and lets the prior's weight fall as `1/D` against both. That is the ELBO's own behavior — more pixels are more evidence against the prior — so it is stated in the spec as intended, not hidden.

This lands *first*, before annealing or uncertainty weighting is layered on, so that the learned `sigma`s are introduced on the per-pixel scale they are specified on.

**Alternatives considered:**
- **Summed (per-image) reconstruction likelihood** — `sum_SE / (2 * sigma_r^2) + (D/2) * log(sigma_r^2)` with the per-image KL at weight 1, the conventional full ELBO. Rejected: it is the same reconstruction:KL balance scaled by `D`, but because learned weights absorb constant multipliers (Decision 4) the `D/2` coefficient on `log(sigma_r^2)` cannot be scaled away — reconstruction then outweighs classification by a factor of `D` (768 at 16x16x3). The classification `sigma_c` would have to shrink by that factor to compensate, which the `log(sigma^2)` floor (Decision 4) would prevent, so classification could be permanently outweighed. The per-pixel form is the same ELBO without that side effect.
- **Keep mean-reduced reconstruction with a hand-tuned `BETA`** — rejected: it has the same form as the chosen design but with the KL weight tuned rather than derived, and a fixed `BETA` does not track the `1/D` the likelihood prescribes when resolution changes.
- **A scalar learned `sigma` over a summed reconstruction without the `D/2` coefficient** — rejected: the learned `sigma^2` then converges to `D * MSE`, carrying a `log(D)` offset that makes the `log(sigma^2)` floor mean something different at every resolution — the "`sigma` quietly absorbs a resolution artifact" failure this decision exists to avoid.

### 6. The VAE plugs into Phase 3b's model-agnostic interface; no new verbs or training loop

Phase 3b's refactor is kept unchanged in shape. Every behavior in this design is expressed through it:

- **Loss** lives in `_VAE.training_step` (normalized reconstruction + annealed KL + uncertainty-weighted classification) and `_Autoencoder.training_step` (reconstruction only). The shared `train()` loop does not branch on model type.
- **Validation** is the inherited `_Model.check_data`: image shape always, class count whenever `model.num_classes` is set. The VAE sets it; the autoencoder no longer does, so it accepts any labeled `Dataset` and ignores the labels. The asymmetry — labels ignored for one model type, required for another, from the same public call — is deliberate and documented rather than smoothed over, because the autoencoder is unsupervised (Decision 7).
- **Inference** is the VAE's `classify()` capability, which reads `mu` through the latent head (Decision 1). The shared `evaluate`/`predict` in `model_api` need no change; `picoface.generator` re-exports them alongside `train`/`generate`, so the student imports everything for their model from one module. The autoencoder drops the `classify` capability, so those verbs raise `GeneratorError` for it (a `CapabilityError` subclass, so 3b's single-`except` guarantee holds).
- **Scheduling** is the one extension the interface needs. Annealing (Decision 3) is a function of training progress, which `training_step` currently cannot see, and progress must be relative to the caller's `epochs` argument since students choose it. `train()` therefore informs the model of the current epoch and total epochs before each epoch (a no-op hook on `_Model`, overridden by the VAE). This keeps the schedule owned by the model and the loop model-agnostic, and gives Decision 3's optional warmup freeze of the uncertainty parameters a place to live.
- **Diagnostics** are new `TrainingHistory` fields populated via the existing metric-name mechanism; models that do not report them leave them as empty lists, per the shared-history contract.

**Alternatives considered:**
- **Revert to per-arm verbs and a type-switching `_train_loop`** (the Phase 3 shape this design was originally drafted against) — rejected: 3b's interface already delivers the single `train()` this design needs, and the per-model `training_step` is the natural home for a loss that only the VAE has.
- **A new unified module (`picoface.model`)** — deferred: attractive alongside a `build_model()` rename, and both belong in Phase 7's naming pass rather than here.

### 7. The CNN stays as the capstone's independent judge; the autoencoder returns to reconstruction-only

Both Phase 2's CNN and Phase 3's plain autoencoder remain in the codebase, with explicitly different reasons.

**The CNN has a real job.** `classify_generated()` measures something only while the judge is independent of the judged. With one model, "the classifier recognizes the generated images" is near-tautological — the decoder was trained to produce output whose encoding lands in the right cluster, and the head reads that same encoding. A separately-trained CNN scoring the VAE's samples is an honest external measurement. So the CNN is not demoted to API completeness; it is repositioned, and it must keep working against the same `Dataset` contract and the same public call shapes as the rest of the project.

**The autoencoder does not have a job, and its docs should say so.** It was justified as a pedagogical stepping stone — one new idea at a time on the way to the VAE. That rationale is gone: students never see the architecture, so there is no progression for them to walk. It stays for API completeness and unsupervised: Phase 3b's classification head and `classify` capability are removed from it, returning it to Phase 3's reconstruction-only behavior, since a second classifying model with a different head placement would contradict Decision 1 and give students two answers to one question. Its docstring and the spec now state plainly that it is not a required step rather than implying a progression that no longer exists.

### 8. Diagnostics are recorded; accuracy is asserted only loosely

`TrainingHistory` gains per-epoch reconstruction loss, KL divergence, classification loss, the annealed KL weight actually applied, and the learned uncertainty weights. Tests record held-out accuracy and wall-clock time alongside them.

Recording rather than asserting is the point. Pinning an accuracy number now would pin it against a stub dataset that is not the real content, and Phase 6 has no basis to judge how much tuning effort is warranted without a baseline to compare against. The recorded numbers are that baseline.

One loose assertion is kept — held-out accuracy above chance by a margin, on a seeded run — because a test that asserts nothing catches nothing, and the repo currently has exactly this problem: `assert 0.0 <= accuracy <= 1.0` appears three times in `tests/test_classifier.py` and is true of any float in range. There is no accuracy regression protection anywhere today. The bar is set well below observed behavior so it guards against breakage without pinning a target.

Two things make the recorded numbers trustworthy and are required, not optional:
- **A held-out split.** Every `evaluate()` call in the existing tests passes the training data. The student-facing claim is about classifying *novel* data; measuring on the training set measures memorization. A second stub with a different seed is sufficient.
- **A seed on torch initialization.** `make_stub_dataset` seeds its own RNG but weight initialization is unseeded, so nothing is comparable run to run and any tight assertion would flake.

### 9. The stub dataset gains classes separable only by spatial arrangement

`make_stub_dataset` currently fills each class with a constant gray value plus noise of +/-20, with class means far enough apart to never overlap — its docstring says "trivially separable", and it is: a threshold on image mean solves it in one dimension.

That makes it useless for Decision 8's purpose. A latent bottleneck would score near-perfectly on a one-dimensional task and report a number that says nothing about real shapes, which differ by *spatial arrangement* at comparable brightness. A falsely reassuring baseline is worse than no baseline.

The generator therefore gains shape-bearing classes — simple filled figures rendered at matched mean brightness, so only arrangement distinguishes them. This is the mitigation ROADMAP.md already prescribes for a risk it already identified ("the stub dataset should include at least one deliberately non-trivial synthetic class") and which was never implemented.

This edges toward Phase 5's territory and stops well short of it: a few lines of numpy, no taxonomy commitment, no resolution or color-depth decision, no rendering machinery. Phase 5 still owns real content. The existing brightness-separated mode is kept for the shape-agnosticism and plumbing tests that do not need difficulty.

## Risks / Trade-offs

- **Uncertainty weighting can starve reconstruction** (Decision 4: effective weight proportional to `1 / L`, so the already-solved task gains weight). → Documented, surfaced through recorded per-epoch weights, with a `log(sigma^2)` floor as the lever; Phase 6 owns the value. This is the most likely source of a "trains fine, generates nothing" outcome.
- **Classification through a probabilistic bottleneck caps accuracy**, and sampled-`z` gradients are noisier than deterministic ones. → Accepted deliberately (Decision 1): it is the only variant where supervision reaches what `generate()` samples. Unpinning `latent_dim` (Decision 2) is the main relief valve, and the recorded baseline tells Phase 6 whether more is needed.
- **Equal reconstruction/classification strength is a choice, not a derivation** (Decision 4). → Recorded as such; it is resolution-invariant, and the recorded learned weights show whether it serves real data in Phase 6.
- **The prior's weight falls as `1/D` against both tasks** (Decision 5). → Intended ELBO behavior, stated in the spec; if Phase 5 picks a much larger resolution, the recorded KL values show whether the latent still regularizes.
- **Two new schedules and a learned balance interact**, and the model must now reach two objectives in one run within the CPU budget. → The budget requirement is re-validated in this change rather than assumed; the recorded wall-clock number makes a regression visible immediately.
- **Removing `show_latent_space()` deletes delivered, spec'd behavior.** → Accepted; ranked nice-to-have, never in a student notebook, no migration needed. The capability to reintroduce a latent visualization is unaffected by anything here.
- **Retiring arm separation loosens a constraint two phases defended.** → Scoped narrowly (Decision 6): the principle is retired, the existing duplication stays, and the CNN stays genuinely independent because Decision 7 gives it a role that requires independence.
- **An improved stub dataset is still synthetic.** → It makes the Phase 6 baseline meaningful, not predictive. Phase 6 remains the phase where real numbers arrive; nothing here claims otherwise.

## Open Questions

None blocking implementation. Left to implementation-time experiment and recorded for Phase 6: the latent dimensionality, the annealing schedule's length and shape (its end point is fixed at the ELBO weight of 1 by Decision 3), the `log(sigma^2)` floor, whether uncertainty parameters are frozen during warmup (Decision 3), and the specific figures the stub renders. Consistent with how Phases 2 and 3 left their architecture constants to implementation-time tuning.
