## Context

Phase 3 built the generator arm and archived it. Its VAE is unsupervised: `_train_loop` receives `data.images` only, wraps them in a single-tensor `TensorDataset`, and optimizes `mse_loss(recon, x) + BETA * kl`. `data.labels` never enters the arm; the only label contact anywhere is `show_latent_space()` using them as matplotlib scatter colors after training finishes.

A `/opsx:explore` session established that this does not match the project's intent. The intended deliverable is one supervised model that a student trains once and then uses as a classifier and as a generator. Phase 2's standalone CNN and Phase 3's plain autoencoder both remain in the codebase but neither is the student's path any longer.

That reframe invalidates several Phase 3 decisions and two of its spec requirements, and it pulls forward one decision Phase 3 explicitly deferred (KL annealing). It also relaxes a constraint Phases 2 and 3 both spent design text defending — "the arms share no model code" — because a single model that classifies and generates is, definitionally, not two arms.

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

The KL weight ramps from zero to a ceiling over an internal schedule. Warming up without the prior lets reconstruction and classification establish a useful latent before the prior starts pulling it toward `N(0, I)`; this is the standard mitigation for posterior collapse and is exactly what Phase 3's design deferred ("a real technique for exactly this failure mode ... worth trying during Phase 6").

Uncertainty weighting is **not** extended to the KL term, though it is superficially tempting to weight all three the same way. Kendall et al.'s derivation applies to task losses with a likelihood interpretation and an observation-noise parameter to learn. The KL divergence is a regularizer against the prior; it has no such parameter. Giving it a learned weight also creates an obvious degenerate pull — the model can make the regularizer cheap by inflating its uncertainty — which is the failure annealing exists to prevent, reintroduced through the mechanism meant to remove hand-tuning.

The two mechanisms therefore occupy disjoint roles: **annealing schedules the regularizer, uncertainty weighting balances the tasks.**

Whether the learned uncertainty parameters are frozen during the annealing warmup or allowed to re-equilibrate as the KL weight rises is left to implementation-time experiment and recorded in the diagnostics (Decision 8). Freezing is the more conservative default.

### 4. Uncertainty weighting (Kendall et al., 2018) balances reconstruction against classification

The two task losses are combined as learned homoscedastic uncertainties: each task `i` contributes `(1 / (2 * sigma_i^2)) * L_i + log(sigma_i)`, with `log(sigma_i^2)` a learned parameter of the model. This removes the need for any hand-picked reconstruction-vs-classification weight, which matters more here than in the paper's setting: the balance depends on the dataset, and neither an instructor nor a student will tune it per course.

**A known failure mode is accepted and documented rather than solved now.** Minimizing that expression over `sigma` yields `sigma_i^2 = L_i`, so the effective gradient weight on each task is `1 / (2 * L_i)` — *lower loss earns higher weight*. On easily-classified data, cross-entropy falls fast while reconstruction MSE plateaus, so the weighting can pour capacity into the already-solved task and starve generation. This runs against the roadmap's stated goal of a generative arm that is "reliable to train, not merely impressive when it happens to converge."

The mitigation lever is a floor on `log(sigma^2)`, which reintroduces exactly one hand-tuned constant — a much weaker one than the per-dataset weight it replaces, since it only bounds a pathology rather than setting the balance. Its value is a Phase 6 tuning question; the diagnostics in Decision 8 are what make it visible. Recording the learned weights per epoch is not optional polish here, it is how this risk gets caught before a student's laptop finds it.

**Alternatives considered:**
- **A fixed hand-tuned classification weight alongside the existing `BETA`** — rejected: two hand-tuned constants, both invalid the moment Phase 5 changes the data.
- **GradNorm or a similar gradient-magnitude balancer** — rejected: more machinery and another schedule to tune, for a two-task problem where the simpler method is well-understood and cheap.

### 5. Loss reductions are normalized to a common per-image scale before anything is weighted

Today reconstruction uses `mse_loss` with mean reduction over `N*C*H*W` (per-pixel) while KL sums over latent dimensions and means over the batch (per-image). The two terms live on scales separated by a factor of `H*W*C`, so `BETA = 0.01` at 16x16x3 is equivalent to a conventional beta near 7.7 — strong regularization, not the light touch the current code comment reasons its way to, and an explanation for why larger trial values collapsed the KL so hard.

Worse for what comes next, the effective regularization strength is **resolution-dependent**: Phase 5 choosing a different image size would silently change it. All three terms are therefore normalized to a per-image scale before annealing or uncertainty weighting is applied.

This must happen *first*. Layered on top of the current mismatch, the learned `sigma` parameters would quietly absorb a resolution-scaling artifact into their values — the model would still train, the bug would be invisible, and the recorded diagnostics that Phase 6 depends on would be measuring the wrong thing.

### 6. `train()`, `evaluate()`, and `predict()` serve VAE models; the "arms share no code" constraint is retired for the unified model

`train(model, data)` keeps its signature. For VAE models it now requires labels and validates class count against the model, raising the generator arm's `ShapeError` the way `picoface.classifier.train()` already does — currently the generator's `train()` accepts any `Dataset` without checking anything but image shape.

`evaluate(model, data)` and `predict(model, image)` are added to `picoface.generator`, reading `mu` through the classification head (Decision 1). They mirror the classifier arm's semantics exactly: accuracy in `[0, 1]`, and a class *name* rather than an index.

`_train_loop` now dispatches three ways: plain autoencoder (reconstruction only, labels ignored), VAE (normalized reconstruction + annealed KL + uncertainty-weighted classification, labels required), with shape and class-count validation ahead of both. The asymmetry — labels ignored for one model type, required for another, from the same public call — is deliberate and documented rather than smoothed over, because the autoencoder is frozen (Decision 7).

Phase 2 and Phase 3 both argued for strict arm separation, including a deliberately duplicated `_preprocess` and a separately-defined `ShapeError`. A single model that classifies and generates makes that premise obsolete. This design does not go on to merge the two internals modules or unify the exception types: that is cleanup with no behavioral payoff, and the CNN remains genuinely independent by Decision 7. The constraint is retired as a *principle*; the existing duplication stays until something needs it gone.

**Alternatives considered:**
- **Generalize `picoface.classifier.evaluate()/predict()` to accept either model type** — rejected: makes the classifier module the home of generator behavior and inverts the dependency the project has kept clean.
- **A new unified module (`picoface.model`)** — deferred: attractive alongside a `build_model()` rename, and both belong in Phase 7's naming pass rather than here.

### 7. The CNN stays as the capstone's independent judge; the autoencoder stays frozen

Both Phase 2's CNN and Phase 3's plain autoencoder remain in the codebase, with explicitly different reasons.

**The CNN has a real job.** `classify_generated()` measures something only while the judge is independent of the judged. With one model, "the classifier recognizes the generated images" is near-tautological — the decoder was trained to produce output whose encoding lands in the right cluster, and the head reads that same encoding. A separately-trained CNN scoring the VAE's samples is an honest external measurement. So the CNN is not demoted to API completeness; it is repositioned, and it must keep working against the same `Dataset` contract and the same public call shapes as the rest of the project.

**The autoencoder does not have a job, and its docs should say so.** It was justified as a pedagogical stepping stone — one new idea at a time on the way to the VAE. That rationale is gone: students never see the architecture, so there is no progression for them to walk. It stays for API completeness, unchanged and unsupervised, and its docstring and the spec now state plainly that it is not a required step rather than implying a progression that no longer exists.

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

- **Uncertainty weighting can starve reconstruction** (Decision 4's `1 / (2 * L)` dynamic). → Documented, surfaced through recorded per-epoch weights, with a `log(sigma^2)` floor as the lever; Phase 6 owns the value. This is the most likely source of a "trains fine, generates nothing" outcome.
- **Classification through a probabilistic bottleneck caps accuracy**, and sampled-`z` gradients are noisier than deterministic ones. → Accepted deliberately (Decision 1): it is the only variant where supervision reaches what `generate()` samples. Unpinning `latent_dim` (Decision 2) is the main relief valve, and the recorded baseline tells Phase 6 whether more is needed.
- **Two new schedules and a learned balance interact**, and the model must now reach two objectives in one run within the CPU budget. → The budget requirement is re-validated in this change rather than assumed; the recorded wall-clock number makes a regression visible immediately.
- **Removing `show_latent_space()` deletes delivered, spec'd behavior.** → Accepted; ranked nice-to-have, never in a student notebook, no migration needed. The capability to reintroduce a latent visualization is unaffected by anything here.
- **Retiring arm separation loosens a constraint two phases defended.** → Scoped narrowly (Decision 6): the principle is retired, the existing duplication stays, and the CNN stays genuinely independent because Decision 7 gives it a role that requires independence.
- **An improved stub dataset is still synthetic.** → It makes the Phase 6 baseline meaningful, not predictive. Phase 6 remains the phase where real numbers arrive; nothing here claims otherwise.

## Open Questions

None blocking implementation. Left to implementation-time experiment and recorded for Phase 6: the latent dimensionality, the annealing schedule's length and ceiling, the `log(sigma^2)` floor, whether uncertainty parameters are frozen during warmup (Decision 3), and the specific figures the stub renders. Consistent with how Phases 2 and 3 left their architecture constants to implementation-time tuning.
