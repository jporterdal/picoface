## Why

The VAE's latent is nominally 128-dimensional (`LATENT_DIM`, set in Phase 6). The dataset it models has roughly ten generating factors: position, size, rotation, stroke width, background and contrast, and class identity. The VAE literature predicts that a VAE with a learned decoder variance uses only as many latent dimensions as the data needs, and collapses the rest to the prior (Dai & Wipf, 2019; Rolínek et al., 2019; Lucas et al., 2019). If that holds here, 128 is harmless headroom that costs parameters: `fc_mu`, `fc_logvar` and the decoder's input layer hold about 300K of the model's 327K parameters. If it does not hold, the loss is keeping unneeded dimensions alive, which bears on how well `generate()`'s samples from N(0, I) match the region real images encode to.

Phase 6 chose 128 from downstream agreement scores (`picoface-phase6/diagnostics.md`, `latent_dim` sweep) without looking inside the latent. Nobody has measured how many of the 128 dimensions the trained model actually uses. This change records that as an open question and starts answering it with the standard measurement: counting **active units** (Burda et al., 2016).

## What Changes

- Add an active-units measurement to the Forge's smoke check (`dataset_forge/smoke.py`), reported alongside its existing numbers. For every latent dimension it reports the variance of the encoder mean across the test split, the mean posterior variance, and the mean KL. It also reports the active-unit count at the conventional threshold of 0.01 and at neighbouring thresholds.
- Run the measurement on the shipped model (default export, `train()` defaults, 3 seeds). Also sweep the nominal latent size (8 to 192, the same range as Phase 6's sweep) to see whether the active count grows with the nominal size or levels off.
- Check that "inactive" means "unused": replace the inactive dimensions of each test image's latent mean with the prior mean (0) and confirm reconstruction agreement does not change.
- Record the results, and which of the pre-registered outcomes they support (design.md, Decision 5), in a diagnostics record for this change, following Phase 5 and Phase 6's format.
- Add the question to `openspec/ROADMAP.md`'s Open Questions, pointing at this change's findings.

This change **measures and recommends; it does not retune.** If the findings argue for a different `LATENT_DIM` or a different loss weighting, they are recorded as a recommendation for a follow-up change, not applied here.

## Capabilities

### New Capabilities
None.

### Modified Capabilities
None. `LATENT_DIM` is an internal constant that no spec names. The smoke check is instructor tooling with no spec requirement, and adding a report to it changes no student-facing behavior. The change sets `skip_specs: true` accordingly.

## Impact

- `dataset_forge/smoke.py`: a new active-units report, included in `smoke()`'s results and `format_results()`'s Markdown output. No new CLI flag: the measurement is one forward pass over the test split.
- `dataset_forge/tests/test_smoke.py`: tests for the new report.
- `openspec/changes/latent-active-units/diagnostics.md`: new record of the measurements and their interpretation.
- `openspec/ROADMAP.md`: one new Open Questions entry.
- No change to `src/picoface/`: no public API, internal constant, or model behavior changes.

## References

- Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. *ICLR*. arXiv:1312.6114
- Burda, Y., Grosse, R., & Salakhutdinov, R. (2016). Importance Weighted Autoencoders. *ICLR*. arXiv:1509.00519
- Dai, B., & Wipf, D. (2019). Diagnosing and Enhancing VAE Models. *ICLR*. arXiv:1903.05789
- Rolínek, M., Zietlow, D., & Martius, G. (2019). Variational Autoencoders Pursue PCA Directions (by Accident). *CVPR*. arXiv:1812.06775
- Lucas, J., Tucker, G., Grosse, R., & Norouzi, M. (2019). Don't Blame the ELBO! A Linear VAE Perspective on Posterior Collapse. *NeurIPS*. arXiv:1911.02469
