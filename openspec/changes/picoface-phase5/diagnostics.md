# picoface-phase5 diagnostics record

A feasibility smoke check of picoface's defaults on Dataset Forge's default export (design Decision 7). **This is a measurement for Phase 6, not tuning.** Nothing in `src/picoface/` was changed.

## Setup

- **Machine:** AMD Ryzen 7 7800X3D (8 cores), 16 GB RAM visible to WSL2 (WSL2 caps this below the host's actual total by default, so the host may have more), CPU only. PyTorch 2.13.0+cpu, numpy 2.5.2, Pillow 12.3.0. The machine was otherwise idle. An earlier run under heavy load was discarded: its timings were up to 3× slower and inconsistent.
- **Data:** `configs/default.json`, 7 classes, 1,000 train / 200 test per class, seeds 0–2. The 28×28 comparison uses the same config with only `height`/`width` changed.
- **Models:** `build_classifier` (the CNN) and `build_vae` (the student's model), each trained with `train()` defaults (10 epochs, batch size 16, learning rate 1e-3). That is about 440 optimizer steps per epoch, against 1 for the Phase 3c default stub. Seeded with `torch.manual_seed(0)`.
- **Command:** `python -m dataset_forge.smoke dataset_forge/output/default-seed<N>`. Held-out accuracy and confusion are measured on the export's test split. `classify_generated()` uses `n=20` per class.

## Export validation baselines

Chance is 0.143. The mean-brightness gate is < 0.243.

| Resolution | Seed | Mean brightness only (gated) | Ink fraction only (baseline) |
|---|---|---|---|
| 24×24 | 0 | 0.179 | 0.392 |
| 24×24 | 1 | 0.167 | 0.412 |
| 24×24 | 2 | 0.179 | 0.389 |
| 28×28 | 0 | 0.180 | 0.404 |
| 28×28 | 1 | 0.171 | 0.412 |
| 28×28 | 2 | 0.164 | 0.414 |

Both models beat the ink-fraction baseline by a wide margin (below), so they are recognizing shapes, not just counting ink.

## Training time and held-out accuracy

PyTorch's default thread count (8 here):

| Resolution | Seed | CNN time (s) | CNN accuracy | VAE time (s) | VAE accuracy |
|---|---|---|---|---|---|
| 24×24 | 0 | 6.2 | 0.947 | 10.6 | 0.864 |
| 24×24 | 1 | 5.7 | 0.961 | 14.0 | 0.846 |
| 24×24 | 2 | 7.1 | 0.959 | 11.0 | 0.881 |
| 28×28 | 0 | 5.8 | 0.974 | 10.5 | 0.954 |
| 28×28 | 1 | 6.7 | 0.969 | 11.1 | 0.954 |
| 28×28 | 2 | 6.0 | 0.946 | 10.3 | 0.941 |

- **Time budget:** well within "seconds to minutes". Even allowing an old laptop core 3–5× slower than this one, each model would train in well under a minute or two.
- **Threads:** a single thread (`torch.set_num_threads(1)`) was as fast as or faster than the default: CNN 5.7–6.5 s, VAE 9.0–9.7 s, with the same accuracy to ±0.005. At batch size 16 these models are too small to gain from more threads. Nothing needs to change, but Phase 6 should know that core count doesn't matter much here: single-core speed does.
- **Resolution cost:** 28×28 costs about the same time as 24×24 on this machine.

## Per-class results

Ranges are over seeds 0–2, default threads. The CNN and VAE columns are the per-class held-out accuracy (the confusion-matrix diagonal, out of 200).

**24×24**

| Class | CNN | VAE | `classify_generated()` agreement |
|---|---|---|---|
| square | 0.76–0.92 | 0.81–0.84 | 0.75–0.80 |
| ring | 0.96–0.99 | 0.94–0.99 | 0.80–0.90 |
| circle | 0.93–0.96 | 0.81–0.91 | 0.90 |
| triangle | 0.93–0.97 | 0.41–0.72 | 0.15–0.35 |
| star | 0.99 | 0.70–0.94 | 0.05–0.10 |
| smiley | 0.99–1.00 | 0.87–0.98 | 0.00 |
| negative_smiley | 0.98–1.00 | 0.96–0.99 | 0.05–0.15 |
| overall | | | 0.41–0.43 |

Largest confusions, summed over 3 seeds (600 test images per class):
- **CNN:** square→triangle 68, circle→square 33, square→circle 32, triangle→square 15, triangle→star 14.
- **VAE:** triangle→star 222, star→triangle 90, square→triangle 71, circle→square 61, smiley→ring 32.

**28×28**

| Class | CNN | VAE | `classify_generated()` agreement |
|---|---|---|---|
| square | 0.86–0.91 | 0.91–0.94 | 0.55–0.70 |
| ring | 0.97–0.99 | 0.94–0.97 | 0.65–0.70 |
| circle | 0.96–0.98 | 0.94–0.97 | 0.80–0.95 |
| triangle | 0.82–0.95 | 0.82–0.89 | 0.10–0.15 |
| star | 0.97–1.00 | 0.95–0.99 | 0.00 |
| smiley | 0.99–1.00 | 0.97–0.99 | 0.00 |
| negative_smiley | 0.99–1.00 | 0.97–1.00 | 0.00–0.05 |
| overall | | | 0.32–0.35 |

Largest confusions, summed over 3 seeds:
- **CNN:** triangle→square 51, square→circle 37, square→triangle 26, circle→square 16, ring→smiley 10.
- **VAE:** triangle→square 54, triangle→star 31, square→circle 29, ring→smiley 25, circle→square 24.

## Observations

- **The face classes are not the hard ones to classify.** The design flagged smiley/circle and smiley/ring as the pairs to watch (Risks). In practice both models classify `smiley` and `negative_smiley` at 0.87–1.00 at 24×24. The CNN almost never confuses them with their plain counterparts (at most 10 of 600 either way). The VAE's one notable face confusion is smiley→ring (32 of 600 at 24×24, falling at 28×28).
- **The filled polygons are where classification struggles.** Square/triangle/circle for the CNN, and triangle/star for the VAE at 24×24: its triangle accuracy drops to 0.41 on one seed. At 24×24 with a radius of about 7–9 px, a triangle's and a star's corners are only a pixel or two across, so the fine detail that separates them is barely there.
- **28×28 helps the student's model most.** VAE held-out accuracy rises from 0.85–0.88 to 0.94–0.95, mostly on triangle and star. The CNN gains a little (0.95–0.96 → 0.95–0.97). Neither pays a time cost.
- **`classify_generated()` is low at both sizes, and the cause is the VAE's decoder, not how the capstone samples.** The `generated.png` figure (see Figures, below) shows, per class (rows, in config order), 4 real training images and then 12 generated ones. Every generated image is a soft round blob: squares, triangles, and stars lose their corners and points, and both smileys lose their features. Only rings and circles, which are round anyway, come out recognizable, and they are the only classes the CNN agrees on. What the CNN calls the generated images (24×24, seed 0): stars → triangle 12, square 7, star 1; smileys → circle 10, ring 5, negative_smiley 4; negative smileys → circle 17.
- **Sampling around a class's cluster is not to blame.** One hypothesis was that full-circle rotation makes each class's latent cluster a mix of all angles, so sampling near its centre would average the rotations into a blob. It was tested by decoding the latent mean of individual *real* test images (one specific angle each) instead. Those reconstructions are blobs too (the `reconstructed.png` figure: 6 real images per class, then their reconstructions). CNN agreement on them is 0.34 at 24×24 and 0.26 at 28×28, lower than for class-cluster samples. The decoder cannot yet draw corners, points, or faces from any latent.
- **The latent carries the class; the reconstruction does not carry the shape.** The VAE's classification head reads the same latent at 0.85–0.95 held-out, while its decoder turns that latent into blobs. This matches the Phase 3c risk that learned uncertainty weighting can pour capacity into the easier classification task and starve reconstruction. With about 440 steps per epoch instead of the stub's 1, this is the first dataset large enough to show it. It is a Phase 6 tuning question: the log-variance floor, `latent_dim`, epoch count, the decoder's capacity. It is not a Forge or data-contract issue.
- **At 28×28, `classify_generated()` agreement drops (0.41–0.43 → 0.32–0.35)**, even though the VAE classifies better. More pixels to reconstruct, with no more decoder capacity, gives blurrier blobs, and fewer of them sit close enough to a square for the CNN to agree.

## Decision: 28×28

There is ample time headroom for 28×28: training time is unchanged at these model sizes, well under a minute per model on this machine. 28×28 clearly improves the student's model at classification (VAE 0.85–0.88 → 0.94–0.95), which is the part of the course that works today. It does not help the capstone. `classify_generated()` is limited by the decoder at either size, and its tuning in Phase 6 will have to happen at whichever size is chosen.

The recommendation was to move to 28×28 before Phase 6 starts, so Phase 6 tunes the decoder at the final size rather than retuning after a later switch. **Decided: 28×28 is the default** (`configs/default.json`, and the `dataset-forge` spec's default-resolution scenario). The 28×28 numbers above were measured on exactly the config that is now the default.

## Does more training data help?

At the 28×28 default, on export seeds 0 and 1, with an idle machine and default threads:
- **A:** the default 1,000 training images per class, 10 epochs.
- **B:** twice the data, 2,000 per class, 10 epochs.
- **C:** a control with the default data but 20 epochs, so the same number of optimizer steps as B without new images.

All three are scored on the same test split: the Forge's test split doesn't change when the training count does.

| Run | CNN time (s) | CNN accuracy | VAE time (s) | VAE accuracy | `classify_generated()` | Reconstruction agreement |
|---|---|---|---|---|---|---|
| A: 1,000 × 10 epochs | 5.6–5.8 | 0.969–0.974 | 9.9–10.8 | 0.954 | 0.32–0.35 | 0.26–0.31 |
| B: 2,000 × 10 epochs | 11.2–11.3 | 0.991–0.992 | 20.1–22.0 | 0.983–0.987 | 0.26–0.34 | 0.28–0.29 |
| C: 1,000 × 20 epochs | 10.8 | 0.988–0.991 | 21.0–22.2 | 0.946–0.973 | 0.29–0.34 | 0.28 |

- **Training time scales linearly with the number of optimizer steps**, so doubling either the data or the epochs doubles it. B stays well within budget: about 11 s (CNN) and 21 s (VAE) here.
- **More data helps classification, and helps the VAE more than extra epochs do.** Both B and C lift the CNN to about 0.99. The VAE reaches 0.98–0.99 with twice the data, but only 0.95–0.97 with twice the epochs on the same data. So its remaining classification errors are partly a data limit, not just a training-length limit.
- **More data does not help generation at all.** For the problem classes (triangle, star, smiley, negative_smiley), `classify_generated()` and reconstruction agreement stay at 0.00–0.10 in every run. B's reconstructions are the same soft blobs as A's. Only the round classes, plus square on some seeds, are ever recognized. Neither doubling the data nor doubling the training steps moves the decoder, so its limit is not data quantity or training length at this scale. That points Phase 6 at the objective's balance (the Phase 3c uncertainty weighting and log-variance floor) and the decoder's capacity.
- **A reproduces the earlier 28×28 seed-0 and seed-1 results exactly** (e.g. seed 0: CNN 0.974, VAE 0.954, `classify_generated()` 0.35), so the smoke check is deterministic for a given export, seed, and machine.

The default stays at 1,000 training images per class for now. 2,000 is a cheap classification gain, but it doesn't touch the capstone's problem. Whether to adopt it is a config change best decided in Phase 6, alongside the decoder work (Open Questions in design.md).

## Phase 6 inputs from this record

- Why the VAE's reconstructions are blobs while its classification head is accurate: check the learned task weights against the Phase 3c floor, the decoder's capacity, `latent_dim`, and the epoch count, on this dataset rather than the stub.
- `classify_generated()` agreement as the capstone's main quality number: 0.32–0.35 at the 28×28 default today, with only the round classes above 0.5. Reconstruction agreement (the smoke check reports it alongside) shows how much of that is the decoder rather than the sampling.

## Figures

Figures are not committed. Regenerate them with the smoke check's `--figures` option, into the gitignored output folder:

```bash
python -m dataset_forge.smoke dataset_forge/output/default-seed0 --figures dataset_forge/output/figures/seed0
```

This writes `generated.png` (per class: 4 real training images, then 12 `classify_generated()` images) and `reconstructed.png` (per class: 6 real test images, then the VAE's reconstructions of them). The observations above came from the 24×24 export. At the 28×28 default the figures show the same blobs.
