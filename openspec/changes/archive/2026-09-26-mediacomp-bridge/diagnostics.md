# mediacomp-bridge diagnostics

Smoke-check results for the flipped (dark on light) dataset, and the `activation_maximize()` edge-darkening diagnostic (tasks 2.2 and 2.3).

## Setup

- **Machine:** AMD Ryzen 7 7800X3D (8 cores), 15 GB RAM, WSL2 on Linux 6.6.87.2. CPU only. Python 3.13.5, PyTorch 2.13.0+cpu, numpy 2.5.2, Pillow 12.3.0.
- **Code:** commit `19381f0` plus this change's uncommitted work (tasks 1.1–2.1, 2.4, 3.1–3.9 and 3.11 applied), for every run.
- **Command:** `python -m dataset_forge.smoke dataset_forge/output/<export> --seed <N> --figures dataset_forge/output/figures/<name>`. Both models use `train()`'s defaults, and `classify_generated()` uses `n=20` per class. The four runs ran one after another on an otherwise idle machine.
- **Exports:** all use `configs/default.json`: 7 classes, 28×28 grayscale, 2,000 train and 200 test images per class. All passed validation.

| Run | Export directory | Background range | Export seed | Training seed | Export commit | Figures |
|---|---|---|---|---|---|---|
| flipped, seed 0 | `default-seed0` | 96–255 | 0 | 0 | `19381f0` + uncommitted | `figures/mediacomp-seed0/` |
| flipped, seed 1 | `default-seed1` | 96–255 | 1 | 1 | `19381f0` + uncommitted | `figures/mediacomp-seed1/` |
| flipped, seed 2 | `default-seed2` | 96–255 | 2 | 2 | `19381f0` + uncommitted | `figures/mediacomp-seed2/` |
| old polarity, seed 0 | `default-seed0-old-polarity` | 0–159 | 0 | 0 | `c12ef68` + uncommitted (the Phase 6 export, copied) | `figures/mediacomp-seed0-old-polarity/` |

The flipped seed-0 export is not a pixel-exact negative of the old one. The labels match, but the flipped renderer draws its shades differently, so the pixels differ (by up to 198 from `255 − x`). The shade distribution is the mirror image, which is what the spike tested (design.md Decision 8).

## Results (task 2.2)

### Held-out accuracy and training time

| Run | CNN accuracy | VAE accuracy | CNN train (s) | VAE train (s) |
|---|---|---|---|---|
| flipped, seed 0 | 0.986 | 0.970 | 14.8 | 38.0 |
| flipped, seed 1 | 0.997 | 0.984 | 13.0 | 32.0 |
| flipped, seed 2 | 0.999 | 0.979 | 12.5 | 37.4 |
| old polarity, seed 0 | 0.991 | 0.994 | 13.8 | 34.8 |

Training times are longer than Phase 6's ~9.5 s and ~21 s. That comes from this machine and its load, not the flip: the old-polarity run, on the same machine and code, took as long.

### CNN agreement on the VAE's images

`classify_generated()` is the CNN's agreement with the VAE's class-targeted generated images; reconstruction is its agreement with the VAE's reconstructions of real test images.

| Class | flipped 0: generated | recon | flipped 1: generated | recon | flipped 2: generated | recon | old 0: generated | recon |
|---|---|---|---|---|---|---|---|---|
| square | 1.00 | 0.85 | 1.00 | 0.95 | 1.00 | 0.90 | 1.00 | 0.95 |
| ring | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| circle | 0.95 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| triangle | 0.95 | 0.75 | 0.90 | 0.80 | 0.90 | 0.90 | 0.90 | 1.00 |
| star | 0.55 | 0.50 | 0.65 | 0.45 | 0.70 | 0.55 | 0.60 | 0.60 |
| smiley | 0.80 | 0.80 | 0.70 | 0.75 | 1.00 | 0.80 | 0.55 | 0.45 |
| negative_smiley | 1.00 | 0.85 | 0.95 | 0.95 | 0.90 | 0.75 | 0.85 | 0.95 |
| **overall** | **0.89** | **0.82** | **0.89** | **0.84** | **0.93** | **0.84** | **0.84** | **0.85** |

### Confusion, flipped seed 0 (test set, 200 per class)

The VAE's main error was 23 `circle` images predicted as `square`. The CNN's was 13 `square` images predicted as `triangle`.

| CNN | square | ring | circle | triangle | star | smiley | negative_smiley |
|---|---|---|---|---|---|---|---|
| square | 186 | 0 | 1 | 13 | 0 | 0 | 0 |
| ring | 0 | 200 | 0 | 0 | 0 | 0 | 0 |
| circle | 0 | 0 | 200 | 0 | 0 | 0 | 0 |
| triangle | 1 | 0 | 0 | 199 | 0 | 0 | 0 |
| star | 0 | 0 | 0 | 0 | 200 | 0 | 0 |
| smiley | 0 | 0 | 0 | 0 | 0 | 200 | 0 |
| negative_smiley | 0 | 0 | 0 | 0 | 1 | 3 | 196 |

| VAE | square | ring | circle | triangle | star | smiley | negative_smiley |
|---|---|---|---|---|---|---|---|
| square | 196 | 0 | 0 | 4 | 0 | 0 | 0 |
| ring | 0 | 198 | 0 | 0 | 1 | 1 | 0 |
| circle | 23 | 0 | 177 | 0 | 0 | 0 | 0 |
| triangle | 0 | 0 | 0 | 198 | 2 | 0 | 0 |
| star | 0 | 0 | 0 | 5 | 195 | 0 | 0 |
| smiley | 0 | 0 | 0 | 1 | 3 | 195 | 1 |
| negative_smiley | 1 | 0 | 0 | 0 | 0 | 0 | 199 |

### Ranges for README "What to expect" (three flipped runs)

| Metric | Range |
|---|---|
| CNN held-out accuracy | 0.986–0.999 |
| VAE held-out accuracy | 0.970–0.984 |
| Reconstruction agreement, overall | 0.82–0.84 |
| `classify_generated()` overall agreement | 0.89–0.93 |

### Seed-0 comparison with the old polarity

The criterion was: the flipped seed-0 CNN and VAE held-out accuracies are each within 0.01 of, or above, the old-polarity seed-0 run's.

- CNN: 0.986 vs 0.991 (−0.005). **Passes.**
- VAE: 0.970 vs 0.994 (−0.024). **Fails.**

To see whether the VAE gap comes from the polarity or from training-seed noise, both seed-0 exports were trained again with training seeds 1, 2 and 3. These runs used the same code and machine and `train()`'s defaults, and measured held-out accuracy only:

| Training seed | flipped: CNN | VAE | old polarity: CNN | VAE |
|---|---|---|---|---|
| 0 (smoke run above) | 0.986 | 0.970 | 0.991 | 0.994 |
| 1 | 0.988 | 0.988 | 0.991 | 0.986 |
| 2 | 0.987 | 0.983 | 0.979 | 0.987 |
| 3 | 0.996 | 0.995 | 0.998 | 0.991 |
| **mean** | **0.989** | **0.984** | **0.990** | **0.990** |

With training seeds 1–3, the flipped VAE is above the old one twice and 0.004 below it once. The flipped VAE's accuracy varies from 0.970 to 0.995 across training seeds on the same export, so the seed-0 gap falls within that seed-to-seed spread. The 4-seed mean gap (−0.005 CNN, −0.006 VAE) is also within the criterion. The spike found the same (design.md Decision 8: VAE 0.987 ± .006 flipped vs 0.989 ± .004).

**Decision (2026-09-25):** the user accepted the seed-0 VAE gap as falling within seed-to-seed variance, so the flipped dataset passes the sanity check.

## `activation_maximize()` edge darkening (tasks 2.1 and 2.3)

Each value is the mean of the outermost 2-pixel ring minus the mean of the 2-pixel ring inside it, over 4 random starts, in 0–255 units. Negative means the edges are darker than just inside them.

| Class | flipped 0: CNN | VAE | flipped 1: CNN | VAE | flipped 2: CNN | VAE | old 0: CNN | VAE |
|---|---|---|---|---|---|---|---|---|
| square | +1.4 | −57.5 | −6.0 | −67.4 | +27.8 | −47.4 | +18.4 | −20.3 |
| ring | −12.4 | −10.1 | −11.7 | −23.6 | −7.5 | −37.8 | −32.9 | −53.7 |
| circle | +24.3 | −31.5 | +31.1 | −20.8 | +41.4 | −5.0 | −14.0 | −46.0 |
| triangle | −38.5 | −45.9 | −44.2 | −44.8 | −44.4 | −51.6 | −5.7 | −11.5 |
| star | −29.1 | −30.1 | −43.3 | −36.4 | −43.4 | −40.4 | −18.3 | −26.6 |
| smiley | −17.2 | −4.0 | −6.5 | −2.6 | +10.5 | +8.7 | −59.8 | −56.7 |
| negative_smiley | +27.2 | +9.5 | +24.2 | +13.1 | +36.3 | +22.9 | −47.3 | −69.6 |
| **mean** | −6.3 | −24.2 | −8.1 | −26.1 | +3.0 | −21.5 | −22.8 | −40.6 |

Darker edges show up under both polarities, and on average the old polarity's are darker. The flip did not introduce edge darkening. What differs is which classes show it: under the flipped data, `triangle`, `star`, and the VAE's `square` have the darkest edges.

### Experiment: an L2 pull on the pixels

The user asked whether a weight-decay-style penalty would clean up the noisy background. A scratch copy of `_ascend()`, with everything else unchanged, subtracts `λ · Σ (x − t)²` from the target logit. Here `x` is the pixel values in [0, 1] and `t` is a target shade: 0 (plain weight decay, which pulls toward black), 0.5, or 1 (white). The CNN and VAE were trained on `default-seed0` with training seed 0, and the ascent used 4 fixed random starts per class. For comparison, the real training images' outer 4-pixel border averages 172.

| Pull | CNN: border mean | CNN: target prob | VAE: border mean | VAE: target prob |
|---|---|---|---|---|
| none (current) | 97 | 0.77 | 99 | 1.00 |
| toward 0, λ = 0.1 | 72 | 0.73 | 77 | 1.00 |
| toward 0.5, λ = 1 | 119 | 0.75 | 114 | 0.97 |
| toward 1, λ = 0.1 | 144 | 0.82 | 133 | 1.00 |
| toward 1, λ = 0.3 | 188 | 0.84 | 166 | 0.96 |
| toward 1, λ = 1 | 215 | 0.75 | 203 | 0.84 |
| toward 1, λ = 3 | 225 | 0.59 | 220 | 0.70 |

"Border mean" is the mean of the outer 4 pixels, in 0–255 units. "Target prob" is the mean softmax probability of the target class. The figures are in `dataset_forge/output/figures/mediacomp-seed0-pixel-decay/`: `weak.png` for λ ≤ 0.1 and `strong.png` for λ ≥ 0.3, with the script alongside. Each has one column pair (CNN, VAE) per setting and one row per class.

- Plain weight decay (toward 0) makes the background darker, which is the opposite of what light-background data needs.
- A pull toward white at λ ≈ 0.3 gives a background about as light as the real data's (188 and 166 vs 172), while keeping the target class's probability (0.84 and 0.96). Above λ = 1 the image fades to near-white and the probability drops.
- The pull cannot tell background from figure. On dark-on-light data the figure is the dark part, so every figure pixel costs as much as a dark background pixel. Without the pull, `square` and `circle` (and the CNN's `triangle`) show a dark, roughly figure-sized blob. As λ grows the blob shrinks to a small dark spot in the centre: the fewest dark pixels that still raise the target score.
- Classes that were diagonal stripes without the pull (`ring`, `star`, `smiley`, and most of `negative_smiley`) stay diagonal stripes, on a lighter background. The stripes were never hidden shapes. They are a texture the classifier's filters respond to, and it recurs across classes, so it is not class-specific to a human eye. This is the known behaviour of activation maximization without strong image priors. A pixel penalty does not change it.
- A pull toward white assumes the dataset's polarity, which `activation_maximize(model, target_class)` does not know. A polarity-neutral target of 0.5 only turns the background gray.

**Decision (2026-09-25):** the user reviewed the flipped and old-polarity `activation_maximize.png` figures, the edge-darkening table, and this experiment, and chose no action in this change. The ascent's blur keeps its zero padding, and `linkage_internals.py` is unchanged. Regularizing `activation_maximize()` is left to a future change. The candidates are optimising the VAE's latent code instead of the pixels, standard regularisers (jitter, total variation, a stronger blur, replicate padding in the blur), and a pull toward the background shade combined with one of those.
