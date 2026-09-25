## Context

See `proposal.md` for the motivation. This design is based on mediaComp 0.4.15, read from its PyPI wheel during exploration, and on the current repository.

- **mediaComp `Picture`** wraps a Pillow image and exposes it through `getImage()`. Its constructor accepts a Pillow image directly: `Picture(pil_image)`. But mediaComp exports only functions at its top level (`makePicture`, `getPixels`, ...), not the class.
- **mediaComp's pixel functions** (`getRed`, `setColor`, ...) call `getpixel()` and expect an RGB triple. `makePicture()` keeps whatever mode the file has. So a grayscale (`"L"` mode) picture breaks mediaComp's own pixel functions.
- **Distribution**: mediaComp is published on PyPI as `mediaComp` (`pip install mediaComp`; import name `mediaComp`). Students install it the same way as picoface, through Thonny's package manager (Tools → Manage packages) or pip. picoface doesn't install it for them.
- **mediaComp's dependencies**: GPL-3.0-or-later, plus wxPython (no Linux wheels), pygame-ce, sounddevice, and tkinter. `show()` runs a wx window in a subprocess.
- **Where students run it**: Thonny on Windows, having learned mediaComp first.
- **picoface's image format** is uint8 N×H×W×C. The default Forge export is 28×28×1. Today the Forge draws a light figure (foreground) on a darker background, at least `min_contrast` = 96 lighter, with backgrounds in 0–159. It anti-aliases by averaging a 4× supersampled canvas.
- **Pillow** is already installed with picoface because matplotlib requires it, but picoface doesn't declare it. `tests/test_packaging.py:30` asserts that it isn't declared.

## Goals / Non-Goals

**Goals:**
- A student who knows mediaComp can go from a drawing to a prediction in a few visible steps: draw, convert to grayscale (the student's own code), `crop_and_center`, `scale_down`, `predict`.
- picoface's generated images open in mediaComp.
- picoface never imports or depends on mediaComp.
- The default dataset looks like mediaComp drawings, with model quality no worse than today.

**Non-Goals:**
- Converting pictures to grayscale for students.
- Showing images inside picoface (mediaComp's `show()` does that).
- Anything involving mediaComp's `Sound`.
- Changing model architectures, including the zero padding in the convolution layers.
- Making student drawings match training data statistically, for example by adding noise or shade jitter to them.
- A model save format.

## Decisions

### 1. Duck typing, with no mediaComp import
A picture is any object with a `getImage()` method that returns a Pillow image. A new picture is made with `type(picture)(new_pil_image)`.

Alternatives considered:
- **A hard dependency.** Rejected: it would pull wxPython and a GUI stack into picoface, and it raises GPL questions for an MIT package.
- **An optional `picoface[mediacomp]` extra.** Rejected: CI would still need wxPython to test it, and the licence question remains.
- **Contributing converters upstream to mediaComp.** Rejected for now: we have no relationship with the maintainers.

Duck typing keeps picoface independent of mediaComp's release cycle. The coupling is two touchpoints, `getImage()` and the Pillow-image constructor, which a small fake class can test.

Because mediaComp is on PyPI, the same tests can also run against the real `Picture` class whenever mediaComp is installed. An optional test module skips itself otherwise. It must skip, not fail, on any import error: mediaComp imports tkinter, pygame, and sounddevice at import time, and on a headless Linux machine without the PortAudio library, sounddevice raises `OSError` rather than `ImportError`. mediaComp stays out of `dev` extras, so a plain development install doesn't have to build wxPython.

Verified during exploration with mediaComp 0.4.15: importing it never imports `wx` (only `show()` and `pictureTool()` need it). So the real-mediaComp tests run without wxPython, as long as tkinter and the PortAudio library are present. A test environment was built on the development machine (Debian trixie, no sudo) at `~/.venvs/picoface-mediacomp/`, as follows:
- a venv from the system Python 3.13;
- `tkinter` and `_tkinter` copied in from the `python3.13-tk` .deb;
- `libportaudio2` and `libjack-jackd2-0` unpacked from their .debs with `dpkg -x` into `sysroot/`, plus a `libportaudio.so` symlink, because `ctypes.util.find_library()` looks for the unversioned name;
- torch (CPU), `pip install -e ".[dev]"`, `pygame-ce` and `sounddevice`, then `pip install --no-deps mediaComp`, which skips wxPython.

Its `run` script puts `sysroot/` on `LD_LIBRARY_PATH`. In that environment the existing suite passed, and real `makeEmptyPicture`, `addOvalFilled`, `getPixelAt`, `makePicture`, `getImage()` and `Picture(pil_image)` all worked. `makePicture()` kept an `"L"`-mode PNG in mode `"L"`, which confirms the Context note. On a machine with `sudo`, `apt install python3-tk libportaudio2` followed by `pip install --no-deps mediaComp pygame-ce sounddevice` does the same.

### 2. Helpers return a new image of the kind they were given
Given a picture, the helpers return a new picture of the same type, always in RGB mode because mediaComp's pixel functions expect RGB. Given an array, they return an array.

Alternatives considered:
- **Changing the picture in place through `setImage()`.** This is how mediaComp's own `addRect` works. Rejected: the instructor wants helpers that return values, and in-place changes make "compare before and after" exercises awkward.
- **Always returning arrays.** Rejected: students would lose `show()` and `pictureTool()` on the result.

Internally every helper works on a uint8 H×W×C numpy array (C = 1 or 3) and converts only at the edges: picture to array on the way in, array to the input's kind on the way out. The helpers read a picture as H×W×3 RGB with no grayscale check, after Pillow's `convert("RGB")`, so pictures in `"L"`, `"P"`, or `"RGBA"` mode work too (alpha is ignored). An H×W array gets a channel axis on the way in, and the result is checked with `dataset-validation`'s `check_image_array(..., batch=False)`. Only `picture_to_array()`, and `predict()` through it, use the strict grayscale reader (Decision 3).

The helpers accept colour so that preparation steps commute. A student's own grayscale function can run before or after `crop_and_center()` and `scale_down()`: `scale_down(make_greyscale(pic))` and `make_greyscale(scale_down(pic))` both work. The two orders agree up to rounding, because averaging and a channel-mean grayscale are both linear. They are not guaranteed to be bit-identical, and the tests allow for that. Grayscale is required only at the model boundary, where the image becomes H×W×1. The shared conversion lives in a new internal module, `_internals/picture_internals.py`, which both `picoface.pictures` and `model_api.predict()` import. `predict()` converts a picture before its image-array check (`_check_single_image()`, from `dataset-validation`), so a converted picture passes the same checks as an array, and both that check and `_preprocess_images()` stay unchanged and strict.

### 3. Grayscale is strictly checked, never converted
`picture_to_array()` requires R == G == B at every pixel and reads one channel. Students' mediaComp grayscale loops set all three channels to the same integer. Pillow's drawing functions keep gray colours gray, and so does `save_images()` output. An exact check therefore fits how pictures are really made. The error message names the fix ("convert it to grayscale first") without doing it.

Alternative considered: a tolerance, such as a maximum channel spread of 2. Deferred; see Open Questions.

### 4. `crop_and_center` estimates the figure's circumscribed circle
The steps:
1. **Background:** the per-channel median of the outermost rows and columns.
2. **Figure:** every pixel whose brightness (the mean of its channels) differs from the background's brightness by more than a threshold. A starting value is 48, half the Forge's `min_contrast`. It rejects the Forge's noise level (σ = 6) and faint scanning artifacts. For a grayscale image this is simply the pixel's difference from the background shade.
3. **Centre and radius:** the smallest circle holding every figure pixel, found with Welzl's algorithm. Only each row's outermost figure pixels can touch that circle, so only those are searched, and the radius grows by half a pixel to cover the pixels themselves.
4. **Canvas:** the side is `ceil(2·radius / 0.65)`, never smaller than needed to hold the figure unshrunk. It is filled with the background colour, with the circle's centre at the canvas centre.

The circle is used because the Forge sizes every class by its circumscribed radius (`radius_fraction` = 0.65 of half the side). Framing by bounding box instead would make triangles and stars larger than their training counterparts. The exact smallest circle is needed, not an approximation: an earlier draft took the bounding box's centre and its farthest figure pixel, which overestimates an upright triangle's radius by about 14% and so framed triangles about 12% smaller than circles of the same size (found while implementing task 3.4). Because the background comes from the border and the figure from the size of the difference, the frame works for either polarity.

Brightness is the channel mean, not the largest single-channel difference, so that `crop_and_center()` finds the same figure whether the student converts to grayscale before or after it (Decision 2), assuming the usual average-of-channels grayscale. The cost: a figure with the same brightness as its background, such as mid-red on mid-green, isn't found. It wouldn't be found after a grayscale conversion either, so the two orders at least fail the same way. The largest-channel difference finds such figures, but makes the result depend on the order of the steps.

The constant 0.65 duplicates the Forge's default `radius_fraction`, because picoface can't import the Forge. A Forge test (the Forge may import picoface) asserts that the two match, so they can't drift apart unnoticed.

### 5. `scale_down` uses Pillow's box filter
`Image.resize((size, size), Image.Resampling.BOX)` makes each output pixel the area-weighted mean of the input pixels it covers, separately in each channel. Every channel of a gray RGB picture goes through the same arithmetic, so a gray picture stays exactly gray (R == G == B) and still passes `picture_to_array()`. That is the same averaging the Forge uses to anti-alias its supersampled canvas, and it handles size ratios that aren't whole numbers.

Alternatives considered:
- **NEAREST.** Thin outlines vanish.
- **BILINEAR or BICUBIC without reduction.** These alias when shrinking by large factors.
- **LANCZOS.** Its overshoot and ringing leave halos around crisp student edges that training images never show.

Non-square input is rejected rather than padded. Squaring the image is `crop_and_center`'s visible job, and the error points there. Input smaller than `size` is rejected, because enlarging isn't "scaling down".

### 6. Pillow becomes a declared dependency
`picoface.pictures` imports Pillow directly, so `pyproject.toml` declares `pillow>=10` rather than relying on matplotlib bringing it in. The Forge keeps `Pillow==12.3.0`. `test_packaging.py` changes from "pillow is not declared" to "no dependency in `pyproject.toml` carries an exact pin taken from `dataset_forge/requirements.txt`". The `dataset-forge` isolation requirement is reworded to match: shared libraries are allowed, the Forge's pins are not (see the spec delta).

### 7. `save_images` writes RGB PNGs
Each image is written as an 8-bit RGB PNG, enlarged by a whole-number `scale` with nearest-neighbour resampling (every pixel becomes a solid block), and named `image_000.png`, `image_001.png`, and so on, zero-padded to the batch size. The PNGs must be RGB because `makePicture()` keeps the file's mode, and mediaComp's pixel functions need RGB (Context). A round trip back through `scale_down` is exact, because averaging a solid block returns its value. The returned paths are absolute: mediaComp 0.4.15's `makePicture()` puts its media folder in front of a relative path, and fails with a `NameError` if none has been set. Task 4.1's README test found this.

### 8. The Forge flips polarity at the shade-sampling step
Only the Forge's shade sampling changes:
- `sample_params()` draws `background ~ U(low, high)`, then `foreground ~ U(0, background − min_contrast)`.
- The default `background_range` becomes (96, 255).
- `ForgeConfig` validation checks that `low − min_contrast ≥ 0`, with a matching error message.
- Draw functions, coverage, noise, and the data contract are untouched.

This gives exactly the distribution the exploration spike tested (`255 − x` applied to the current default export). Mirroring a uniform range gives a uniform range, and noise with zero mean is symmetric.

`validate.ink_fraction()` counts pixels *below* the image's percentile midpoint. The shortcut-check results can't change, because 1 − f ranks images in the same order. The stub dataset's shade constants swap (background 192, figure 64), to keep development data consistent with real data.

Alternatives considered:
- **Inverting the finished image** in `shade()`. Rejected: the config's shade ranges would then describe the opposite of the saved images.
- **A `polarity` config switch.** Rejected: it adds a second mode to test for no current need. Old exports stay reproducible from their commit.
- **Mirroring inside the models** (training on data flipped back internally). Rejected after the spike showed no need for it.

Exploration spike (3 training seeds, picoface defaults, `default-2x-seed0` against its `255 − x` copy):

| Variant | CNN acc | VAE acc | `classify_generated` | Reconstruction |
|---|---|---|---|---|
| current polarity | 0.987 ± .006 | 0.989 ± .004 | 0.80 ± .03 | 0.81 ± .03 |
| flipped | 0.992 ± .001 | 0.987 ± .006 | 0.85 ± .02 | 0.81 ± .02 |
| flipped + first-conv padding with 255 | 0.993 ± .002 | 0.984 ± .012 | 0.84 ± .03 | 0.81 ± .04 |

For seed 0, the flipped model's generated grid was almost a pixel-for-pixel negative of the current one. The per-class numbers (20 images per class per seed) vary by about ±0.1 between seeds, so the flip is "no worse", not proven better.

### 9. Zero padding in the convolution layers stays unchanged for now
After a flip, zero padding (black) no longer matches the background (white). The spike measured no effect, including when the first layer was padded with 255. Changing it is recorded in `ROADMAP.md` as a possible future change. Replicate padding is the preferred option if it's ever needed: it works for either polarity, because the Forge keeps the outer rows and columns free of ink.

### 10. A smoke-check diagnostic for `activation_maximize()`
`activation_maximize()` periodically blurs its image, using `F.conv2d` with zero padding (`linkage_internals.py:133`). On a light-background model, that could darken the image edges. The diagnostic:
- **Figure:** `dataset_forge.smoke` writes `activation_maximize.png`, with one row per class and a few random starts each from the CNN and then the VAE, enlarged for viewing.
- **Measure:** it reports each class's *edge darkening*: the mean of the outermost 2-pixel ring minus the mean of the 2-pixel ring inside it. A strongly negative value means the edges are darker than just inside them.
- **Comparison:** it runs on both the flipped export and the current-polarity export, so a darkening caused by the flip shows up as a difference, not as an absolute number that is hard to judge.

The results and figures go in this change's `diagnostics.md`. The instructor then inspects them and decides (tasks.md gate). If a fix is wanted, the candidate is replicate padding for the blur only, a one-line internal change. The decision is recorded either way.

## Risks / Trade-offs

- **[Student drawings still don't look like training data** (crisp, one flat shade, no noise), so predictions may still be wrong.] → The README treats a wrong prediction as something to investigate, not a bug. Both helpers produce images inside the training distribution for size, framing, polarity, and antialiasing, which are the largest gaps.
- **[The `crop_and_center` threshold picks up stray marks** (a signature, a stray line), which inflates the circle.] → The error for a blank image and the README tell students to draw one figure. The threshold is a module constant that can be tuned without a spec change.
- **[The two preparation orders differ slightly]**: grayscale-then-scale and scale-then-grayscale can differ by rounding, and a student's weighted grayscale (e.g. by luminance) can shift `crop_and_center()`'s threshold decisions at the figure's edges. → Both orders give valid, model-sized images; the README says the order is free but grayscale must come before `predict()`. Tests compare the orders with a tolerance.
- **[The exact grayscale check rejects pictures loaded from JPEG photos]**, whose colour compression leaves small differences between channels. → The error message says what to do (run the grayscale conversion). A tolerance can be added later (Open Questions).
- **[mediaComp changes its `Picture` constructor or `getImage()`** (it is at version 0.4.x).] → Only those two touchpoints are used. The fake class in the tests documents the assumed interface. An optional test against the real PyPI release runs wherever mediaComp is installed, and a manual check on Windows is a task.
- **[picoface's 0.65 framing constant drifts from the Forge's default]** → A Forge test asserts they are equal (Decision 4).
- **[Old exports and models trained on them have the old polarity]** → Nothing ships them to students yet. Each export's manifest records its `background_range`, which tells old and new apart. Regenerate `dataset_forge/output/`.
- **[`save_images` overwrites files of the same name]** → The behaviour is documented. Students choose the folder.

## Migration Plan

1. Land the Forge flip and regenerate the default export (seed 0).
2. Re-run the smoke check over 3 seeds, and update README "What to expect" and `diagnostics.md`.
3. Run the `activation_maximize` diagnostic, followed by the instructor's decision.
4. Land `picoface.pictures`, the `predict()` change, and the Pillow dependency.

Rollback is `git revert`. Exports aren't committed, so re-exporting from the reverted commit restores the old polarity.

## Open Questions

- Should the grayscale check allow a small spread between channels, for example for photos loaded from JPEG? This can be decided after classroom use without changing the approach.
- Should `save_images` name files by class when given a `classify_generated()` report (e.g. `03_star_seen_as_circle.png`, for the collage exercise)? This is an additive option that can come later.
