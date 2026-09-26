## Why

The students who use `picoface` will already have learned mediaComp, the Python 3 port of Guzdial and Ericson's Media Computation library. They work in Thonny on Windows. From mediaComp they know how to draw on a `Picture`, transform its pixels, and view it with `show()`. `picoface` currently ignores all of that:
- Its images are numpy arrays that mediaComp can't open.
- It has no image viewer.
- Its dataset is the photographic negative of what a mediaComp drawing looks like: a light figure on a dark background, where mediaComp's defaults draw black on white.

Connecting the two libraries lets students test their trained models on pictures they drew themselves, and view and edit generated images with tools they already know.

## What Changes

- **BREAKING (dataset content): Dataset Forge renders dark figures on light backgrounds.** The figure shade is always darker than the background by at least the configured minimum contrast. The default background range moves from 0–159 to 96–255. Every class flips together, including `negative_smiley`: it is defined by its shape (a filled disc with the face cut out), not by polarity, so it needs no exception. Earlier exports stay valid data-contract files, but models trained on them expect the old polarity.
  - An exploration spike trained on a flipped copy of the current default export: 3 seeds, picoface defaults, no model changes. Results matched or beat the current polarity on every measure. Mean CNN held-out accuracy was 0.992 vs 0.987, VAE 0.987 vs 0.989, `classify_generated()` agreement 0.85 vs 0.80, reconstruction agreement 0.81 vs 0.81. Training times were unchanged. `design.md` records the numbers.
  - The Forge's ink-fraction measure, config validation, tests, and docs follow the new polarity. The internal stub dataset is flipped to match.
- **New `picoface.pictures` module** connecting mediaComp and picoface without importing mediaComp. It works with any object that behaves like a mediaComp `Picture`: one with `getImage()` that returns a Pillow image, and a constructor that accepts one.
  - `crop_and_center(picture)`: crops to the figure and centres it on a square canvas of background colour, framed the way Forge figures are framed.
  - `scale_down(picture, size=28)`: shrinks the image to `size`×`size` with area-averaging antialiasing, the same averaging the Forge uses when it renders.
  - `picture_to_array(picture)`: converts to picoface's image format (uint8 H×W×1). If the picture isn't grayscale, it raises a clear error telling the student to convert it first. Grayscale conversion stays the student's job, done with the pixel loops they learned in mediaComp.
  - `save_images(images, folder, scale=8)`: writes picoface images as upscaled RGB PNGs and returns their paths, ready for mediaComp's `makePicture()`.
  - The helpers return a **new** Picture of the same kind when given a Picture, and an array when given an array. Students' originals are never modified.
  - `crop_and_center()` and `scale_down()` work on colour and grayscale images alike. Students can therefore run their own grayscale conversion before or after either helper: `scale_down(make_greyscale(pic))` and `make_greyscale(scale_down(pic))` both work. Only `picture_to_array()` and `predict()` require grayscale.
- **`predict()` accepts a Picture** as well as an array, converting it with `picture_to_array()`. It still requires the picture to be grayscale and the model's size, so the preprocessing steps stay visible to students.
- **Pillow becomes a declared picoface dependency** (a minimum version, not a pin). picoface already receives it through matplotlib; it is now imported directly. The Forge keeps its own exact pin.
- **A diagnostic check on `activation_maximize()`.** The Forge's smoke check also renders an `activation_maximize()` image per class for both models and reports a border-darkening measure. The concern: its blur step pads with zeros, i.e. black, which on light backgrounds could darken the image edges. The instructor inspects the figure visually and decides whether a follow-up is needed.
- **Documentation:**
  - a "Using picoface with mediaComp" README section. It says that mediaComp is installed separately from PyPI (`pip install mediaComp`), and gives notes for Thonny on Windows;
  - re-measured "What to expect" numbers;
  - a note in `ROADMAP.md` recording zero padding in the conv layers as a possible future change, not changed here.

## Capabilities

### New Capabilities
- `mediacomp-bridge`: moving images between mediaComp `Picture` objects (or anything that behaves like one) and picoface's image arrays. Covers the student-facing crop-and-centre and scale-down helpers, grayscale checking, and saving picoface images as PNGs that mediaComp can open.

### Modified Capabilities
- `dataset-forge`: figure polarity reverses (dark figure on a light background). The isolation requirement is clarified: a library both sides use, such as numpy or Pillow, may appear in both dependency lists, but the Forge's exact pins never apply to picoface.
- `model-interface`: `predict()` accepts a Picture-like object as well as an image array.

## Impact

- `dataset_forge/`: `config.py` (default `background_range`, the contrast check and its message, the docstring), `render.py` (figure shade drawn below the background), `validate.py` (`ink_fraction` counts dark pixels), `smoke.py` (new `activation_maximize()` figure and border measure), `README.md`, `configs/default.json`, and the tests that fix polarity (`test_render.py`, `test_validate.py`, `test_cli.py`, `test_config.py`).
- `src/picoface/`: new `pictures.py`; `predict()` in `_internals/model_api.py` accepts Pictures; `_internals/stub_data.py` shade constants flipped.
- `pyproject.toml`: adds `pillow` to `dependencies`. `tests/test_packaging.py`'s Pillow check becomes a check that picoface does not carry the Forge's exact pins.
- New tests for `picoface.pictures`, using a small fake Picture class. An optional test module also runs them against the real PyPI mediaComp when it is installed, and skips itself otherwise. mediaComp, wxPython, and a display are never required in CI.
- Docs: `README.md` (mediaComp section, re-measured metrics), `openspec/ROADMAP.md` (mediaComp integration and the zero-padding note).
- Unchanged: no mediaComp import or dependency anywhere; the data contract format; model architectures and training defaults. Existing exports under `dataset_forge/output/` should be regenerated.
