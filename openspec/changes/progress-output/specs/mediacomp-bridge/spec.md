## MODIFIED Requirements

### Requirement: Saving picoface images for mediaComp
The system SHALL provide `save_images(images, folder, scale=8, verbose=True)`. It writes each image as an RGB PNG file in `folder`, creating the folder if needed, and returns the written file paths in image order. It SHALL accept one image or a batch of images in picoface's image format, which covers the output of `generate()`, `activation_maximize()`, and a `classify_generated()` report's images. Each image SHALL be enlarged by the whole-number factor `scale` with no blending between pixels, so every original pixel becomes a solid `scale`×`scale` block. `verbose` controls only what `save_images()` prints (Requirement: Output is on by default and can be turned off per call, in `progress-output`); it SHALL NOT affect the files written or the paths returned.

#### Scenario: Generated images open in mediaComp
- **WHEN** a student saves the output of `generate(model, 8)` and opens the first returned path with mediaComp's `makePicture()`
- **THEN** the result SHALL be a 224×224 RGB picture (for 28×28 images at the default scale) that mediaComp can show and inspect

#### Scenario: Saving and re-preparing round-trips
- **WHEN** a saved image is loaded as a picture, passed through `scale_down()` to its original size, and converted with `picture_to_array()`
- **THEN** the result SHALL equal the original image

#### Scenario: Paths come back in order
- **WHEN** a student saves a batch of `n` images
- **THEN** `save_images()` SHALL return `n` paths, the `i`-th holding the `i`-th image
