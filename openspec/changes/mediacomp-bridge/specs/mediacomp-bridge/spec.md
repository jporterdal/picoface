## Purpose

Lets students who already know mediaComp use its `Picture` objects with picoface. A student can draw a picture, prepare it for a trained model in visible steps, classify it, and open picoface's own images in mediaComp. picoface never depends on mediaComp to do this.

## ADDED Requirements

### Requirement: Picture-like objects are recognized without depending on mediaComp
The system SHALL treat as a picture any object that returns a Pillow image from a `getImage()` method and can be constructed from a Pillow image, as mediaComp's `Picture` can. The installable package SHALL NOT import mediaComp or declare it as a dependency, and nothing in this capability SHALL require a display or a GUI toolkit.

#### Scenario: A mediaComp Picture is accepted
- **WHEN** a student passes a mediaComp `Picture` to any function of this capability
- **THEN** the function SHALL accept it as a picture

#### Scenario: picoface installs and imports without mediaComp
- **WHEN** picoface is installed and every public module is imported in an environment without mediaComp
- **THEN** the installation and the imports SHALL succeed, and mediaComp SHALL NOT be installed as a result

#### Scenario: Any object with the same interface is accepted
- **WHEN** a function of this capability is given a stand-in object that is not a mediaComp `Picture` but offers the same `getImage()` method and constructor
- **THEN** the function SHALL treat it exactly as it treats a mediaComp `Picture`

### Requirement: Helpers return the kind of image they were given
`crop_and_center()` and `scale_down()` SHALL each accept a picture or an image array (uint8, H×W, H×W×1, or H×W×3) and return a new image of the same kind. Given a picture, a helper SHALL return a new picture of the same type as its input, in RGB mode so that mediaComp's pixel functions work on it. Given an array, it SHALL return a uint8 array with the input's channel count: H×W×3 for an H×W×3 input, and H×W×1 for an H×W×1 or H×W input. A helper SHALL NOT modify the image it was given.

The helpers SHALL accept colour and grayscale images alike, so that a student can convert a picture to grayscale before or after either helper. Only `picture_to_array()` and `predict()` SHALL require a grayscale picture.

#### Scenario: A picture in gives a picture out
- **WHEN** a student calls `scale_down(pic)` with a mediaComp `Picture`
- **THEN** the result SHALL be a new object of the same type as `pic`, in RGB mode, and `pic` SHALL be unchanged

#### Scenario: An array in gives an array out
- **WHEN** a student calls `crop_and_center(image)` with a uint8 image array
- **THEN** the result SHALL be a uint8 H×W×1 array, and `image` SHALL be unchanged

#### Scenario: A colour array keeps its channels
- **WHEN** a student calls `scale_down(image)` with a square uint8 H×W×3 image array
- **THEN** the result SHALL be a uint8 array of shape (28, 28, 3)

#### Scenario: Grayscale conversion can come before or after scaling
- **WHEN** a student scales down a colour picture and then converts it to grayscale by averaging each pixel's red, green, and blue values, and separately converts the same picture to grayscale first and then scales it down
- **THEN** both orders SHALL succeed, both results SHALL be 28×28 grayscale pictures, and they SHALL differ by no more than rounding at any pixel

### Requirement: Converting a picture to picoface's image format
The system SHALL provide `picture_to_array(picture)`, which returns the picture as a uint8 H×W×1 array in picoface's image format. Grayscale conversion SHALL remain the student's job. A picture whose red, green, and blue values differ at any pixel SHALL be rejected with an error saying the picture must be converted to grayscale first. The function SHALL NOT convert the picture itself.

#### Scenario: A grayscale picture converts
- **WHEN** a student converts a picture whose pixels all have equal red, green, and blue values
- **THEN** the result SHALL be a uint8 array of shape (height, width, 1) holding those gray values, where row `y`, column `x` of the array corresponds to mediaComp's pixel at `(x, y)`

#### Scenario: A color picture is rejected with guidance
- **WHEN** a student converts a picture that has at least one pixel whose red, green, and blue values are not all equal
- **THEN** the system SHALL raise an error stating that the picture must be converted to grayscale first, and SHALL NOT return an array

### Requirement: Cropping and centering a figure
The system SHALL provide `crop_and_center(image)`, which finds the figure in a colour or grayscale image, crops to it, and places it centered on a new square canvas. The background colour SHALL be estimated from the image's outermost rows and columns. The figure SHALL be every pixel whose brightness (the mean of its channels) differs clearly from the background's. The new canvas SHALL be filled with the background colour, and the figure SHALL fill about the same share of it as a figure fills a default Dataset Forge image. The output's side length SHALL be large enough that the figure is not shrunk.

#### Scenario: An off-center drawing is centered
- **WHEN** a student calls `crop_and_center()` on a 200×100 picture with a dark filled circle drawn near its top-left corner on a white background
- **THEN** the result SHALL be square, white outside the circle, with the circle centered, and the circle's diameter SHALL be about the same fraction of the side length as a default Dataset Forge figure's

#### Scenario: A blank image is rejected
- **WHEN** a student calls `crop_and_center()` on an image with no pixel that differs clearly from its background
- **THEN** the system SHALL raise an error saying no figure was found

### Requirement: Scaling a picture down with antialiasing
The system SHALL provide `scale_down(image, size=28)`, which shrinks a square colour or grayscale image to `size`×`size` pixels. Each output pixel SHALL be, in each channel, the area-weighted average of the input pixels it covers, the same antialiasing Dataset Forge uses when rendering. A non-square input SHALL be rejected with an error that points the student to `crop_and_center()`. An input smaller than `size` SHALL be rejected with an error saying it can only be shrunk.

#### Scenario: A large square drawing is shrunk to the model's size
- **WHEN** a student calls `scale_down()` on a 200×200 grayscale picture
- **THEN** the result SHALL be 28×28, and a thin dark outline in the input SHALL survive as a visible gray outline rather than vanishing

#### Scenario: A non-square image is rejected with guidance
- **WHEN** a student calls `scale_down()` on a 200×100 picture
- **THEN** the system SHALL raise an error saying the image must be square, and naming `crop_and_center()` as the way to make it so

### Requirement: Saving picoface images for mediaComp
The system SHALL provide `save_images(images, folder, scale=8)`. It writes each image as an RGB PNG file in `folder`, creating the folder if needed, and returns the written files' absolute paths in image order. It SHALL accept one image or a batch of images in picoface's image format, which covers the output of `generate()`, `activation_maximize()`, and a `classify_generated()` report's images. Each image SHALL be enlarged by the whole-number factor `scale` with no blending between pixels, so every original pixel becomes a solid `scale`×`scale` block.

#### Scenario: Generated images open in mediaComp
- **WHEN** a student saves the output of `generate(model, 8)` and opens the first returned path with mediaComp's `makePicture()`
- **THEN** the result SHALL be a 224×224 RGB picture (for 28×28 images at the default scale) that mediaComp can show and inspect

#### Scenario: Saving and re-preparing round-trips
- **WHEN** a saved image is loaded as a picture, passed through `scale_down()` to its original size, and converted with `picture_to_array()`
- **THEN** the result SHALL equal the original image

#### Scenario: Paths come back in order
- **WHEN** a student saves a batch of `n` images
- **THEN** `save_images()` SHALL return `n` paths, the `i`-th holding the `i`-th image
