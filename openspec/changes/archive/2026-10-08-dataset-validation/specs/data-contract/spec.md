## MODIFIED Requirements

### Requirement: Dataset interchange format
The system SHALL define a dataset interchange format consisting of an `.npz` file containing an `images` array (uint8, shape N×H×W×C, with N ≥ 1 and C either 1 for grayscale or 3 for RGB) and a `labels` array (integer, shape N, every value a valid class index), accompanied by a `classes.json` file in the same directory. `classes.json` SHALL be a JSON object whose keys are exactly the strings `"0"` through `"K-1"` for some K ≥ 2, each mapped to a distinct, non-empty class-name string.

#### Scenario: Loading a well-formed dataset bundle
- **WHEN** a dataset bundle conforming to the schema is loaded
- **THEN** `load_dataset()` SHALL return images, labels, and class names without requiring the caller to specify image dimensions or class count in advance

#### Scenario: Grayscale images match the mediaComp bridge's image format
- **WHEN** a bundle's images have shape N×H×W×1
- **THEN** each image SHALL have the same H×W×1 uint8 form that the `mediacomp-bridge` capability's `picture_to_array()` produces

### Requirement: Format is size- and class-agnostic
The dataset interchange format SHALL NOT hardcode image resolution or number of classes; these SHALL be inferred from the bundle's contents at load time. Color depth SHALL be inferred as well, from the two the format allows: grayscale (C = 1) or RGB (C = 3).

#### Scenario: Two datasets with different shapes both load correctly
- **WHEN** two dataset bundles with different image resolutions, class counts, or color depths (grayscale or RGB) are each loaded with `load_dataset()`
- **THEN** both SHALL load successfully without any code changes to `load_dataset()` or to the calling code

### Requirement: Shared dataset return type
The system SHALL define a single, immutable dataset value type (numpy-only fields: `images`, `labels`, `class_names`; no framework-specific types such as `torch` tensors) that `load_dataset()` returns, so that all consumers of dataset data share one contract regardless of framework choices made in later phases.

The type SHALL enforce the interchange format's array rules whenever a value is created, however it is created, so that every dataset value in the system is known to be well-formed. A violation SHALL raise the dataset error type (Requirement: Malformed datasets are rejected with explanatory errors).

The type SHALL expose `height`, `width`, and `num_classes` as read-only attributes derived from its images and class names. They SHALL always agree with `images` and `class_names`.

#### Scenario: Inspecting a loaded dataset does not dump raw pixel data
- **WHEN** a loaded dataset value is printed or repr'd (e.g. at a REPL or in a notebook)
- **THEN** the output SHALL show array shapes and dtypes (e.g. image count, dimensions, class list) rather than raw array contents, and the labels shape shown SHALL be the labels array's own shape

#### Scenario: Reading image size and class count
- **WHEN** a student loads a bundle of 28×28 grayscale images with three classes
- **THEN** `data.height` SHALL be 28, `data.width` SHALL be 28, and `data.num_classes` SHALL be 3

#### Scenario: A dataset built directly is checked too
- **WHEN** a dataset value is constructed directly, not through `load_dataset()`, from images of shape N×H×W with no channel axis
- **THEN** construction SHALL raise the dataset error type, and no dataset value SHALL be created

## ADDED Requirements

### Requirement: Malformed datasets are rejected with explanatory errors
The system SHALL provide a public dataset error type, a subclass of `ValueError`, importable from the same module as `load_dataset()`. Every rule of the interchange format that a bundle or dataset value breaks SHALL raise this type. The one exception is a missing file, which SHALL raise `FileNotFoundError`. Each message SHALL state what was expected, what was found, and how to fix it. When the problem was found while loading a bundle, the message SHALL name the file it concerns.

#### Scenario: Images without a channel axis
- **WHEN** a bundle's `images` array has shape N×H×W
- **THEN** the error SHALL report the shape found, state that images must be N×H×W×C with C = 1 (grayscale) or 3 (RGB), and show how to add a channel axis

#### Scenario: Unsupported channel count
- **WHEN** a bundle's `images` array has C other than 1 or 3 (for example, 4 for RGBA)
- **THEN** the error SHALL report the channel count found and state that only grayscale (1) or RGB (3) are supported

#### Scenario: Float images
- **WHEN** a bundle's `images` array has a floating-point dtype
- **THEN** the error SHALL state that images must be whole numbers from 0 to 255 (uint8), and SHALL suggest how to convert from a 0-to-1 float range

#### Scenario: Images of different sizes
- **WHEN** a bundle's `images` was saved as an array of separate, differently-sized arrays rather than one regular array
- **THEN** the error SHALL explain that every image must have the same height, width, and channel count

#### Scenario: Missing array in the .npz
- **WHEN** a bundle's `.npz` file does not contain an `images` array or a `labels` array
- **THEN** the error SHALL name the missing array and list the arrays the file does contain

#### Scenario: Labels do not match images
- **WHEN** a bundle's `labels` is not one-dimensional, or its length differs from the number of images
- **THEN** the error SHALL report both shapes and state that there must be exactly one label per image

#### Scenario: Label outside the class range
- **WHEN** a bundle's `labels` contains a value less than 0 or at least the number of classes
- **THEN** the error SHALL report the offending value, how many labels have it, and the valid range given by `classes.json`

#### Scenario: Gaps or extra keys in classes.json
- **WHEN** a bundle's `classes.json` keys are not exactly `"0"` through `"K-1"` (for example `{"0": "a", "2": "b"}`), or it is not a JSON object
- **THEN** the error SHALL name `classes.json`, report the keys (or JSON type) found, and state the expected form

#### Scenario: Duplicate class names
- **WHEN** two entries in a bundle's `classes.json`, or in a dataset value's class names, have the same name
- **THEN** the error SHALL name the duplicated class and the indices that share it

#### Scenario: Missing companion file
- **WHEN** `load_dataset()` is given an `.npz` path with no `classes.json` in the same directory
- **THEN** the system SHALL raise `FileNotFoundError` with a message stating that `classes.json` is expected in the same directory as the `.npz`, naming that directory

#### Scenario: Errors can be caught as ValueError
- **WHEN** a student wraps `load_dataset()` in `except ValueError`
- **THEN** every dataset error SHALL be caught by that handler

### Requirement: Lossless conversion at load time
`load_dataset()` SHALL convert a bundle's arrays to the interchange format's dtypes when, and only when, the conversion loses no information: integer-typed images whose values all lie within 0–255 SHALL become uint8, and integer-typed labels SHALL become the library's standard integer type. `load_dataset()` SHALL NOT convert floating-point arrays, add or remove axes, or rescale values; those cases SHALL be rejected with the dataset error type.

#### Scenario: Integer images within range are accepted
- **WHEN** a bundle's `images` is an int64 array whose values all lie within 0–255
- **THEN** `load_dataset()` SHALL return uint8 images with the same values

#### Scenario: Integer images out of range are rejected
- **WHEN** a bundle's `images` is an integer array containing a value above 255 or below 0
- **THEN** `load_dataset()` SHALL raise the dataset error type, reporting the minimum and maximum values found

### Requirement: Warning for classes with no examples
`load_dataset()` SHALL issue a warning, of a public warning type that subclasses `UserWarning`, when one or more classes listed in `classes.json` have no images in the bundle. The dataset SHALL still load. Dataset values constructed directly (for example, by splitting a loaded dataset) SHALL NOT issue this warning.

#### Scenario: A class with no images
- **WHEN** a bundle's `classes.json` lists classes `a`, `b`, and `c`, and no label in the bundle refers to `c`
- **THEN** `load_dataset()` SHALL return the dataset with all three class names and issue a warning naming `c`
