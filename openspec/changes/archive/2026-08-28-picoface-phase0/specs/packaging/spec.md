## ADDED Requirements

### Requirement: Consistent naming across repo, import, and distribution
The project's GitHub repository name, its Python import name, and its PyPI distribution name SHALL all be `picoface`.

#### Scenario: Name consistency check
- **WHEN** a student installs the package and imports it
- **THEN** the command they used to install it and the name they use in `import picoface` SHALL be textually identical (aside from install-command syntax), with no separate "install name" to remember

### Requirement: Installable via multiple channels
The package SHALL be structured (via `pyproject.toml` and a standard src layout) so it can be installed via `pip install` from PyPI, from a local path or zip archive, or via a Colab cell cloning the GitHub repository, without requiring different code for each channel.

#### Scenario: Same package installs via three channels
- **WHEN** the package is installed via PyPI, via a local zip/folder, or via a Colab git-clone-then-pip-install-e flow
- **THEN** all three SHALL result in a working `import picoface` using the same package source

### Requirement: Dataset Forge excluded from the installable package
Dataset Forge SHALL live outside the installable `picoface` package (a separate top-level directory) so that installing `picoface` never pulls in Dataset Forge's unrestricted/heavier dependencies.

#### Scenario: Installing picoface does not install Dataset Forge
- **WHEN** a student runs `pip install picoface` (via any supported channel)
- **THEN** no Dataset Forge code or dependencies SHALL be included in the installed package
