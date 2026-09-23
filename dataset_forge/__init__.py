"""Dataset Forge: renders the course's real shape dataset (instructor-only).

Procedurally draws one grayscale figure per image, with randomized rotation,
size, position, shades, and noise, and exports seeded, class-balanced train
and test splits in picoface's data-contract format. Run from the repo root
with `python -m dataset_forge`; see README.md.

Not part of the installable `picoface` package: the Forge may import
picoface, but picoface never imports the Forge.
"""
