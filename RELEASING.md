# Releasing picoface

This is the maintainer-only runbook for publishing `picoface` to PyPI. It's
separate from `README.md` because it's not something a student ever needs to
read — see `openspec/ROADMAP.md` for why PyPI is the chosen target channel.

Nothing here runs automatically. Every step below is a manual command you
run yourself, with your own PyPI account and credentials.

## 1. Verify the name is actually free on PyPI

Do this **before anything else below**. This project already renamed itself
once (`tinyface` → `picoface`) after discovering a name collision on PyPI, so
don't assume `picoface` is still available — check first:

```bash
pip index versions picoface
# or just visit https://pypi.org/project/picoface/
```

If the name is taken, stop here and pick a new distribution name (the import
name and repo name don't have to match the PyPI name, but keeping them
aligned, as this project already does, is worth preserving if possible).

## 2. Create a PyPI account and an API token

Register at [pypi.org](https://pypi.org) if you don't already have an
account. Once registered, create an API token (Account settings → API
tokens) scoped to the `picoface` project once it exists, or to your whole
account for the very first upload. **Use a token, not your account
password** — `twine` supports token auth natively, and a leaked token is
easy to revoke without touching your login credentials.

Store the token somewhere `twine` can read it, e.g. in `~/.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-...your-token...
```

## 3. (Recommended) Dry run on TestPyPI first

TestPyPI is a separate instance for exactly this purpose — it lets you
rehearse the full upload and install flow without touching the real index or
burning a real version number if something's wrong.

```bash
pip install -e ".[dev]"
python -m build
twine upload --repository testpypi dist/*
```

This needs its own TestPyPI account and token (they are not shared with
real PyPI) — register at
[test.pypi.org](https://test.pypi.org) and add a `[testpypi]` section to
`~/.pypirc` the same way as step 2.

Then verify the install actually works from TestPyPI in a clean virtual
environment:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ picoface
python -c "import picoface"
```

The `--extra-index-url` is needed because TestPyPI doesn't mirror PyPI's
other packages (torch, numpy, matplotlib) — only `picoface` itself comes
from TestPyPI in this command.

## 4. Build

From a clean repo root (no stale `dist/` from a previous build):

```bash
rm -rf dist build src/picoface.egg-info
python -m build
twine check dist/*
```

`twine check` should report both the sdist and the wheel as `PASSED` before
you proceed.

## 5. Upload to PyPI

```bash
twine upload dist/*
```

This is the actual, irreversible publish — PyPI does not allow re-uploading
a version number once it's taken, even if you delete the release, so double
check `pyproject.toml`'s `version` first.

## 6. Versioning and tagging

Bump `version` in `pyproject.toml` for every release using
[semantic versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`). After a
successful upload, tag the commit that was built:

```bash
git tag v<version>
git push origin v<version>
```

There is no CI/release-automation workflow (no publish-on-tag GitHub
Action) — that's a deliberate choice to keep the release process a manual,
reviewed action rather than something that fires automatically on a push.

## A note on the CPU-only torch install

`pip install picoface` (from PyPI, once published) will **not** force the
CPU-only torch wheel — PyPI's default `torch` build pulls in CUDA-enabled
binaries regardless of which index the rest of the install comes from.
The two-step install order documented in `README.md`'s Install section
(install the CPU wheel first, then install `picoface`) remains necessary
for every install channel, including a future PyPI release. There is no way
to bake a CPU-only dependency constraint into `picoface`'s own PyPI metadata
that overrides this.
