"""Render, export, and validate a dataset.

    python -m dataset_forge [--config PATH] [--seed N] [--out DIR]

Defaults to `configs/default.json`, seed 0, and
`output/<config name>-seed<N>/`. Exits non-zero if the config is invalid or
any validation check fails; a failed export's folder is kept for inspection,
with the failure recorded in its manifest.
"""

import argparse
import sys
from pathlib import Path

from dataset_forge.config import ForgeConfig
from dataset_forge.export import DEFAULT_CONFIG, export
from dataset_forge.validate import validate_and_record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dataset_forge", description="Render, export, and validate a dataset."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        config = ForgeConfig.load(args.config)
        out_dir = export(config, seed=args.seed, out_dir=args.out)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"Exported to {out_dir}")
    report = validate_and_record(out_dir)
    print(report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
