"""Fill a Section 5.4 architecture template from finished runs.

Replaces every ``REPLACE_WITH_<SOURCE>_<field>`` placeholder in the template:

- ``PRIMARY``: read from ``<primary run>/provenance.json``;
- ``STAGE_A`` / ``STAGE_B``: read from ``<stage run>/architecture_decision.json``,
  where ``WINNER`` means the decision's ``selected`` candidate.

The filled config is written next to the template (so its ``include`` still
resolves) as ``<template stem>.filled.yaml`` unless ``--output`` is given.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PLACEHOLDER = re.compile(r"REPLACE_WITH_(PRIMARY|STAGE_A|STAGE_B)_(\w+)")
SOURCES = {
    "PRIMARY": ("primary", "provenance.json"),
    "STAGE_A": ("stage_a", "architecture_decision.json"),
    "STAGE_B": ("stage_b", "architecture_decision.json"),
}


def fill(template: str, runs: dict[str, Path | None]) -> str:
    cache: dict[str, dict] = {}

    def lookup(match: re.Match[str]) -> str:
        source, field = match.groups()
        option, filename = SOURCES[source]
        run = runs.get(option)
        if run is None:
            raise SystemExit(f"the template needs --{option.replace('_', '-')} for {match[0]}")
        if source not in cache:
            path = run / filename
            if not path.is_file():
                raise SystemExit(f"{path} does not exist; has that run finished?")
            cache[source] = json.loads(path.read_text(encoding="utf-8"))
        key = "selected" if field == "WINNER" else field
        if key not in cache[source]:
            raise SystemExit(f"{run / filename} has no field {key!r}")
        return str(cache[source][key])

    return PLACEHOLDER.sub(lookup, template)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("template", type=Path)
    parser.add_argument("--primary", type=Path, required=True, help="primary study output dir")
    parser.add_argument("--stage-a", type=Path, help="Stage A study output dir")
    parser.add_argument("--stage-b", type=Path, help="Stage B study output dir")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    runs = {"primary": args.primary, "stage_a": args.stage_a, "stage_b": args.stage_b}
    filled = fill(args.template.read_text(encoding="utf-8"), runs)
    output = args.output or args.template.with_name(f"{args.template.stem}.filled.yaml")
    output.write_text(filled, encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
