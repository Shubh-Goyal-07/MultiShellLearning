"""Plot 3D (HTML) and 2D (PNG) embedding spaces for every row of finished studies."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multishell.visualize import visualize_study  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("studies", nargs="+", type=Path, help="study output dirs (artifacts/...)")
    parser.add_argument("--partition", choices=("validation", "test", "all"), default="all")
    parser.add_argument("--method", action="append", help="only these methods (repeatable)")
    parser.add_argument("--seed", action="append", type=int, help="only these seeds (repeatable)")
    parser.add_argument("--max-points", type=int, default=4000, help="points per plot")
    parser.add_argument(
        "--output", type=Path, help="output dir (default: <study>/visualizations; one study only)"
    )
    parser.add_argument(
        "--offline", action="store_true", help="embed plotly.js so 3D files open without internet"
    )
    args = parser.parse_args()
    partitions = ("validation", "test") if args.partition == "all" else (args.partition,)
    if args.output is not None and len(args.studies) > 1:
        parser.error("--output takes a single study")
    for study in args.studies:
        index = visualize_study(
            study,
            args.output,
            partitions=partitions,
            methods=args.method,
            seeds=args.seed,
            max_points=args.max_points,
            offline=args.offline,
        )
        print(f"wrote {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
