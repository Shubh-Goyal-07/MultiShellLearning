"""Run the final ShellMetric study from a single restartable command."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multishell.shellmetric.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
