"""
Console entry point for the AML bandits example runner.

No GoF pattern applies here: this module is a thin command-line shim that delegates to the
repository example script so the packaged `aml-bandits` entry point stays usable.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    """Run the repository example script."""
    example_path = Path(__file__).resolve().parents[2] / "examples" / "run_bandits.py"
    if not example_path.exists():
        raise FileNotFoundError(f"Example runner not found: {example_path}")
    runpy.run_path(str(example_path), run_name="__main__")
