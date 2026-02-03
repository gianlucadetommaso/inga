"""Pre-commit script that runs format, lint fix, type check, and tests."""

import subprocess
import sys


def check() -> None:
    """Run all pre-commit steps: format, lint fix, type check, and tests."""
    commands = [
        (["ruff", "format", "."], "Format"),
        (["ruff", "check", "--fix", "."], "Lint fix"),
        (["mypy", "steindag", "tests"], "Type check"),
        (["python", "-m", "pytest", "tests", "-v"], "Tests"),
    ]

    for cmd, name in commands:
        print(f"\n{'=' * 60}")
        print(f"Running: {name}")
        print(f"{'=' * 60}\n")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n❌ {name} failed!")
            sys.exit(result.returncode)

    print(f"\n{'=' * 60}")
    print("✅ All checks passed!")
    print(f"{'=' * 60}\n")
