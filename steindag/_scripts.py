"""Development scripts for formatting, linting, and testing."""

import subprocess
import sys

# Available commands and their implementations
COMMANDS: dict[str, tuple[list[str], str]] = {
    "fmt": (["ruff", "format", "."], "Format"),
    "lint": (["ruff", "check", "--fix", "."], "Lint fix"),
    "typecheck": (["mypy", "steindag", "tests"], "Type check"),
    "test": (["python", "-m", "pytest", "tests", "-v"], "Tests"),
}


def _run_commands(names: list[str]) -> None:
    """Run the specified commands in order.

    Args:
        names: List of command names to run (e.g., ["fmt", "lint", "test"]).
    """
    for name in names:
        if name not in COMMANDS:
            print(f"ERROR: Unknown command: {name}")
            print(f"Available commands: {', '.join(COMMANDS.keys())}")
            sys.exit(1)

        cmd, description = COMMANDS[name]
        print(f"\n{'=' * 60}")
        print(f"Running: {description}")
        print(f"{'=' * 60}\n")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nERROR: {description} failed!")
            sys.exit(result.returncode)

    print(f"\n{'=' * 60}")
    print("All steps completed!")
    print(f"{'=' * 60}\n")


def dev() -> None:
    """Run development commands specified as arguments.

    Usage:
        uv run dev fmt          # Format only
        uv run dev lint         # Lint only
        uv run dev test         # Test only
        uv run dev fmt lint     # Format and lint
        uv run dev fmt lint test  # Format, lint, and test
        uv run dev              # With no args, runs all: fmt, lint, typecheck, test

    Available commands: fmt, lint, typecheck, test
    """
    args = sys.argv[1:]

    if not args:
        # No arguments: run all commands
        _run_commands(["fmt", "lint", "typecheck", "test"])
    else:
        _run_commands(args)
