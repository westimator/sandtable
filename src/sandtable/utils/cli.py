"""
src/sandtable/utils/cli.py

ANSI escape codes and helpers for terminal output.
"""

# ANSI escape codes
BOLD = "\033[1m"
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


def color_value(val: float, fmt: str, *, invert: bool = False) -> str:
    """Color a formatted value green (positive) or red (negative).

    Args:
        val: The numeric value to determine color from.
        fmt: The already-formatted string to wrap.
        invert: If True, flip the color logic (e.g. for drawdown).
    """
    positive = val > 0
    if invert:
        positive = not positive
    color = GREEN if positive else RED if val != 0 else ""
    reset = RESET if color else ""
    return f"{color}{fmt}{reset}"
