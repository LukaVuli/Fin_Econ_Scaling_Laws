"""Size parsing and human-friendly parameter-count formatting helpers."""

from __future__ import annotations

from typing import Union


def parse_size(s: Union[str, int]) -> int:
    """Convert size string to integer."""
    if isinstance(s, int):
        return s
    s = str(s).upper()
    if 'B' in s:
        return int(float(s.replace('B', '')) * 1_000_000_000)
    if 'M' in s:
        return int(float(s.replace('M', '')) * 1_000_000)
    elif 'K' in s:
        return int(float(s.replace('K', '')) * 1_000)
    return int(s)


def format_params(n: int) -> str:
    """Format parameter count as human-readable string."""
    if n < 1_000:
        return str(n)
    if n < 1_000_000:
        k = n / 1_000
        return f"{k:g}K"
    if n < 1_000_000_000:
        m = n / 1_000_000
        return f"{m:g}M"
    b = n / 1_000_000_000
    return f"{b:g}B"
