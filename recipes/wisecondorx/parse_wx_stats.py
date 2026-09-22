#!/usr/bin/env python3
"""Parse WisecondorX per-chromosome z-scores. Stdlib only.

Looks for v1.2.x ``*_statistics.txt`` / ``*_chr_statistics.txt`` and the
current ``*_statistics.bed`` name. Prints chr 21 / 18 / 13 z and the
published synthetic T21 annotation. This is a pipeline smoke test, not a
clinical call.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CHRS = ("21", "18", "13")
PUBLISHED_T21_Z = "4.7–6.5"
STAT_GLOBS = (
    "*_statistics.bed",
    "*_statistics.txt",
    "*_chr_statistics.bed",
    "*_chr_statistics.txt",
    "*statistics.bed",
    "*statistics.txt",
)


def _norm_chr(token: str) -> str:
    token = token.strip()
    if token.lower().startswith("chr"):
        token = token[3:]
    return token.upper() if token.upper() in {"X", "Y"} else token


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def find_stats_file(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"no such file or directory: {path}")
    matches: list[Path] = []
    for pattern in STAT_GLOBS:
        matches.extend(sorted(path.glob(pattern)))
        matches.extend(sorted(path.glob(f"**/{pattern}")))
    # Unique, prefer names that look like chromosome statistics
    seen: list[Path] = []
    for item in matches:
        if item not in seen:
            seen.append(item)
    if not seen:
        raise FileNotFoundError(
            f"no *_statistics.bed / *_statistics.txt under {path}"
        )
    preferred = [
        p
        for p in seen
        if "statistics" in p.name.lower() and "bin" not in p.name.lower()
    ]
    return preferred[0] if preferred else seen[0]


def parse_zscores(path: Path) -> dict[str, float]:
    """Return chr -> zscore from a WisecondorX statistics table."""
    zscores: dict[str, float] = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    z_col = 3  # default: chr ratio.mean ratio.median zscore
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if not parts:
            continue
        headerish = [p.lower() for p in parts]
        if headerish[0] in {"chr", "chrom", "chromosome"}:
            for i, name in enumerate(headerish):
                if name in {"zscore", "z", "z-score", "z_score"}:
                    z_col = i
            continue
        chrom = _norm_chr(parts[0])
        if not chrom or not (chrom.isdigit() or chrom in {"X", "Y"}):
            continue
        if len(parts) <= z_col or not _is_float(parts[z_col]):
            # Fall back to last numeric field
            nums = [p for p in parts[1:] if _is_float(p)]
            if not nums:
                continue
            zscores[chrom] = float(nums[-1])
            continue
        zscores[chrom] = float(parts[z_col])
    return zscores


def format_report(zscores: dict[str, float], stats_path: Path) -> str:
    lines = [
        f"WisecondorX statistics: {stats_path}",
        f"{'chr':<6}{'z':>10}",
    ]
    for chrom in CHRS:
        if chrom in zscores:
            lines.append(f"{chrom:<6}{zscores[chrom]:>10.3f}")
        else:
            lines.append(f"{chrom:<6}{'n/a':>10}")
    lines.extend(
        [
            "",
            f"Published synthetic T21 z at 8% FF was {PUBLISHED_T21_Z}.",
            "This run is a pipeline smoke test, not a clinical call.",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print chr 21 / 18 / 13 z from WisecondorX statistics."
    )
    parser.add_argument(
        "path",
        help="Path to *_statistics.bed / *_statistics.txt, or a directory",
    )
    args = parser.parse_args(argv)
    try:
        stats_path = find_stats_file(Path(args.path))
        zscores = parse_zscores(stats_path)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_report(zscores, stats_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
