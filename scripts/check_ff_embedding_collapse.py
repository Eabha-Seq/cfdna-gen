#!/usr/bin/env python3
"""Check whether a checkpoint's continuous FF embedding is collapsed.

Not run in CI: it needs a real checkpoint (local directory or Hugging Face
id), typically hundreds of MB. Unit tests cover the same math on tiny
random / zeroed heads in tests/test_model.py.

Usage:
    python scripts/check_ff_embedding_collapse.py
    python scripts/check_ff_embedding_collapse.py --model ./models/v15
    python scripts/check_ff_embedding_collapse.py --model eabhaseq/cfdna-gen

Exit status 1 if the continuous path is below the collapse threshold.
See docs/FF_CONDITIONING_FIX.md.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="eabhaseq/cfdna-gen",
        help="Local model directory or Hugging Face repo id (default: eabhaseq/cfdna-gen)",
    )
    parser.add_argument(
        "--ff-low",
        type=float,
        default=0.01,
        help="Low fetal fraction for the continuous-head delta (default: 0.01)",
    )
    parser.add_argument(
        "--ff-high",
        type=float,
        default=0.10,
        help="High fetal fraction for the continuous-head delta (default: 0.10)",
    )
    args = parser.parse_args(argv)

    from cfdna_gen.model import CfDNACausalLM, diagnose_ff_conditioning

    print(f"Loading {args.model} …")
    model = CfDNACausalLM.from_pretrained(args.model, device="cpu")
    report = diagnose_ff_conditioning(model, ff_low=args.ff_low, ff_high=args.ff_high)

    print()
    print("FF conditioning diagnostic")
    print("--------------------------")
    print(
        f"  continuous ||FF({report['ff_low']})-FF({report['ff_high']})|| "
        f"= {report['continuous_l2']:.6e}"
    )
    print(
        f"  GC comparison ||GC(0.40)-GC(0.50)|| "
        f"= {report['gc_l2_0_40_vs_0_50']:.6e}"
    )
    print(f"  collapse threshold = {report['collapse_l2_threshold']:.6e}")
    print(f"  continuous collapsed                          = {report['continuous_collapsed']}")
    print(f"  mean off-diagonal FF-bin-token cosine         = {report['mean_ff_token_cosine']:.4f}")
    print()

    if report["continuous_collapsed"]:
        print(
            "FAIL: continuous FF path is near-constant. "
            "target_ff alone will not change fragment realism. "
            "See docs/FF_CONDITIONING_FIX.md."
        )
        return 1

    print(
        "Continuous FF path varies. Still run logit/motif/composition "
        "gates at pinned length and GC before shipping (FF_CONDITIONING_FIX.md)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
