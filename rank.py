#!/usr/bin/env python3
"""
rank.py — Redrob Hackathon submission CLI.

Single command to go from a candidate pool to a validator-passing
submission CSV:

    python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv

Accepts .jsonl, .jsonl.gz, or the bundle's pretty-printed sample_candidates.json
(useful for quick local smoke tests with --top-k below 100).

No third-party dependencies. No network calls. No GPU. Designed to run in a
few seconds for 50k candidates and comfortably under the 5-minute / 16GB
budget for the full 100k pool.
"""

import argparse
import json
import sys
import time

from redrob_ranker import pipeline


def parse_args():
    p = argparse.ArgumentParser(description="Rank candidates for the Redrob hackathon JD.")
    p.add_argument("--candidates", required=True, help="Path to candidates.jsonl[.gz] or sample_candidates.json")
    p.add_argument("--out", required=True, help="Path to write the submission CSV")
    p.add_argument("--top-k", type=int, default=100, help="Number of ranked rows to output (default 100)")
    p.add_argument("--quiet", action="store_true", help="Suppress progress logging on stderr")
    p.add_argument("--stats-json", default=None, help="Optional path to dump run stats as JSON")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    stats = pipeline.run(
        candidates_path=args.candidates,
        output_path=args.out,
        top_k=args.top_k,
        verbose=not args.quiet,
    )
    elapsed = time.time() - t0

    if elapsed > 300:
        print(
            f"[rank.py] WARNING: total runtime {elapsed:.1f}s exceeds the "
            f"5-minute compute budget in submission_spec.md Section 3.",
            file=sys.stderr,
        )

    if args.stats_json:
        with open(args.stats_json, "w") as f:
            json.dump(stats, f, indent=2)

    print(f"Done. Wrote {args.top_k} ranked rows to {args.out} in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
