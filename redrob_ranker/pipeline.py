"""
pipeline.py
===========
Orchestrates the end-to-end ranking run:
  load -> score every candidate -> sort -> drop honeypots -> take top-K
  -> generate reasoning -> write CSV

Designed to run well inside the hackathon's compute envelope (<=5 min wall
clock, <=16GB RAM, CPU only, no network) even for the full 100,000-candidate
pool: scoring is O(n) pure-Python string/dict work with no model loading, no
external calls, and no per-candidate I/O.
"""

from __future__ import annotations

import sys
import time
from datetime import date
from typing import Any, Dict, List

from . import features, io_utils, reasoning, scoring


def _reference_date(candidates: List[Dict[str, Any]]) -> date:
    """The 'as of today' anchor for recency scoring. Derived from the data
    itself (max last_active_date seen) rather than hardcoded, so the
    pipeline behaves correctly no matter when it's actually run.
    """
    best = None
    for c in candidates:
        d = features.parse_date((c.get("redrob_signals") or {}).get("last_active_date"))
        if d and (best is None or d > best):
            best = d
    return best or date.today()


def run(
    candidates_path: str,
    output_path: str,
    top_k: int = 100,
    verbose: bool = True,
) -> Dict[str, Any]:
    t0 = time.time()

    candidates = list(io_utils.load_any_candidate_file(candidates_path))
    n_loaded = len(candidates)
    if verbose:
        print(f"[pipeline] loaded {n_loaded} candidates in {time.time()-t0:.1f}s", file=sys.stderr)

    if n_loaded < top_k:
        raise ValueError(
            f"Need at least {top_k} candidates to produce a top-{top_k} "
            f"submission; only {n_loaded} were loaded from {candidates_path}."
        )

    ref_date = _reference_date(candidates)
    if verbose:
        print(f"[pipeline] reference date for recency scoring: {ref_date}", file=sys.stderr)

    scored: List[Dict[str, Any]] = []
    seen_ids = set()
    n_dupe_ids = 0
    n_honeypot = 0

    for c in candidates:
        cid = c.get("candidate_id")
        if not cid:
            continue
        if cid in seen_ids:
            n_dupe_ids += 1
            continue
        seen_ids.add(cid)

        breakdown = scoring.score_candidate(c, ref_date)
        if breakdown["is_honeypot"]:
            n_honeypot += 1
        scored.append({"candidate": c, "breakdown": breakdown})

    if verbose:
        print(
            f"[pipeline] scored {len(scored)} unique candidates "
            f"({n_dupe_ids} duplicate ids skipped, "
            f"{n_honeypot} honeypot-flagged & excluded) "
            f"in {time.time()-t0:.1f}s total",
            file=sys.stderr,
        )

    eligible = [s for s in scored if not s["breakdown"]["is_honeypot"]]
    if len(eligible) < top_k:
        raise RuntimeError(
            f"Only {len(eligible)} non-honeypot candidates available, need "
            f"{top_k}. Check INTEGRITY thresholds in config.py — they may be "
            f"too aggressive for this pool."
        )

    # Deterministic ordering: score desc, then candidate_id asc as the
    # required tie-break (submission_spec.md Section 3).
    eligible.sort(key=lambda s: (-s["breakdown"]["final_score"], s["candidate"]["candidate_id"]))
    top = eligible[:top_k]

    # Normalize scores into a clean, strictly-helpful-looking [0,1] band for
    # the CSV `score` column while preserving rank order exactly (min-max
    # over just the selected top-K, since that's what's actually published).
    raw_scores = [t["breakdown"]["final_score"] for t in top]
    lo, hi = min(raw_scores), max(raw_scores)
    span = (hi - lo) or 1.0

    rows = []
    for i, item in enumerate(top, start=1):
        c = item["candidate"]
        breakdown = item["breakdown"]
        norm_score = 0.40 + 0.59 * (breakdown["final_score"] - lo) / span
        rows.append({
            "candidate_id": c["candidate_id"],
            "rank": i,
            "score": norm_score,
            "reasoning": reasoning.build_reasoning(c, breakdown),
        })
        
    # Guarantee strict monotonic non-increase even after rounding to the
    # CSV's 4-decimal display precision (validator checks this exactly).
    for i in range(1, len(rows)):
        if rows[i]["score"] > rows[i - 1]["score"]:
            rows[i]["score"] = rows[i - 1]["score"]
        rows[i - 1]["score"] = round(rows[i - 1]["score"], 4)
    rows[-1]["score"] = round(rows[-1]["score"], 4)

    # Rounding to 4 decimals can collapse two *distinct* raw scores into the
    # same displayed value, creating a tie the validator can see even though
    # the original sort only enforced candidate_id-ascending for exact raw
    # ties. Re-sort each contiguous block of equal *displayed* score by
    # candidate_id ascending so the output satisfies the tie-break rule
    # regardless of why the tie appeared.
    i = 0
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and rows[j + 1]["score"] == rows[i]["score"]:
            j += 1
        if j > i:
            rows[i:j + 1] = sorted(rows[i:j + 1], key=lambda r: r["candidate_id"])
            for k, row in enumerate(rows[i:j + 1], start=i):
                row["rank"] = k + 1
        i = j + 1

    io_utils.write_submission_csv(output_path, rows)

    elapsed = time.time() - t0
    if verbose:
        print(f"[pipeline] wrote {len(rows)} rows to {output_path} ({elapsed:.1f}s total)", file=sys.stderr)

    return {
        "n_loaded": n_loaded,
        "n_scored": len(scored),
        "n_honeypot_excluded": n_honeypot,
        "n_duplicate_ids": n_dupe_ids,
        "top_k": top_k,
        "elapsed_seconds": elapsed,
        "reference_date": str(ref_date),
    }
