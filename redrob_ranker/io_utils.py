"""
io_utils.py
===========
Loading the candidate pool (plain or gzipped JSONL) and writing the
submission CSV. Kept dependency-free (standard library only) so Stage 3
reproduction has nothing to install incorrectly.
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path
from typing import Iterator, Dict, Any, List


def iter_candidates(path: str) -> Iterator[Dict[str, Any]]:
    """Yield candidate dicts from a .jsonl or .jsonl.gz file, one at a time.

    Streaming (rather than loading the whole 100k-candidate / ~465MB
    uncompressed file into a single list before processing) keeps peak
    memory well under the 16GB budget regardless of pool size.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")

    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(
                    f"[io_utils] WARNING: skipping malformed JSON on line "
                    f"{line_num}: {e}",
                    file=sys.stderr,
                )
                continue


def load_candidates_json_array(path: str) -> List[Dict[str, Any]]:
    """Load a pretty-printed JSON *array* of candidates (e.g.
    sample_candidates.json), as opposed to JSONL. Used mainly for local
    testing against the bundle's sample file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_any_candidate_file(path: str) -> Iterator[Dict[str, Any]]:
    """Convenience dispatcher: accepts .jsonl, .jsonl.gz, or a pretty JSON
    array (.json), and always yields candidate dicts one at a time.
    """
    p = Path(path)
    if p.suffix == ".json" and not p.name.endswith(".jsonl"):
        # Could be a JSON array (sample_candidates.json) — sniff first
        # non-whitespace character to decide.
        with open(p, "r", encoding="utf-8") as f:
            head = f.read(1)
        if head == "[":
            for c in load_candidates_json_array(str(p)):
                yield c
            return
    yield from iter_candidates(str(p))


REQUIRED_CSV_HEADER = ["candidate_id", "rank", "score", "reasoning"]


def write_submission_csv(path: str, ranked_rows: List[Dict[str, Any]]) -> None:
    """Write the final submission CSV.

    ranked_rows: list of dicts with keys candidate_id, rank, score, reasoning,
    already in rank-ascending (1..N) order.
    """
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(REQUIRED_CSV_HEADER)
        for row in ranked_rows:
            writer.writerow([
                row["candidate_id"],
                row["rank"],
                f"{row['score']:.4f}",
                row["reasoning"],
            ])
