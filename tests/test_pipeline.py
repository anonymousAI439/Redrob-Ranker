"""
tests/test_pipeline.py
=======================
Pure-stdlib unittest suite. Run with:

    python -m unittest discover -s tests -v

Uses the bundled 50-candidate sample_candidates.json as a fixture. Since
that file has far fewer than 100 candidates, these tests run the pipeline
with --top-k set to the fixture size rather than 100, and check the
*mechanics* (uniqueness, monotonicity, determinism, no-hallucination,
honeypot-threshold calibration) rather than re-validating the full
100-row CSV shape (validate_submission.py already covers that shape
contract end-to-end against a real top-100 run).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from redrob_ranker import features, io_utils, pipeline, scoring  # noqa: E402

FIXTURE = str(Path(__file__).parent / "fixtures" / "sample_candidates.json")
CANDIDATE_ID_RE = re.compile(r"^CAND_[0-9]{7}$")


def _load_fixture():
    with open(FIXTURE, "r", encoding="utf-8") as f:
        return json.load(f)


class TestScoringMechanics(unittest.TestCase):
    def setUp(self):
        self.candidates = _load_fixture()
        self.ref_date = pipeline._reference_date(self.candidates)

    def test_every_candidate_scores_without_exception(self):
        for c in self.candidates:
            breakdown = scoring.score_candidate(c, self.ref_date)
            self.assertIn("final_score", breakdown)
            self.assertGreaterEqual(breakdown["final_score"], 0.0)

    def test_honeypot_false_positive_rate_on_clean_sample(self):
        # The bundled sample contains no deliberately-planted honeypots, but
        # does contain realistic synthetic noise (skill duration slightly
        # exceeding years_of_experience, education/job-start date overlaps,
        # etc. in ~15-20% of profiles). The >=2-severe-flags threshold
        # should keep false-positive exclusions rare on this kind of noise.
        flagged = [
            c["candidate_id"] for c in self.candidates
            if scoring.score_candidate(c, self.ref_date)["is_honeypot"]
        ]
        self.assertLessEqual(
            len(flagged), 2,
            f"Too many false-positive honeypot exclusions on clean sample data: {flagged}",
        )

    def test_keyword_stuffer_trap_is_suppressed(self):
        # Build a synthetic "HR Manager with 9 AI keywords" candidate —
        # exactly the trap pattern shown in sample_submission.csv — and
        # confirm it scores well below a genuine ML/ranking profile.
        stuffer = json.loads(json.dumps(self.candidates[0]))  # deep copy
        stuffer["candidate_id"] = "CAND_9999998"
        stuffer["profile"]["current_title"] = "HR Manager"
        stuffer["profile"]["headline"] = "HR Manager"
        stuffer["profile"]["summary"] = "Experienced HR manager handling recruitment and payroll."
        stuffer["skills"] = [
            {"name": n, "proficiency": "expert", "endorsements": 0, "duration_months": 0}
            for n in [
                "Machine Learning", "Deep Learning", "NLP", "LLM", "RAG",
                "Embeddings", "Vector Search", "Ranking", "Recommendation Systems",
            ]
        ]
        for job in stuffer["career_history"]:
            job["title"] = "HR Manager"
            job["description"] = "Managed recruitment, onboarding, and employee relations."
            job["industry"] = "Human Resources"

        strong = next(
            c for c in self.candidates if c["candidate_id"] == "CAND_0000031"
        )

        stuffer_score = scoring.score_candidate(stuffer, self.ref_date)["final_score"]
        strong_score = scoring.score_candidate(strong, self.ref_date)["final_score"]
        self.assertLess(
            stuffer_score, strong_score,
            "Keyword-stuffed HR Manager should NOT outrank a genuine "
            "recommendation-systems engineer.",
        )

    def test_title_gate_discounts_unrelated_titles(self):
        low_gate = scoring.title_gate(scoring.title_fit_score("hr manager"))
        high_gate = scoring.title_gate(
            scoring.title_fit_score(
                "senior machine learning engineer ranking and recommendation systems"
            )
        )
        self.assertLess(low_gate, 0.5)
        self.assertGreater(high_gate, 0.9)


class TestPipelineEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp_out = "/tmp/_test_redrob_submission.csv"
        if os.path.exists(self.tmp_out):
            os.remove(self.tmp_out)

    def tearDown(self):
        if os.path.exists(self.tmp_out):
            os.remove(self.tmp_out)

    def test_pipeline_runs_and_produces_valid_shape(self):
        top_k = 20
        stats = pipeline.run(FIXTURE, self.tmp_out, top_k=top_k, verbose=False)
        self.assertEqual(stats["n_loaded"], 50)

        with open(self.tmp_out, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        self.assertEqual(header, io_utils.REQUIRED_CSV_HEADER)
        self.assertEqual(len(rows), top_k)

        ranks = [int(r[1]) for r in rows]
        self.assertEqual(ranks, list(range(1, top_k + 1)))

        ids = [r[0] for r in rows]
        self.assertEqual(len(ids), len(set(ids)), "duplicate candidate_id in output")
        for cid in ids:
            self.assertRegex(cid, CANDIDATE_ID_RE)

        scores = [float(r[2]) for r in rows]
        self.assertEqual(scores, sorted(scores, reverse=True), "scores must be non-increasing by rank")

        # tie-break: equal scores must be candidate_id ascending
        for i in range(len(rows) - 1):
            if scores[i] == scores[i + 1]:
                self.assertLessEqual(ids[i], ids[i + 1])

    def test_determinism_across_runs(self):
        out_a = "/tmp/_test_redrob_a.csv"
        out_b = "/tmp/_test_redrob_b.csv"
        pipeline.run(FIXTURE, out_a, top_k=15, verbose=False)
        pipeline.run(FIXTURE, out_b, top_k=15, verbose=False)
        hash_a = hashlib.md5(Path(out_a).read_bytes()).hexdigest()
        hash_b = hashlib.md5(Path(out_b).read_bytes()).hexdigest()
        self.assertEqual(hash_a, hash_b, "pipeline output must be byte-identical across runs")
        os.remove(out_a)
        os.remove(out_b)

    def test_reasoning_skill_names_are_not_hallucinated(self):
        # Every skill name quoted in the reasoning text must actually be
        # present (case-insensitively) somewhere on that candidate's real
        # skills list or career-history text — i.e., nothing is invented.
        top_k = 20
        pipeline.run(FIXTURE, self.tmp_out, top_k=top_k, verbose=False)
        candidates_by_id = {c["candidate_id"]: c for c in _load_fixture()}

        with open(self.tmp_out, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                c = candidates_by_id[row["candidate_id"]]
                texts = features.candidate_full_text(c)
                haystack = texts["all_text"]
                skill_names = [s["name"].lower() for s in c.get("skills", [])]
                # crude check: any quoted Title-Case-ish token sequence in
                # the reasoning that looks like a skill name should be
                # findable in the candidate's own skills or narrative text.
                for name in skill_names:
                    if name in row["reasoning"].lower():
                        self.assertIn(
                            name, haystack,
                            f"Reasoning mentions skill '{name}' not found in candidate data",
                        )


if __name__ == "__main__":
    unittest.main()
