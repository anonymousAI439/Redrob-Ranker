# Redrob Hackathon — Candidate Ranking Engine

A deterministic, CPU-only, network-free ranking pipeline for the
*Intelligent Candidate Discovery & Ranking Challenge*. Built to match the
job description for a Senior AI Engineer (Search & Recommendation Systems)
at Redrob, and to be robust against the keyword-stuffing and honeypot traps
the challenge dataset is designed to contain.

## tl;dr

```bash
pip install -r requirements.txt   # optional — only needed for the sandbox app, see below
python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv
python validate_submission.py ./submission.csv
```

No GPU, no API keys, no network calls during ranking.
~1.9GB peak RAM for 100,000 candidates on a single CPU core

## Repo layout

```
rank.py                       # CLI entrypoint — the reproduce_command
redrob_ranker/
  config.py                   # every weight, taxonomy term, and threshold — read this first
  io_utils.py                 # candidate loading (.jsonl/.jsonl.gz/.json), CSV writing
  features.py                 # text normalization, concept matching, anti-pattern + honeypot detection
  scoring.py                  # the six scoring components, title gate, behavioral multiplier
  reasoning.py                # deterministic, fact-grounded reasoning-text generator
  pipeline.py                 # orchestrates load -> score -> filter -> rank -> write
tests/
  test_pipeline.py            # unittest suite against the bundled 50-candidate sample
  fixtures/sample_candidates.json
app.py                        # streamlit sandbox app 
Dockerfile                    # alternative sandbox option
requirements.txt              # empty for rank.py itself; streamlit+pyyaml only for app.py
submission_metadata.yaml      
validate_submission.py        # the organizers' validator, included for convenience
candidate_schema.json         # the organizers' schema, included for reference
sample_candidates.json        # the organizers' 50-candidate sample, included for reference
```

## How to actually use this

1. **Read `redrob_ranker/config.py` top to bottom.** Every scoring weight,
   every keyword/phrase in the concept taxonomy, every anti-pattern rule,
   and every honeypot threshold lives there, with comments explaining the
   *why*, not just the *what*. This is the file you'll be asked about in
   the Stage 5 interview.
2. **Run the test suite**: `python -m unittest discover -s tests -v`. It
   exercises the keyword-stuffer trap directly (a synthetic "HR Manager
   with 9 AI keywords" candidate, mirroring the pattern in
   `sample_submission.csv`, must score below a genuine ranking/retrieval
   engineer), checks the honeypot detector's false-positive rate on the
   clean sample, and checks output determinism.
3. **Run it on the real `candidates.jsonl[.gz]`** once you have it, inspect
   the output, and *change things*. The weights and taxonomy in
   `config.py` encode one defensible interpretation of the JD — you should
   look at what actually comes out the top, decide whether you agree with
   it, and adjust.
4. **Fill in `submission_metadata.yaml`** — team info, your repo URL, your
   deployed sandbox URL, and your actual local compute specs are all
   placeholder text right now.

## Methodology

Six weighted components combine into a base fit score, which is then scaled
by a behavioral multiplier and an anti-pattern multiplier, then gated by an
integrity/honeypot check. Full detail and rationale is in `config.py` and
`scoring.py`; summary:

| Component | Weight | What it measures |
|---|---|---|
| **Narrative** | 30% | Scans **career_history descriptions** (not just the skills list) for embeddings/retrieval/ranking/LLM substance. This is the component built specifically for the JD's core insight: a candidate who never writes "RAG" but whose job descriptions show they shipped embedding-based retrieval should score well here. |
| **Title** | 15% | Matches `current_title`/`headline` against the JD's role vocabulary. Also acts as a **gate** on the skills score (see below). |
| **Skills** | 20% | Trust-weighted concept match over the `skills` array — weighted by proficiency **and** by `endorsements`, `duration_months`, and Redrob's own `skill_assessment_scores`, so an "expert" skill with zero endorsements and zero duration barely counts. |
| **Experience** | 15% | Soft peak at 6–8 years (JD's stated ideal), gentle decay outside the 5–9 band — never a hard cutoff, per the JD's own note that seniority doesn't map linearly to years. |
| **Location** | 12% | Pune/Noida highest, other JD-welcomed Tier-1 cities next, elsewhere in India lower, outside India lowest (no visa sponsorship), with a partial credit for `willing_to_relocate`. |
| **Education** | 8% | Deliberately low weight — the JD never gates on pedigree. Institution tier + relevant field of study. |

**Title gate.** `gated_skills_score = raw_skills_score * title_gate(title_score)`.
A strongly matching title leaves the skills score untouched; a completely
unrelated title (e.g. "HR Manager") cuts it to roughly 20% of its raw
value. This is the direct countermeasure to `sample_submission.csv`'s own
example trap — an HR Manager with 9 AI keywords in their skills section.

**Behavioral multiplier.** Multiplies the base fit by a factor in
[0.45, 1.20] built from `last_active_date` recency (decayed relative to the
*data's own* max date, not a hardcoded "today"), `recruiter_response_rate`,
`interview_completion_rate`, `open_to_work_flag`, `notice_period_days`, and
verification flags. Per the JD: a great-on-paper candidate who's gone dark
for months is **down-weighted, not zeroed out**.

**Anti-pattern multiplier.** Independently detects and penalizes (not
hard-excludes) the JD's explicit non-fits: pure-research-only career
history, consulting-only career with no product-company experience,
CV/speech/robotics skill concentration with zero NLP/IR exposure, frequent
job-hopping (avg. tenure <18mo over the last 4 roles), architecture/
management titles with no recent hands-on coding, and 5+ years of
closed-source work with no GitHub/certification signal at all (only
checked for plausibly technical titles — an accountant having no GitHub
isn't a red flag).

**Honeypot / integrity gate.** A candidate is excluded entirely if it trips
**two or more** independent hard-impossibility checks: a single job lasting
longer than the candidate's whole declared experience, several "expert"-
proficiency skills used for ~0 months, employment predating a known
company's actual founding year, or malformed education date ranges. The
two-flag threshold is deliberate: a single mild inconsistency shows up in
~15–20% of the provided sample's *ordinary* synthetic data (it's noise, not
a trap), so a one-flag hard exclusion would have an unacceptable
false-positive rate. One flag alone instead applies a 0.55x penalty rather
than a full exclusion. See `features.integrity_flags()` for the exact
checks and `config.INTEGRITY` for the thresholds.

## Performance

Benchmarked against a 100,000-row synthetic candidate file (~484MB
uncompressed / ~56MB gzipped, generated by replicating the provided sample
with unique IDs purely to test *throughput*, not ranking quality):

```
loaded 100,000 candidates in 26.7s
scored 100,000 candidates in 213.5s total
wrote 100 rows to submission.csv (215.2s total)
```

Well inside the 5-minute / 16GB CPU-only budget
There is no model loading and no per-candidate I/O, so this scales linearly and predictably.

## Deploying the sandbox

`app.py` is a streamlit app that imports and calls the **exact same**
`redrob_ranker.pipeline.run()` used by `rank.py` — no forked logic, so the
sandbox can never disagree with your actual submission.

**HuggingFace Spaces/ Streamlit** (recommended, matches `submission_metadata_template.yaml`'s
own example): create a new Space (SDK: Streamlit), push this repo's contents
(including `app.py`, `requirements.txt`, and `sample_candidates.json` for
the default demo), and the Space will build automatically.

**Docker** (fallback): `docker build -t redrob-ranker . && docker run --rm -v "$PWD":/data redrob-ranker python rank.py --candidates /data/candidates.jsonl.gz --out /data/submission.csv`

## Honest limitations

The system relies on a manually designed concept taxonomy and phrase matching to identify relevant skills and experiences. Although the taxonomy captures many important concepts, it cannot understand unseen terminology, synonyms, or implicit semantic relationships that are not explicitly encoded. Matching relies on normalized text and concept matching. The system cannot capture deeper semantic similarity between different but related concepts.Ranking is static once configuration is fixed.The system does not improve automatically based on recruiter interactions or hiring outcomes.
