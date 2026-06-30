"""
app.py — Streamlit sandbox demo for Redrob Candidate Ranker

It uses the same redrob_ranker.pipeline.run() logic to ensure parity
with rank.py and HF Spaces behavior.
"""

import json
import tempfile
from pathlib import Path

import streamlit as st

from redrob_ranker import pipeline

EXAMPLE_PATH = Path(__file__).parent / "sample_candidates.json"


def write_uploaded_file(uploaded_file):
    """Write Streamlit UploadedFile to a temp file and return its path."""
    suffix = Path(uploaded_file.name).suffix or ".json"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(uploaded_file.read())
    tmp.flush()
    return tmp.name


def run_ranking(file_path, top_k):
    # Count candidates
    with open(file_path, "r", encoding="utf-8") as f:
        head = f.read(1)

    if head == "[":
        with open(file_path, "r", encoding="utf-8") as f:
            n = len(json.load(f))
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            n = sum(1 for line in f if line.strip())

    top_k = max(1, min(int(top_k), n))

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        out_path = tmp.name

    stats = pipeline.run(file_path, out_path, top_k=top_k, verbose=False)

    import csv
    with open(out_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    stats_text = (
        f"Loaded {stats['n_loaded']} candidates · "
        f"{stats['n_honeypot_excluded']} honeypot-flagged & excluded · "
        f"{stats['n_duplicate_ids']} duplicate ids skipped · "
        f"reference date {stats['reference_date']} · "
        f"{stats['elapsed_seconds']:.2f}s elapsed (CPU, no network)"
    )

    return rows[1:], rows[0], stats_text, out_path


# ---------------- UI ---------------- #

st.set_page_config(page_title="Redrob Candidate Ranker — Sandbox", layout="wide")

st.title("Redrob Candidate Ranker — Sandbox")

st.markdown(
    """
Upload a `.json` (array) or `.jsonl` candidate file matching `candidate_schema.json`.
This runs the exact same `redrob_ranker` pipeline as `rank.py` fully on CPU.

If no file is uploaded, a bundled sample dataset is used.
"""
)

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload candidates.json / candidates.jsonl",
        type=["json", "jsonl"]
    )

with col2:
    top_k = st.number_input("Top K", min_value=1, value=10, step=1)

if st.button("Rank", type="primary"):

    # Decide input file
    if uploaded_file is None:
        file_path = str(EXAMPLE_PATH)
    else:
        file_path = write_uploaded_file(uploaded_file)

    rows, header, stats_text, out_path = run_ranking(file_path, top_k)

    st.subheader("Stats")
    st.markdown(stats_text)

    st.subheader("Results")
    st.dataframe(rows, use_container_width=True)

    # Load CSV for download
    with open(out_path, "rb") as f:
        csv_bytes = f.read()

    st.download_button(
        label="Download submission CSV",
        data=csv_bytes,
        file_name="submission.csv",
        mime="text/csv",
    )