# Alternative sandbox option (submission_spec.md lists Docker as acceptable
# alongside HuggingFace Spaces / Streamlit Cloud / Replit / Colab / Binder).
#
# Build:  docker build -t redrob-ranker .
# Run:    docker run --rm -v "$PWD":/data redrob-ranker \
#             python rank.py --candidates /data/candidates.jsonl.gz --out /data/submission.csv
#
# No network access is required at run time — this image only needs the
# network once, at build time, to pull the base image and (if you choose to
# run app.py instead of rank.py) install gradio.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# rank.py itself needs nothing from requirements.txt (stdlib only); this
# install is only exercised if you run the optional app.py sandbox UI.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: print usage. Override the command to actually rank, e.g.
#   docker run --rm -v "$PWD":/data redrob-ranker \
#       python rank.py --candidates /data/candidates.jsonl.gz --out /data/submission.csv
CMD ["python", "rank.py", "--help"]
