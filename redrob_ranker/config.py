"""
config.py
=========

Nothing in this file calls the network and nothing here is randomized: re-running
the pipeline on the same candidates.jsonl always produces the same scores.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. Top-level component weights
# ---------------------------------------------------------------------------
# These five buckets mirror the JD almost line-for-line:
#   - "narrative" is the single highest-weighted bucket on purpose. The JD is
#     explicit that the right answer requires reading the GAP between what a
#     candidate's skills section says and what their career history shows.
#     A candidate who never writes the word "RAG" but whose job descriptions
#     show they shipped embedding-based retrieval should score well here.
#   - "title" is kept separate and smaller, but it acts as a GATE (see
#     scoring.py: title_gate()) on the skills bucket — this is what stops a
#     "Marketing Manager" with 9 AI keywords in their skills list from
#     outranking a real ML engineer.
#   - "skills" is trust-weighted (endorsements + duration_months), not a raw
#     keyword count, to specifically defeat keyword-stuffing.
WEIGHTS = {
    "title": 0.15,
    "narrative": 0.30,
    "skills": 0.20,
    "experience": 0.15,
    "location": 0.12,
    "education": 0.08,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

# ---------------------------------------------------------------------------
# 2. Concept taxonomy
# ---------------------------------------------------------------------------
# Each group is: weight (relative importance, JD-derived) + a list of
# lowercase phrases/substrings. Matching is substring-based over normalized
# text (title/headline/summary/career-history descriptions/skill names) —
# deliberately simple and auditable rather than a black-box embedding model,
# per the JD's own warning against "doing keyword embedding" instead of
# actually reading the profile. We use this taxonomy as one INPUT signal
# among several (title gating, skill trust, behavioral signals), not as the
# entire ranking method.
#
# Weights map directly to the JD's two skill tiers:
#   "absolutely need"      -> 1.0 - 1.3
#   "would like, won't reject for" -> 0.5 - 0.7
CONCEPT_GROUPS = {
    "embeddings_retrieval": {
        "weight": 1.3,
        "terms": [
            "embedding", "embeddings", "sentence-transformers",
            "sentence transformers", "dense retrieval", "semantic search",
            "vector search", "approximate nearest neighbor", "ann search",
            "knn search", "bi-encoder", "cross-encoder", "bge embedding",
            "e5 embedding", "openai embedding", "text embedding",
        ],
    },
    "vector_db_hybrid_search": {
        "weight": 1.2,
        "terms": [
            "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
            "elasticsearch", "faiss", "vector database", "vector db",
            "hybrid search", "hybrid retrieval", "bm25",
        ],
    },
    "ranking_recsys_ltr": {
        "weight": 1.3,
        "terms": [
            "ranking model", "ranking layer", "ranking pipeline",
            "re-ranking", "reranking", "recommendation system",
            "recommender system", "learning to rank", "learning-to-rank",
            "ltr model", "xgboost", "lightgbm", "click-through",
            "click through rate", "ctr", "discovery feed",
            "relevance model", "search relevance", "candidate generation",
            "two-tower", "two tower model",
        ],
    },
    "eval_frameworks": {
        "weight": 1.0,
        "terms": [
            "ndcg", "mrr", "map@", "mean average precision", "a/b test",
            "ab test", "offline-online correlation", "offline to online",
            "evaluation framework", "relevance labeling",
            "offline eval", "online eval", "online evaluation",
        ],
    },
    "llm_finetune": {
        "weight": 1.1,
        "terms": [
            "llm", "large language model", "fine-tun", "finetun", "lora",
            "qlora", "peft", "rag", "retrieval augmented",
            "retrieval-augmented", "prompt engineering",
            "hugging face transformers", "transformer model", "gpt-",
            "llm-based re-rank", "llm re-rank", "instruction tuning",
        ],
    },
    "nlp_ir_fundamentals": {
        "weight": 0.9,
        "terms": [
            "information retrieval", "nlp", "natural language processing",
            "tf-idf", "tokeniz", "named entity", "search engine", "lucene",
            "solr", "query understanding", "text classification",
            "topic modeling",
        ],
    },
    "production_ml_systems": {
        "weight": 0.8,
        "terms": [
            "production", "deployed", "shipped", "scale", "mlops",
            "model serving", "inference latency", "real-time", "real time",
            "online serving", "feature pipeline", "model registry",
            "a/b testing", "canary deploy", "model monitoring",
        ],
    },
    # "Liked but won't reject" tier from the JD
    "llm_advanced_finetune": {
        "weight": 0.6,
        "terms": ["lora", "qlora", "peft", "distillation", "rlhf"],
    },
    "distributed_systems": {
        "weight": 0.5,
        "terms": [
            "distributed system", "kafka", "spark streaming", "kubernetes",
            "large-scale inference", "low-latency serving", "sharding",
            "load balancing", "horizontal scaling",
        ],
    },
    "hr_tech_marketplace": {
        "weight": 0.5,
        "terms": [
            "recruiting platform", "hr-tech", "hr tech", "marketplace",
            "talent platform", "job matching", "candidate matching",
        ],
    },
    "open_source": {
        "weight": 0.5,
        "terms": ["open-source", "open source contribution", "github stars",
                   "maintainer of", "published package", "pypi package"],
    },
    "python_general": {
        "weight": 0.4,
        "terms": ["python"],
    },
}

# Groups that contribute to the "core JD must-have" subtotal used for the
# title/skills gate (everything the JD lists under "absolutely need").
CORE_GROUPS = {
    "embeddings_retrieval", "vector_db_hybrid_search", "ranking_recsys_ltr",
    "eval_frameworks", "nlp_ir_fundamentals",
}

# ---------------------------------------------------------------------------
# 3. Anti-pattern signals — domains the JD explicitly says are NOT a fit,
#    used to scan skills/industries (NOT to gate title, which has its own
#    logic in title_terms below).
# ---------------------------------------------------------------------------
CV_SPEECH_ROBOTICS_TERMS = [
    "image classification", "computer vision", "object detection", "gan",
    "gans", "speech recognition", "tts", "text-to-speech", "robotics",
    "slam", "autonomous driving", "ocr", "pose estimation",
    "image segmentation",
]

ACADEMIC_TERMS = [
    "phd", "ph.d", "postdoc", "post-doc", "professor", "academia",
    "published paper", "journal article", "conference paper", "research lab",
    "research institute", "university research",
]

ACADEMIC_INDUSTRIES = {
    "academia", "research", "education", "research & development",
    "r&d", "higher education", "academic research",
}

# IT services / pure-consulting firms named (or implied) in the JD as a
# disqualifier UNLESS the candidate also has product-company experience.
CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "tech mahindra", "hcl", "hcltech", "mindtree", "wipro technologies",
    "l&t infotech", "ltimindtree", "mphasis",
}

# ---------------------------------------------------------------------------
# 4. Title taxonomy — drives the title GATE described in scoring.py.
# ---------------------------------------------------------------------------
TITLE_POSITIVE_TERMS = [
    "machine learning", "ml engineer", "ai engineer", "applied scientist",
    "applied ml", "applied ai", "recommendation", "recommender",
    "search engineer", "ranking engineer", "retrieval engineer",
    "nlp engineer", "data scientist", "research engineer", "ml scientist",
    "ai researcher", "search & ranking", "search and ranking",
    "deep learning engineer",
]

TITLE_NEGATIVE_TERMS = [
    "marketing", "human resources", " hr manager", "hr manager",
    "accountant", "accounting", "customer support", "customer service",
    "operations manager", "business analyst", "civil engineer",
    "mechanical engineer", "graphic designer", ".net developer",
    "mobile developer", "qa engineer", "project manager", "sales",
    "recruiter", "financial analyst", "office manager", "administrator",
]

# Titles that are management/architecture-only — used for the "tech lead who
# hasn't shipped code in 18 months" disqualifier from the JD.
NO_CODE_TITLE_TERMS = [
    "architect", "director", " vp ", "vp of", "head of", "tech lead",
    "engineering manager",
]
CODE_QUALIFIER_TERMS = ["engineer", "developer", "scientist", "ic"]

# ---------------------------------------------------------------------------
# 5. Location tiers (JD section "On location, comp, and logistics")
# ---------------------------------------------------------------------------
PRIMARY_CITIES = {"pune", "noida"}
WELCOMED_CITIES = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "new delhi", "gurgaon",
    "gurugram", "bangalore", "bengaluru", "chennai",
}

# ---------------------------------------------------------------------------
# 6. Education tiers
# ---------------------------------------------------------------------------
EDUCATION_TIER_SCORE = {
    "tier_1": 1.00,
    "tier_2": 0.85,
    "tier_3": 0.65,
    "tier_4": 0.50,
    "unknown": 0.60,
}
RELEVANT_FIELDS_OF_STUDY = [
    "computer", "information technology", "data science", "statistics",
    "mathematics", "artificial intelligence", "machine learning",
    "software engineering", "electronics",
]

# ---------------------------------------------------------------------------
# 7. Experience-band fit (JD: "5-9 years... ideal candidate is 6-8")
# ---------------------------------------------------------------------------
EXPERIENCE_BAND = {
    "full_score_low": 6.0,
    "full_score_high": 8.0,
    "soft_low": 4.0,
    "soft_high": 11.0,
    "hard_floor_years": 1.0,   # below this, score bottoms out at floor_score
    "hard_ceiling_years": 18.0,  # above this, score bottoms out at floor_score
    "floor_score": 0.35,
}

# ---------------------------------------------------------------------------
# 8. Behavioral-signal multiplier tuning (JD + redrob_signals_doc)
# ---------------------------------------------------------------------------
BEHAVIORAL = {
    # last_active_date recency decay, in days since the reference date
    # (reference date = max(last_active_date) seen in the loaded file, so
    # this is robust to whenever the pipeline is actually run).
    "recency_full_score_days": 14,
    "recency_half_life_days": 120,
    "recency_floor_multiplier": 0.55,   # JD: down-weight, don't zero out
    "notice_period_sweet_spot_days": 30,
    "notice_period_penalty_per_30d": 0.04,
    "open_to_work_bonus": 0.05,
    "open_to_work_penalty": 0.12,
    "verified_bonus_each": 0.02,        # email/phone/linkedin, up to 3
    "multiplier_floor": 0.45,
    "multiplier_cap": 1.20,
}

# ---------------------------------------------------------------------------
# 9. Known company founding years — used ONLY as a soft honeypot signal
#    ("8 years of experience at a company founded 3 years ago" from the JD's
#    own honeypot example). We only check companies we have high confidence
#    about; unknown companies (including the dataset's fictional ones, e.g.
#    Hooli/Pied Piper/Stark Industries/Wayne Enterprises/Globex/Initech/
#    Dunder Mifflin/Acme Corp) are skipped rather than guessed at, since a
#    wrong guess here is worse than no check at all.
KNOWN_FOUNDING_YEARS = {
    "tcs": 1968, "infosys": 1981, "wipro": 1945, "cognizant": 1994,
    "tech mahindra": 1986, "hcl": 1976, "mindtree": 1999,
    "swiggy": 2014, "zomato": 2008, "ola": 2010, "razorpay": 2014,
    "cred": 2018, "flipkart": 2007, "uber": 2009, "paytm": 2010,
    "byjus": 2011, "freshworks": 2010, "zoho": 1996, "meesho": 2015,
    "phonepe": 2015, "nykaa": 2012, "policybazaar": 2008, "urban company": 2014,
    "delhivery": 2011, "dunzo": 2015, "groww": 2016, "cure.fit": 2016,
    "google": 1998, "microsoft": 1975, "amazon": 1994, "meta": 2004,
    "facebook": 2004, "netflix": 1997, "apple": 1976, "openai": 2015,
    "anthropic": 2021,
}

# ---------------------------------------------------------------------------
# 10. Honeypot / integrity-risk thresholds (see features.py: integrity_flags)
# ---------------------------------------------------------------------------
INTEGRITY = {
    # A single job can't outlast the candidate's whole declared experience by
    # more than this many months of slack.
    "single_job_overflow_months_slack": 9,
    # "expert"/"advanced" proficiency claimed with near-zero time-on-skill.
    "instant_expert_max_months": 1,
    "instant_expert_min_count": 3,
    # Total claimed career-history months vastly exceeding declared
    # years_of_experience (heavy overlap / padding).
    "total_overflow_ratio": 2.2,
    "total_overflow_min_months_over": 30,
    # Number of independent "severe" flags required to treat a profile as a
    # honeypot exclusion vs. a softer integrity penalty. Calibrated against
    # the provided 50-candidate sample, where *single* mild inconsistencies
    # (e.g., one skill's duration slightly exceeding YoE, or graduation date
    # not lining up with first job) appear in ~15-20% of normal synthetic
    # profiles — i.e., dataset noise, not honeypots. Requiring >=2
    # independent severe signals avoids false-positiving on that noise while
    # still catching genuinely "subtly impossible" honeypots, which by
    # construction stack several impossibilities at once.
    "severe_flags_for_exclusion": 2,
    "single_flag_penalty_multiplier": 0.55,
}

# ---------------------------------------------------------------------------
# 11. Proficiency / trust multipliers
# ---------------------------------------------------------------------------
PROFICIENCY_WEIGHT = {
    "beginner": 0.40,
    "intermediate": 0.70,
    "advanced": 1.00,
    "expert": 1.20,
}
