"""
reasoning.py
============
Generates the `reasoning` column. Every clause is built directly from data
actually present on the candidate's record (years_of_experience, current_title,
literal skill names, literal matched concept phrases, literal redrob_signal
values). Nothing here is invented, and nothing here calls an LLM — it's
template assembly over extracted facts, which keeps it:
  - deterministic (same input -> same output, every run, for Stage 3 repro)
  - hallucination-free (Stage 4 check: "every claim corresponds to something
    actually in the profile")
  - varied (Stage 4 check: phrasing choice is driven by a hash of
    candidate_id, not by a single fixed template with name-swap)
  - rank-consistent (the "concern" clause is included precisely when a real
    weak component exists, which correlates with low rank — and omitted for
    genuinely strong candidates, which correlates with high rank)
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List

from . import config


def _stable_hash(s: str) -> int:
    """A hash that is stable across processes and Python versions, unlike
    the builtin hash() for str (which is salted per-process by default via
    PYTHONHASHSEED for security reasons). Determinism across reruns matters
    here so the submission CSV is byte-for-byte reproducible, as required
    for Stage 3.
    """
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def _pick(options: List[str], seed_key: str) -> str:
    idx = _stable_hash(seed_key) % len(options)
    return options[idx]


_GROUP_LABEL = {
    "embeddings_retrieval": "embeddings/retrieval",
    "vector_db_hybrid_search": "vector-DB/hybrid-search",
    "ranking_recsys_ltr": "ranking/recsys",
    "eval_frameworks": "ranking-evaluation",
    "llm_finetune": "LLM/fine-tuning",
    "nlp_ir_fundamentals": "NLP/IR",
    "production_ml_systems": "production-ML",
    "distributed_systems": "distributed-systems",
}


def _strength_clause(candidate: Dict[str, Any], breakdown: Dict[str, Any], seed: str) -> str:
    matches = breakdown.get("narrative_matches", {})
    core_groups_present = [g for g in config.CORE_GROUPS if g in matches]

    if core_groups_present:
        # Prefer the highest-weighted core group present.
        best_group = max(
            core_groups_present, key=lambda g: config.CONCEPT_GROUPS[g]["weight"]
        )
        terms = matches[best_group][:2]
        label = _GROUP_LABEL.get(best_group, best_group)
        options = [
            f"career history shows hands-on {label} work (mentions {', '.join(terms)})",
            f"their role descriptions reference real {label} work, specifically {', '.join(terms)}",
            f"prior roles demonstrate {label} experience ({', '.join(terms)})",
        ]
        return _pick(options, seed + "strength_narrative")

    skill_contribs = breakdown.get("skill_contributions", [])
    if skill_contribs:
        top_names = [name for name, _ in skill_contribs[:2] if name]
        if top_names:
            options = [
                f"lists {', and '.join(top_names)} among their skills",
                f"their skill set includes {', and '.join(top_names)}",
            ]
            return _pick(options, seed + "strength_skills")

    return "profile shows broadly adjacent technical experience"


def _concern_clause(
    candidate: Dict[str, Any], breakdown: Dict[str, Any], seed: str
) -> str:
    comps = breakdown["components"]
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {}) or {}
    concerns: List[str] = []

    if breakdown.get("anti_pattern_flags"):
        flag = breakdown["anti_pattern_flags"][0]
        flag_text = {
            "pure_research_only": "career history is entirely research/academic, with no production deployment experience",
            "consulting_only_no_product_experience": "entire career has been at IT-services/consulting firms with no product-company experience",
            "cv_speech_robotics_without_nlp_ir": "skill set is concentrated in computer vision/speech with no NLP or retrieval exposure",
            "frequent_job_hopping": "recent role tenure averages under 18 months, a job-hopping pattern",
            "architecture_role_no_recent_code": "current title is architecture/management-focused with limited recent hands-on coding",
            "closed_source_no_external_validation": "5+ years of experience but minimal GitHub activity or certifications to validate it",
        }.get(flag, flag.replace("_", " "))
        concerns.append(flag_text)

    if comps["location_score"] < 0.5:
        country = profile.get("country", "their country")
        concerns.append(f"based in {country}, outside India, and the role doesn't sponsor visas")

    notice = signals.get("notice_period_days")
    if notice and notice > 60:
        concerns.append(f"a {notice}-day notice period, well above the JD's sub-30-day preference")

    days_inactive = breakdown.get("behavioral_detail", {}).get("days_inactive")
    if days_inactive is not None and days_inactive > 90:
        concerns.append(f"hasn't been active on the platform in {days_inactive} days")

    response_rate = signals.get("recruiter_response_rate")
    if response_rate is not None and response_rate < 0.15:
        concerns.append(f"a low recruiter response rate ({response_rate:.2f})")

    if comps["experience_score"] < 0.6:
        yoe = profile.get("years_of_experience", 0)
        concerns.append(f"{yoe} years of experience sits outside the JD's 5-9 year band")

    if breakdown.get("integrity_flags") and not breakdown.get("is_honeypot"):
        concerns.append("one profile field doesn't fully line up internally (flagged for manual review)")

    if not concerns:
        return ""

    return _pick(
        [
            "main concern: {c}",
            "one caveat: {c}",
            "worth noting: {c}",
        ],
        seed + "concern_frame",
    ).format(c=concerns[0])


def build_reasoning(candidate: Dict[str, Any], breakdown: Dict[str, Any]) -> str:
    profile = candidate.get("profile", {})
    seed = str(candidate.get("candidate_id", ""))
    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "professional")
    location = profile.get("location", "")
    signals = candidate.get("redrob_signals", {}) or {}

    opener_options = [
        f"{title} with {yoe:.1f} years of experience",
        f"{yoe:.1f}-year {title}",
        f"{title} ({yoe:.1f} yrs experience)",
    ]
    opener = _pick(opener_options, seed + "opener")

    strength = _strength_clause(candidate, breakdown, seed)

    behavior_bits = []
    rr = signals.get("recruiter_response_rate")
    if rr is not None and rr >= 0.5:
        behavior_bits.append(f"recruiter response rate {rr:.2f}")
    if signals.get("open_to_work_flag"):
        behavior_bits.append("marked open to work")
    behavior_clause = ""
    if behavior_bits:
        behavior_clause = "; " + ", ".join(behavior_bits)

    location_bit = f"; {location}" if location else ""

    concern = _concern_clause(candidate, breakdown, seed)
    concern_clause = f". {concern.capitalize()}." if concern else "."

    sentence = f"{opener}; {strength}{behavior_clause}{location_bit}{concern_clause}"
    # Keep it to roughly 1-2 sentences / a reasonable length for the CSV.
    if len(sentence) > 320:
        sentence = sentence[:317].rsplit(" ", 1)[0] + "..."
    return sentence
