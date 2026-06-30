"""
features.py
============
Pure feature-extraction helpers: turning a raw candidate dict into the
signals scoring.py needs. No scoring/weighting decisions live here — only
"what is true about this profile."
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from . import config

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^a-z0-9\s/&.+#-]")


def normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    t = text.lower()
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t)
    return t.strip()


def concept_hits(text: str, terms: List[str]) -> List[str]:
    """Return the subset of `terms` that appear as substrings of `text`."""
    return [term for term in terms if term in text]


def score_concept_groups(text: str) -> Tuple[float, Dict[str, List[str]]]:
    """Score a normalized text blob against every concept group in
    config.CONCEPT_GROUPS. Returns (raw_weighted_score, {group: matched_terms}).
    """
    total = 0.0
    matches: Dict[str, List[str]] = {}
    for group, spec in config.CONCEPT_GROUPS.items():
        hits = concept_hits(text, spec["terms"])
        if hits:
            matches[group] = hits
            # log-scaled so 1 hit already counts a lot, but 5 hits doesn't
            # count 5x as much (defeats simple repetition/keyword stuffing)
            total += spec["weight"] * (1.0 + math.log1p(len(hits) - 1))
    return total, matches


def core_concept_strength(matches: Dict[str, List[str]]) -> float:
    """Weighted strength restricted to the JD's 'absolutely need' groups."""
    total = 0.0
    for group in config.CORE_GROUPS:
        if group in matches:
            spec = config.CONCEPT_GROUPS[group]
            total += spec["weight"] * (1.0 + math.log1p(len(matches[group]) - 1))
    return total


def candidate_full_text(candidate: Dict[str, Any]) -> Dict[str, str]:
    """Build the normalized text blobs used across scoring:
    title_text, narrative_text (career history + summary), skills_text.
    """
    profile = candidate.get("profile", {})
    title_text = normalize_text(
        " ".join(filter(None, [profile.get("current_title"), profile.get("headline")]))
    )

    narrative_parts = [profile.get("summary", "")]
    for job in candidate.get("career_history", []) or []:
        narrative_parts.append(job.get("title", ""))
        narrative_parts.append(job.get("description", ""))
    narrative_text = normalize_text(" ".join(filter(None, narrative_parts)))

    skill_names = [s.get("name", "") for s in candidate.get("skills", []) or []]
    skills_text = normalize_text(" ".join(skill_names))

    return {
        "title_text": title_text,
        "narrative_text": narrative_text,
        "skills_text": skills_text,
        "all_text": normalize_text(
            title_text + " " + narrative_text + " " + skills_text
        ),
    }


# ---------------------------------------------------------------------------
# Anti-pattern detection
# ---------------------------------------------------------------------------

def title_term_balance(title_text: str) -> Tuple[int, int]:
    pos = sum(1 for t in config.TITLE_POSITIVE_TERMS if t in title_text)
    neg = sum(1 for t in config.TITLE_NEGATIVE_TERMS if t in title_text)
    return pos, neg


def is_pure_research_only(candidate: Dict[str, Any]) -> bool:
    history = candidate.get("career_history", []) or []
    if not history:
        return False
    academic_count = 0
    for job in history:
        industry = (job.get("industry") or "").strip().lower()
        title = normalize_text(job.get("title", ""))
        desc = normalize_text(job.get("description", ""))
        is_academic_industry = industry in config.ACADEMIC_INDUSTRIES
        has_academic_term = any(
            term in title or term in desc for term in config.ACADEMIC_TERMS
        )
        if is_academic_industry or has_academic_term:
            academic_count += 1
    return academic_count == len(history)


def is_consulting_only(candidate: Dict[str, Any]) -> bool:
    history = candidate.get("career_history", []) or []
    profile = candidate.get("profile", {})
    companies = [profile.get("current_company", "")] + [
        j.get("company", "") for j in history
    ]
    companies = [c.strip().lower() for c in companies if c]
    if not companies:
        return False
    in_consulting = [c in config.CONSULTING_FIRMS for c in companies]
    return len(in_consulting) > 0 and all(in_consulting)


def is_cv_speech_robotics_without_nlp(candidate: Dict[str, Any]) -> bool:
    skills = candidate.get("skills", []) or []
    skill_names = [normalize_text(s.get("name", "")) for s in skills]
    cv_hits = sum(
        1 for name in skill_names
        if any(term in name for term in config.CV_SPEECH_ROBOTICS_TERMS)
    )
    nlp_hits = sum(
        1 for name in skill_names
        if any(
            term in name
            for term in config.CONCEPT_GROUPS["nlp_ir_fundamentals"]["terms"]
            + config.CONCEPT_GROUPS["embeddings_retrieval"]["terms"]
            + config.CONCEPT_GROUPS["ranking_recsys_ltr"]["terms"]
        )
    )
    return cv_hits >= 2 and nlp_hits == 0


def is_job_hopper(candidate: Dict[str, Any]) -> bool:
    history = candidate.get("career_history", []) or []
    if len(history) < 4:
        return False
    recent = sorted(history, key=lambda j: j.get("start_date") or "", reverse=True)[:4]
    durations = [j.get("duration_months", 0) or 0 for j in recent]
    if not durations:
        return False
    avg = sum(durations) / len(durations)
    return avg < 18


def is_no_code_architect(candidate: Dict[str, Any]) -> bool:
    profile = candidate.get("profile", {})
    title = normalize_text(profile.get("current_title", ""))
    if not any(term in title for term in config.NO_CODE_TITLE_TERMS):
        return False
    if any(term in title for term in config.CODE_QUALIFIER_TERMS):
        return False
    history = candidate.get("career_history", []) or []
    current = next((j for j in history if j.get("is_current")), None)
    duration = (current or {}).get("duration_months", 0) or 0
    return duration >= 18


def is_closed_source_no_validation(candidate: Dict[str, Any]) -> bool:
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience", 0) or 0
    signals = candidate.get("redrob_signals", {}) or {}
    github_score = signals.get("github_activity_score", -1)
    certs = candidate.get("certifications", []) or []
    if yoe < 5 or certs:
        return False
    if github_score is not None and github_score > 5:
        return False
    # Only meaningful for candidates whose work is plausibly software/ML in
    # the first place — an accountant or civil engineer having no GitHub
    # activity isn't a "no external validation" red flag, it's just normal.
    # Deliberately checked against current_title (a clean structured field)
    # rather than career_history descriptions, which in this dataset reuse
    # generic boilerplate paragraphs across many unrelated job titles.
    title_text = normalize_text(profile.get("current_title", ""))
    non_software_engineering = ["civil engineer", "mechanical engineer", "electrical engineer", "chemical engineer", "structural engineer"]
    if any(term in title_text for term in non_software_engineering):
        return False
    tech_title_terms = [
        "engineer", "developer", "scientist", "programmer", "architect",
        "ml ", "ai ", "machine learning", "data ", "devops", "sde",
    ]
    return any(term in title_text for term in tech_title_terms)


def anti_pattern_multiplier(candidate: Dict[str, Any]) -> Tuple[float, List[str]]:
    """Returns (multiplier in (0,1], list of triggered anti-pattern names)
    for the JD's explicit "things we explicitly do NOT want" section.
    Multiplicative, with a floor so nothing collapses to literal zero.
    """
    mult = 1.0
    triggered = []

    if is_pure_research_only(candidate):
        mult *= 0.15
        triggered.append("pure_research_only")
    if is_consulting_only(candidate):
        mult *= 0.35
        triggered.append("consulting_only_no_product_experience")
    if is_cv_speech_robotics_without_nlp(candidate):
        mult *= 0.55
        triggered.append("cv_speech_robotics_without_nlp_ir")
    if is_job_hopper(candidate):
        mult *= 0.75
        triggered.append("frequent_job_hopping")
    if is_no_code_architect(candidate):
        mult *= 0.80
        triggered.append("architecture_role_no_recent_code")
    if is_closed_source_no_validation(candidate):
        mult *= 0.88
        triggered.append("closed_source_no_external_validation")

    return max(mult, 0.05), triggered


# ---------------------------------------------------------------------------
# Integrity / honeypot detection
# ---------------------------------------------------------------------------

def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


# Public alias — used across modules (scoring.py, pipeline.py) to avoid
# reaching into a "private" name from outside this module.
parse_date = _parse_date


def integrity_flags(candidate: Dict[str, Any]) -> List[str]:
    """Return the list of *severe* internal-consistency violations.

    Calibrated deliberately conservatively: every check here is a hard
    logical contradiction (a single job lasting longer than the candidate's
    whole career, a skill an "expert" used for zero time, a company being
    worked at before it was founded, education ending before it started) —
    not a mild rounding mismatch. See config.INTEGRITY for thresholds and
    the rationale for requiring >=2 of these before excluding a candidate.
    """
    flags: List[str] = []
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience", 0) or 0
    yoe_months = yoe * 12
    history = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    education = candidate.get("education", []) or []

    cfg = config.INTEGRITY

    # 1. A single job that alone outlasts the candidate's total experience.
    for job in history:
        dur = job.get("duration_months", 0) or 0
        if dur > yoe_months + cfg["single_job_overflow_months_slack"]:
            flags.append(f"single_job_exceeds_total_experience({job.get('company')})")
            break

    # 2. Total career-history months wildly exceeding declared experience
    #    (heavy unexplained overlap), beyond plain rounding noise.
    total_months = sum((j.get("duration_months", 0) or 0) for j in history)
    if (
        total_months > yoe_months * cfg["total_overflow_ratio"]
        and total_months - yoe_months > cfg["total_overflow_min_months_over"]
    ):
        flags.append("career_history_total_far_exceeds_experience")

    # 3. Multiple "instant experts" — advanced/expert proficiency claimed
    #    with ~zero time on the skill.
    instant_experts = [
        s.get("name") for s in skills
        if s.get("proficiency") in ("advanced", "expert")
        and (s.get("duration_months", 0) or 0) <= cfg["instant_expert_max_months"]
    ]
    if len(instant_experts) >= cfg["instant_expert_min_count"]:
        flags.append(f"instant_expert_skills({len(instant_experts)})")

    # 4. Known company worked at before it was founded.
    for job in history:
        company = (job.get("company") or "").strip().lower()
        founding_year = config.KNOWN_FOUNDING_YEARS.get(company)
        start = _parse_date(job.get("start_date"))
        if founding_year and start and start.year < founding_year:
            flags.append(f"worked_at_{company}_before_founding({start.year})")

    # 5. Education end date before its own start date, or an absurdly long
    #    single degree (>10 years).
    for edu in education:
        sy, ey = edu.get("start_year"), edu.get("end_year")
        if sy and ey:
            if ey < sy:
                flags.append("education_end_before_start")
            elif ey - sy > 10:
                flags.append("education_duration_implausible")

    # 6. Duplicate skill name listed twice with materially different
    #    proficiency (internally contradictory profile).
    seen: Dict[str, str] = {}
    for s in skills:
        name = (s.get("name") or "").strip().lower()
        prof = s.get("proficiency")
        if name in seen and seen[name] != prof:
            flags.append(f"contradictory_duplicate_skill({name})")
        seen[name] = prof

    return flags


def integrity_gate(candidate: Dict[str, Any]) -> Tuple[bool, float, List[str]]:
    """Returns (is_honeypot_exclusion, penalty_multiplier, flags)."""
    flags = integrity_flags(candidate)
    cfg = config.INTEGRITY
    if len(flags) >= cfg["severe_flags_for_exclusion"]:
        return True, 0.0, flags
    if len(flags) == 1:
        return False, cfg["single_flag_penalty_multiplier"], flags
    return False, 1.0, flags
