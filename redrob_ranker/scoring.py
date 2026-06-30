"""
scoring.py
==========
Turns the features extracted in features.py into:
  1. five 0..~1.3 component scores (title, narrative, skills, experience,
     location, education)
  2. a title -> skills GATE (the anti-keyword-stuffing mechanism)
  3. a behavioral multiplier (redrob_signals)
  4. an anti-pattern multiplier and integrity gate (features.py)
and combines them into one final, normalized score per candidate, plus a
"breakdown" dict that reasoning.py uses to write honest, fact-grounded
explanations.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Dict, List, Tuple

from . import config, features


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Title fit + gate
# ---------------------------------------------------------------------------

def title_fit_score(title_text: str) -> float:
    pos, neg = features.title_term_balance(title_text)
    # Also let core-concept language in the title itself count (e.g. a
    # headline like "Search, Ranking & Retrieval" with no literal "ML
    # engineer" string).
    core_score, _ = features.score_concept_groups(title_text)
    core_score = min(core_score, 1.5)  # title text is short; cap contribution
    raw = 0.55 * min(pos, 3) + 0.45 * core_score - 0.6 * min(neg, 2)
    # Map roughly onto [0, 1.3]
    return _clamp(raw / 2.0 + 0.15, 0.0, 1.3)


def title_gate(title_score: float) -> float:
    """Smooth gate applied to the skills component. A perfectly matched
    title leaves skills scoring untouched (gate ~1.15); a totally unrelated
    title heavily discounts the skills bucket but doesn't zero it outright
    (e.g. an HR Manager who happens to list 9 AI keywords gets the skills
    bucket cut to ~20% of its raw value — exactly the sample_submission.csv
    trap this dataset is designed to punish).
    """
    return _clamp(0.20 + title_score * 0.85, 0.20, 1.15)


# ---------------------------------------------------------------------------
# Narrative fit (career history + summary) — the highest-weighted bucket
# ---------------------------------------------------------------------------

def narrative_fit_score(narrative_text: str) -> Tuple[float, Dict[str, List[str]]]:
    raw, matches = features.score_concept_groups(narrative_text)
    core = features.core_concept_strength(matches)
    # Reward depth (core concepts) more than breadth (everything else).
    combined = 0.7 * core + 0.3 * raw
    return _clamp(combined / 3.0, 0.0, 1.3), matches


# ---------------------------------------------------------------------------
# Skills fit (trust-weighted, not a raw keyword count)
# ---------------------------------------------------------------------------

def _skill_trust_factor(skill: Dict[str, Any], assessment_scores: Dict[str, float]) -> float:
    endorsements = skill.get("endorsements", 0) or 0
    duration = skill.get("duration_months", 0) or 0
    name = skill.get("name", "")
    proficiency = skill.get("proficiency", "intermediate")

    endorsement_term = min(math.log1p(endorsements) / math.log1p(50), 1.0)
    duration_term = min(duration, 36) / 36.0
    trust = 0.30 + 0.40 * endorsement_term + 0.30 * duration_term

    # Cross-check against the platform's own (objective) assessment score,
    # if Redrob ran one for this skill — this is the strongest anti-stuffing
    # signal available, since it isn't self-reported.
    assess = assessment_scores.get(name)
    if assess is not None:
        if assess >= 70:
            trust = min(trust * 1.25, 1.4)
        elif assess < 30 and proficiency in ("advanced", "expert"):
            trust *= 0.5  # claims expertise the assessment doesn't support

    return _clamp(trust, 0.15, 1.4)


def skills_fit_score(
    candidate: Dict[str, Any],
) -> Tuple[float, List[Tuple[str, float]]]:
    """Returns (score in [0, ~1.3], [(matched_skill_name, contribution), ...]
    sorted by contribution desc) for use in reasoning generation.
    """
    skills = candidate.get("skills", []) or []
    signals = candidate.get("redrob_signals", {}) or {}
    assessment_scores = signals.get("skill_assessment_scores", {}) or {}

    total = 0.0
    contributions: List[Tuple[str, float]] = []

    for skill in skills:
        name_norm = features.normalize_text(skill.get("name", ""))
        if not name_norm:
            continue
        best_group_weight = 0.0
        for group, spec in config.CONCEPT_GROUPS.items():
            if any(term in name_norm for term in spec["terms"]):
                best_group_weight = max(best_group_weight, spec["weight"])
        if best_group_weight == 0.0:
            continue

        prof_weight = config.PROFICIENCY_WEIGHT.get(
            skill.get("proficiency", "intermediate"), 0.6
        )
        trust = _skill_trust_factor(skill, assessment_scores)
        contribution = best_group_weight * prof_weight * trust
        total += contribution
        contributions.append((skill.get("name", ""), contribution))

    contributions.sort(key=lambda kv: kv[1], reverse=True)
    score = _clamp(total / 4.0, 0.0, 1.3)
    return score, contributions


# ---------------------------------------------------------------------------
# Experience fit
# ---------------------------------------------------------------------------

def experience_fit_score(years_of_experience: float) -> float:
    band = config.EXPERIENCE_BAND
    y = years_of_experience or 0.0

    if band["full_score_low"] <= y <= band["full_score_high"]:
        return 1.0
    if y < band["full_score_low"]:
        if y <= band["hard_floor_years"]:
            return band["floor_score"]
        if y >= band["soft_low"]:
            span = band["full_score_low"] - band["soft_low"]
            return 0.75 + 0.25 * (y - band["soft_low"]) / max(span, 1e-6)
        span = band["soft_low"] - band["hard_floor_years"]
        return band["floor_score"] + (0.75 - band["floor_score"]) * (
            y - band["hard_floor_years"]
        ) / max(span, 1e-6)
    else:
        if y >= band["hard_ceiling_years"]:
            return band["floor_score"]
        if y <= band["soft_high"]:
            span = band["soft_high"] - band["full_score_high"]
            return 1.0 - 0.25 * (y - band["full_score_high"]) / max(span, 1e-6)
        span = band["hard_ceiling_years"] - band["soft_high"]
        return 0.75 - (0.75 - band["floor_score"]) * (
            y - band["soft_high"]
        ) / max(span, 1e-6)


# ---------------------------------------------------------------------------
# Location fit
# ---------------------------------------------------------------------------

def location_fit_score(candidate: Dict[str, Any]) -> float:
    profile = candidate.get("profile", {})
    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").strip().lower()
    relocate = bool((candidate.get("redrob_signals", {}) or {}).get("willing_to_relocate"))

    if country != "india":
        base = 0.35
        return _clamp(base + (0.15 if relocate else 0.0), 0.0, 0.55)

    if any(city in location for city in config.PRIMARY_CITIES):
        return 1.0
    if any(city in location for city in config.WELCOMED_CITIES):
        return 0.90
    base = 0.70
    return _clamp(base + (0.10 if relocate else 0.0), 0.0, 0.85)


# ---------------------------------------------------------------------------
# Education fit
# ---------------------------------------------------------------------------

def education_fit_score(candidate: Dict[str, Any]) -> float:
    education = candidate.get("education", []) or []
    if not education:
        return config.EDUCATION_TIER_SCORE["unknown"] * 0.9

    best = 0.0
    for edu in education:
        tier = edu.get("tier") or "unknown"
        score = config.EDUCATION_TIER_SCORE.get(tier, config.EDUCATION_TIER_SCORE["unknown"])
        field = (edu.get("field_of_study") or "").lower()
        if any(term in field for term in config.RELEVANT_FIELDS_OF_STUDY):
            score = min(score + 0.10, 1.0)
        best = max(best, score)
    return best


# ---------------------------------------------------------------------------
# Behavioral multiplier
# ---------------------------------------------------------------------------

def behavioral_multiplier(
    candidate: Dict[str, Any], reference_date: date
) -> Tuple[float, Dict[str, Any]]:
    signals = candidate.get("redrob_signals", {}) or {}
    cfg = config.BEHAVIORAL
    detail: Dict[str, Any] = {}

    mult = 1.0

    last_active = features.parse_date(signals.get("last_active_date"))
    if last_active:
        days_inactive = max((reference_date - last_active).days, 0)
        detail["days_inactive"] = days_inactive
        if days_inactive <= cfg["recency_full_score_days"]:
            recency_mult = 1.0
        else:
            decay_days = days_inactive - cfg["recency_full_score_days"]
            recency_mult = 0.5 ** (decay_days / cfg["recency_half_life_days"])
            recency_mult = max(recency_mult, cfg["recency_floor_multiplier"])
        mult *= recency_mult
        detail["recency_mult"] = round(recency_mult, 3)

    response_rate = signals.get("recruiter_response_rate")
    if response_rate is not None:
        # 0.0 -> 0.85x, 0.5 -> 1.0x, 1.0 -> 1.10x
        rr_mult = 0.85 + 0.25 * _clamp(response_rate, 0, 1)
        mult *= rr_mult
        detail["response_rate_mult"] = round(rr_mult, 3)

    interview_rate = signals.get("interview_completion_rate")
    if interview_rate is not None:
        ic_mult = 0.90 + 0.15 * _clamp(interview_rate, 0, 1)
        mult *= ic_mult
        detail["interview_completion_mult"] = round(ic_mult, 3)

    if signals.get("open_to_work_flag"):
        mult *= 1.0 + cfg["open_to_work_bonus"]
    else:
        mult *= 1.0 - cfg["open_to_work_penalty"]

    notice = signals.get("notice_period_days")
    if notice is not None:
        extra_30d_blocks = max(0, (notice - cfg["notice_period_sweet_spot_days"])) / 30.0
        mult *= max(1.0 - extra_30d_blocks * cfg["notice_period_penalty_per_30d"], 0.8)
        detail["notice_period_days"] = notice

    verified_count = sum(
        1 for k in ("verified_email", "verified_phone", "linkedin_connected")
        if signals.get(k)
    )
    mult *= 1.0 + verified_count * cfg["verified_bonus_each"]

    mult = _clamp(mult, cfg["multiplier_floor"], cfg["multiplier_cap"])
    detail["final_multiplier"] = round(mult, 3)
    return mult, detail


# ---------------------------------------------------------------------------
# Full pipeline for one candidate
# ---------------------------------------------------------------------------

def score_candidate(
    candidate: Dict[str, Any], reference_date: date
) -> Dict[str, Any]:
    texts = features.candidate_full_text(candidate)
    profile = candidate.get("profile", {})

    title_score = title_fit_score(texts["title_text"])
    gate = title_gate(title_score)

    narrative_score, narrative_matches = narrative_fit_score(texts["narrative_text"])
    raw_skills_score, skill_contributions = skills_fit_score(candidate)
    gated_skills_score = raw_skills_score * gate

    experience_score = experience_fit_score(profile.get("years_of_experience", 0))
    location_score = location_fit_score(candidate)
    education_score = education_fit_score(candidate)

    weights = config.WEIGHTS
    base_fit = (
        weights["title"] * title_score
        + weights["narrative"] * narrative_score
        + weights["skills"] * gated_skills_score
        + weights["experience"] * experience_score
        + weights["location"] * location_score
        + weights["education"] * education_score
    )

    anti_mult, anti_flags = features.anti_pattern_multiplier(candidate)
    behavior_mult, behavior_detail = behavioral_multiplier(candidate, reference_date)
    is_honeypot, integrity_mult, integrity_flags_list = features.integrity_gate(candidate)

    pre_integrity_score = base_fit * anti_mult * behavior_mult
    final_score = 0.0 if is_honeypot else pre_integrity_score * integrity_mult

    return {
        "candidate_id": candidate.get("candidate_id"),
        "final_score": final_score,
        "is_honeypot": is_honeypot,
        "components": {
            "title_score": round(title_score, 4),
            "title_gate": round(gate, 4),
            "narrative_score": round(narrative_score, 4),
            "raw_skills_score": round(raw_skills_score, 4),
            "gated_skills_score": round(gated_skills_score, 4),
            "experience_score": round(experience_score, 4),
            "location_score": round(location_score, 4),
            "education_score": round(education_score, 4),
            "base_fit": round(base_fit, 4),
        },
        "anti_pattern_multiplier": round(anti_mult, 4),
        "anti_pattern_flags": anti_flags,
        "behavioral_multiplier": round(behavior_mult, 4),
        "behavioral_detail": behavior_detail,
        "integrity_flags": integrity_flags_list,
        "integrity_multiplier": round(integrity_mult, 4),
        "narrative_matches": narrative_matches,
        "skill_contributions": skill_contributions[:6],
    }
