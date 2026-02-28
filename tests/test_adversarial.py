"""
test_adversarial.py — Adversarial and boundary condition tests.
"""
import pytest
from score import compute_risk_score


def test_slow_fraudster_caught_at_t2():
    """A slow fraudster (waits 3hr) with moderate signals should still be caught at T2+."""
    slow_fraud = {
        "time_to_first_min": 180,
        "imei_cardinality": 4,
        "imei_match": False,
        "imei_swap_corr": 0.45,
        "avg_dwell_variance": 1.5,   # < 2.0 → rehearsed; also triggers BETA
        "intent_purity": 0.85,
        "non_financial_count": 1,
        "session_count_1h": 0,
        "all_recipients_unknown": True,
        "max_drain_ratio": 0.76,
        "displacement_km": 85.0,
        "agent_risk": 0.55,
        "cumulative_drain": 0.76,
    }
    result = compute_risk_score(slow_fraud)
    assert result.score >= 41, (
        f"Slow fraudster should be caught at T2+ (score≥41), got {result.score}"
    )


def test_legit_emergency_drain_below_t3():
    """A legitimate emergency drain to a known recipient should stay below T3 (score < 61)."""
    legit_emergency = {
        "time_to_first_min": 25,
        "imei_match": False,
        "imei_cardinality": 1,
        "imei_swap_corr": 0.0,
        "avg_dwell_variance": 8.5,
        "intent_purity": 0.6,
        "non_financial_count": 2,
        "session_count_1h": 2,
        "all_recipients_unknown": False,
        "any_unknown_recipient": False,
        "max_drain_ratio": 0.83,
        "displacement_km": 3.0,
        "agent_risk": 0.30,
        "cumulative_drain": 0.83,
        "avg_recipient_age_days": 450,
    }
    result = compute_risk_score(legit_emergency)
    assert result.score < 61, (
        f"Legit emergency drain should be below T3 (score<61), got {result.score}"
    )


def test_demographic_fairness_msisdn_agnostic(fraud_features):
    """Scoring must be MSISDN-agnostic: same features, different MSISDNs → same score."""
    features_a = {**fraud_features, "msisdn": "254722000001"}
    features_b = {**fraud_features, "msisdn": "254799999999"}
    result_a = compute_risk_score(features_a)
    result_b = compute_risk_score(features_b)
    assert result_a.score == result_b.score, (
        f"Scores differ by MSISDN: {result_a.score} vs {result_b.score}"
    )
