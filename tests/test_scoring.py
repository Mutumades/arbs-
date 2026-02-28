"""
test_scoring.py — Core scoring logic tests.
"""
import pytest
from score import (
    compute_risk_score,
    compute_time_decay,
    determine_action,
    evaluate_combinations,
)


# ─── compute_risk_score ──────────────────────────────────────────────

def test_compute_risk_score_fraud_is_high(fraud_features):
    result = compute_risk_score(fraud_features)
    assert result.score >= 81, f"Expected score ≥ 81, got {result.score}"
    assert result.action == "T4_FREEZE"


def test_compute_risk_score_legit_is_low(legit_features):
    result = compute_risk_score(legit_features)
    assert result.score <= 20, f"Expected score ≤ 20, got {result.score}"
    assert result.action == "T0_ALLOW"


# ─── determine_action boundary values ────────────────────────────────

@pytest.mark.parametrize("score,expected", [
    (0,   "T0_ALLOW"),
    (20,  "T0_ALLOW"),
    (21,  "T1_OBSERVE"),
    (40,  "T1_OBSERVE"),
    (41,  "T2_FRICTION"),
    (60,  "T2_FRICTION"),
    (61,  "T3_STEP_UP"),
    (80,  "T3_STEP_UP"),
    (81,  "T4_FREEZE"),
    (100, "T4_FREEZE"),
])
def test_determine_action_boundaries(score, expected):
    assert determine_action(score) == expected


# ─── compute_time_decay ──────────────────────────────────────────────

@pytest.mark.parametrize("minutes,expected", [
    (30,   1.00),   # < 1 hour
    (120,  0.85),   # < 4 hours
    (600,  0.65),   # < 12 hours
    (1200, 0.45),   # < 24 hours
    (3000, 0.25),   # < 72 hours
    (5000, 0.10),   # >= 72 hours
])
def test_compute_time_decay_brackets(minutes, expected):
    assert compute_time_decay(minutes) == expected


# ─── evaluate_combinations ──────────────────────────────────────────

def test_evaluate_combinations_alpha():
    features = {
        "time_to_first_min": 30,
        "imei_swap_corr": 0.8,
        "intent_purity": 0.95,
        "all_recipients_unknown": True,
    }
    triggered = evaluate_combinations(features)
    assert "ALPHA" in triggered


def test_evaluate_combinations_all_six():
    """Each combination triggers with a crafted input that satisfies its conditions."""
    cases = {
        "ALPHA": {
            "time_to_first_min": 30, "imei_swap_corr": 0.8,
            "intent_purity": 0.95, "all_recipients_unknown": True,
        },
        "BETA": {
            "time_to_first_min": 120, "avg_dwell_variance": 1.5,
            "max_drain_ratio": 0.8, "all_recipients_unknown": True,
        },
        "GAMMA": {
            "imei_cardinality": 5, "imei_swap_corr": 0.6,
            "time_to_first_min": 100, "max_drain_ratio": 0.1,
        },
        "DELTA": {
            "time_to_first_min": 60, "non_financial_count": 0,
            "session_count_1h": 2, "cumulative_drain": 0.8,
        },
        "EPSILON": {
            "time_to_first_min": 200, "displacement_km": 100,
            "all_recipients_unknown": True, "max_drain_ratio": 0.6,
        },
        "ZETA": {
            "time_to_first_min": 15, "non_financial_count": 0,
            "max_drain_ratio": 0.8, "all_recipients_unknown": True,
            "avg_dwell_variance": 2.0,
        },
    }
    for combo_name, features in cases.items():
        triggered = evaluate_combinations(features)
        assert combo_name in triggered, (
            f"Expected {combo_name} to trigger with features {features}, got {triggered}"
        )


# ─── Regression: demo_fraud scenario produces T4_FREEZE ──────────────

def test_demo_fraud_regression(fraud_features):
    """Classic fraud demo scenario must always produce score ≥ 81."""
    result = compute_risk_score(fraud_features)
    assert result.score >= 81
