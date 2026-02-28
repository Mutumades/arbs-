"""
conftest.py — Shared pytest fixtures for BioGuard tests.
"""
import pytest


@pytest.fixture
def fraud_features():
    """Classic fast-fraudster feature dict (mirrors demo_fraud scenario)."""
    return {
        "time_to_first_min": 7,
        "imei_swap_corr": 0.83,
        "imei_cardinality": 6,
        "imei_match": False,
        "avg_dwell_variance": 0.9,
        "intent_purity": 1.0,
        "non_financial_count": 0,
        "session_count_1h": 4,
        "all_recipients_unknown": True,
        "any_unknown_recipient": True,
        "max_drain_ratio": 0.941,
        "distinct_recipients": 2,
        "avg_recipient_age_days": 8,
        "displacement_km": 142.0,
        "agent_risk": 0.78,
        "time_of_day_risk": 0.9,
        "cumulative_drain": 0.94,
        "device_familiarity": 0.2,
        "amount": 45000,
        "balance_before": 47800,
    }


@pytest.fixture
def legit_features():
    """Normal legitimate-user feature dict (mirrors demo_legit scenario)."""
    return {
        "time_to_first_min": 1440,
        "imei_swap_corr": 0.0,
        "imei_cardinality": 1,
        "imei_match": True,
        "avg_dwell_variance": 12.5,
        "intent_purity": 0.3,
        "non_financial_count": 8,
        "session_count_1h": 0,
        "all_recipients_unknown": False,
        "any_unknown_recipient": False,
        "max_drain_ratio": 0.057,
        "distinct_recipients": 1,
        "avg_recipient_age_days": 890,
        "displacement_km": 2.1,
        "agent_risk": 0.15,
        "time_of_day_risk": 0.4,
        "cumulative_drain": 0.06,
        "device_familiarity": 1.0,
        "amount": 2000,
        "balance_before": 35000,
    }
