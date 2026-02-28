"""
test_features.py — Feature engineering unit tests.
"""
import pytest
from score import compute_time_of_day_risk, compute_device_familiarity, compute_velocity


# ─── compute_time_of_day_risk ────────────────────────────────────────

@pytest.mark.parametrize("timestamp,expected", [
    ("2025-01-15T02:00:00Z", 0.9),   # hour 2 → night
    ("2025-01-15T06:00:00Z", 0.6),   # hour 6 → early morning
    ("2025-01-15T12:00:00Z", 0.4),   # hour 12 → midday (lowest risk)
    ("2025-01-15T22:00:00Z", 0.7),   # hour 22 → late night
])
def test_compute_time_of_day_risk(timestamp, expected):
    result = compute_time_of_day_risk(timestamp)
    assert abs(result - expected) < 0.05, f"Expected ~{expected} for {timestamp}, got {result}"


# ─── compute_device_familiarity ──────────────────────────────────────

def test_device_familiarity_known_device():
    """Matching IMEI + single MSISDN + zero correlation → high familiarity (1.0)."""
    result = compute_device_familiarity(
        imei_match=True, imei_cardinality=1, imei_swap_corr=0.0
    )
    assert result >= 0.95


def test_device_familiarity_unfamiliar_device():
    """Mismatched IMEI + high cardinality + high correlation → low familiarity."""
    result = compute_device_familiarity(
        imei_match=False, imei_cardinality=10, imei_swap_corr=1.0
    )
    assert result < 0.1


# ─── compute_velocity ────────────────────────────────────────────────

def test_compute_velocity_normal():
    """5 transactions in 60 minutes → 5.0 txn/hr."""
    txns = [{"amount": "1000"}] * 5
    result = compute_velocity(txns, time_to_first_min=60)
    assert result == pytest.approx(5.0, abs=0.01)


def test_compute_velocity_empty_list():
    """No transactions → velocity is 0.0."""
    assert compute_velocity([], time_to_first_min=60) == 0.0
