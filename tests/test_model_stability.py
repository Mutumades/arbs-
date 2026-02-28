"""
test_model_stability.py — Model stability and configuration integrity tests.
"""
import pytest
from score import load_ml_models, score_with_ensemble
from config import FEATURE_COLS, ENSEMBLE_WEIGHTS, COMBO_BONUSES, COMBO_NAMES
import numpy as np


def test_load_ml_models_returns_dict_with_expected_keys():
    """load_ml_models() always returns a dict with the four expected keys."""
    models = load_ml_models()
    assert isinstance(models, dict)
    for key in ("xgboost", "isolation_forest", "scaler", "metadata"):
        assert key in models


def test_score_with_ensemble_fallback_when_models_none():
    """score_with_ensemble() falls back gracefully when ML models are not loaded."""
    models = {"xgboost": None, "isolation_forest": None, "scaler": None, "metadata": None}
    dummy_array = np.zeros(len(FEATURE_COLS))
    result = score_with_ensemble(dummy_array, models, rule_score=50)
    assert "ensemble_score" in result
    assert "ensemble_prediction" in result


def test_feature_cols_has_21_features():
    assert len(FEATURE_COLS) == 21


def test_ensemble_weights_sum_to_one():
    total = sum(ENSEMBLE_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-9, f"ENSEMBLE_WEIGHTS sum to {total}, expected 1.0"


def test_combo_bonuses_keys_match_combo_names():
    assert set(COMBO_BONUSES.keys()) == set(COMBO_NAMES.keys())
