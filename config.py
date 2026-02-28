"""
config.py — Central configuration for BioGuard fraud detection system.
All thresholds, weights, paths, and tunable parameters in one place.
Supports .env overrides via os.environ.
"""
import os
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR
MODEL_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "bioguard_audit.db"

# ─── Data files ──────────────────────────────────────────────────────
SIM_SWAPS_CSV = DATA_DIR / "sim_swaps.csv"
USSD_SESSIONS_CSV = DATA_DIR / "ussd_sessions.csv"
TRANSACTIONS_CSV = DATA_DIR / "transactions.csv"
SCORED_EVENTS_CSV = DATA_DIR / "scored_events.csv"
AI_SCORES_CSV = DATA_DIR / "ai_scores.csv"

# ─── Model artifacts ────────────────────────────────────────────────
XGBOOST_MODEL_PATH = MODEL_DIR / "xgboost_model.joblib"
ISOLATION_FOREST_PATH = MODEL_DIR / "isolation_forest.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"

# ─── Action tier thresholds ──────────────────────────────────────────
TIER_THRESHOLDS = {
    "T4_FREEZE":   81,
    "T3_STEP_UP":  61,
    "T2_FRICTION": 41,
    "T1_OBSERVE":  21,
    "T0_ALLOW":    0,
}

TIER_DESCRIPTIONS = {
    "T0_ALLOW":    "Transaction processed normally.",
    "T1_OBSERVE":  "3-min delay applied. SMS alert sent.",
    "T2_FRICTION": "Limit reduced to KES 10K. PIN re-confirm required.",
    "T3_STEP_UP":  "Transaction held. USSD knowledge challenge issued.",
    "T4_FREEZE":   "All outbound frozen. IVR call. Case created.",
}

TIER_DISPLAY = {
    "T0_ALLOW":    ("✅ Allow",    "#2ecc71"),
    "T1_OBSERVE":  ("👁️ Observe",  "#f39c12"),
    "T2_FRICTION": ("⚠️ Friction", "#e67e22"),
    "T3_STEP_UP":  ("🔐 Step-Up",  "#e74c3c"),
    "T4_FREEZE":   ("🚨 Freeze",   "#c0392b"),
}

# ─── Combination names ───────────────────────────────────────────────
COMBO_NAMES = {
    "ALPHA":   "Wallet Key on Fraud Phone",
    "BETA":    "Rehearsed Drain",
    "GAMMA":   "Industrial Assembly Line",
    "DELTA":   "Ghost SIM Drain",
    "EPSILON": "Displacement + Drain",
    "ZETA":    "Behavioral Drain (clean device)",
}

# ─── Combination bonus scores ────────────────────────────────────────
COMBO_BONUSES = {
    "ALPHA":   50,
    "BETA":    45,
    "GAMMA":   55,
    "DELTA":   35,
    "EPSILON": 35,
    "ZETA":    45,
}

# ─── Individual signal scores ────────────────────────────────────────
SIGNAL_SCORES = {
    "time_critical":          10,   # < 15 min
    "time_high":               6,   # < 60 min
    "time_moderate":           2,   # < 240 min
    "imei_corr_high":         12,   # > 0.5
    "imei_card_high":          8,   # > 3
    "imei_card_moderate":      3,   # > 2
    "imei_mismatch":           3,
    "dwell_var_low":           8,   # < 2.0s
    "dwell_var_moderate":      4,   # < 5.0s
    "intent_purity_extreme":   6,   # > 0.95
    "intent_purity_high":      3,   # > 0.8
    "no_non_financial":        5,   # == 0
    "low_non_financial":       2,   # <= 1
    "sessions_burst":          5,   # > 3 in 1h
    "sessions_moderate":       2,   # > 1 in 1h
    "all_unknown_recip":      10,
    "any_unknown_recip":       3,
    "drain_extreme":           8,   # > 0.8
    "drain_high":              5,   # > 0.6
    "drain_moderate":          2,   # > 0.4
    "many_recipients":         4,   # > 2
    "young_recipient":         3,   # < 30d
    "displacement_extreme":    5,   # > 100km
    "displacement_high":       3,   # > 50km
    "agent_risk_high":         4,   # > 0.6
    "agent_risk_moderate":     2,   # > 0.4
    "night_txn":               3,   # 00:00-05:00
    "high_velocity":           4,   # > 3 txn/hr
    "device_unfamiliar":       3,   # composite
}

# ─── Time decay brackets ─────────────────────────────────────────────
TIME_DECAY = [
    (1,    1.00),  # < 1 hour
    (4,    0.85),  # < 4 hours
    (12,   0.65),  # < 12 hours
    (24,   0.45),  # < 24 hours
    (72,   0.25),  # < 72 hours
    (None, 0.10),  # >= 72 hours
]

# ─── Ensemble weights ────────────────────────────────────────────────
ENSEMBLE_WEIGHTS = {
    "rules":            0.30,
    "xgboost":          0.45,
    "isolation_forest":  0.25,
}

# ─── XGBoost hyperparameters ─────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":      80,
    "max_depth":          3,
    "learning_rate":     0.05,
    "eval_metric":       "aucpr",
    "random_state":      42,
    "reg_alpha":         2.0,
    "reg_lambda":        5.0,
    "min_child_weight":  8,
    "subsample":         0.6,
    "colsample_bytree":  0.6,
    "gamma":             1.0,
}
XGB_THRESHOLD = 0.55
ENSEMBLE_FPR_LIMIT = 0.005

# ─── Isolation Forest hyperparameters ────────────────────────────────
ISO_PARAMS = {
    "n_estimators":   200,
    "contamination":  0.05,
    "random_state":   42,
    "max_features":   0.7,
}

# ─── Feature noise levels (for synthetic robustness) ─────────────────
NOISE_LEVELS = {
    "time_to_first_min":      0.20,
    "session_count_1h":       0.05,
    "imei_match":             0.0,
    "imei_cardinality":       0.10,
    "imei_swap_corr":         0.15,
    "avg_dwell_variance":     0.25,
    "avg_mean_dwell":         0.25,
    "avg_directness":         0.10,
    "non_financial_count":    0.10,
    "intent_purity":          0.10,
    "max_drain_ratio":        0.08,
    "cumulative_drain":       0.08,
    "all_unknown":            0.0,
    "any_unknown":            0.0,
    "distinct_recipients":    0.05,
    "avg_recipient_age":      0.15,
    "displacement_km":        0.30,
    "agent_risk":             0.10,
    "txn_velocity":           0.15,
    "time_of_day_risk":       0.05,
    "device_familiarity":     0.0,
}

# ─── Feature column list ─────────────────────────────────────────────
FEATURE_COLS = [
    "time_to_first_min", "session_count_1h", "imei_match", "imei_cardinality",
    "imei_swap_corr", "avg_dwell_variance", "avg_mean_dwell", "avg_directness",
    "non_financial_count", "intent_purity", "max_drain_ratio", "cumulative_drain",
    "all_unknown", "any_unknown", "distinct_recipients", "avg_recipient_age",
    "displacement_km", "agent_risk", "txn_velocity", "time_of_day_risk",
    "device_familiarity",
]

# ─── Data generation ─────────────────────────────────────────────────
NUM_EVENTS = 2000
FRAUD_RATE = 0.03
DATA_RANDOM_SEED = 2025

# ─── API settings ────────────────────────────────────────────────────
API_HOST = os.environ.get("BIOGUARD_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("BIOGUARD_PORT", "8000"))
API_KEY = os.environ.get("BIOGUARD_API_KEY", "bg-dev-key-2026")
RATE_LIMIT_PER_MINUTE = int(os.environ.get("BIOGUARD_RATE_LIMIT", "60"))

# ─── Logging ──────────────────────────────────────────────────────────
import logging

LOG_LEVEL = os.environ.get("BIOGUARD_LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s"

def get_logger(name: str) -> logging.Logger:
    """Get a configured logger for the given module name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    return logger
