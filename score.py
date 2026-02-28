"""
score.py — Single authoritative scoring module for BioGuard.

All scoring logic lives here. Imported by api.py, dashboard.py, and scoring_engine.py.
Eliminates the previous 3× code duplication problem.

Provides:
  - compute_risk_score(features) → RiskResult
  - evaluate_combinations(features) → list[str]
  - compute_time_decay(minutes) → float
  - determine_action(score) → str
  - load_ml_models() → dict
  - score_with_ensemble(features, models) → EnsembleResult
  - compute_new_features(features, swap) → dict  (velocity, time-of-day, device familiarity)
"""
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from config import (
    TIER_THRESHOLDS, TIER_DESCRIPTIONS, TIER_DISPLAY,
    COMBO_NAMES, COMBO_BONUSES, SIGNAL_SCORES, TIME_DECAY,
    ENSEMBLE_WEIGHTS, FEATURE_COLS,
    XGBOOST_MODEL_PATH, ISOLATION_FOREST_PATH, SCALER_PATH,
    MODEL_METADATA_PATH, get_logger,
)

logger = get_logger("score")


# ─── Result dataclasses ──────────────────────────────────────────────

@dataclass
class RiskResult:
    """Result from rule-based scoring."""
    score: int
    action: str
    action_description: str
    triggered_combinations: list = field(default_factory=list)
    explanation: list = field(default_factory=list)
    drain_ratio: float = 0.0

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "action": self.action,
            "action_description": self.action_description,
            "triggered_combinations": self.triggered_combinations,
            "explanation": self.explanation,
            "drain_ratio": self.drain_ratio,
        }


@dataclass
class EnsembleResult:
    """Result from ML ensemble scoring."""
    rule_score: int
    xgb_probability: float
    iso_anomaly_score: float
    ensemble_score: float
    ensemble_prediction: int
    action: str
    action_description: str
    triggered_combinations: list = field(default_factory=list)
    explanation: list = field(default_factory=list)
    shap_values: Optional[dict] = None

    def to_dict(self) -> dict:
        d = {
            "rule_score": self.rule_score,
            "xgb_probability": round(self.xgb_probability, 4),
            "iso_anomaly_score": round(self.iso_anomaly_score, 4),
            "ensemble_score": round(self.ensemble_score, 4),
            "ensemble_prediction": self.ensemble_prediction,
            "action": self.action,
            "action_description": self.action_description,
            "triggered_combinations": self.triggered_combinations,
            "explanation": self.explanation,
        }
        if self.shap_values:
            d["shap_values"] = self.shap_values
        return d


# ─── Time decay ──────────────────────────────────────────────────────

def compute_time_decay(minutes: float) -> float:
    """Apply temporal decay: risk diminishes as time since SIM swap increases."""
    hrs = minutes / 60
    for threshold_hrs, decay_val in TIME_DECAY:
        if threshold_hrs is None or hrs < threshold_hrs:
            return decay_val
    return TIME_DECAY[-1][1]


# ─── Action determination ────────────────────────────────────────────

def determine_action(score: int) -> str:
    """Map a risk score (0-100) to an action tier."""
    if score >= TIER_THRESHOLDS["T4_FREEZE"]:
        return "T4_FREEZE"
    elif score >= TIER_THRESHOLDS["T3_STEP_UP"]:
        return "T3_STEP_UP"
    elif score >= TIER_THRESHOLDS["T2_FRICTION"]:
        return "T2_FRICTION"
    elif score >= TIER_THRESHOLDS["T1_OBSERVE"]:
        return "T1_OBSERVE"
    else:
        return "T0_ALLOW"


# ─── Combination evaluation ──────────────────────────────────────────

def _eval_alpha(f: dict) -> bool:
    """Wallet Key on Fraud Phone: correlated IMEI + pure intent + unknown recip + fast."""
    t = f.get("time_to_first_min", f.get("t", 9999))
    ic = f.get("imei_swap_corr", f.get("ic", 0))
    ip = f.get("intent_purity", f.get("ip", 0))
    kr = f.get("all_recipients_unknown", not f.get("kr", True))
    return t < 60 and ic > 0.5 and ip > 0.9 and kr


def _eval_beta(f: dict) -> bool:
    """Rehearsed Drain: low dwell variance + high drain + unknown recip + within 4h."""
    t = f.get("time_to_first_min", f.get("t", 9999))
    dv = f.get("avg_dwell_variance", f.get("dv", 99))
    drain = f.get("max_drain_ratio", f.get("_drain", 0))
    kr = f.get("all_recipients_unknown", not f.get("kr", True))
    return t < 240 and dv < 3.0 and drain > 0.7 and kr


def _eval_gamma(f: dict) -> bool:
    """Industrial Assembly Line: serial IMEI reuse across victims."""
    ik = f.get("imei_cardinality", f.get("ik", 0))
    ic = f.get("imei_swap_corr", f.get("ic", 0))
    t = f.get("time_to_first_min", f.get("t", 9999))
    drain = f.get("max_drain_ratio", f.get("_drain", 0))
    return ik > 3 and ic > 0.4 and t < 1440 and drain > 0


def _eval_delta(f: dict) -> bool:
    """Ghost SIM Drain: no non-financial activity, session burst, high drain."""
    t = f.get("time_to_first_min", f.get("t", 9999))
    nf = f.get("non_financial_count", f.get("nf", 99))
    s1 = f.get("session_count_1h", f.get("s1", 0))
    cd = f.get("cumulative_drain", f.get("cd", 0))
    return t < 360 and nf <= 1 and s1 > 1 and cd > 0.6


def _eval_epsilon(f: dict) -> bool:
    """Displacement + Drain: geographic displacement + drain + unknown recip."""
    t = f.get("time_to_first_min", f.get("t", 9999))
    dk = f.get("displacement_km", f.get("dk", 0))
    kr = f.get("all_recipients_unknown", not f.get("kr", True))
    drain = f.get("max_drain_ratio", f.get("_drain", 0))
    return t < 1440 and dk > 50 and kr and drain > 0.5


def _eval_zeta(f: dict) -> bool:
    """Behavioral Drain: fast + zero non-fin + drain + unknown + low dwell (clean device)."""
    t = f.get("time_to_first_min", f.get("t", 9999))
    nf = f.get("non_financial_count", f.get("nf", 99))
    drain = f.get("max_drain_ratio", f.get("_drain", 0))
    kr = f.get("all_recipients_unknown", not f.get("kr", True))
    dv = f.get("avg_dwell_variance", f.get("dv", 99))
    return t < 30 and nf == 0 and drain > 0.75 and kr and dv < 4.0


COMBINATIONS = {
    "ALPHA":   _eval_alpha,
    "BETA":    _eval_beta,
    "GAMMA":   _eval_gamma,
    "DELTA":   _eval_delta,
    "EPSILON": _eval_epsilon,
    "ZETA":    _eval_zeta,
}


def evaluate_combinations(features: dict) -> list:
    """Evaluate all 6 named detection combinations. Returns list of triggered combo names."""
    triggered = []
    for name, func in COMBINATIONS.items():
        if func(features):
            triggered.append(name)
    return triggered


# ─── New feature computation ─────────────────────────────────────────

def compute_velocity(txns: list, time_to_first_min: float) -> float:
    """Transactions per hour post-swap."""
    if not txns or time_to_first_min <= 0:
        return 0.0
    hours = max(0.1, time_to_first_min / 60)
    return round(len(txns) / hours, 3)


def compute_time_of_day_risk(swap_ts_str: str) -> float:
    """Risk multiplier based on hour. Fraud peaks at night/early morning."""
    try:
        if "T" in str(swap_ts_str):
            hour = int(str(swap_ts_str).split("T")[1].split(":")[0])
        else:
            hour = 12
    except (IndexError, ValueError):
        hour = 12
    # Night (00-05): high risk, Early morning (05-07): moderate, Day: low
    if 0 <= hour < 5:
        return 0.9
    elif 5 <= hour < 7:
        return 0.6
    elif 22 <= hour <= 23:
        return 0.7
    else:
        return round(0.1 + (0.3 * (1 - abs(hour - 12) / 12)), 2)


def compute_device_familiarity(imei_match: bool, imei_cardinality: int,
                                imei_swap_corr: float) -> float:
    """Composite device familiarity: 1.0 = fully familiar, 0.0 = totally unfamiliar."""
    match_score = 1.0 if imei_match else 0.0
    card_score = max(0, 1.0 - (imei_cardinality - 1) / 10)
    corr_score = 1.0 - imei_swap_corr
    return round(0.4 * match_score + 0.3 * card_score + 0.3 * corr_score, 3)


# ─── Core scoring ────────────────────────────────────────────────────

def compute_risk_score(features: dict) -> RiskResult:
    """
    Compute rule-based risk score from behavioral features.

    Accepts features in either:
      - Full feature dict (from engineer_features): keys like time_to_first_min, imei_swap_corr
      - Dashboard short form: keys like t, ic, dv, ip, etc.

    Returns a RiskResult with score, action, triggered combos, and explanations.
    """
    score = 0
    triggered = []
    reasons = []

    # Normalize keys — support both full and short forms
    t = features.get("time_to_first_min", features.get("t", 9999))
    ic = features.get("imei_swap_corr", features.get("ic", 0))
    ik = features.get("imei_cardinality", features.get("ik", 0))
    im = features.get("imei_match", features.get("im", True))
    dv = features.get("avg_dwell_variance", features.get("dv", 10))
    ip = features.get("intent_purity", features.get("ip", 0.5))
    nf = features.get("non_financial_count", features.get("nf", 5))
    s1 = features.get("session_count_1h", features.get("s1", 0))
    amount = features.get("amount", 0)
    balance = features.get("balance_before", features.get("balance", 1))
    kr_known = features.get("recipient_is_known", features.get("kr", True))
    all_unknown = features.get("all_recipients_unknown", not kr_known)
    any_unknown = features.get("any_unknown_recipient", not kr_known)
    drain = features.get("max_drain_ratio", features.get("_drain", 0))
    if drain == 0 and amount > 0 and balance > 0:
        drain = amount / max(1, balance)
    ra = features.get("avg_recipient_age_days", features.get("ra", 999))
    rc = features.get("distinct_recipients", features.get("rc", 1))
    cd = features.get("cumulative_drain", features.get("cd", 0))
    dk = features.get("displacement_km", features.get("dk", 0))
    ar = features.get("agent_risk", features.get("ar", 0.2))
    txn_vel = features.get("txn_velocity", 0)
    tod_risk = features.get("time_of_day_risk", 0)
    dev_fam = features.get("device_familiarity", 1.0)

    # Store drain for combo evaluation
    features["_drain"] = drain

    # ── Temporal signals ──
    if t < 15:
        score += SIGNAL_SCORES["time_critical"]
        reasons.append(f"SIM swapped {t:.0f}min ago (critical window)")
    elif t < 60:
        score += SIGNAL_SCORES["time_high"]
        reasons.append(f"SIM swapped {t:.0f}min ago (high-risk)")
    elif t < 240:
        score += SIGNAL_SCORES["time_moderate"]

    # ── Device signals ──
    if ic > 0.5:
        score += SIGNAL_SCORES["imei_corr_high"]
        reasons.append(f"IMEI swap correlation {ic:.2f}")
    if ik > 3:
        score += SIGNAL_SCORES["imei_card_high"]
        reasons.append(f"IMEI hosted {ik} MSISDNs in 90d")
    elif ik > 2:
        score += SIGNAL_SCORES["imei_card_moderate"]
    if not im:
        score += SIGNAL_SCORES["imei_mismatch"]

    # ── Behavioral signals ──
    if dv < 2.0:
        score += SIGNAL_SCORES["dwell_var_low"]
        reasons.append(f"Dwell variance {dv:.1f}s (rehearsed)")
    elif dv < 5.0:
        score += SIGNAL_SCORES["dwell_var_moderate"]
    if ip > 0.95:
        score += SIGNAL_SCORES["intent_purity_extreme"]
        reasons.append("SIM used exclusively for mobile money")
    elif ip > 0.8:
        score += SIGNAL_SCORES["intent_purity_high"]
    if nf == 0:
        score += SIGNAL_SCORES["no_non_financial"]
        reasons.append("Zero non-financial activity")
    elif nf <= 1:
        score += SIGNAL_SCORES["low_non_financial"]
    if s1 > 3:
        score += SIGNAL_SCORES["sessions_burst"]
    elif s1 > 1:
        score += SIGNAL_SCORES["sessions_moderate"]

    # ── Transaction signals ──
    if all_unknown:
        score += SIGNAL_SCORES["all_unknown_recip"]
        reasons.append("Unknown recipient")
    elif any_unknown:
        score += SIGNAL_SCORES["any_unknown_recip"]
    if drain > 0.8:
        score += SIGNAL_SCORES["drain_extreme"]
        reasons.append(f"Drain ratio {drain:.0%}")
    elif drain > 0.6:
        score += SIGNAL_SCORES["drain_high"]
    elif drain > 0.4:
        score += SIGNAL_SCORES["drain_moderate"]
    if rc > 2:
        score += SIGNAL_SCORES["many_recipients"]
    if ra < 30:
        score += SIGNAL_SCORES["young_recipient"]
        reasons.append(f"Recipient account {ra:.0f}d old")

    # ── Location signals ──
    if dk > 100:
        score += SIGNAL_SCORES["displacement_extreme"]
        reasons.append(f"{dk:.0f}km geographic displacement")
    elif dk > 50:
        score += SIGNAL_SCORES["displacement_high"]

    # ── Agent risk ──
    if ar > 0.6:
        score += SIGNAL_SCORES["agent_risk_high"]
    elif ar > 0.4:
        score += SIGNAL_SCORES["agent_risk_moderate"]

    # ── New features ──
    if tod_risk > 0.7:
        score += SIGNAL_SCORES["night_txn"]
        reasons.append("Transaction during high-risk hours")
    if txn_vel > 3:
        score += SIGNAL_SCORES["high_velocity"]
        reasons.append(f"High transaction velocity ({txn_vel:.1f}/hr)")
    if dev_fam < 0.3:
        score += SIGNAL_SCORES["device_unfamiliar"]
        reasons.append("Device unfamiliar to this account")

    # ── Named combinations ──
    triggered = evaluate_combinations(features)
    for combo in triggered:
        score += COMBO_BONUSES[combo]

    # ── Time decay ──
    decay = compute_time_decay(t)
    score = min(100, int(score * decay))

    # ── Action tier ──
    action = determine_action(score)

    return RiskResult(
        score=score,
        action=action,
        action_description=TIER_DESCRIPTIONS[action],
        triggered_combinations=triggered,
        explanation=reasons,
        drain_ratio=round(drain, 3),
    )


# ─── ML model loading ────────────────────────────────────────────────

def load_ml_models() -> dict:
    """Load persisted ML models from disk. Returns dict with model objects or None."""
    models = {"xgboost": None, "isolation_forest": None, "scaler": None, "metadata": None}
    try:
        import joblib
        if XGBOOST_MODEL_PATH.exists():
            models["xgboost"] = joblib.load(XGBOOST_MODEL_PATH)
            logger.info("Loaded XGBoost model from %s", XGBOOST_MODEL_PATH)
        if ISOLATION_FOREST_PATH.exists():
            models["isolation_forest"] = joblib.load(ISOLATION_FOREST_PATH)
            logger.info("Loaded Isolation Forest from %s", ISOLATION_FOREST_PATH)
        if SCALER_PATH.exists():
            models["scaler"] = joblib.load(SCALER_PATH)
            logger.info("Loaded scaler from %s", SCALER_PATH)
        if MODEL_METADATA_PATH.exists():
            with open(MODEL_METADATA_PATH) as f:
                models["metadata"] = json.load(f)
            logger.info("Loaded model metadata")
    except ImportError:
        logger.warning("joblib not installed — ML model loading unavailable")
    except Exception as e:
        logger.error("Failed to load ML models: %s", e)
    return models


def score_with_ensemble(features_array: np.ndarray, models: dict,
                        rule_score: int) -> dict:
    """
    Score using the weighted ensemble (rules + XGBoost + Isolation Forest).

    Args:
        features_array: 1D numpy array of feature values in FEATURE_COLS order
        models: dict from load_ml_models()
        rule_score: pre-computed rule-based score (0-100)

    Returns:
        dict with ensemble scores and predictions
    """
    if models["xgboost"] is None or models["scaler"] is None:
        logger.warning("ML models not loaded — falling back to rules only")
        return {"ensemble_score": rule_score / 100.0, "ensemble_prediction": 1 if rule_score >= 41 else 0}

    X = features_array.reshape(1, -1)
    X_scaled = models["scaler"].transform(X)

    # XGBoost probability
    xgb_proba = models["xgboost"].predict_proba(X)[:, 1][0]

    # Isolation Forest anomaly score (flip so higher = more anomalous)
    iso_score_raw = models["isolation_forest"].decision_function(X_scaled)[0]
    metadata = models.get("metadata", {})
    iso_min = metadata.get("iso_score_min", -0.5)
    iso_max = metadata.get("iso_score_max", 0.5)
    iso_norm = 1 - (iso_score_raw - iso_min) / max(0.001, iso_max - iso_min)
    iso_norm = max(0, min(1, iso_norm))

    # Weighted ensemble
    rule_norm = rule_score / 100.0
    weights = ENSEMBLE_WEIGHTS
    ensemble_val = (
        weights["rules"] * rule_norm +
        weights["xgboost"] * xgb_proba +
        weights["isolation_forest"] * iso_norm
    )

    threshold = metadata.get("ensemble_threshold", 0.5)
    prediction = 1 if ensemble_val >= threshold else 0

    return {
        "rule_score_norm": round(rule_norm, 4),
        "xgb_probability": round(xgb_proba, 4),
        "iso_anomaly_score": round(iso_norm, 4),
        "ensemble_score": round(ensemble_val, 4),
        "ensemble_prediction": prediction,
    }
