# BioGuard — System Architecture

## System Overview

```
 ┌──────────────────────────────────────────────────────────┐
 │                     BioGuard Platform                    │
 │                                                          │
 │  ┌──────────┐    ┌──────────────┐    ┌───────────────┐  │
 │  │ Raw Data │───▶│   Feature    │───▶│ Rule Scoring  │  │
 │  │ (events) │    │ Engineering  │    │  (score.py)   │  │
 │  └──────────┘    └──────────────┘    └───────┬───────┘  │
 │                                              │           │
 │                                     ┌────────▼────────┐  │
 │                                     │  ML Ensemble    │  │
 │                                     │ (ai_scoring.py) │  │
 │                                     └────────┬────────┘  │
 │                                              │           │
 │                                     ┌────────▼────────┐  │
 │                                     │  Action Tier    │  │
 │                                     │ T0–T4 Decision  │  │
 │                                     └─────────────────┘  │
 └──────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
Raw Events (SIM swap + USSD sessions + transactions)
    │
    ▼
Feature Engineering (scoring_engine.py / score.py)
    │   • Temporal: time_to_first_min, time_of_day_risk
    │   • Device:   imei_match, imei_cardinality, imei_swap_corr, device_familiarity
    │   • Behavior: avg_dwell_variance, intent_purity, non_financial_count, txn_velocity
    │   • Financial: max_drain_ratio, cumulative_drain, distinct_recipients
    │   • Location: displacement_km
    │   • Agent:    agent_risk
    ▼
Rule-Based Scoring (score.py → compute_risk_score)
    │   • Signal scores accumulated per feature
    │   • 6 named detection combinations evaluated (ALPHA–ZETA)
    │   • Combination bonus added if triggered
    │   • Time-decay applied (score × decay_factor)
    ▼
ML Ensemble (ai_scoring.py → score_with_ensemble)
    │   • XGBoost fraud probability  (weight: 0.45)
    │   • Isolation Forest anomaly   (weight: 0.25)
    │   • Rule score normalised      (weight: 0.30)
    │   • Weighted sum → ensemble_score
    ▼
Action Tier Decision (score.py → determine_action)
    │   T0_ALLOW    score  0–20   Normal processing
    │   T1_OBSERVE  score 21–40   3-min delay + SMS alert
    │   T2_FRICTION score 41–60   Limit + PIN re-confirm
    │   T3_STEP_UP  score 61–80   Hold + USSD challenge
    │   T4_FREEZE   score 81–100  Freeze + IVR + case
    ▼
Response / Audit (api.py → SQLite audit_log)
```

---

## Component Descriptions

| File | Role |
|---|---|
| `config.py` | Central configuration: thresholds, weights, paths, logging |
| `score.py` | Authoritative scoring module: all rule logic and ML wrappers |
| `api.py` | FastAPI REST API: `/score`, `/score/ensemble`, demo endpoints |
| `scoring_engine.py` | CLI batch runner: loads CSVs, engineers features, scores, exports |
| `ai_scoring.py` | ML training pipeline: XGBoost + Isolation Forest + SHAP |
| `generate_synthetic_data.py` | Synthetic dataset generator for development and testing |

---

## Detection Combinations

Six named patterns detect distinct fraud archetypes:

| Name | Description | Key Conditions |
|---|---|---|
| **ALPHA** | Wallet Key on Fraud Phone | t<60min, IMEI corr>0.5, pure intent>0.9, unknown recipient |
| **BETA** | Rehearsed Drain | t<4hr, dwell variance<3s, drain>70%, unknown recipient |
| **GAMMA** | Industrial Assembly Line | IMEI cardinality>3, corr>0.4, any drain present |
| **DELTA** | Ghost SIM Drain | t<6hr, ≤1 non-financial action, session burst, cumulative drain>60% |
| **EPSILON** | Displacement + Drain | t<24hr, >50km displacement, unknown recipient, drain>50% |
| **ZETA** | Behavioral Drain (clean device) | t<30min, 0 non-financial, drain>75%, unknown recipient, low dwell |

Each triggered combination adds a bonus to the risk score (35–55 points).

---

## ML Pipeline

### Training (`ai_scoring.py`)
1. Load `scored_events.csv` (output of `scoring_engine.py`)
2. Engineer feature array (`FEATURE_COLS`, 21 features)
3. Train **XGBoost** classifier (fraud/legit binary)
   - Hyperparameters in `config.py` (`XGB_PARAMS`)
   - Threshold optimised for FPR ≤ 0.5%
4. Train **Isolation Forest** anomaly detector
5. Calibrate ensemble weights on validation set
6. Save models to `models/` directory with metadata JSON

### Inference (`score.py → score_with_ensemble`)
```
feature_array (21 values, FEATURE_COLS order)
    │
    ├── XGBoost.predict_proba → xgb_probability  × 0.45
    ├── IsolationForest.decision_function → iso_norm  × 0.25
    └── rule_score / 100                           × 0.30
         │
         └── ensemble_score → determine_action → action_tier
```

### SHAP Explanations
XGBoost SHAP values are computed per prediction and returned in the
`/score/ensemble` response for model interpretability.

---

## Deployment Architecture

```
Internet
    │
    ▼
┌─────────────────────────────────────────┐
│            Docker Compose               │
│                                         │
│  ┌────────────────┐  ┌───────────────┐  │
│  │   API Service  │  │   Dashboard   │  │
│  │  FastAPI/uvicorn│  │  Streamlit    │  │
│  │  port 8000     │  │  port 8501    │  │
│  └───────┬────────┘  └───────┬───────┘  │
│          │                   │          │
│          └─────────┬─────────┘          │
│                    │                    │
│           ┌────────▼────────┐           │
│           │  SQLite DB      │           │
│           │  (audit_log)    │           │
│           └─────────────────┘           │
└─────────────────────────────────────────┘
```

- **API** (`api.py`): REST endpoints, API-key auth, rate limiting, audit trail
- **Dashboard**: Interactive Streamlit UI for analysts
- **Audit DB** (`bioguard_audit.db`): Persistent SQLite log of all scored events
