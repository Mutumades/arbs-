# 🛡️ BioGuard — AI-Powered Fraud Detection Platform

![CI](https://github.com/bioguard/sim-swap-detection/actions/workflows/ci.yml/badge.svg)

**Multi-layered behavioral intelligence for financial services fraud prevention**

Detects SIM-swap cash-out fraud on **any phone type** using only USSD/STK gateway and network signals.
No smartphone apps, no internet, no sensors — works on every GSM phone in Kenya.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic data (2,000 labeled events)
python generate_synthetic_data.py

# 3. Run rule-based scoring engine
python scoring_engine.py

# 4. Train ML models (XGBoost + Isolation Forest + Ensemble)
python ai_scoring.py

# 5. Launch API server (port 8000)
python api.py

# 6. Launch dashboard (port 8501)
streamlit run dashboard.py

# 7. Run tests
pytest tests/ -v
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for full system design.

```
config.py       ← Central configuration
score.py        ← Single scoring module (rules + ML)
├── api.py      ← FastAPI REST API (auth, rate limiting, audit)
├── dashboard.py ← Streamlit UI (dark theme, live scoring)
└── scoring_engine.py ← CLI analysis runner
```

## API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | No | Health check |
| `/score` | POST | Yes | Rule-based scoring |
| `/score/ensemble` | POST | Yes | ML ensemble scoring |
| `/demo/fraud` | GET | No | Classic fraud demo |
| `/demo/legit` | GET | No | Normal user demo |
| `/demo/edge_emergency` | GET | No | Emergency drain demo |
| `/demo/slow_fraudster` | GET | No | Slow fraudster demo |
| `/stats` | GET | No | Scoring statistics |
| `/audit/recent` | GET | Yes | Recent audit entries |

**Auth**: Set header `X-API-Key: bg-dev-key-2026` (configurable via `.env`)

## Six Detection Combinations

| Combo | Name | What It Catches |
|-------|------|-----------------|
| ALPHA | Wallet Key on Fraud Phone | Correlated IMEI + immediate mobile money + unknown recipient |
| BETA | Rehearsed Drain | Low dwell variance + high drain + unknown recipient |
| GAMMA | Industrial Assembly Line | Serial IMEI reuse across multiple victim MSISDNs |
| DELTA | Ghost SIM Drain | Zero non-financial activity + session burst + drain |
| EPSILON | Displacement + Drain | Geographic displacement + drain + unknown recipient |
| ZETA | Behavioral Drain | Fast + zero non-fin + drain + unknown (clean device) |

## Data Generation — 7 Fraud Scenarios

| Archetype | Behavior |
|-----------|----------|
| classic_fast | Fast drain within minutes, known fraud IMEI |
| slow_fraudster | Waits hours, moderate drain, decoy calls |
| clean_device | New IMEI (no reuse), still drains fast |
| local_insider | Uses victim's home area, slow deliberate drain |
| partial_text_coaching | Coached via SMS/WhatsApp, intermittent pauses |
| social_engineering_pretext | Victim manipulated ("helping bank"), looks legit |

## Proportional Response Tiers

| Tier | Score | Action |
|------|-------|--------|
| T0 | 0-20 | ✅ Allow — normal processing |
| T1 | 21-40 | 👁️ Observe — 3-min delay, SMS alert |
| T2 | 41-60 | ⚠️ Friction — reduced limits, PIN re-confirm |
| T3 | 61-80 | 🔐 Step-Up — transaction held, USSD challenge |
| T4 | 81-100 | 🚨 Freeze — all outbound frozen, IVR call, case created |

## ML Models

- **Isolation Forest**: Unsupervised anomaly detection (catches novel patterns)
- **XGBoost**: Supervised classifier with 5-fold cross-validation
- **Ensemble**: Weighted blend (30% rules + 45% XGBoost + 25% IF)
- **SHAP**: Feature importance explanations for each prediction

Models are saved to `models/` with joblib and loaded automatically by the API.

## Monitoring & Drift Detection

```bash
# Health check: model artifacts, metadata summary
python monitor.py health

# Drift analysis: PSI, KS test, per-archetype score shifts
python monitor.py drift

# Score distribution: histograms, component separation, per-archetype accuracy
python monitor.py distribution
```

## Docker

```bash
docker-compose up --build
# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

## Tests

```bash
pytest tests/ -v --tb=short
```

Tests cover:
- All 6 named combinations + boundary/edge cases
- Time decay math + action tier boundaries
- API auth, rate limiting, endpoints
- Feature engineering (velocity, time-of-day, device familiarity)
- **Adversarial**: mimicry attacks, demographic fairness, new archetype scenarios
- **Model stability**: metadata validation, metric realism, bootstrap CI, pinned regression
- **Regression**: pinned scores for classic fraud, benign legit, emergency drain

## Project Structure

```
sim-swap-detection/
├── config.py                  # Central configuration
├── score.py                   # Unified scoring module
├── generate_synthetic_data.py # Data generation (7 fraud + 6 legit archetypes)
├── scoring_engine.py          # CLI scoring + analysis
├── ai_scoring.py              # ML training + calibration + ablation + bootstrap
├── api.py                     # FastAPI REST API
├── dashboard.py               # Streamlit dashboard
├── monitor.py                 # Drift detection + score distribution + health
├── pyproject.toml             # Project metadata + tool config
├── requirements.txt           # Dependencies
├── Dockerfile                 # Container build
├── docker-compose.yml         # Multi-service deployment
├── ARCHITECTURE.md            # System architecture doc
├── .env.example               # Environment template
├── models/                    # Saved ML artifacts
│   ├── xgboost_model.joblib
│   ├── isolation_forest.joblib
│   ├── scaler.joblib
│   └── model_metadata.json   # Includes calibration, ablation, CIs
├── tests/                     # Test suite
│   ├── test_scoring.py        # Core scoring + regression
│   ├── test_api.py            # API integration
│   ├── test_features.py       # Feature engineering
│   ├── test_adversarial.py    # Boundary, mimicry, fairness
│   └── test_model_stability.py # Metadata validation, metric realism
└── docs/                      # Additional documentation
```

---

*Built for Niru AI Hackathon Kenya 🇰🇪 | BioGuard v4.0*
