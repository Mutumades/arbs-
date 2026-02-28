"""
api.py — REST API for real-time SIM-swap fraud scoring.

Features:
  - /score (rule-based) and /score/ensemble (ML models)
  - API key authentication
  - Rate limiting
  - CORS middleware
  - SQLite audit trail
  - Structured logging
  - Demo endpoints for preset scenarios
"""
import time
import sqlite3
import json
from datetime import datetime
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn
import numpy as np
import csv

from config import (
    API_HOST, API_PORT, API_KEY, RATE_LIMIT_PER_MINUTE,
    TIER_DESCRIPTIONS, DB_PATH, FEATURE_COLS, get_logger,
    SCORED_EVENTS_CSV, MODEL_METADATA_PATH,
)
from pathlib import Path
from score import (
    compute_risk_score, load_ml_models, score_with_ensemble,
    compute_time_of_day_risk, compute_device_familiarity,
    RiskResult,
)

logger = get_logger("api")

# ─── Globals ──────────────────────────────────────────────────────────

ml_models = {}


# ─── Database setup ───────────────────────────────────────────────────

def init_db():
    """Create audit trail table if it doesn't exist."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            msisdn TEXT NOT NULL,
            risk_score INTEGER NOT NULL,
            action TEXT NOT NULL,
            triggered_combinations TEXT,
            explanation TEXT,
            processing_time_ms REAL,
            endpoint TEXT,
            api_key_used TEXT
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Audit database initialized at %s", DB_PATH)


def log_to_audit(result: dict, endpoint: str, api_key: str = ""):
    """Log a scoring result to the SQLite audit trail."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(
            """INSERT INTO audit_log
               (timestamp, msisdn, risk_score, action, triggered_combinations,
                explanation, processing_time_ms, endpoint, api_key_used)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.utcnow().isoformat(),
                result.get("msisdn", ""),
                result.get("risk_score", 0),
                result.get("action", ""),
                json.dumps(result.get("triggered_combinations", [])),
                json.dumps(result.get("explanation", [])),
                result.get("processing_time_ms", 0),
                endpoint,
                api_key[:8] + "..." if api_key else "",
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error("Audit log write failed: %s", e)


# ─── Rate limiting ────────────────────────────────────────────────────

rate_limit_store = defaultdict(list)


def check_rate_limit(client_ip: str) -> bool:
    """Simple in-memory sliding window rate limiter."""
    now = time.time()
    window = 60  # seconds
    # Clean old entries
    rate_limit_store[client_ip] = [
        t for t in rate_limit_store[client_ip] if now - t < window
    ]
    if len(rate_limit_store[client_ip]) >= RATE_LIMIT_PER_MINUTE:
        return False
    rate_limit_store[client_ip].append(now)
    return True


# ─── App lifecycle ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB and load ML models."""
    global ml_models
    init_db()
    ml_models = load_ml_models()
    if ml_models.get("xgboost"):
        logger.info("ML models loaded — ensemble scoring available")
    else:
        logger.warning("ML models not found — run ai_scoring.py first for ensemble scoring")
    yield
    logger.info("Shutting down BioGuard API")


# ─── App setup ────────────────────────────────────────────────────────

app = FastAPI(
    title="BioGuard — SIM-Swap Fraud Detection API",
    description="AI-powered behavioral fraud scoring for mobile money transactions",
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(request: Request, api_key: str = Security(api_key_header)):
    """Validate API key from header. Demo endpoints bypass auth."""
    if request.url.path.startswith("/demo") or request.url.path in ("/health", "/docs", "/openapi.json", "/redoc"):
        return api_key or ""
    if not api_key or api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key. Set X-API-Key header.")
    return api_key


# ─── Request/Response models ─────────────────────────────────────────

class TransactionEvent(BaseModel):
    msisdn: str = Field(..., example="254722123456", description="Mobile subscriber number")
    time_since_swap_min: float = Field(..., ge=0, description="Minutes since SIM swap")
    swap_channel: str = Field("agent", description="Channel: agent, retail_store, call_center")
    agent_risk_score: float = Field(0.2, ge=0, le=1, description="Risk score of the swap agent")
    imei_match_pre_swap: bool = Field(False, description="Does post-swap IMEI match pre-swap?")
    imei_msisdn_cardinality_90d: int = Field(1, ge=0, description="MSISDNs seen on this IMEI in 90d")
    imei_swap_correlation: float = Field(0.0, ge=0, le=1, description="Correlation of IMEI with swaps")
    avg_dwell_variance_s: float = Field(10.0, ge=0, description="Average dwell time variance (seconds)")
    avg_mean_dwell_s: float = Field(7.0, ge=0, description="Average mean dwell time (seconds)")
    intent_purity: float = Field(0.5, ge=0, le=1, description="Fraction of sessions that are financial")
    non_financial_activity_count: int = Field(3, ge=0, description="Non-financial USSD activities")
    session_count_first_hour: int = Field(0, ge=0, description="USSD sessions in first hour post-swap")
    avg_path_directness: float = Field(0.8, ge=0, le=1, description="Menu path directness")
    txn_type: str = Field("SEND_MONEY", description="Transaction type")
    amount: int = Field(..., gt=0, description="Transaction amount in KES")
    balance_before: int = Field(..., gt=0, description="Balance before transaction in KES")
    recipient_is_known: bool = Field(True, description="Is recipient in known contacts?")
    recipient_account_age_days: int = Field(365, ge=0, description="Recipient account age in days")
    distinct_recipients_post_swap: int = Field(1, ge=0, description="Distinct recipients post-swap")
    cumulative_drain_ratio: float = Field(0.0, ge=0, le=1, description="Total drain ratio post-swap")
    displacement_km: float = Field(0.0, ge=0, description="Distance from home location in km")
    swap_timestamp: str = Field("", description="Swap timestamp for time-of-day analysis (ISO format)")


class RiskResponse(BaseModel):
    msisdn: str
    risk_score: int
    action: str
    action_description: str
    triggered_combinations: list[str]
    explanation: list[str]
    processing_time_ms: float


class EnsembleResponse(RiskResponse):
    xgb_probability: float = 0.0
    iso_anomaly_score: float = 0.0
    ensemble_score: float = 0.0
    ensemble_prediction: int = 0


# ─── Helper ───────────────────────────────────────────────────────────

def event_to_features(evt: TransactionEvent) -> dict:
    """Convert API event model to scoring feature dict."""
    tod_risk = compute_time_of_day_risk(evt.swap_timestamp)
    dev_fam = compute_device_familiarity(
        evt.imei_match_pre_swap, evt.imei_msisdn_cardinality_90d, evt.imei_swap_correlation
    )
    return {
        "time_to_first_min": evt.time_since_swap_min,
        "session_count_1h": evt.session_count_first_hour,
        "imei_match": evt.imei_match_pre_swap,
        "imei_cardinality": evt.imei_msisdn_cardinality_90d,
        "imei_swap_corr": evt.imei_swap_correlation,
        "avg_dwell_variance": evt.avg_dwell_variance_s,
        "avg_mean_dwell": evt.avg_mean_dwell_s,
        "avg_directness": evt.avg_path_directness,
        "non_financial_count": evt.non_financial_activity_count,
        "intent_purity": evt.intent_purity,
        "max_drain_ratio": evt.amount / max(1, evt.balance_before),
        "cumulative_drain": evt.cumulative_drain_ratio,
        "all_recipients_unknown": not evt.recipient_is_known,
        "any_unknown_recipient": not evt.recipient_is_known,
        "distinct_recipients": evt.distinct_recipients_post_swap,
        "avg_recipient_age_days": evt.recipient_account_age_days,
        "displacement_km": evt.displacement_km,
        "agent_risk": evt.agent_risk_score,
        "amount": evt.amount,
        "balance_before": evt.balance_before,
        "txn_velocity": 0,
        "time_of_day_risk": tod_risk,
        "device_familiarity": dev_fam,
        "msisdn": evt.msisdn,
    }


# ─── Stats ────────────────────────────────────────────────────────────

stats = {"total": 0, "by_tier": {t: 0 for t in TIER_DESCRIPTIONS}}


# ─── Endpoints ────────────────────────────────────────────────────────

@app.post("/score", response_model=RiskResponse, tags=["Scoring"])
async def score_endpoint(
    event: TransactionEvent,
    request: Request,
    api_key: str = Security(verify_api_key),
):
    """Score a transaction using rule-based behavioral analysis."""
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")

    start = time.time()
    features = event_to_features(event)
    result = compute_risk_score(features)
    elapsed_ms = round((time.time() - start) * 1000, 2)

    response = {
        "msisdn": event.msisdn,
        "risk_score": result.score,
        "action": result.action,
        "action_description": result.action_description,
        "triggered_combinations": result.triggered_combinations,
        "explanation": result.explanation,
        "processing_time_ms": elapsed_ms,
    }

    stats["total"] += 1
    stats["by_tier"][result.action] += 1
    log_to_audit(response, "/score", api_key)
    logger.info("Scored %s → %d (%s) in %.1fms", event.msisdn, result.score, result.action, elapsed_ms)

    return response


@app.post("/score/ensemble", response_model=EnsembleResponse, tags=["Scoring"])
async def score_ensemble_endpoint(
    event: TransactionEvent,
    request: Request,
    api_key: str = Security(verify_api_key),
):
    """Score using the full ML ensemble (rules + XGBoost + Isolation Forest)."""
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")

    if not ml_models.get("xgboost"):
        raise HTTPException(
            status_code=503,
            detail="ML models not loaded. Run ai_scoring.py first to train and save models.",
        )

    start = time.time()
    features = event_to_features(event)
    rule_result = compute_risk_score(features)

    # Build feature array for ML
    feature_array = np.array([features.get(c, 0) for c in FEATURE_COLS], dtype=float)
    # Convert booleans
    for i, col in enumerate(FEATURE_COLS):
        if isinstance(feature_array[i], bool):
            feature_array[i] = float(feature_array[i])

    ensemble = score_with_ensemble(feature_array, ml_models, rule_result.score)
    elapsed_ms = round((time.time() - start) * 1000, 2)

    # Use ensemble prediction for action
    ens_score = int(ensemble["ensemble_score"] * 100)
    from score import determine_action
    action = determine_action(ens_score)

    response = {
        "msisdn": event.msisdn,
        "risk_score": rule_result.score,
        "action": action,
        "action_description": TIER_DESCRIPTIONS[action],
        "triggered_combinations": rule_result.triggered_combinations,
        "explanation": rule_result.explanation,
        "processing_time_ms": elapsed_ms,
        "xgb_probability": ensemble["xgb_probability"],
        "iso_anomaly_score": ensemble["iso_anomaly_score"],
        "ensemble_score": ensemble["ensemble_score"],
        "ensemble_prediction": ensemble["ensemble_prediction"],
    }

    stats["total"] += 1
    log_to_audit(response, "/score/ensemble", api_key)
    logger.info("Ensemble scored %s → %.3f (%s) in %.1fms",
                event.msisdn, ensemble["ensemble_score"], action, elapsed_ms)

    return response


@app.get("/health", tags=["System"])
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": "bioguard-sim-swap-v4.0",
        "ml_models_loaded": ml_models.get("xgboost") is not None,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/stats", tags=["System"])
async def get_stats():
    """Get scoring statistics."""
    return stats


@app.get("/audit/recent", tags=["System"])
async def get_recent_audit(limit: int = 20, api_key: str = Security(verify_api_key)):
    """Get recent audit log entries."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audit log error: {e}")


# ─── Demo endpoints (no auth required) ───────────────────────────────

def _score_demo(event: TransactionEvent, endpoint: str) -> dict:
    start = time.time()
    features = event_to_features(event)
    result = compute_risk_score(features)
    elapsed_ms = round((time.time() - start) * 1000, 2)
    return {
        "msisdn": event.msisdn,
        "risk_score": result.score,
        "action": result.action,
        "action_description": result.action_description,
        "triggered_combinations": result.triggered_combinations,
        "explanation": result.explanation,
        "processing_time_ms": elapsed_ms,
    }


@app.get("/demo/fraud", tags=["Demo"])
async def demo_fraud():
    """Demo: Classic fast fraudster scenario."""
    return _score_demo(TransactionEvent(
        msisdn="254722999888", time_since_swap_min=7, agent_risk_score=0.78,
        imei_match_pre_swap=False, imei_msisdn_cardinality_90d=6, imei_swap_correlation=0.83,
        avg_dwell_variance_s=0.9, intent_purity=1.0, non_financial_activity_count=0,
        session_count_first_hour=4, avg_path_directness=1.0, amount=45000,
        balance_before=47800, recipient_is_known=False, recipient_account_age_days=8,
        distinct_recipients_post_swap=2, cumulative_drain_ratio=0.94, displacement_km=142.0,
        swap_timestamp="2025-01-15T02:30:00Z",
    ), "/demo/fraud")


@app.get("/demo/legit", tags=["Demo"])
async def demo_legit():
    """Demo: Normal legitimate user (24hr post-swap)."""
    return _score_demo(TransactionEvent(
        msisdn="254722111222", time_since_swap_min=1440, agent_risk_score=0.15,
        imei_match_pre_swap=True, imei_msisdn_cardinality_90d=1, imei_swap_correlation=0.0,
        avg_dwell_variance_s=12.5, intent_purity=0.3, non_financial_activity_count=8,
        session_count_first_hour=0, avg_path_directness=0.65, amount=2000,
        balance_before=35000, recipient_is_known=True, recipient_account_age_days=890,
        distinct_recipients_post_swap=1, cumulative_drain_ratio=0.06, displacement_km=2.1,
        swap_timestamp="2025-01-15T14:00:00Z",
    ), "/demo/legit")


@app.get("/demo/edge_emergency", tags=["Demo"])
async def demo_edge():
    """Demo: Legitimate emergency drain (hardest false positive case)."""
    return _score_demo(TransactionEvent(
        msisdn="254722333444", time_since_swap_min=25, agent_risk_score=0.30,
        imei_match_pre_swap=False, imei_msisdn_cardinality_90d=1, imei_swap_correlation=0.0,
        avg_dwell_variance_s=8.5, intent_purity=0.6, non_financial_activity_count=2,
        session_count_first_hour=2, avg_path_directness=0.75, amount=35000,
        balance_before=42000, recipient_is_known=True, recipient_account_age_days=450,
        distinct_recipients_post_swap=1, cumulative_drain_ratio=0.83, displacement_km=3.0,
        swap_timestamp="2025-01-15T15:30:00Z",
    ), "/demo/edge_emergency")


@app.get("/demo/slow_fraudster", tags=["Demo"])
async def demo_slow():
    """Demo: Slow fraudster who waits 3 hours."""
    return _score_demo(TransactionEvent(
        msisdn="254722555666", time_since_swap_min=180, agent_risk_score=0.55,
        imei_match_pre_swap=False, imei_msisdn_cardinality_90d=4, imei_swap_correlation=0.45,
        avg_dwell_variance_s=3.5, intent_purity=0.85, non_financial_activity_count=1,
        session_count_first_hour=0, avg_path_directness=1.0, amount=52000,
        balance_before=68000, recipient_is_known=False, recipient_account_age_days=15,
        distinct_recipients_post_swap=1, cumulative_drain_ratio=0.76, displacement_km=85.0,
        swap_timestamp="2025-01-15T03:00:00Z",
    ), "/demo/slow_fraudster")


@app.post("/demo/score", tags=["Demo"])
async def demo_score(event: TransactionEvent):
    """Score a custom event (no auth required, for dashboard use)."""
    start = time.time()
    features = event_to_features(event)
    result = compute_risk_score(features)
    elapsed_ms = round((time.time() - start) * 1000, 2)
    return {
        "msisdn": event.msisdn,
        "risk_score": result.score,
        "action": result.action,
        "action_description": result.action_description,
        "triggered_combinations": result.triggered_combinations,
        "explanation": result.explanation,
        "processing_time_ms": elapsed_ms,
    }


@app.get("/analytics/data", tags=["Analytics"])
async def analytics_data():
    """Return scored events analytics for the dashboard."""
    try:
        rows = []
        with open(str(SCORED_EVENTS_CSV), newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        raise HTTPException(status_code=404, detail="Run scoring_engine.py first")

    fraud = [r for r in rows if r.get("label") == "fraud"]
    legit = [r for r in rows if r.get("label") == "legit"]
    fraud_scores = [int(r["risk_score"]) for r in fraud]
    legit_scores = [int(r["risk_score"]) for r in legit]

    caught_t2 = sum(1 for s in fraud_scores if s >= 41)
    caught_t4 = sum(1 for s in fraud_scores if s >= 81)
    fp_t2 = sum(1 for s in legit_scores if s >= 41)

    # Detection by archetype
    arch_names = ["classic_fast", "slow_fraudster", "clean_device", "local_insider"]
    archetypes = {}
    for a in arch_names:
        arch_rows = [r for r in fraud if r.get("archetype") == a]
        if arch_rows:
            archetypes[a] = sum(1 for r in arch_rows if int(r["risk_score"]) >= 41) / len(arch_rows) * 100
        else:
            archetypes[a] = 0

    # Radar data
    def avg_feat(data, feat):
        vals = [float(r[feat]) for r in data if r.get(feat)]
        return sum(vals) / max(1, len(vals))

    radar_fraud = [
        1 - min(1, avg_feat(fraud, "time_to_first_min") / 60),
        avg_feat(fraud, "imei_swap_corr"),
        1 - min(1, avg_feat(fraud, "avg_dwell_variance") / 20),
        avg_feat(fraud, "intent_purity"),
        avg_feat(fraud, "max_drain_ratio"),
        min(1, avg_feat(fraud, "displacement_km") / 200),
    ]
    radar_legit = [
        1 - min(1, avg_feat(legit, "time_to_first_min") / 60),
        avg_feat(legit, "imei_swap_corr"),
        1 - min(1, avg_feat(legit, "avg_dwell_variance") / 20),
        avg_feat(legit, "intent_purity"),
        avg_feat(legit, "max_drain_ratio"),
        min(1, avg_feat(legit, "displacement_km") / 200),
    ]

    # FP breakdown
    legit_archs = ["normal", "urgent_mobile_money", "rural_shared", "power_user", "emergency_drain", "new_recipient"]
    fp_breakdown = []
    for a in legit_archs:
        arch_rows = [r for r in legit if r.get("archetype") == a]
        if arch_rows:
            t1 = sum(1 for r in arch_rows if int(r["risk_score"]) >= 21)
            t2 = sum(1 for r in arch_rows if int(r["risk_score"]) >= 41)
            fp_breakdown.append({
                "archetype": a, "count": len(arch_rows),
                "t1_plus": f"{t1} ({t1/len(arch_rows)*100:.1f}%)",
                "t2_plus": f"{t2} ({t2/len(arch_rows)*100:.1f}%)",
            })

    return {
        "total": len(rows), "fraud_count": len(fraud), "legit_count": len(legit),
        "caught_t2": caught_t2, "caught_t4": caught_t4, "fp_t2": fp_t2,
        "fraud_scores": fraud_scores, "legit_scores": legit_scores,
        "archetypes": archetypes,
        "radar": {"fraud": radar_fraud, "legit": radar_legit},
        "fp_breakdown": fp_breakdown,
    }


@app.get("/analytics/metadata", tags=["Analytics"])
async def analytics_metadata():
    """Return model metadata for the dashboard."""
    try:
        with open(str(MODEL_METADATA_PATH)) as f:
            return json.load(f)
    except Exception:
        raise HTTPException(status_code=404, detail="Run ai_scoring.py first")


# ─── Root + Static files ─────────────────────────────────────────────

@app.get("/", tags=["UI"], include_in_schema=False)
async def root():
    """Serve the dashboard."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


# ─── Request logging middleware ───────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with timing."""
    start = time.time()
    response = await call_next(request)
    elapsed = round((time.time() - start) * 1000, 1)
    logger.debug("%s %s -> %d (%.1fms)", request.method, request.url.path, response.status_code, elapsed)
    return response


# ─── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting BioGuard API on %s:%d", API_HOST, API_PORT)
    uvicorn.run(app, host=API_HOST, port=API_PORT)
