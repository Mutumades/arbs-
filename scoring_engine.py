"""
scoring_engine.py — CLI analysis runner for rule-based fraud scoring.

Loads synthetic event data, engineers features, scores each event using the
unified score.py module, and prints detection/FP analysis by archetype.
"""
import csv
from collections import defaultdict, Counter
from config import (
    SIM_SWAPS_CSV, USSD_SESSIONS_CSV, TRANSACTIONS_CSV,
    SCORED_EVENTS_CSV, COMBO_NAMES, get_logger,
)
from score import compute_risk_score, compute_velocity, compute_time_of_day_risk, compute_device_familiarity

logger = get_logger("scoring_engine")

# ─── Data loading ────────────────────────────────────────────────────

def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

swaps = load_csv(str(SIM_SWAPS_CSV))
sessions = load_csv(str(USSD_SESSIONS_CSV))
txns = load_csv(str(TRANSACTIONS_CSV))

sessions_by_msisdn = defaultdict(list)
for s in sessions:
    sessions_by_msisdn[s["msisdn"]].append(s)
txns_by_msisdn = defaultdict(list)
for t in txns:
    txns_by_msisdn[t["msisdn"]].append(t)

# ─── Feature engineering ─────────────────────────────────────────────

def engineer_features(swap):
    """Extract behavioral features from raw swap/session/transaction data."""
    msisdn = swap["msisdn"]
    my_sessions = sessions_by_msisdn.get(msisdn, [])
    my_txns = txns_by_msisdn.get(msisdn, [])

    time_to_first_min = float(swap["time_to_first_session_min"])
    sess_1h = int(swap["session_count_first_hour"])
    imei_match = swap["imei_match"] == "True"
    imei_card = int(swap["imei_msisdn_cardinality_90d"])
    imei_corr = float(swap["imei_swap_correlation"])
    non_fin = int(swap["non_financial_activity_count"])
    displacement = float(swap["displacement_km"])
    agent_risk = float(swap["agent_risk_score"])

    if my_sessions:
        dwell_vars = [float(s["dwell_variance"]) for s in my_sessions]
        mean_dwells = [float(s["mean_dwell"]) for s in my_sessions]
        directs = [float(s["path_directness"]) for s in my_sessions]
        avg_dwell_var = sum(dwell_vars) / len(dwell_vars)
        avg_mean_dwell = sum(mean_dwells) / len(mean_dwells)
        avg_direct = sum(directs) / len(directs)
    else:
        avg_dwell_var, avg_mean_dwell, avg_direct = 10.0, 7.0, 0.8

    intent_purity = len(my_sessions) / max(1, len(my_sessions) + non_fin)

    if my_txns:
        max_drain = max(float(t["drain_ratio"]) for t in my_txns)
        total_amt = sum(int(t["amount"]) for t in my_txns)
        first_bal = int(my_txns[0]["balance_before"])
        cum_drain = total_amt / max(1, first_bal)
        all_unknown = all(t["recipient_is_known"] == "False" for t in my_txns)
        any_unknown = any(t["recipient_is_known"] == "False" for t in my_txns)
        distinct_recip = len(set(t["recipient_msisdn"] for t in my_txns))
        avg_recip_age = sum(int(t["recipient_account_age_days"]) for t in my_txns) / len(my_txns)
    else:
        max_drain = cum_drain = 0
        all_unknown = any_unknown = False
        distinct_recip = 0
        avg_recip_age = 999

    # New features
    txn_velocity = compute_velocity(my_txns, time_to_first_min)
    time_of_day_risk = compute_time_of_day_risk(swap.get("swap_ts", ""))
    device_familiarity = compute_device_familiarity(imei_match, imei_card, imei_corr)

    return {
        "msisdn": msisdn,
        "label": swap["label"],
        "archetype": swap.get("fraud_archetype", "unknown"),
        "time_to_first_min": time_to_first_min,
        "session_count_1h": sess_1h,
        "imei_match": imei_match,
        "imei_cardinality": imei_card,
        "imei_swap_corr": imei_corr,
        "avg_dwell_variance": round(avg_dwell_var, 2),
        "avg_mean_dwell": round(avg_mean_dwell, 2),
        "avg_directness": round(avg_direct, 2),
        "non_financial_count": non_fin,
        "intent_purity": round(intent_purity, 3),
        "max_drain_ratio": round(max_drain, 3),
        "cumulative_drain": round(cum_drain, 3),
        "all_recipients_unknown": all_unknown,
        "any_unknown_recipient": any_unknown,
        "distinct_recipients": distinct_recip,
        "avg_recipient_age_days": round(avg_recip_age, 0),
        "displacement_km": displacement,
        "agent_risk": agent_risk,
        "txn_velocity": txn_velocity,
        "time_of_day_risk": time_of_day_risk,
        "device_familiarity": device_familiarity,
    }


# ─── Score all events ────────────────────────────────────────────────

feature_matrix = [engineer_features(s) for s in swaps]
fraud_rows = [f for f in feature_matrix if f["label"] == "fraud"]
legit_rows = [f for f in feature_matrix if f["label"] == "legit"]

for f in feature_matrix:
    result = compute_risk_score(f)
    f["risk_score"] = result.score
    f["action"] = result.action
    f["triggered_combos"] = result.triggered_combinations

# ─── Analysis output ─────────────────────────────────────────────────

logger.info("=" * 75)
logger.info("COMBINATION PERFORMANCE (on realistic data)")
logger.info("=" * 75)

from score import COMBINATIONS as COMBO_FUNCS

header = f"{'Combo':<10} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>5} {'Prec':>7} {'Recall':>7} {'F1':>7} {'FPR':>8}"
logger.info(header)
logger.info("-" * 75)

for name, func in COMBO_FUNCS.items():
    tp = sum(1 for f in fraud_rows if func(f))
    fp = sum(1 for f in legit_rows if func(f))
    fn = len(fraud_rows) - tp
    tn = len(legit_rows) - fp
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(0.001, prec + rec)
    logger.info(f"{name:<10} {tp:>4} {fp:>4} {fn:>4} {tn:>5} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f} {fp / max(1, fp + tn):>8.4f}")

logger.info("")
logger.info("=" * 75)
logger.info("SCORE DISTRIBUTION BY ACTION TIER")
logger.info("=" * 75)
tiers = ["T0_ALLOW", "T1_OBSERVE", "T2_FRICTION", "T3_STEP_UP", "T4_FREEZE"]
for tier in tiers:
    fc = sum(1 for f in feature_matrix if f["action"] == tier and f["label"] == "fraud")
    lc = sum(1 for f in feature_matrix if f["action"] == tier and f["label"] == "legit")
    logger.info(f"  {tier:<15} Fraud: {fc:>3}  Legit: {lc:>4}")

logger.info("")
logger.info("=" * 75)
logger.info("DETECTION BY FRAUD ARCHETYPE")
logger.info("=" * 75)
for arch in ["classic_fast", "slow_fraudster", "clean_device", "local_insider"]:
    rows = [f for f in fraud_rows if f["archetype"] == arch]
    if not rows:
        continue
    caught_t2 = sum(1 for f in rows if f["risk_score"] >= 41)
    caught_t3 = sum(1 for f in rows if f["risk_score"] >= 61)
    caught_t4 = sum(1 for f in rows if f["risk_score"] >= 81)
    avg_score = sum(f["risk_score"] for f in rows) / len(rows)
    combos_used = Counter()
    for f in rows:
        for c in f["triggered_combos"]:
            combos_used[c] += 1
    logger.info(f"\n  {arch} (n={len(rows)}):")
    logger.info(f"    Avg score: {avg_score:.0f}")
    logger.info(f"    Caught at T2+: {caught_t2}/{len(rows)} ({caught_t2 / len(rows) * 100:.0f}%)")
    logger.info(f"    Caught at T3+: {caught_t3}/{len(rows)} ({caught_t3 / len(rows) * 100:.0f}%)")
    logger.info(f"    Caught at T4:  {caught_t4}/{len(rows)} ({caught_t4 / len(rows) * 100:.0f}%)")
    logger.info(f"    Top combos: {dict(combos_used.most_common(3))}")

logger.info("")
logger.info("=" * 75)
logger.info("FALSE POSITIVE ANALYSIS BY LEGIT ARCHETYPE")
logger.info("=" * 75)
for arch in ["normal", "urgent_mobile_money", "rural_shared", "power_user", "emergency_drain", "new_recipient"]:
    rows = [f for f in legit_rows if f["archetype"] == arch]
    if not rows:
        continue
    fp_t1 = sum(1 for f in rows if f["risk_score"] >= 21)
    fp_t2 = sum(1 for f in rows if f["risk_score"] >= 41)
    fp_t3 = sum(1 for f in rows if f["risk_score"] >= 61)
    logger.info(f"  {arch:<20} n={len(rows):>4}  T1+: {fp_t1:>3} ({fp_t1 / len(rows) * 100:>4.1f}%)  "
                f"T2+: {fp_t2:>3} ({fp_t2 / len(rows) * 100:>4.1f}%)  T3+: {fp_t3:>3} ({fp_t3 / len(rows) * 100:>4.1f}%)")

logger.info("")
logger.info("=" * 75)
logger.info("OVERALL PERFORMANCE METRICS")
logger.info("=" * 75)
for threshold_name, threshold in [("T1+ (observe)", 21), ("T2+ (friction)", 41), ("T3+ (step-up)", 61), ("T4 (freeze)", 81)]:
    tp = sum(1 for f in fraud_rows if f["risk_score"] >= threshold)
    fp = sum(1 for f in legit_rows if f["risk_score"] >= threshold)
    tpr = tp / max(1, len(fraud_rows))
    fpr = fp / max(1, len(legit_rows))
    prec = tp / max(1, tp + fp)
    logger.info(f"  {threshold_name:<22} Detection: {tp:>2}/{len(fraud_rows)} ({tpr * 100:>5.1f}%)  "
                f"FP: {fp:>3}/{len(legit_rows)} ({fpr * 100:>5.2f}%)  Precision: {prec * 100:>5.1f}%")

missed = [f for f in fraud_rows if f["risk_score"] < 41]
if missed:
    logger.info("\n--- Missed Fraud Cases (score < 41) ---")
    for f in sorted(missed, key=lambda x: -x["risk_score"]):
        logger.info(f"  {f['msisdn']} arch={f['archetype']:<15} score={f['risk_score']:>3} "
                    f"drain={f['max_drain_ratio']:.2f} t2first={f['time_to_first_min']:.0f}m "
                    f"imei_corr={f['imei_swap_corr']:.2f} displace={f['displacement_km']:.0f}km "
                    f"combos={f['triggered_combos']}")

# ─── Export scored events ────────────────────────────────────────────

with open(str(SCORED_EVENTS_CSV), "w", newline="") as f:
    fields = [
        "msisdn", "label", "archetype", "risk_score", "action", "triggered_combos",
        "time_to_first_min", "imei_swap_corr", "imei_cardinality",
        "avg_dwell_variance", "intent_purity", "max_drain_ratio",
        "cumulative_drain", "all_recipients_unknown", "non_financial_count",
        "displacement_km", "agent_risk", "txn_velocity", "time_of_day_risk",
        "device_familiarity",
    ]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for row in feature_matrix:
        out = {k: row.get(k, "") for k in fields}
        out["triggered_combos"] = "|".join(row.get("triggered_combos", []))
        w.writerow(out)

logger.info(f"\nDone: Scored {len(feature_matrix)} events → {SCORED_EVENTS_CSV.name}")
