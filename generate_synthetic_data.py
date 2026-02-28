"""
generate_synthetic_data.py

Generates 2000 synthetic SIM-swap events with labeled fraud/legit outcomes.

Fraud archetypes (6):
  classic_fast, slow_fraudster, clean_device, local_insider,
  partial_text_coaching (NEW), social_engineering_pretext (NEW)
Legit archetypes (6):
  normal, urgent_mobile_money, rural_shared, power_user,
  emergency_drain, new_recipient

Enhancements over v1:
  - DeviceProfile class with East African device noise characteristics
  - Environmental context enum (INDOOR_STILL, WALKING, MATATU, OUTDOOR_WIND)
  - Heavy cross-class contamination (12-15% legit overlap, 8-10% fraud overlap)
  - Label noise: 2-3% random label flips to simulate annotation errors
  - Temporal patterns: time-of-day affects session pacing
  - Validation statistics output (distribution overlap metrics)

Outputs: sim_swaps.csv, ussd_sessions.csv, transactions.csv
"""

import csv
import random
import uuid
import math
from datetime import datetime, timedelta
from collections import defaultdict, Counter

random.seed(2025)

# ─── Device Profiles ─────────────────────────────────────────────────

class DeviceProfile:
    """Models per-device sensor noise floors for realistic dwell-time variance."""
    def __init__(self, name, dwell_noise_floor, session_jitter_pct, is_basic):
        self.name = name
        self.dwell_noise_floor = dwell_noise_floor    # added to every dwell time
        self.session_jitter_pct = session_jitter_pct  # % jitter on dwell
        self.is_basic = is_basic

DEVICE_PROFILES = {
    # Basic phones — higher latency, larger jitter (slower USSD response)
    "itel_5027":       DeviceProfile("itel_5027",       1.5, 0.45, True),
    "Nokia_105":       DeviceProfile("Nokia_105",       1.2, 0.40, True),
    "Nokia_110":       DeviceProfile("Nokia_110",       1.3, 0.40, True),
    "Tecno_T301":      DeviceProfile("Tecno_T301",      1.4, 0.42, True),
    "itel_2160":       DeviceProfile("itel_2160",        1.6, 0.48, True),
    "itel_2163":       DeviceProfile("itel_2163",        1.7, 0.50, True),
    # Smartphones — lower latency, tighter jitter
    "Samsung_A14":     DeviceProfile("Samsung_A14",      0.5, 0.25, False),
    "Tecno_Spark_10":  DeviceProfile("Tecno_Spark_10",   0.6, 0.28, False),
    "Infinix_Hot_30":  DeviceProfile("Infinix_Hot_30",   0.6, 0.30, False),
    "Redmi_12C":       DeviceProfile("Redmi_12C",        0.4, 0.22, False),
    "Samsung_A04":     DeviceProfile("Samsung_A04",      0.5, 0.26, False),
}
DEVICE_NAMES = list(DEVICE_PROFILES.keys())
BASIC_PHONES = [n for n, p in DEVICE_PROFILES.items() if p.is_basic]
SMARTPHONES = [n for n, p in DEVICE_PROFILES.items() if not p.is_basic]
ALL_DEVICES = DEVICE_NAMES

# ─── Environmental Context ───────────────────────────────────────────

ENV_CONTEXTS = {
    "INDOOR_STILL":  {"dwell_bias": 0.0,  "variance_mult": 1.0,  "weight": 0.50},
    "WALKING":       {"dwell_bias": 1.5,  "variance_mult": 1.3,  "weight": 0.20},
    "MATATU":        {"dwell_bias": 2.5,  "variance_mult": 1.8,  "weight": 0.18},
    "OUTDOOR_WIND":  {"dwell_bias": 1.0,  "variance_mult": 1.2,  "weight": 0.12},
}
ENV_NAMES = list(ENV_CONTEXTS.keys())
ENV_WEIGHTS = [ENV_CONTEXTS[e]["weight"] for e in ENV_NAMES]

# ─── Time-of-day session pacing ──────────────────────────────────────

def time_of_day_pace_factor(hour):
    """Simulate slower interactions at night, faster during business hours."""
    if 0 <= hour < 5:
        return 1.4    # groggy, slow
    elif 5 <= hour < 8:
        return 1.1    # waking up
    elif 8 <= hour < 17:
        return 0.9    # alert, fastest
    elif 17 <= hour < 21:
        return 1.0    # relaxed
    else:
        return 1.2    # late, slower

# ─── Geography ───────────────────────────────────────────────────────

COUNTIES = [
    ("Nairobi", -1.2921, 36.8219), ("Mombasa", -4.0435, 39.6682),
    ("Kisumu", -0.1022, 34.7617), ("Nakuru", -0.3031, 36.0800),
    ("Eldoret", 0.5143, 35.2698), ("Thika", -1.0396, 37.0900),
    ("Nyeri", -0.4197, 36.9511), ("Machakos", -1.5177, 37.2634),
    ("Meru", 0.0480, 37.6559), ("Kisii", -0.6817, 34.7668),
    ("Malindi", -3.2138, 40.1169), ("Garissa", -0.4532, 39.6461),
    ("Turkana", 3.1166, 35.5966), ("Narok", -1.0876, 35.8600),
]
FRAUD_HUBS = [
    ("Kayole", -1.2756, 36.9122), ("Eastleigh", -1.2720, 36.8460),
    ("Kondele", -0.0917, 34.7500), ("Langas", 0.5050, 35.2550),
    ("Umoja", -1.2690, 36.8930),
]

BASE_DATE = datetime(2025, 1, 1, 6, 0, 0)
FRAUD_IMEIS = [f"35289{random.randint(100,999)}{random.randint(10000000,99999999)}" for _ in range(8)]
fraud_imei_usage = {imei: 0 for imei in FRAUD_IMEIS}

# ─── Helper functions ────────────────────────────────────────────────

def gen_msisdn():
    return random.choice(["2547", "2541", "2540"]) + str(random.randint(10000000, 99999999))

def gen_imei(prefix=None):
    if prefix: return prefix + str(random.randint(10000000, 99999999))
    return "35" + str(random.randint(2000000000000, 9999999999999))

def jitter(base, pct=0.3):
    return max(1, int(base * (1 + random.uniform(-pct, pct))))

def add_latency_noise(dwell, is_rural=False, device_profile=None, env_context=None):
    """Add realistic latency noise based on device, environment, and network."""
    noise = random.uniform(0, 2) if not is_rural else random.uniform(1, 6)
    if device_profile:
        noise += device_profile.dwell_noise_floor
    if env_context and env_context in ENV_CONTEXTS:
        noise += ENV_CONTEXTS[env_context]["dwell_bias"]
    return max(1, int(dwell + noise))

def gen_cgi(lat, lon):
    return (f"639-02-{random.randint(1000,9999)}-{random.randint(10000,99999)}",
            lat + random.uniform(-0.02, 0.02), lon + random.uniform(-0.02, 0.02))

def gen_known_recipients(n=5):
    return [gen_msisdn() for _ in range(n)]

def compute_variance(arr):
    if len(arr) < 2: return 0.0
    m = sum(arr) / len(arr)
    return round(sum((x - m) ** 2 for x in arr) / len(arr), 2)

def pick_env_context():
    return random.choices(ENV_NAMES, weights=ENV_WEIGHTS, k=1)[0]

def pick_device(prefer_basic=False):
    if prefer_basic:
        return random.choice(BASIC_PHONES)
    return random.choice(ALL_DEVICES)

# ─── Data containers ─────────────────────────────────────────────────

sim_swaps, ussd_sessions, transactions = [], [], []
NUM_EVENTS = 2000
FRAUD_RATE = 0.03

# ─── Label noise tracking ────────────────────────────────────────────
LABEL_NOISE_RATE = 0.025  # 2.5% label flips
label_flips = 0

def add_session(sid, msisdn, imei, cgi, ts, dwells, path, directness, end_reason, pin_att, label):
    ussd_sessions.append({
        "session_id": sid, "msisdn": msisdn, "imei": imei, "cgi": cgi,
        "shortcode": "*334#", "session_start_ts": ts.isoformat() + "Z",
        "total_steps": len(dwells), "dwell_times": dwells,
        "dwell_variance": compute_variance(dwells),
        "mean_dwell": round(sum(dwells) / max(1, len(dwells)), 1),
        "menu_path": path, "path_directness": directness,
        "session_duration_s": sum(dwells) + len(dwells) * 2,
        "session_end_reason": end_reason, "pin_attempts": pin_att, "label": label,
    })

def add_txn(sid, msisdn, txn_type, amount, bal_before, recipient, known, recip_age, ts, label):
    transactions.append({
        "txn_id": f"TXN-{ts.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}",
        "session_id": sid, "msisdn": msisdn, "txn_type": txn_type,
        "amount": amount, "balance_before": bal_before,
        "balance_after": bal_before - amount,
        "drain_ratio": round(amount / max(1, bal_before), 3),
        "recipient_msisdn": recipient, "recipient_is_known": known,
        "recipient_account_age_days": recip_age,
        "txn_ts": ts.isoformat() + "Z", "label": label,
    })

# ─── Main generation loop ────────────────────────────────────────────

for i in range(NUM_EVENTS):
    is_fraud = random.random() < FRAUD_RATE
    msisdn = gen_msisdn()
    known_recips = gen_known_recipients(random.randint(3, 10))
    home = random.choice(COUNTIES)
    home_name, home_lat, home_lon = home
    is_rural = home_name in ("Turkana", "Garissa", "Narok", "Malindi")
    swap_ts = BASE_DATE + timedelta(days=random.randint(0, 60), hours=random.randint(6, 22), minutes=random.randint(0, 59))
    pre_swap_imei = gen_imei()

    # Pick environmental context and device for this event
    env_ctx = pick_env_context()
    hour = swap_ts.hour
    pace = time_of_day_pace_factor(hour)

    if is_fraud:
        archetype = random.choices(
            ["classic_fast", "slow_fraudster", "clean_device", "local_insider",
             "partial_text_coaching", "social_engineering_pretext"],
            weights=[0.28, 0.22, 0.14, 0.16, 0.10, 0.10]
        )[0]
        label = "fraud"

        if archetype == "classic_fast":
            post_imei = random.choice(FRAUD_IMEIS[:5])
            fraud_imei_usage[post_imei] += 1
            imei_card = fraud_imei_usage[post_imei]
            imei_corr = round(random.uniform(0.6, 0.95), 2)
            hub = random.choice(FRAUD_HUBS)
            post_cgi, post_lat, post_lon = gen_cgi(hub[1], hub[2])
            first_gap = timedelta(minutes=random.randint(1, 12))
            agent_risk = round(random.uniform(0.5, 0.95), 2)
            balance = random.randint(15000, 200000)
            non_fin = 0; sess_1h = random.randint(2, 6)
            base_dwell = random.uniform(2, 4) * pace
            drain_pct = random.uniform(0.85, 0.98)
            pin_attempts = 1 if random.random() > 0.15 else 2
            device_type = pick_device(prefer_basic=True)

        elif archetype == "slow_fraudster":
            post_imei = random.choice(FRAUD_IMEIS[3:7])
            fraud_imei_usage[post_imei] += 1
            imei_card = fraud_imei_usage[post_imei]
            imei_corr = round(random.uniform(0.15, 0.55), 2)
            hub = random.choice(FRAUD_HUBS)
            post_cgi, post_lat, post_lon = gen_cgi(hub[1], hub[2])
            first_gap = timedelta(hours=random.randint(2, 12))
            agent_risk = round(random.uniform(0.15, 0.55), 2)
            balance = random.randint(20000, 150000)
            non_fin = random.choice([1, 2, 2, 3, 4])
            sess_1h = 0
            base_dwell = random.uniform(5, 10) * pace  # deliberately slow — overlaps legit
            drain_pct = random.uniform(0.45, 0.80)
            pin_attempts = random.choice([1, 1, 1, 2])
            device_type = pick_device(prefer_basic=True)

        elif archetype == "clean_device":
            post_imei = gen_imei()
            imei_card = 1; imei_corr = 0.0
            hub = random.choice(FRAUD_HUBS)
            post_cgi, post_lat, post_lon = gen_cgi(hub[1], hub[2])
            first_gap = timedelta(minutes=random.randint(2, 20))
            agent_risk = round(random.uniform(0.45, 0.85), 2)
            balance = random.randint(25000, 180000)
            non_fin = 0; sess_1h = random.randint(2, 5)
            base_dwell = random.uniform(2, 4) * pace
            drain_pct = random.uniform(0.82, 0.97)
            pin_attempts = 1
            device_type = pick_device()

        elif archetype == "local_insider":
            post_imei = random.choice(FRAUD_IMEIS[5:])
            fraud_imei_usage[post_imei] += 1
            imei_card = fraud_imei_usage[post_imei]
            imei_corr = round(random.uniform(0.1, 0.45), 2)
            post_cgi, post_lat, post_lon = gen_cgi(home_lat, home_lon)
            first_gap = timedelta(minutes=random.randint(30, 180))
            agent_risk = round(random.uniform(0.3, 0.75), 2)
            balance = random.randint(30000, 250000)
            non_fin = random.choice([2, 3, 3, 4, 5])
            sess_1h = random.randint(0, 2)
            base_dwell = random.uniform(5, 12) * pace
            drain_pct = random.uniform(0.45, 0.80)
            pin_attempts = 1
            device_type = pick_device(prefer_basic=True)

        elif archetype == "partial_text_coaching":
            # NEW: Fraudster sends WhatsApp/SMS instructions between call segments
            # Intermittent call overlap, pauses in motion, mixed session timing
            post_imei = random.choice(FRAUD_IMEIS[:6])
            fraud_imei_usage[post_imei] += 1
            imei_card = fraud_imei_usage[post_imei]
            imei_corr = round(random.uniform(0.3, 0.7), 2)
            hub = random.choice(FRAUD_HUBS)
            post_cgi, post_lat, post_lon = gen_cgi(hub[1], hub[2])
            first_gap = timedelta(minutes=random.randint(5, 45))
            agent_risk = round(random.uniform(0.3, 0.70), 2)
            balance = random.randint(20000, 150000)
            # Some non-financial activity (the SMS/WhatsApp instruction checking)
            non_fin = random.choice([1, 2, 3])
            sess_1h = random.randint(1, 3)
            # Variable dwell — pauses to read instructions then rushes
            base_dwell = random.uniform(4, 9) * pace
            drain_pct = random.uniform(0.60, 0.92)
            pin_attempts = random.choice([1, 1, 2])
            device_type = pick_device()

        else:  # social_engineering_pretext
            # NEW: Victim believes they're helping police/bank
            # Calm but sustained call, minimal switching, moderate pace
            post_imei = gen_imei()  # clean device — victim doing it themselves
            imei_card = 1; imei_corr = 0.0
            post_cgi, post_lat, post_lon = gen_cgi(home_lat, home_lon)  # local — victim at home
            first_gap = timedelta(minutes=random.randint(15, 90))  # slower start
            agent_risk = round(random.uniform(0.10, 0.40), 2)  # legit-looking agent
            balance = random.randint(30000, 200000)
            non_fin = random.choice([3, 4, 5, 6])  # victim makes normal calls too
            sess_1h = random.randint(0, 2)
            base_dwell = random.uniform(6, 14) * pace  # victim deliberates — overlaps heavily with legit
            drain_pct = random.uniform(0.50, 0.85)
            pin_attempts = random.choice([1, 1, 2, 2])  # some fumbling
            device_type = pick_device()

        swap_channel = "agent"
        imei_match = False
        dev_profile = DEVICE_PROFILES.get(device_type)
        dwell_gen = lambda bd=base_dwell, dp=dev_profile, ec=env_ctx: add_latency_noise(
            jitter(bd, dp.session_jitter_pct if dp else 0.2), is_rural, dp, ec)

        s1_ts = swap_ts + first_gap
        s1_id = f"USS-{uuid.uuid4().hex[:8].upper()}"
        s1_dwells = [dwell_gen(), dwell_gen(), dwell_gen()]
        add_session(s1_id, msisdn, post_imei, post_cgi, s1_ts, s1_dwells,
                    "main>my_account>balance>pin", 1.0, "complete", pin_attempts, label)

        # Partial text coaching has longer inter-session gaps (reading instructions)
        if archetype == "partial_text_coaching":
            gap2 = random.randint(60, 300)  # longer gap — reading texts
        else:
            gap2 = random.randint(8, 90)

        s2_ts = s1_ts + timedelta(seconds=sum(s1_dwells) + gap2)
        s2_id = f"USS-{uuid.uuid4().hex[:8].upper()}"
        s2_dwells = [dwell_gen() for _ in range(5)]

        # Social engineering: victim deliberates more, adds hesitation dwells
        if archetype == "social_engineering_pretext":
            s2_dwells[2] = add_latency_noise(jitter(base_dwell * 2.5, 0.4), is_rural, dev_profile, env_ctx)
            s2_dwells[3] = add_latency_noise(jitter(base_dwell * 1.8, 0.3), is_rural, dev_profile, env_ctx)

        add_session(s2_id, msisdn, post_imei, post_cgi, s2_ts, s2_dwells,
                    "main>send_money>number>amount>pin>confirm", 1.0, "complete", 1, label)

        send_amount = int(balance * drain_pct / 1000) * 1000
        mule = gen_msisdn()
        txn_ts = s2_ts + timedelta(seconds=sum(s2_dwells) + 5)
        if random.random() < 0.3:
            mule_age = random.randint(90, 800)
        else:
            mule_age = random.randint(3, 60)
        add_txn(s2_id, msisdn, "SEND_MONEY", send_amount, balance, mule, False,
                mule_age, txn_ts, label)

        remaining = balance - send_amount
        if remaining > 2000 and random.random() > 0.35:
            s3_ts = txn_ts + timedelta(seconds=random.randint(15, 60))
            s3_id = f"USS-{uuid.uuid4().hex[:8].upper()}"
            s3_dwells = [dwell_gen() for _ in range(5)]
            add_session(s3_id, msisdn, post_imei, post_cgi, s3_ts, s3_dwells,
                        "main>send_money>number>amount>pin>confirm", 1.0, "complete", 1, label)
            s2_amt = int(remaining * random.uniform(0.7, 0.95) / 100) * 100
            mule2_age = random.randint(90, 600) if random.random() < 0.3 else random.randint(5, 50)
            add_txn(s3_id, msisdn, "SEND_MONEY", s2_amt, remaining, gen_msisdn(), False,
                    mule2_age, s3_ts + timedelta(seconds=sum(s3_dwells)), label)
            sess_1h = max(sess_1h, 3)

        time_to_first = first_gap.total_seconds() / 60
        displacement = ((post_lat - home_lat)**2 + (post_lon - home_lon)**2)**0.5 * 111

        # ─── Cross-class contamination: 8-10% of fraud events get softer signals ──
        if random.random() < 0.09:
            # Make this fraud case look more legitimate
            non_fin = random.randint(3, 8)       # add decoy non-financial activity
            drain_pct = random.uniform(0.25, 0.50)  # lower drain
            base_dwell = base_dwell * 1.5            # slower, more deliberate

    else:
        label = "legit"
        archetype = random.choices(
            ["normal", "urgent_mobile_money", "rural_shared", "power_user", "emergency_drain", "new_recipient"],
            weights=[0.45, 0.08, 0.12, 0.10, 0.05, 0.20]
        )[0]
        swap_channel = random.choice(["agent", "agent", "agent", "retail_store", "call_center"])
        agent_risk = round(random.uniform(0.03, 0.45), 2)
        if random.random() < 0.05: agent_risk = round(random.uniform(0.45, 0.70), 2)
        device_type = pick_device()
        dev_profile = DEVICE_PROFILES.get(device_type)

        if archetype == "normal":
            post_imei = pre_swap_imei if random.random() > 0.5 else gen_imei()
            first_gap = timedelta(hours=random.randint(2, 72))
            balance = random.randint(500, 100000); non_fin = random.randint(3, 15)
            sess_1h = 0; base_dwell = random.uniform(5, 15) * pace
            drain_pct = random.uniform(0.02, 0.30)
            send_to_known = random.random() > 0.15
            recip_age = random.randint(180, 3000)
        elif archetype == "urgent_mobile_money":
            post_imei = gen_imei()
            first_gap = timedelta(minutes=random.randint(5, 45))
            balance = random.randint(3000, 80000); non_fin = random.randint(0, 3)
            sess_1h = random.randint(1, 3); base_dwell = random.uniform(5, 12) * pace
            drain_pct = random.uniform(0.10, 0.45)
            send_to_known = random.random() > 0.25
            recip_age = random.randint(90, 2000)
        elif archetype == "rural_shared":
            post_imei = gen_imei()
            first_gap = timedelta(hours=random.randint(1, 48))
            balance = random.randint(200, 20000); non_fin = random.randint(2, 10)
            sess_1h = random.randint(0, 1); base_dwell = random.uniform(8, 25) * pace
            drain_pct = random.uniform(0.05, 0.40)
            send_to_known = random.random() > 0.20
            recip_age = random.randint(100, 2500)
        elif archetype == "power_user":
            post_imei = pre_swap_imei if random.random() > 0.4 else gen_imei()
            first_gap = timedelta(minutes=random.randint(20, 240))
            balance = random.randint(20000, 300000); non_fin = random.randint(5, 20)
            sess_1h = random.randint(0, 3); base_dwell = random.uniform(3, 7) * pace
            drain_pct = random.uniform(0.05, 0.25)
            send_to_known = random.random() > 0.10
            recip_age = random.randint(200, 3000)
        elif archetype == "emergency_drain":
            post_imei = gen_imei()
            first_gap = timedelta(minutes=random.randint(8, 90))
            balance = random.randint(10000, 100000); non_fin = random.randint(0, 4)
            sess_1h = random.randint(1, 3); base_dwell = random.uniform(4, 12) * pace
            drain_pct = random.uniform(0.55, 0.95)
            send_to_known = random.random() > 0.40
            recip_age = random.randint(15, 1500)
        else:  # new_recipient
            post_imei = pre_swap_imei if random.random() > 0.5 else gen_imei()
            first_gap = timedelta(hours=random.randint(1, 48))
            balance = random.randint(2000, 80000); non_fin = random.randint(3, 12)
            sess_1h = random.randint(0, 1); base_dwell = random.uniform(6, 18) * pace
            drain_pct = random.uniform(0.05, 0.35)
            send_to_known = False
            recip_age = random.choice([
                random.randint(5, 60),
                random.randint(30, 180),
                random.randint(90, 2000),
            ])

        imei_match = post_imei == pre_swap_imei
        imei_card = random.randint(2, 5) if archetype == "rural_shared" else random.randint(1, 2)
        imei_corr = round(random.uniform(0.0, 0.12), 2)
        if random.random() < 0.02: imei_corr = round(random.uniform(0.15, 0.35), 2)

        post_cgi, post_lat, post_lon = gen_cgi(home_lat, home_lon)
        if random.random() < 0.04:
            td = random.choice(COUNTIES)
            post_cgi, post_lat, post_lon = gen_cgi(td[1], td[2])

        # ─── Cross-class contamination: 12-15% of legit events get fraud-like signals ──
        if random.random() < 0.13:
            # Make this legit case look suspicious
            if random.random() < 0.4:
                first_gap = timedelta(minutes=random.randint(3, 25))  # fast post-swap
            if random.random() < 0.3:
                drain_pct = random.uniform(0.65, 0.92)  # high drain
            if random.random() < 0.25:
                non_fin = random.randint(0, 1)  # low non-financial
            if random.random() < 0.2:
                sess_1h = random.randint(2, 4)  # burst sessions

        s1_ts = swap_ts + first_gap
        s1_id = f"USS-{uuid.uuid4().hex[:8].upper()}"
        if random.random() > 0.55:
            path = "main>my_account>balance>back>send_money>number>amount>pin>confirm"
            directness = round(5/9, 2); n_steps = 7
        elif random.random() > 0.3:
            path = "main>send_money>number>amount>pin>confirm"
            directness = 1.0; n_steps = 5
        else:
            path = "main>my_account>balance"
            directness = 1.0; n_steps = 3

        dwells = [add_latency_noise(jitter(base_dwell, dev_profile.session_jitter_pct if dev_profile else 0.4),
                                     is_rural, dev_profile, env_ctx) for _ in range(n_steps)]
        if n_steps >= 5 and random.random() > 0.3:
            dwells[2] = add_latency_noise(jitter(base_dwell * 2.5, 0.3), is_rural, dev_profile, env_ctx)

        end_reason = "complete" if random.random() > 0.05 else random.choice(["timeout", "user_cancel"])
        pin_att = 1 if random.random() > 0.08 else 2

        add_session(s1_id, msisdn, post_imei, post_cgi, s1_ts, dwells, path, directness,
                    end_reason, pin_att, label)

        send_amount = max(100, int(balance * drain_pct / 100) * 100)
        recipient = random.choice(known_recips) if send_to_known else gen_msisdn()
        txn_type = random.choices(["SEND_MONEY", "BUY_AIRTIME", "PAY_BILL", "PAY_MERCHANT"],
                                  weights=[0.5, 0.15, 0.2, 0.15])[0]
        if end_reason == "complete":
            txn_ts = s1_ts + timedelta(seconds=sum(dwells) + 10)
            add_txn(s1_id, msisdn, txn_type, send_amount, balance, recipient,
                    send_to_known, recip_age, txn_ts, label)

        time_to_first = first_gap.total_seconds() / 60
        displacement = ((post_lat - home_lat)**2 + (post_lon - home_lon)**2)**0.5 * 111
        non_fin_final = non_fin

    # Missing IMEI noise (3%)
    if random.random() < 0.03:
        post_imei_store = "UNKNOWN"; imei_card = 0; imei_corr = 0.0; imei_match = False
    else:
        post_imei_store = post_imei

    # ─── Label noise: 2.5% random flips ──────────────────────────────
    final_label = label
    if random.random() < LABEL_NOISE_RATE:
        final_label = "legit" if label == "fraud" else "fraud"
        label_flips += 1

    sim_swaps.append({
        "msisdn": msisdn, "swap_ts": swap_ts.isoformat() + "Z",
        "swap_channel": swap_channel,
        "agent_risk_score": agent_risk,
        "pre_swap_imei": pre_swap_imei, "post_swap_imei": post_imei_store,
        "imei_match": imei_match,
        "imei_msisdn_cardinality_90d": imei_card,
        "imei_swap_correlation": imei_corr,
        "device_type": device_type, "home_county": home_name,
        "displacement_km": round(displacement, 1),
        "non_financial_activity_count": non_fin,
        "session_count_first_hour": sess_1h,
        "time_to_first_session_min": round(time_to_first, 1),
        "label": final_label, "fraud_archetype": archetype,
    })

# ─── Write CSVs ──────────────────────────────────────────────────────

def write_csv(filename, rows, fieldnames):
    with open(filename, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: str(v) if isinstance(v, list) else v for k, v in row.items()})

write_csv("sim_swaps.csv", sim_swaps, [
    "msisdn","swap_ts","swap_channel","agent_risk_score","pre_swap_imei","post_swap_imei",
    "imei_match","imei_msisdn_cardinality_90d","imei_swap_correlation","device_type",
    "home_county","displacement_km","non_financial_activity_count",
    "session_count_first_hour","time_to_first_session_min","label","fraud_archetype"])
write_csv("ussd_sessions.csv", ussd_sessions, [
    "session_id","msisdn","imei","cgi","shortcode","session_start_ts","total_steps",
    "dwell_times","dwell_variance","mean_dwell","menu_path","path_directness",
    "session_duration_s","session_end_reason","pin_attempts","label"])
write_csv("transactions.csv", transactions, [
    "txn_id","session_id","msisdn","txn_type","amount","balance_before","balance_after",
    "drain_ratio","recipient_msisdn","recipient_is_known","recipient_account_age_days",
    "txn_ts","label"])

# ─── Validation statistics ───────────────────────────────────────────

fraud_count = sum(1 for s in sim_swaps if s["label"] == "fraud")
legit_count = len(sim_swaps) - fraud_count
print(f"Generated {len(sim_swaps)} events: {legit_count} legit, {fraud_count} fraud ({fraud_count/len(sim_swaps)*100:.1f}%)")
print(f"Sessions: {len(ussd_sessions)}, Transactions: {len(transactions)}")
print(f"\nFraud archetypes: {dict(Counter(s['fraud_archetype'] for s in sim_swaps if s['label']=='fraud'))}")
print(f"Legit archetypes: {dict(Counter(s['fraud_archetype'] for s in sim_swaps if s['label']=='legit'))}")

# Overlap zone analysis
legit_fast = sum(1 for s in sim_swaps if s['label']=='legit' and float(s['time_to_first_session_min'])<30)
legit_drain = sum(1 for t in transactions if t['label']=='legit' and float(t['drain_ratio'])>0.6)
legit_unknown = sum(1 for t in transactions if t['label']=='legit' and t['recipient_is_known']==False)
print(f"\n── Overlap Zones (cross-class contamination) ──")
print(f"  Legit <30min to first session: {legit_fast}")
print(f"  Legit drain>60%: {legit_drain}")
print(f"  Legit unknown recipient: {legit_unknown}")
print(f"  Label noise flips: {label_flips} ({label_flips/NUM_EVENTS*100:.1f}%)")

print(f"\n── Distribution Overlap Metrics ──")
fraud_times = [float(s["time_to_first_session_min"]) for s in sim_swaps if s["label"]=="fraud"]
legit_times = [float(s["time_to_first_session_min"]) for s in sim_swaps if s["label"]=="legit"]
if fraud_times and legit_times:
    f_mean, l_mean = sum(fraud_times)/len(fraud_times), sum(legit_times)/len(legit_times)
    f_std = (sum((x-f_mean)**2 for x in fraud_times)/len(fraud_times))**0.5
    l_std = (sum((x-l_mean)**2 for x in legit_times)/len(legit_times))**0.5
    if f_std + l_std > 0:
        separation = abs(f_mean - l_mean) / ((f_std + l_std) / 2)
        print(f"  time_to_first_min: fraud μ={f_mean:.1f} σ={f_std:.1f} | legit μ={l_mean:.1f} σ={l_std:.1f} | separation={separation:.2f}σ")

fraud_drains = [float(t["drain_ratio"]) for t in transactions if t["label"]=="fraud"]
legit_drains = [float(t["drain_ratio"]) for t in transactions if t["label"]=="legit"]
if fraud_drains and legit_drains:
    fd_mean, ld_mean = sum(fraud_drains)/len(fraud_drains), sum(legit_drains)/len(legit_drains)
    fd_std = (sum((x-fd_mean)**2 for x in fraud_drains)/len(fraud_drains))**0.5
    ld_std = (sum((x-ld_mean)**2 for x in legit_drains)/len(legit_drains))**0.5
    if fd_std + ld_std > 0:
        separation = abs(fd_mean - ld_mean) / ((fd_std + ld_std) / 2)
        print(f"  drain_ratio: fraud μ={fd_mean:.2f} σ={fd_std:.2f} | legit μ={ld_mean:.2f} σ={ld_std:.2f} | separation={separation:.2f}σ")

print(f"\nFraud IMEI reuse:")
for imei, c in sorted(fraud_imei_usage.items(), key=lambda x: -x[1]):
    if c > 0: print(f"  {imei[:10]}...={c}")
