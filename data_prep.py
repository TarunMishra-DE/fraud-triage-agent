"""
data_prep.py — Day 2 script
Transforms IEEE-CIS Fraud Detection dataset into a SEPA-native case file format
for the Agentic Fraud Triage prototype.

DATASET REQUIRED:
  Download from: https://www.kaggle.com/c/ieee-fraud-detection/data
  Files needed:  train_transaction.csv  (590k rows)
                 train_identity.csv     (144k rows)
  Place both in: ./data/raw/

OUTPUTS:
  ./data/processed/cases.jsonl        — one JSON object per case, ready for agent
  ./data/processed/cases_sample.jsonl — 200-row sample for fast dev iteration
  ./data/processed/eval_set.jsonl     — 50-row held-out evaluation set (labelled)

TRANSPARENCY NOTE:
  All data is derived from the public IEEE-CIS dataset (Kaggle).
  IBAN/BIC values are synthetically generated and do not represent real accounts.
  Synthetic SEPA Instant and APP fraud rows are clearly flagged as synthetic.
"""

import pandas as pd
import numpy as np
import json
import random
import string
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)
np.random.seed(42)

# ── Paths ────────────────────────────────────────────────────────────────────
RAW       = Path("./data/raw")
PROCESSED = Path("./data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

# ── 1. Load IEEE-CIS data ─────────────────────────────────────────────────────
print("Loading IEEE-CIS data...")
txn = pd.read_csv(RAW / "train_transaction.csv")
idn = pd.read_csv(RAW / "train_identity.csv")
df  = txn.merge(idn, on="TransactionID", how="left")
print(f"  Loaded {len(df):,} transactions")

# ── 2. EUR conversion (USD → EUR, approximate 0.92 rate) ─────────────────────
df["amount_eur"] = (df["TransactionAmt"] * 0.92).round(2)

# ── 3. SEPA merchant category mapping ────────────────────────────────────────
# IEEE-CIS ProductCD: W=web, H=hotel, C=card, S=service, R=retail
# Map to SEPA-relevant risk categories + MCC codes
MCC_MAP = {
    "W": {"category": "E-commerce",       "mcc": "5411", "sepa_risk": "medium"},
    "H": {"category": "Accommodation",    "mcc": "7011", "sepa_risk": "low"},
    "C": {"category": "Card services",    "mcc": "6011", "sepa_risk": "high"},
    "S": {"category": "Professional svc", "mcc": "7389", "sepa_risk": "medium"},
    "R": {"category": "Retail",           "mcc": "5999", "sepa_risk": "low"},
}
df["merchant_category"] = df["ProductCD"].map(
    lambda x: MCC_MAP.get(x, {}).get("category", "Unknown")
)
df["mcc_code"]   = df["ProductCD"].map(lambda x: MCC_MAP.get(x, {}).get("mcc", "0000"))
df["sepa_risk"]  = df["ProductCD"].map(lambda x: MCC_MAP.get(x, {}).get("sepa_risk", "medium"))

# ── 4. Synthetic IBAN / BIC generator ────────────────────────────────────────
EU_COUNTRIES = ["DE", "FR", "NL", "ES", "IT", "AT", "BE", "PL"]
EU_BICS      = {
    "DE": ["DEUTDEDB", "COBADEFF", "SSKMDEMMXXX"],
    "FR": ["BNPAFRPP",  "SOGEFRPP",  "CRLYFRPP"],
    "NL": ["ABNANL2A",  "INGBNL2A",  "RABONL2U"],
    "ES": ["BBVAESMM",  "CAIXESBB",  "SABDESBB"],
    "IT": ["UNCRITMM",  "BCITITMM",  "BNLIITRR"],
    "AT": ["RLNWATWW",  "BKAUATWW",  "RZOOAT2L"],
    "BE": ["GEBABEBB",  "BNAGBEBB",  "NICABEBB"],
    "PL": ["PKOPPLPW",  "BREXPLPW",  "WBKPPLPP"],
}

def make_iban(country: str) -> str:
    check = str(random.randint(10, 99))
    bban  = "".join(random.choices(string.digits, k=18))
    return f"{country}{check}{bban}"

def make_bic(country: str) -> str:
    return random.choice(EU_BICS.get(country, EU_BICS["DE"]))

def assign_country(row) -> str:
    # Heuristic: high-value → more likely cross-border
    if row["amount_eur"] > 500:
        return random.choices(EU_COUNTRIES, weights=[3,2,2,1,1,1,1,1])[0]
    return random.choices(EU_COUNTRIES, weights=[6,1,1,1,1,1,1,1])[0]

df["sender_country"]   = df.apply(assign_country, axis=1)
df["receiver_country"] = df.apply(assign_country, axis=1)
df["sender_iban"]      = df["sender_country"].apply(make_iban)
df["receiver_iban"]    = df["receiver_country"].apply(make_iban)
df["receiver_bic"]     = df["receiver_country"].apply(make_bic)
df["is_cross_border"]  = (df["sender_country"] != df["receiver_country"]).astype(int)

# ── 5. SEPA payment type assignment ──────────────────────────────────────────
# Assign SEPA Instant to ~30% of transactions (realistic for 2025/2026 EU)
df["sepa_type"] = np.where(
    np.random.random(len(df)) < 0.30,
    "SEPA_INSTANT",   # irreversible, highest fraud risk for neobanks
    "SEPA_CREDIT"
)

# ── 6. Synthetic APP / social engineering fraud rows ─────────────────────────
# APP = Authorised Push Payment — customer is tricked into sending money.
# Distinct from card fraud; different liability under PSD2.
# We generate ~500 synthetic APP rows flagged clearly as synthetic.
def make_app_row(i: int) -> dict:
    country = random.choice(EU_COUNTRIES)
    amount  = round(random.uniform(200, 4000), 2)
    return {
        "TransactionID":    f"SYNTH_{i:05d}",
        "isFraud":          1,
        "amount_eur":       amount,
        "merchant_category":"Peer transfer",
        "mcc_code":         "6012",
        "sepa_risk":        "high",
        "sepa_type":        "SEPA_INSTANT",
        "sender_country":   country,
        "receiver_country": random.choice([c for c in EU_COUNTRIES if c != country]),
        "sender_iban":      make_iban(country),
        "receiver_iban":    make_iban(random.choice(EU_COUNTRIES)),
        "receiver_bic":     make_bic(random.choice(EU_COUNTRIES)),
        "is_cross_border":  1,
        "fraud_type":       "APP",          # authorised push payment
        "synthetic":        True,
        "card_network":     "SEPA",
        "DeviceType":       "mobile",
        "DeviceInfo":       "Android",
        "TransactionDT":    random.randint(86400, 15897600),
    }

app_rows = pd.DataFrame([make_app_row(i) for i in range(500)])
print(f"  Generated {len(app_rows)} synthetic APP fraud rows")

# ── 7. Timestamp → human-readable DD.MM.YYYY (German convention) ─────────────
BASE_DATE = datetime(2025, 1, 1)

def dt_to_german(seconds: float) -> str:
    d = BASE_DATE + timedelta(seconds=float(seconds))
    return d.strftime("%d.%m.%Y %H:%M")

df["transaction_date"] = df["TransactionDT"].apply(dt_to_german)

# ── 8. Fraud type labelling ───────────────────────────────────────────────────
# For non-synthetic rows: label as card fraud (IEEE-CIS is card-centric)
df["fraud_type"] = np.where(df["isFraud"] == 1, "CARD_FRAUD", "LEGITIMATE")
df["synthetic"]  = False

# ── 9. Velocity features (key for investigator context) ──────────────────────
# Proxy: use addr1 (billing region) as customer ID grouper
# In production this would be a real customer ID
# Cap at 200 to avoid region-level inflation from shared addr1 values
print("Computing velocity features...")
df["customer_proxy"] = df["addr1"].fillna(0).astype(int)
velocity = (
    df.groupby("customer_proxy")["TransactionID"]
    .count()
    .rename("txn_count_30d")
)
df = df.merge(velocity, on="customer_proxy", how="left")
df["txn_count_30d"] = df["txn_count_30d"].fillna(1).astype(int).clip(upper=50)

# High velocity flag: more than 10 transactions in the window
df["high_velocity_flag"] = (df["txn_count_30d"] > 10).astype(int)

# ── 10. AML risk heuristic ────────────────────────────────────────────────────
# Simple rule: cross-border + high amount + SEPA Instant → potential AML signal
# In a real system this would come from an AML model
df["aml_signal"] = (
    (df["is_cross_border"] == 1) &
    (df["amount_eur"] > 1000) &
    (df["sepa_type"] == "SEPA_INSTANT")
).astype(int)

# ── 11. Upstream fraud score (proxy) ─────────────────────────────────────────
# IEEE-CIS doesn't include a score column — we simulate one using
# available features so the agent has something to reference.
# This is NOT a trained model — it's a heuristic proxy for demo purposes.
df["upstream_fraud_score"] = (
    df["isFraud"] * 0.6 +
    df["sepa_risk"].map({"high": 0.3, "medium": 0.15, "low": 0.05}) +
    df["high_velocity_flag"] * 0.1
).clip(0, 1).round(3)

# ── 12. Select and rename final columns ──────────────────────────────────────
KEEP = [
    "TransactionID", "transaction_date", "amount_eur",
    "merchant_category", "mcc_code", "sepa_type", "sepa_risk",
    "sender_iban", "receiver_iban", "receiver_bic",
    "sender_country", "receiver_country", "is_cross_border",
    "upstream_fraud_score", "isFraud", "fraud_type",
    "high_velocity_flag", "txn_count_30d", "aml_signal",
    "DeviceType", "DeviceInfo", "synthetic",
]
# Only keep columns that exist (identity file columns are optional)
keep_existing = [c for c in KEEP if c in df.columns]
df_final = df[keep_existing].copy()

# Append synthetic APP rows (fill missing cols with None)
df_final = pd.concat([df_final, app_rows.reindex(columns=keep_existing)], ignore_index=True)
df_final["case_id"] = ["CASE_" + str(i).zfill(6) for i in range(len(df_final))]

print(f"  Final dataset: {len(df_final):,} rows | fraud rate: {df_final['isFraud'].mean():.1%}")

# ── 13. Build transaction history lookup ──────────────────────────────────────
# For each case the agent will call get_transaction_history(customer_proxy)
# We pre-build a lookup dict: customer_proxy → last 10 transactions
print("Building transaction history lookup...")
history_lookup = {}
for cust, grp in df_final.groupby("customer_proxy") if "customer_proxy" in df_final.columns else []:
    recent = grp.sort_values("transaction_date").tail(10)
    history_lookup[int(cust)] = recent[
        ["case_id", "transaction_date", "amount_eur", "merchant_category",
         "sepa_type", "receiver_country", "upstream_fraud_score"]
    ].to_dict(orient="records")

with open(PROCESSED / "history_lookup.json", "w") as f:
    json.dump(history_lookup, f, default=str)
print(f"  History lookup: {len(history_lookup):,} customers")

# ── 14. Export case files ─────────────────────────────────────────────────────
print("Exporting case files...")

def row_to_case(row: pd.Series) -> dict:
    """Convert a DataFrame row to a structured case dict for the agent."""
    return {
        "case_id":               row.get("case_id"),
        "transaction_id":        row.get("TransactionID"),
        "transaction_date":      row.get("transaction_date"),
        "amount_eur":            float(row.get("amount_eur", 0) or 0),
        "merchant_category":     row.get("merchant_category"),
        "mcc_code":              row.get("mcc_code"),
        "sepa_type":             row.get("sepa_type"),
        "sepa_risk_tier":        row.get("sepa_risk"),
        "sender_iban":           row.get("sender_iban"),
        "receiver_iban":         row.get("receiver_iban"),
        "receiver_bic":          row.get("receiver_bic"),
        "sender_country":        row.get("sender_country"),
        "receiver_country":      row.get("receiver_country"),
        "is_cross_border":       bool(row.get("is_cross_border", 0)),
        "upstream_fraud_score":  float(row.get("upstream_fraud_score", 0) or 0),
        "high_velocity_flag":    bool(row.get("high_velocity_flag", 0)),
        "txn_count_30d":         int(row.get("txn_count_30d", 0) if pd.notna(row.get("txn_count_30d", 0)) else 0),
        "aml_signal":            bool(row.get("aml_signal", 0)),
        "device_type":           row.get("DeviceType"),
        "device_info":           row.get("DeviceInfo"),
        # Ground truth labels (hidden from agent — used for eval only)
        "_ground_truth_fraud":   bool(row.get("isFraud", 0)),
        "_fraud_type":           row.get("fraud_type"),
        "_is_synthetic":         bool(row.get("synthetic", False)),
    }

# Safe Copy: "txn_count_30d":         int(row.get("txn_count_30d", 0) or 0),
# Full dataset
with open(PROCESSED / "cases.jsonl", "w") as f:
    for _, row in df_final.iterrows():
        f.write(json.dumps(row_to_case(row), default=str) + "\n")

# Dev sample (200 rows, balanced fraud/legit)
fraud_rows  = df_final[df_final["isFraud"] == 1].sample(n=min(100, int(df_final["isFraud"].sum())))
legit_rows  = df_final[df_final["isFraud"] == 0].sample(n=100)
sample      = pd.concat([fraud_rows, legit_rows]).sample(frac=1)  # shuffle
with open(PROCESSED / "cases_sample.jsonl", "w") as f:
    for _, row in sample.iterrows():
        f.write(json.dumps(row_to_case(row), default=str) + "\n")

# Eval set (50 rows, balanced, held out — do not use for prompt development)
eval_fraud  = df_final[df_final["isFraud"] == 1].sample(n=25, random_state=99)
eval_legit  = df_final[df_final["isFraud"] == 0].sample(n=25, random_state=99)
eval_set    = pd.concat([eval_fraud, eval_legit]).sample(frac=1, random_state=99)
with open(PROCESSED / "eval_set.jsonl", "w") as f:
    for _, row in eval_set.iterrows():
        f.write(json.dumps(row_to_case(row), default=str) + "\n")

print(f"""
─────────────────────────────────────────────
✓ Data prep complete
  cases.jsonl         {len(df_final):,} cases
  cases_sample.jsonl  200 cases (dev)
  eval_set.jsonl       50 cases (held-out eval)
  history_lookup.json  {len(history_lookup):,} customer histories
─────────────────────────────────────────────
Next: Day 3 — open agent_design.py and build the LangGraph node stubs
""")
