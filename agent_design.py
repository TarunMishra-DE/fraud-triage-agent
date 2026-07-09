"""
agent_design.py — Day 3 script
LangGraph agent: 5-node fraud triage graph with HITL escalation.

NODES:
  1. retrieve      — load case file + customer transaction history
  2. analyse       — pandas rule engine: velocity, geo, SEPA-specific flags
  3. reason        — LLM call: structured recommendation + reasoning trace
  4. confidence    — apply HITL escalation rules, set final routing
  5. log           — write result to SQLite audit log

Run a single case end-to-end:
  python agent_design.py --case_id CASE_000042

Run 10 test cases:
  python agent_design.py --test
"""

import os
import json
import sqlite3
import argparse
from datetime import datetime
from pathlib import Path
from typing import TypedDict, Literal, Optional

from anthropic import Anthropic
from langgraph.graph import StateGraph, END

# ── Config ────────────────────────────────────────────────────────────────────
MODEL          = "claude-sonnet-4-6"
DATA_DIR       = Path("./data/processed")
DB_PATH        = Path("./data/audit_log.db")
ESCALATE_THRESHOLD_EUR   = 500.0   # auto-escalate declines above this amount
LOW_CONFIDENCE_ESCALATE  = True    # escalate when confidence is low or medium

client = Anthropic()  # reads ANTHROPIC_API_KEY from env

# ── State schema ──────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    case_id:                str
    case:                   dict          # raw case file
    transaction_history:    list[dict]    # last 10 txns for this customer
    structured_findings:    dict          # output of analyse node (rules engine)
    recommendation:         str           # "approve" | "decline" | "escalate"
    confidence:             str           # "high" | "medium" | "low"
    confidence_score:       float         # 0.0–1.0
    reasoning_trace:        str           # LLM's written explanation
    secondary_flag:         str           # "fraud" | "aml_concern" | "ambiguous"
    escalation_required:    bool
    escalation_reason:      Optional[str]
    investigator_decision:  Optional[str] # filled after human review
    error:                  Optional[str]

# ── Node 1: Retrieve ──────────────────────────────────────────────────────────
def retrieve(state: AgentState) -> AgentState:
    """Load case + customer transaction history from disk."""
    case_id = state["case_id"]

    # Load the case
    case = None
    with open(DATA_DIR / "cases_sample.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if row["case_id"] == case_id:
                case = row
                break
    if not case:
        # Fallback: load first case (for testing)
        with open(DATA_DIR / "cases_sample.jsonl") as f:
            case = json.loads(f.readline())

    # Load transaction history
    with open(DATA_DIR / "history_lookup.json") as f:
        lookup = json.load(f)

    # Use sender country as a proxy customer key (demo only)
    history = lookup.get(str(case.get("sender_country", "")), [])[:10]

    return {**state, "case": case, "transaction_history": history}

# ── Node 2: Analyse (rules engine — NO LLM here) ─────────────────────────────
def analyse(state: AgentState) -> AgentState:
    """
    Structured signal extraction using deterministic rules.
    LLMs are unreliable for arithmetic over tabular data — this stays in Python.
    """
    case = state["case"]
    history = state["transaction_history"]

    findings = {}

    # Velocity
    findings["txn_count_30d"]    = case.get("txn_count_30d", 0)
    findings["high_velocity"]    = case.get("high_velocity_flag", False)
    findings["velocity_note"]    = (
        f"Customer had {findings['txn_count_30d']} transactions in the last 30 days — "
        + ("above normal threshold (>10)." if findings["high_velocity"] else "within normal range.")
    )

    # Geographic mismatch
    findings["is_cross_border"]   = case.get("is_cross_border", False)
    findings["sender_country"]    = case.get("sender_country", "?")
    findings["receiver_country"]  = case.get("receiver_country", "?")
    findings["geo_note"] = (
        f"Cross-border transfer: {findings['sender_country']} → {findings['receiver_country']}."
        if findings["is_cross_border"]
        else f"Domestic transfer within {findings['sender_country']}."
    )

    # SEPA-specific
    findings["sepa_type"]         = case.get("sepa_type", "SEPA_CREDIT")
    findings["is_instant"]        = findings["sepa_type"] == "SEPA_INSTANT"
    findings["sepa_note"] = (
        "SEPA Instant transfer — irreversible once processed. Elevated risk."
        if findings["is_instant"]
        else "Standard SEPA Credit Transfer — reversible within T+1."
    )

    # Amount context
    findings["amount_eur"]        = case.get("amount_eur", 0.0)
    findings["amount_note"] = (
        f"Transaction amount: €{findings['amount_eur']:,.2f}. "
        + ("High value — above auto-escalation threshold." if findings["amount_eur"] > ESCALATE_THRESHOLD_EUR else "Below auto-escalation threshold.")
    )

    # AML signal
    findings["aml_signal"]        = case.get("aml_signal", False)
    findings["aml_note"] = (
        "AML signal raised: cross-border SEPA Instant above €1,000. "
        "If confirmed, this routes to SAR filing — NOT customer contact."
        if findings["aml_signal"]
        else "No AML signal detected."
    )

    # Upstream model score
    score = case.get("upstream_fraud_score", 0.0)
    findings["upstream_score"]    = score
    findings["score_note"] = (
        f"Upstream fraud model score: {score:.3f}. "
        + ("High-confidence fraud signal." if score > 0.7
           else "Moderate signal — requires review." if score > 0.4
           else "Low fraud signal from upstream model.")
    )

    # Recent history summary
    if history:
        recent_amounts = [h.get("amount_eur", 0) for h in history]
        avg_amount = sum(recent_amounts) / len(recent_amounts)
        findings["history_note"] = (
            f"Last {len(history)} transactions: avg €{avg_amount:,.2f}. "
            f"Current transaction is {'above' if findings['amount_eur'] > avg_amount * 2 else 'consistent with'} recent behaviour."
        )
    else:
        findings["history_note"] = "No recent transaction history available."

    return {**state, "structured_findings": findings}

# ── Node 3: Reason (LLM call) ─────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a fraud triage assistant for a BaFin-regulated German digital bank.
Your role is to help fraud investigators review flagged SEPA transactions.

You NEVER make final decisions — you provide structured recommendations that a human investigator reviews.
You are a decision-support tool. Human oversight is mandatory on every case.

CONFIDENCE CALIBRATION — follow these rules strictly:
- "high" (0.85–1.0): ALL major signals point the same direction with no conflicting data.
  Every single one of these must be true for HIGH confidence:
  · Upstream score is unambiguously high (>0.80) OR unambiguously low (<0.20)
  · Velocity, geography, and SEPA type all consistently support the same conclusion
  · Transaction history is available and consistent with the conclusion
  · No missing data fields
  If ANY signal conflicts or ANY data is missing → confidence CANNOT be "high"

- "medium" (0.45–0.84): One strong signal with mitigating factors, OR mixed signals.
  Default to medium when uncertain. Most real cases are medium.
  Examples: high score but domestic low-amount transfer; cross-border but low score.

- "low" (0.0–0.44): Contradictory signals, missing history, or genuinely unresolvable.
  Examples: no transaction history + moderate score; signals point in opposite directions.

IMPORTANT: When in doubt between high and medium, always choose medium.
High confidence requires every signal to be unambiguous and consistent.
Missing transaction history alone is sufficient reason to downgrade to medium or low.

RECOMMENDATION RULES:
- "approve": confidence is high AND upstream score <0.35 AND no AML signal AND amount <€200
- "decline": confidence is high AND upstream score >0.75 AND clear fraud signals converge
- "escalate": everything else, and always when confidence is medium or low

SECONDARY FLAG RULES:
- "aml_concern": cross-border AND SEPA Instant AND amount >€1,000 — routes to SAR, not customer contact
- "fraud": clear card fraud or APP fraud signals without AML indicators  
- "ambiguous": signals conflict or insufficient data

IMPORTANT — do not default to medium confidence. Force yourself to evaluate whether 
signals are truly mixed (medium) or whether they clearly converge (high) or clearly 
conflict/are missing (low). A perfect upstream score of 1.0 with high velocity is 
HIGH confidence fraud — not medium. A low upstream score with no anomalies is HIGH 
confidence legitimate — not medium.

Your output must be a valid JSON object with exactly these fields:
{
  "recommendation": "approve" | "decline" | "escalate",
  "confidence": "high" | "medium" | "low",
  "confidence_score": <float 0.0-1.0>,
  "reasoning_trace": "<clear, step-by-step explanation citing specific values>",
  "secondary_flag": "fraud" | "aml_concern" | "ambiguous",
  "key_risk_signals": ["<signal 1>", "<signal 2>"],
  "key_benign_signals": ["<signal 1>", "<signal 2>"],
  "investigator_note": "<one sentence on what the investigator should verify>"
}

Rules:
- If secondary_flag is "aml_concern": note this routes to SAR filing — NOT customer contact.
- Cite specific values in reasoning_trace (amounts, countries, scores, velocity counts).
- reasoning_trace must be readable by a non-technical investigator.
- Respond with ONLY the JSON object. No preamble, no markdown fences."""

def reason(state: AgentState) -> AgentState:
    """LLM reasoning node — produces structured recommendation with trace."""
    case     = state["case"]
    findings = state["structured_findings"]

    user_message = f"""Case ID: {case['case_id']}

TRANSACTION DETAILS:
- Date: {case.get('transaction_date', 'unknown')}
- Amount: €{case.get('amount_eur', 0):,.2f}
- SEPA type: {case.get('sepa_type')}
- Merchant category: {case.get('merchant_category')} (MCC {case.get('mcc_code')})
- Sender IBAN: {case.get('sender_iban')} ({case.get('sender_country')})
- Receiver IBAN: {case.get('receiver_iban')} ({case.get('receiver_country')})
- Receiver BIC: {case.get('receiver_bic')}
- Upstream fraud score: {case.get('upstream_fraud_score', 0):.3f}

STRUCTURED ANALYSIS (rules engine):
- Velocity: {findings['velocity_note']}
- Geography: {findings['geo_note']}
- SEPA: {findings['sepa_note']}
- Amount context: {findings['amount_note']}
- AML signal: {findings['aml_note']}
- Upstream model: {findings['score_note']}
- Transaction history: {findings['history_note']}

Provide your triage recommendation as a JSON object."""

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            temperature = 0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        raw = response.content[0].text.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)

        return {
            **state,
            "recommendation":  result.get("recommendation", "escalate"),
            "confidence":      result.get("confidence", "low"),
            "confidence_score": float(result.get("confidence_score", 0.3)),
            "reasoning_trace": result.get("reasoning_trace", ""),
            "secondary_flag":  result.get("secondary_flag", "ambiguous"),
        }
    except Exception as e:
        # Safe fallback — always escalate on error
        return {
            **state,
            "recommendation":  "escalate",
            "confidence":      "low",
            "confidence_score": 0.0,
            "reasoning_trace": f"Agent error — escalated for manual review. Error: {str(e)}",
            "secondary_flag":  "ambiguous",
            "error":           str(e),
        }

# ── Node 4: Confidence / HITL escalation ─────────────────────────────────────
def apply_hitl_rules(state: AgentState) -> AgentState:
    escalate = False
    reason   = []

    # Escalate low/medium confidence
    if state["confidence"] in ("low", "medium") and LOW_CONFIDENCE_ESCALATE:
        escalate = True
        reason.append(f"Confidence is {state['confidence']} — mandatory human review")

    # Escalate high-value declines
    if (state["recommendation"] == "decline"
            and state["case"].get("amount_eur", 0) > ESCALATE_THRESHOLD_EUR):
        escalate = True
        reason.append(
            f"Decline on high-value transaction "
            f"(€{state['case'].get('amount_eur',0):,.2f} > €{ESCALATE_THRESHOLD_EUR})"
        )

    # Escalate any AML signal
    if state["structured_findings"].get("aml_signal"):
        escalate = True
        reason.append("AML signal — routes to SAR filing, not customer contact")

    if state["secondary_flag"] == "aml_concern":
        escalate = True
        reason.append("AML classification — regulatory reporting obligation triggered")

    if escalate:
        return {
            **state,
            "escalation_required": True,
            "escalation_reason":   " | ".join(reason),
            "recommendation":      "escalate",
        }

    return {**state, "escalation_required": False, "escalation_reason": None}

# ── Node 5: Log (audit trail) ─────────────────────────────────────────────────
def log_to_audit(state: AgentState) -> AgentState:
    """Write recommendation to SQLite audit log (BaFin / MaRisk requirement)."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at             TEXT,
            case_id               TEXT,
            transaction_id        TEXT,
            amount_eur            REAL,
            agent_recommendation  TEXT,
            confidence            TEXT,
            confidence_score      REAL,
            secondary_flag        TEXT,
            escalation_required   INTEGER,
            escalation_reason     TEXT,
            reasoning_trace       TEXT,
            investigator_decision TEXT,
            override_reason       TEXT
        )
    """)
    conn.execute("""
        INSERT INTO audit_log
        (logged_at, case_id, transaction_id, amount_eur, agent_recommendation,
         confidence, confidence_score, secondary_flag, escalation_required,
         escalation_reason, reasoning_trace, investigator_decision, override_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        state["case_id"],
        state["case"].get("transaction_id"),
        state["case"].get("amount_eur"),
        state["recommendation"],
        state["confidence"],
        state["confidence_score"],
        state["secondary_flag"],
        int(state.get("escalation_required", False)),
        state.get("escalation_reason"),
        state["reasoning_trace"],
        state.get("investigator_decision"),
        None,
    ))
    conn.commit()
    conn.close()
    return state

# ── Build the graph ───────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("retrieve",         retrieve)
    graph.add_node("analyse",          analyse)
    graph.add_node("reason",           reason)
    graph.add_node("apply_hitl_rules", apply_hitl_rules)
    graph.add_node("log_to_audit",     log_to_audit)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve",         "analyse")
    graph.add_edge("analyse",          "reason")
    graph.add_edge("reason",           "apply_hitl_rules")
    graph.add_edge("apply_hitl_rules", "log_to_audit")
    graph.add_edge("log_to_audit",     END)

    return graph.compile()

app = build_graph()

# ── CLI runner ────────────────────────────────────────────────────────────────
def run_case(case_id: str) -> dict:
    initial_state: AgentState = {
        "case_id":               case_id,
        "case":                  {},
        "transaction_history":   [],
        "structured_findings":   {},
        "recommendation":        "",
        "confidence":            "",
        "confidence_score":      0.0,
        "reasoning_trace":       "",
        "secondary_flag":        "",
        "escalation_required":   False,
        "escalation_reason":     None,
        "investigator_decision": None,
        "error":                 None,
    }
    result = app.invoke(initial_state)
    return result

def print_result(result: dict):
    print("\n" + "="*60)
    print(f"CASE:           {result['case_id']}")
    print(f"Amount:         €{result['case'].get('amount_eur', 0):,.2f}")
    print(f"SEPA type:      {result['case'].get('sepa_type')}")
    print(f"Country:        {result['case'].get('sender_country')} → {result['case'].get('receiver_country')}")
    print(f"Upstream score: {result['case'].get('upstream_fraud_score', 0):.3f}")
    print("-"*60)
    print(f"RECOMMENDATION: {result['recommendation'].upper()}")
    print(f"CONFIDENCE:     {result['confidence']} ({result['confidence_score']:.2f})")
    print(f"SECONDARY FLAG: {result['secondary_flag']}")
    if result["escalation_required"]:
        print(f"⚠️  ESCALATED:  {result['escalation_reason']}")
    print(f"\nREASONING:\n{result['reasoning_trace']}")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_id", type=str, default=None)
    parser.add_argument("--test",    action="store_true")
    args = parser.parse_args()

    if args.test:
        # Load first 10 cases from sample
        cases = []
        with open(DATA_DIR / "cases_sample.jsonl") as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                cases.append(json.loads(line)["case_id"])

        print(f"Running {len(cases)} test cases...")
        for cid in cases:
            result = run_case(cid)
            print_result(result)
    else:
        cid = args.case_id or "CASE_000000"
        result = run_case(cid)
        print_result(result)
