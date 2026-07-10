"""
app.py — Streamlit investigator UI
Fraud Triage Agent · N26-inspired · AI PM Portfolio

Run with: streamlit run app.py
"""

import streamlit as st
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from agent_design import run_case, build_graph

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Triage Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Global */
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap');
  
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
  
  /* Case card */
  .case-card {
    background: #F7F8FB; border: 1px solid #E2E8F0;
    border-radius: 10px; padding: 20px 22px; margin-bottom: 16px;
  }
  .case-id {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    color: #64748B; letter-spacing: 0.08em; margin-bottom: 4px;
  }
  .case-amount {
    font-size: 32px; font-weight: 300; color: #0A1628;
    letter-spacing: -0.02em; line-height: 1;
  }
  .case-meta { font-size: 13px; color: #64748B; margin-top: 4px; }

  /* Signal row */
  .signal-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 10px; margin: 16px 0;
  }
  .signal-card {
    background: white; border: 1px solid #E2E8F0;
    border-radius: 7px; padding: 10px 14px;
  }
  .signal-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    color: #94A3B8; letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 4px;
  }
  .signal-value { font-size: 13px; font-weight: 500; color: #0F172A; }
  .signal-value.flag { color: #DC2626; }
  .signal-value.warn { color: #D97706; }
  .signal-value.ok   { color: #059669; }

  /* Reasoning block */
  .reasoning-block {
    background: white; border: 1px solid #E2E8F0;
    border-left: 3px solid #1B4FD8;
    border-radius: 0 7px 7px 0;
    padding: 14px 16px; margin: 14px 0;
  }
  .reasoning-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    color: #1B4FD8; letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 8px;
  }
  .reasoning-text { font-size: 13px; line-height: 1.7; color: #0F172A; }

  /* Badges */
  .badge {
    display: inline-block; font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; font-weight: 500; padding: 3px 10px;
    border-radius: 4px; letter-spacing: 0.04em;
  }
  .badge-escalate { background: #FEF3C7; color: #92400E; }
  .badge-decline  { background: #FEE2E2; color: #991B1B; }
  .badge-approve  { background: #D1FAE5; color: #065F46; }
  .badge-aml      { background: #EDE9FE; color: #4C1D95; }
  .badge-fraud    { background: #FEE2E2; color: #991B1B; }
  .badge-ambig    { background: #F1F5F9; color: #475569; }
  .badge-high     { background: #D1FAE5; color: #065F46; }
  .badge-medium   { background: #FEF3C7; color: #92400E; }
  .badge-low      { background: #F1F5F9; color: #64748B; }

  /* HITL notice */
  .hitl-notice {
    background: #FEF3C7; border: 1px solid #FDE68A;
    border-radius: 7px; padding: 12px 16px;
    font-size: 13px; color: #92400E; line-height: 1.55;
    margin: 12px 0;
  }
  .hitl-notice strong { font-weight: 600; }

  /* AML notice */
  .aml-notice {
    background: #EDE9FE; border: 1px solid #C4B5FD;
    border-radius: 7px; padding: 12px 16px;
    font-size: 13px; color: #4C1D95; line-height: 1.55;
    margin: 12px 0;
  }

  /* Audit row */
  .audit-row {
    background: #F7F8FB; border: 1px solid #E2E8F0;
    border-radius: 7px; padding: 10px 14px; margin-bottom: 8px;
    font-size: 12px; color: #475569;
  }
  .audit-case-id {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    color: #0A1628; font-weight: 500;
  }

  /* Sidebar */
  .sidebar-metric {
    text-align: center; padding: 12px;
    background: white; border: 1px solid #E2E8F0;
    border-radius: 8px; margin-bottom: 10px;
  }
  .sidebar-metric-value {
    font-size: 28px; font-weight: 300; color: #0A1628;
    letter-spacing: -0.02em; line-height: 1;
  }
  .sidebar-metric-label {
    font-size: 11px; color: #94A3B8; margin-top: 4px;
    font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.06em;
  }

  /* Divider */
  hr { border: none; border-top: 1px solid #E2E8F0; margin: 16px 0; }
            
/* Hide sticky top navigation header */
  header[data-testid="stHeader"] { display: none !important; }
  div[data-testid="stToolbar"] { display: none !important; }
  .stAppHeader { display: none !important; }
            
</style>
""", unsafe_allow_html=True)

# Hide the sticky duplicate tab header
st.markdown("""
<style>
section[data-testid="stSidebar"] ~ div [data-testid="stHeader"] {
    display: none;
}
div[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
DATA_DIR = Path("./data/processed")
DB_PATH  = Path("./data/audit_log.db")

@st.cache_data
def load_cases():
    cases = []
    path = DATA_DIR / "cases_sample.jsonl"
    if path.exists():
        with open(path) as f:
            for line in f:
                cases.append(json.loads(line))
    return cases

def load_audit_log():
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT logged_at, case_id, agent_recommendation, confidence,
               escalation_required, escalation_reason,
               investigator_decision, override_reason
        FROM audit_log
        ORDER BY logged_at DESC LIMIT 50
    """).fetchall()
    conn.close()
    return rows

def log_investigator_decision(case_id, decision, override_reason=None):
    if not DB_PATH.exists():
        return
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        UPDATE audit_log
        SET investigator_decision = ?, override_reason = ?
        WHERE case_id = ?
        AND id = (SELECT MAX(id) FROM audit_log WHERE case_id = ?)
    """, (decision, override_reason, case_id, case_id))
    conn.commit()
    conn.close()

def rec_badge(rec):
    cls = {"escalate": "escalate", "decline": "decline", "approve": "approve"}.get(rec, "ambig")
    return f'<span class="badge badge-{cls}">{rec.upper()}</span>'

def conf_badge(conf):
    return f'<span class="badge badge-{conf}">{conf.upper()}</span>'

def flag_badge(flag):
    cls = {"aml_concern": "aml", "fraud": "fraud", "ambiguous": "ambig"}.get(flag, "ambig")
    label = {"aml_concern": "AML CONCERN", "fraud": "FRAUD", "ambiguous": "AMBIGUOUS"}.get(flag, flag.upper())
    return f'<span class="badge badge-{cls}">{label}</span>'

def signal_class(value_str):
    v = str(value_str).lower()
    if any(w in v for w in ["above", "high", "instant", "cross", "flag"]):
        return "flag"
    if any(w in v for w in ["moderate", "medium", "warn"]):
        return "warn"
    return "ok"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Fraud Triage Agent")
    st.markdown(
        "<div style='font-size:12px;color:#64748B;margin-bottom:16px'>"
        "N26-inspired · BaFin-regulated · EU AI Act compliant"
        "</div>",
        unsafe_allow_html=True,
    )

    # Stats from audit log
    audit = load_audit_log()
    total     = len(audit)
    escalated = sum(1 for r in audit if r[4] == 1)
    decided   = sum(1 for r in audit if r[6] is not None)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f'<div class="sidebar-metric">'
            f'<div class="sidebar-metric-value">{total}</div>'
            f'<div class="sidebar-metric-label">CASES RUN</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="sidebar-metric">'
            f'<div class="sidebar-metric-value">{escalated}</div>'
            f'<div class="sidebar-metric-label">ESCALATED</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px;color:#94A3B8;font-family:monospace;margin-bottom:8px'>"
        "REGULATORY NOTES"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:12px;color:#475569;line-height:1.6'>"
        "⚖️ EU AI Act Annex III — high-risk classification<br>"
        "👤 Article 14 — human oversight on every case<br>"
        "📋 BaFin / MaRisk — full audit log maintained<br>"
        "🔒 GDPR Art. 22 — no solely-automated decisions"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px;color:#94A3B8;font-family:monospace;margin-bottom:8px'>"
        "DATA TRANSPARENCY"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:11px;color:#94A3B8;line-height:1.6'>"
        "Public IEEE-CIS dataset (Kaggle) + synthetic SEPA rows. "
        "No real customer data. Inspired by published N26 job postings "
        "and BaFin regulatory filings."
        "</div>",
        unsafe_allow_html=True,
    )


# ── Main tabs ─────────────────────────────────────────────────────────────────
# Find this:
tab_review, tab_queue, tab_audit = st.tabs([
    "🔎 Case review", "📋 Case queue", "📁 Audit log"
])

# Replace with:
tab_review, tab_queue, tab_audit = st.tabs([
    "Case review", "Case queue", "Audit log"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Case review
# ══════════════════════════════════════════════════════════════════════════════
with tab_review:
    st.markdown("#### Investigator review panel")
    st.markdown(
        "<div style='font-size:13px;color:#64748B;margin-bottom:20px'>"
        "Select a case and run the agent. Review the recommendation and reasoning "
        "trace, then record your decision. Every action is logged to the audit trail."
        "</div>",
        unsafe_allow_html=True,
    )

    cases = load_cases()
    if not cases:
        st.error("No cases found. Run data_prep.py first.")
        st.stop()

    case_ids = [c["case_id"] for c in cases[:50]]

    col_select, col_run = st.columns([3, 1])
    with col_select:
        selected_id = st.selectbox(
            "Select case", case_ids,
            label_visibility="collapsed",
            placeholder="Choose a case ID..."
        )
    with col_run:
        run_btn = st.button("▶ Run agent", type="primary", use_container_width=True)

    # Retrieve raw case for display
    selected_case = next((c for c in cases if c["case_id"] == selected_id), None)

    if selected_case:
        # Case summary card (always shown)
        amount    = selected_case.get("amount_eur", 0)
        sepa_type = selected_case.get("sepa_type", "—")
        s_country = selected_case.get("sender_country", "?")
        r_country = selected_case.get("receiver_country", "?")
        date      = selected_case.get("transaction_date", "—")
        merchant  = selected_case.get("merchant_category", "—")
        mcc       = selected_case.get("mcc_code", "—")
        score     = selected_case.get("upstream_fraud_score", 0)
        velocity  = selected_case.get("txn_count_30d", 0)
        cross     = selected_case.get("is_cross_border", False)
        aml       = selected_case.get("aml_signal", False)
        s_iban    = selected_case.get("sender_iban", "—")
        r_iban    = selected_case.get("receiver_iban", "—")
        r_bic     = selected_case.get("receiver_bic", "—")

        st.markdown(
            f"""<div class="case-card">
              <div class="case-id">{selected_id} · {date}</div>
              <div class="case-amount">€ {amount:,.2f}</div>
              <div class="case-meta">
                {sepa_type} &nbsp;·&nbsp; {s_country} → {r_country}
                &nbsp;·&nbsp; {merchant} (MCC {mcc})
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

        # Signal grid
        vel_cls   = "flag" if velocity > 10 else "ok"
        score_cls = "flag" if score > 0.7 else ("warn" if score > 0.4 else "ok")
        sepa_cls  = "flag" if sepa_type == "SEPA_INSTANT" else "ok"
        cross_cls = "warn" if cross else "ok"

        st.markdown(
            f"""<div class="signal-grid">
              <div class="signal-card">
                <div class="signal-label">Upstream fraud score</div>
                <div class="signal-value {score_cls}">{score:.3f}</div>
              </div>
              <div class="signal-card">
                <div class="signal-label">SEPA type</div>
                <div class="signal-value {sepa_cls}">{sepa_type.replace('_',' ')}</div>
              </div>
              <div class="signal-card">
                <div class="signal-label">Velocity (30d)</div>
                <div class="signal-value {vel_cls}">{velocity} txn {'⚠ above threshold' if velocity > 10 else '✓ normal'}</div>
              </div>
              <div class="signal-card">
                <div class="signal-label">Geography</div>
                <div class="signal-value {cross_cls}">{'Cross-border ' if cross else 'Domestic '}{s_country} → {r_country}</div>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

        with st.expander("IBAN / BIC details"):
            st.markdown(
                f"""<div style='font-family:monospace;font-size:12px;line-height:2;color:#475569'>
                  <b>Sender IBAN:</b> &nbsp;&nbsp;{s_iban}<br>
                  <b>Receiver IBAN:</b> {r_iban}<br>
                  <b>Receiver BIC:</b> &nbsp;{r_bic}
                </div>""",
                unsafe_allow_html=True,
            )

    # ── Agent result ──────────────────────────────────────────────────────────
    if run_btn and selected_case:
        with st.spinner("Agent reasoning..."):
            result = run_case(selected_id)
        st.session_state["last_result"] = result
        st.session_state["last_case_id"] = selected_id

    if "last_result" in st.session_state and st.session_state.get("last_case_id") == selected_id:
        result = st.session_state["last_result"]
        rec    = result.get("recommendation", "escalate")
        conf   = result.get("confidence", "medium")
        flag   = result.get("secondary_flag", "ambiguous")
        trace  = result.get("reasoning_trace", "")
        esc    = result.get("escalation_required", False)
        esc_r  = result.get("escalation_reason", "")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("**Agent recommendation**")

        # Badges row
        st.markdown(
            f"{rec_badge(rec)} &nbsp; {conf_badge(conf)} &nbsp; {flag_badge(flag)}",
            unsafe_allow_html=True,
        )

        # Reasoning trace
        st.markdown(
            f"""<div class="reasoning-block">
              <div class="reasoning-label">Reasoning trace</div>
              <div class="reasoning-text">{trace}</div>
            </div>""",
            unsafe_allow_html=True,
        )

        # HITL / AML notices
        if flag == "aml_concern":
            st.markdown(
                """<div class="aml-notice">
                  <strong>⚖️ AML concern flagged.</strong> This case routes to
                  SAR filing (Verdachtsmeldung to the FIU) — not customer contact.
                  Confidentiality obligation applies: the customer must not be informed.
                </div>""",
                unsafe_allow_html=True,
            )

        if esc:
            st.markdown(
                f"""<div class="hitl-notice">
                  <strong>👤 Human review required.</strong> {esc_r}.
                  EU AI Act Article 14 — the investigator retains full authority.
                </div>""",
                unsafe_allow_html=True,
            )

        # Decision buttons
        st.markdown("<br>**Your decision**", unsafe_allow_html=True)
        d_col1, d_col2, d_col3 = st.columns(3)

        with d_col1:
            if st.button("✅ Agree with agent", use_container_width=True):
                log_investigator_decision(selected_id, f"agreed:{rec}")
                st.success(f"Logged: agreed with {rec.upper()}")

        with d_col2:
            if st.button("✏️ Override", use_container_width=True):
                st.session_state["show_override"] = True

        with d_col3:
            if st.button("🔺 Escalate to senior", use_container_width=True):
                log_investigator_decision(selected_id, "escalated:senior_review")
                st.warning("Logged: escalated to senior investigator")

        if st.session_state.get("show_override"):
            with st.form("override_form"):
                override_dec = st.selectbox(
                    "Your decision",
                    ["approve", "decline", "escalate"],
                )
                override_reason = st.text_area(
                    "Override reason (required for audit log)",
                    placeholder="Explain why you are overriding the agent recommendation...",
                    height=80,
                )
                submitted = st.form_submit_button("Submit override")
                if submitted:
                    if not override_reason.strip():
                        st.error("Override reason is required for BaFin audit compliance.")
                    else:
                        log_investigator_decision(
                            selected_id, f"override:{override_dec}", override_reason
                        )
                        st.session_state["show_override"] = False
                        st.success(f"Logged: override → {override_dec.upper()}")
                        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Case queue
# ══════════════════════════════════════════════════════════════════════════════
with tab_queue:
    st.markdown("#### Open case queue")
    st.markdown(
        "<div style='font-size:13px;color:#64748B;margin-bottom:20px'>"
        "All available cases. Click a case ID to open it in the review panel."
        "</div>",
        unsafe_allow_html=True,
    )

    cases = load_cases()
    for c in cases[:30]:
        amount  = c.get("amount_eur", 0)
        country = f"{c.get('sender_country','?')} → {c.get('receiver_country','?')}"
        sepa    = c.get("sepa_type", "—")
        score   = c.get("upstream_fraud_score", 0)
        aml     = "🟣 AML" if c.get("aml_signal") else ""
        inst    = "⚡ Instant" if sepa == "SEPA_INSTANT" else ""
        flags   = " &nbsp; ".join(filter(None, [aml, inst]))

        col_id, col_amt, col_geo, col_score, col_flags = st.columns([2, 1, 2, 1, 2])
        with col_id:
            st.markdown(
                f"<span style='font-family:monospace;font-size:12px;color:#0A1628'>{c['case_id']}</span>",
                unsafe_allow_html=True,
            )
        with col_amt:
            st.markdown(f"<span style='font-size:13px'>€{amount:,.0f}</span>", unsafe_allow_html=True)
        with col_geo:
            st.markdown(f"<span style='font-size:12px;color:#64748B'>{country}</span>", unsafe_allow_html=True)
        with col_score:
            color = "#DC2626" if score > 0.7 else ("#D97706" if score > 0.4 else "#059669")
            st.markdown(
                f"<span style='font-size:12px;color:{color};font-family:monospace'>{score:.2f}</span>",
                unsafe_allow_html=True,
            )
        with col_flags:
            st.markdown(f"<span style='font-size:12px'>{flags}</span>", unsafe_allow_html=True)

        st.markdown("<hr style='margin:6px 0'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Audit log
# ══════════════════════════════════════════════════════════════════════════════
with tab_audit:
    st.markdown("#### BaFin audit log")
    st.markdown(
        "<div style='font-size:13px;color:#64748B;margin-bottom:20px'>"
        "Append-only record of every agent recommendation and investigator decision. "
        "Satisfies MaRisk audit trail requirements."
        "</div>",
        unsafe_allow_html=True,
    )

    audit_rows = load_audit_log()

    if not audit_rows:
        st.info("No audit entries yet. Run the agent on a case to generate entries.")
    else:
        for row in audit_rows:
            logged_at, case_id, agent_rec, conf, esc_req, esc_reason, inv_dec, ov_reason = row

            inv_label = ""
            if inv_dec:
                inv_label = f"&nbsp;·&nbsp; <b>Investigator:</b> {inv_dec}"
                if ov_reason:
                    inv_label += f" — {ov_reason}"

            esc_label = ""
            if esc_req:
                esc_label = f"&nbsp;·&nbsp; ⚠ Escalated"

            st.markdown(
                f"""<div class="audit-row">
                  <span class="audit-case-id">{case_id}</span>
                  &nbsp;·&nbsp; {logged_at[:16]}
                  &nbsp;·&nbsp; Agent: <b>{(agent_rec or '—').upper()}</b> / {conf}
                  {esc_label}
                  {inv_label}
                </div>""",
                unsafe_allow_html=True,
            )

        if st.button("Export audit log (CSV)"):
            import csv, io
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow([
                "logged_at", "case_id", "agent_recommendation",
                "confidence", "escalation_required", "escalation_reason",
                "investigator_decision", "override_reason"
            ])
            writer.writerows(audit_rows)
            st.download_button(
                "⬇ Download CSV",
                output.getvalue(),
                file_name=f"fraud_triage_audit_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )