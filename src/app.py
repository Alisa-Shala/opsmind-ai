import os
import sqlite3
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from agent import OpsMindAgent
from batch_analyzer import OpsMindBatchAnalyzer
from genai_assistant import OpsMindGenAIAssistant
from genai_summary import generate_business_summary

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="OpsMind AI", page_icon="🤖", layout="wide")

st.markdown("""
<style>
/* Hide default Streamlit menu */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Sidebar nav styling */
.nav-item {
    padding: 10px 16px;
    border-radius: 10px;
    margin-bottom: 4px;
    cursor: pointer;
    font-size: 15px;
    color: #CBD5E1;
}
.nav-item:hover { background: #1E293B; color: #F8FAFC; }

/* KPI cards */
div[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #1E293B;
    padding: 18px;
    border-radius: 14px;
}
div[data-testid="stMetricLabel"] { color: #94A3B8; font-size: 13px; }
div[data-testid="stMetricValue"] { color: #F8FAFC; font-size: 26px; font-weight: 700; }

/* Section headers */
h2 { color: #F8FAFC !important; }
h3 { color: #E2E8F0 !important; }
</style>
""", unsafe_allow_html=True)

DATABASE_PATH = "database/opsmind.db"


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------
@st.cache_data
def load_data():
    conn = sqlite3.connect(DATABASE_PATH)
    df   = pd.read_sql_query("SELECT * FROM invoices", conn)
    conn.close()
    return df

@st.cache_resource
def load_agent():
    return OpsMindAgent()

@st.cache_resource
def load_assistant():
    return OpsMindGenAIAssistant()


# --------------------------------------------------
# RUNTIME API KEY
# --------------------------------------------------
# Load from Streamlit Secrets if available (for cloud deployment)
try:
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

if "runtime_api_key" in st.session_state:
    key = st.session_state["runtime_api_key"]
    if key.startswith("gsk_"):
        os.environ["GROQ_API_KEY"] = key
    else:
        os.environ["ANTHROPIC_API_KEY"] = key

def _apply_api_key(key: str):
    if key.startswith("gsk_"):
        os.environ["GROQ_API_KEY"] = key
    else:
        os.environ["ANTHROPIC_API_KEY"] = key
    st.session_state["runtime_api_key"] = key
    st.session_state["runtime_key_type"] = "groq" if key.startswith("gsk_") else "anthropic"
    st.cache_resource.clear()
    st.rerun()

try:
    df = load_data()
except Exception as e:
    st.error("Could not load database. Run: py src/load_to_database.py")
    st.stop()


# --------------------------------------------------
# UPLOAD DATASET
# --------------------------------------------------
def process_uploaded_dataset(raw_df, agent):
    df = raw_df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    defaults = {
        "invoice_id": [f"INV-{i+1:04d}" for i in range(len(df))],
        "vendor": "Unknown Vendor", "department": "Unknown",
        "invoice_type": "Standard", "payment_terms_days": 30,
        "has_purchase_order": 1, "duplicate_flag": 0,
        "vendor_delay_history": 0, "sla_breached": 0,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    if "amount" not in df.columns:
        st.error("CSV must have an 'amount' column.")
        st.stop()
    if "approval_days" not in df.columns:
        df["approval_days"] = 7
    df["amount"]        = pd.to_numeric(df["amount"],        errors="coerce").fillna(0)
    df["approval_days"] = pd.to_numeric(df["approval_days"], errors="coerce").fillna(7).astype(int)
    df["approval_delay_flag"] = (df["approval_days"] > 10).astype(int)
    for c in ["has_purchase_order","duplicate_flag","vendor_delay_history","sla_breached"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    risk_levels, fraud_scores, actions, reasons = [], [], [], []
    prog = st.progress(0)
    for i, (_, row) in enumerate(df.iterrows()):
        res = load_agent().analyze_invoice({
            "amount": row["amount"], "approval_days": row["approval_days"],
            "payment_terms_days": row["payment_terms_days"],
            "has_purchase_order": row["has_purchase_order"],
            "duplicate_flag": row["duplicate_flag"],
            "vendor_delay_history": row["vendor_delay_history"],
            "sla_breached": row["sla_breached"],
            "approval_delay_flag": row["approval_delay_flag"],
        })
        risk_levels.append(res["risk_level"])
        fraud_scores.append(res["fraud_score"])
        actions.append(res["recommended_action"])
        reasons.append(" | ".join(res["exception_types"]))
        prog.progress((i+1)/len(df))
    prog.empty()
    df["risk_level"]       = risk_levels
    df["fraud_score"]      = fraud_scores
    df["next_best_action"] = actions
    df["risk_reason"]      = reasons
    return df


agent     = load_agent()
assistant = load_assistant()
analyzer  = OpsMindBatchAnalyzer(agent)


# --------------------------------------------------
# SIDEBAR — NAVIGATION + CONFIG + FILTERS
# --------------------------------------------------
with st.sidebar:
    st.markdown("# 🤖 OpsMind AI")
    st.caption("Agentic AI for Invoice Operations")
    st.divider()

    page = st.radio("", [
        "📊  Overview",
        "🤖  AI Agent",
        "💬  GenAI Assistant",
        "📈  Deep Analytics",
        "📋  Data & Reports",
    ], label_visibility="collapsed")

    st.divider()
    st.markdown("**🔑 AI Configuration**")
    key_input = st.text_input("API Key", type="password",
                               placeholder="gsk_... or sk-ant-...",
                               help="Free key at console.groq.com")
    if key_input:
        if st.button("Apply", type="primary", use_container_width=True):
            _apply_api_key(key_input)

    if assistant.has_api_key():
        st.success("✅ GenAI active")
    else:
        st.info("ℹ️ No key — core features work")

    st.divider()
    st.markdown("**📂 Data Source**")
    data_source = st.radio("", ["Demo dataset", "Upload my CSV"],
                            label_visibility="collapsed")
    if data_source == "Upload my CSV":
        st.caption("Needs: `amount`. Optional: `vendor`, `department`, etc.")
        up_csv = st.file_uploader("Upload CSV", type=["csv"])
        if up_csv:
            if st.session_state.get("upload_name") != up_csv.name:
                with st.spinner("Processing..."):
                    raw = pd.read_csv(up_csv)
                    st.session_state["uploaded_df"]  = process_uploaded_dataset(raw, agent)
                    st.session_state["upload_name"]  = up_csv.name
                st.success(f"✅ {len(st.session_state['uploaded_df'])} invoices loaded!")
            if "uploaded_df" in st.session_state:
                df = st.session_state["uploaded_df"]

    st.divider()
    st.markdown("**⚙️ Filters**")
    dept_f   = st.multiselect("Department", sorted(df["department"].unique()),
                               default=sorted(df["department"].unique()))
    risk_f   = st.multiselect("Risk Level", ["Low","Medium","High"],
                               default=["Low","Medium","High"])
    sla_f    = st.selectbox("SLA Status", ["All","Breached","Not Breached"])
    a_min, a_max = st.slider("Amount ($)",
                              float(df["amount"].min()), float(df["amount"].max()),
                              (float(df["amount"].min()), float(df["amount"].max())))


# --------------------------------------------------
# FILTERED DATA
# --------------------------------------------------
filtered_df = df[
    df["department"].isin(dept_f) &
    df["risk_level"].isin(risk_f) &
    df["amount"].between(a_min, a_max)
].copy()
if sla_f == "Breached":     filtered_df = filtered_df[filtered_df["sla_breached"]==1]
elif sla_f == "Not Breached": filtered_df = filtered_df[filtered_df["sla_breached"]==0]

if filtered_df.empty:
    st.warning("No invoices match the selected filters.")
    st.stop()

total_invoices  = len(filtered_df)
total_amount    = filtered_df["amount"].sum()
high_risk_count = len(filtered_df[filtered_df["risk_level"]=="High"])
sla_breaches    = int(filtered_df["sla_breached"].sum())
missing_po      = len(filtered_df[filtered_df["has_purchase_order"]==0])
dup_count       = int(filtered_df["duplicate_flag"].sum())


# ==================================================
# PAGE: OVERVIEW
# ==================================================
if page == "📊  Overview":

    # ABOUT
    with st.expander("ℹ️ About OpsMind AI", expanded=False):
        st.markdown("""
**OpsMind AI** is an Agentic AI system built for Accounts Payable (AP) operations — the department responsible for reviewing and paying vendor invoices.

**The problem it solves:** Companies receive hundreds or thousands of invoices every month. Manually checking each one for fraud, duplicates, missing documents, or SLA violations is slow and error-prone.

**What OpsMind AI does automatically:**
- 🔍 **Detects risk** — classifies every invoice as Low, Medium, or High risk using Machine Learning
- 🚨 **Flags exceptions** — missing Purchase Orders, duplicate submissions, SLA breaches, vendor delay history
- 🤖 **AI Agent decisions** — recommends next action, assigns the right team, generates manager explanations
- ✉️ **Auto-generates emails** — drafts vendor and manager notifications based on detected issues
- 📊 **Monitors the portfolio** — tracks risk trends, vendor performance, and process health over time
- 💬 **Answers questions** — GenAI assistant responds to business questions about the invoice portfolio
        """)

    st.markdown("## 📊 Overview")
    st.caption("Executive view of the invoice portfolio — risk, compliance, and process health.")

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Invoices",    f"{total_invoices:,}")
    k2.metric("Total Value", f"${total_amount:,.0f}")
    k3.metric("High Risk",   f"{high_risk_count}")
    k4.metric("SLA Breach",  f"{sla_breaches}")
    k5.metric("Missing PO",  f"{missing_po}")
    k6.metric("Duplicates",  f"{dup_count}")

    # AP HEALTH SCORE
    st.divider()
    st.markdown("### 🏥 AP Process Health Score")
    st.caption("A single composite score (0–100) measuring the health of the AP process across four dimensions: risk, compliance, efficiency, and integrity.")

    risk_score       = max(0, 100 - (high_risk_count / total_invoices * 250))
    compliance_score = (1 - missing_po  / total_invoices) * 100
    efficiency_score = (1 - sla_breaches / total_invoices) * 100
    integrity_score  = (1 - dup_count   / total_invoices) * 100
    health_score     = int(
        risk_score * 0.35 +
        compliance_score * 0.25 +
        efficiency_score * 0.25 +
        integrity_score  * 0.15
    )
    health_color = "#22C55E" if health_score >= 75 else ("#F59E0B" if health_score >= 50 else "#EF4444")
    health_label = "Good" if health_score >= 75 else ("Needs Attention" if health_score >= 50 else "Critical")

    h1, h2, h3, h4, h5 = st.columns(5)
    h1.markdown(f"""
<div style='background:#111827;border:2px solid {health_color};border-radius:16px;padding:20px;text-align:center'>
<div style='font-size:13px;color:#94A3B8;margin-bottom:6px'>Overall Health</div>
<div style='font-size:48px;font-weight:800;color:{health_color}'>{health_score}</div>
<div style='font-size:14px;color:{health_color}'>{health_label}</div>
</div>""", unsafe_allow_html=True)
    h2.metric("Risk Score",        f"{int(risk_score)}/100",       help="Based on % high-risk invoices")
    h3.metric("Compliance Score",  f"{int(compliance_score)}/100", help="Based on PO coverage rate")
    h4.metric("Efficiency Score",  f"{int(efficiency_score)}/100", help="Based on SLA compliance rate")
    h5.metric("Integrity Score",   f"{int(integrity_score)}/100",  help="Based on duplicate invoice rate")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        rc = (filtered_df["risk_level"].value_counts()
              .reindex(["Low","Medium","High"], fill_value=0).reset_index())
        rc.columns = ["risk_level","count"]
        fig = px.bar(rc, x="risk_level", y="count", color="risk_level", text="count",
                     color_discrete_map={"Low":"#22C55E","Medium":"#F59E0B","High":"#EF4444"},
                     title="Risk Level Distribution")
        fig.update_layout(template="plotly_dark", showlegend=False, height=340,
                          margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        hd = (filtered_df.groupby(["department","risk_level"]).size().reset_index(name="count"))
        pv = (hd.pivot(index="department", columns="risk_level", values="count")
              .reindex(columns=["Low","Medium","High"]).fillna(0).astype(int))
        if not pv.empty:
            fig2 = px.imshow(pv, color_continuous_scale="RdYlGn_r",
                             title="Risk Heatmap: Department x Risk Level",
                             text_auto=True, aspect="auto")
            fig2.update_layout(template="plotly_dark", height=340,
                               margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        # Risk Trend Over Time
        if "submitted_date" in filtered_df.columns:
            trend = filtered_df.copy()
            trend["month"] = pd.to_datetime(trend["submitted_date"], errors="coerce").dt.to_period("M").astype(str)
            trend_g = (trend.groupby(["month","risk_level"]).size().reset_index(name="count"))
            fig3 = px.line(trend_g, x="month", y="count", color="risk_level",
                           color_discrete_map={"Low":"#22C55E","Medium":"#F59E0B","High":"#EF4444"},
                           markers=True, title="Invoice Risk Trend by Month")
            fig3.update_layout(template="plotly_dark", height=320,
                               margin=dict(l=10,r=10,t=40,b=40),
                               xaxis_tickangle=-30)
            st.plotly_chart(fig3, use_container_width=True)

    with c4:
        vr = (filtered_df[filtered_df["risk_level"]=="High"]
              .groupby("vendor").size().reset_index(name="count")
              .sort_values("count", ascending=False).head(7))
        if not vr.empty:
            fig4 = px.bar(vr, x="count", y="vendor", orientation="h",
                          color="count", color_continuous_scale="Reds",
                          text="count", title="Top High-Risk Vendors")
            fig4.update_layout(template="plotly_dark", height=320,
                               margin=dict(l=10,r=10,t=40,b=10),
                               yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig4, use_container_width=True)

    st.divider()
    st.markdown("### 🚨 Priority Invoice Queue")
    mq = len(filtered_df[filtered_df["next_best_action"]=="Send for manual review"]) if "next_best_action" in filtered_df.columns else 0
    eq = len(filtered_df[filtered_df["next_best_action"]=="Escalate to manager"])    if "next_best_action" in filtered_df.columns else 0
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Manual Review", mq)
    m2.metric("Escalations",   eq)
    m3.metric("Duplicates",    dup_count)
    m4.metric("Missing PO",    missing_po)

    pri = filtered_df[
        (filtered_df["risk_level"]=="High") | (filtered_df["sla_breached"]==1) |
        (filtered_df["duplicate_flag"]==1)  | (filtered_df["has_purchase_order"]==0)
    ]
    if not pri.empty:
        cols = [c for c in ["invoice_id","vendor","department","amount","risk_level",
                             "risk_reason","next_best_action"] if c in pri.columns]
        st.dataframe(pri[cols].head(10), use_container_width=True)
        st.download_button("⬇️ Download Priority Report",
                           data=pri.to_csv(index=False).encode(),
                           file_name="priority_invoices.csv", mime="text/csv")
    else:
        st.success("No priority invoices for the selected filters.")

    # PAYMENT DUE MONITOR
    if "due_date" in filtered_df.columns:
        st.divider()
        st.markdown("### 📅 Payment Due Monitor")
        st.caption("Invoices approaching their payment due date — sorted by urgency and risk level.")

        from datetime import date
        today = pd.Timestamp(date.today())
        due_df = filtered_df.copy()
        due_df["due_date"] = pd.to_datetime(due_df["due_date"], errors="coerce")
        due_df["days_until_due"] = (due_df["due_date"] - today).dt.days
        due_df = due_df[due_df["days_until_due"].between(0, 30)].copy()
        due_df = due_df.sort_values(["days_until_due","risk_level"],
                                     key=lambda x: x if x.name=="days_until_due"
                                     else x.map({"High":0,"Medium":1,"Low":2}))

        d1,d2,d3 = st.columns(3)
        d1.metric("Due in 7 days",  len(due_df[due_df["days_until_due"]<=7]))
        d2.metric("Due in 14 days", len(due_df[due_df["days_until_due"]<=14]))
        d3.metric("Due in 30 days", len(due_df))

        if due_df.empty:
            st.success("No invoices due in the next 30 days.")
        else:
            show_cols = [c for c in ["invoice_id","vendor","department","amount",
                                      "risk_level","days_until_due","next_best_action"]
                         if c in due_df.columns]
            st.dataframe(due_df[show_cols].head(10), use_container_width=True)
            st.download_button("⬇️ Download Due Monitor Report",
                               data=due_df.to_csv(index=False).encode(),
                               file_name="payment_due_monitor.csv", mime="text/csv")


# ==================================================
# PAGE: AI AGENT
# ==================================================
elif page == "🤖  AI Agent":
    agent_page = st.radio("", ["Single Invoice", "Batch Analysis"],
                           horizontal=True, label_visibility="collapsed")

    if agent_page == "Single Invoice":
        st.markdown("## 🤖 AI Agent Decision Center")
        st.caption("Enter invoice details — the agent predicts risk, calculates fraud score, detects exceptions, and recommends the next action.")

        c1, c2 = st.columns(2)
        with c1:
            amount        = st.number_input("Invoice Amount ($)", min_value=0.0, value=6000.0, step=500.0)
            approval_days = st.number_input("Approval Days", min_value=1, value=10)
            pay_terms     = st.selectbox("Payment Terms (days)", [15,30,45,60])
        with c2:
            has_po       = st.selectbox("Has Purchase Order?", ["Yes","No"])
            dup_flag     = st.selectbox("Duplicate Invoice?",  ["No","Yes"])
            vend_delay   = st.selectbox("Vendor Delay History?", ["No","Yes"])
            sla_inp      = st.selectbox("SLA Breached?", ["No","Yes"])

        use_agentic = st.toggle("🔗 Agentic Deep Analysis (AI reasons step-by-step with tools)",
                                 value=False)

        if st.button("▶  Run Analysis", type="primary", use_container_width=True):
            inv = {
                "amount": amount, "approval_days": approval_days,
                "payment_terms_days": pay_terms,
                "has_purchase_order":   1 if has_po=="Yes" else 0,
                "duplicate_flag":       1 if dup_flag=="Yes" else 0,
                "vendor_delay_history": 1 if vend_delay=="Yes" else 0,
                "sla_breached":         1 if sla_inp=="Yes" else 0,
                "approval_delay_flag":  1 if approval_days>10 else 0,
            }
            with st.spinner("Analyzing invoice..."):
                res = (agent.analyze_invoice_agentic(inv)
                       if use_agentic and agent.anthropic_client
                       else agent.analyze_invoice(inv))

            # Approval Time Prediction
            predicted_days = None
            try:
                with open("models/approval_predictor.pkl","rb") as f:
                    apr_model = pickle.load(f)
                with open("models/approval_predictor_features.pkl","rb") as f:
                    apr_feats = pickle.load(f)
                apr_input = pd.DataFrame([{
                    "amount":               inv["amount"],
                    "payment_terms_days":   inv["payment_terms_days"],
                    "has_purchase_order":   inv["has_purchase_order"],
                    "duplicate_flag":       inv["duplicate_flag"],
                    "vendor_delay_history": inv["vendor_delay_history"],
                    "sla_breached":         inv["sla_breached"]
                }])
                predicted_days = int(round(apr_model.predict(apr_input)[0]))
            except Exception:
                pass

            r1,r2,r3,r4,r5 = st.columns(5)
            r1.metric("Risk Level",      res["risk_level"])
            r2.metric("Fraud Score",     f"{res['fraud_score']}/100")
            r3.metric("Priority",        res["action_priority"])
            r4.metric("Team",            res["assigned_team"].split(" ")[0])
            if predicted_days is not None:
                r5.metric("Predicted Approval", f"{predicted_days} days",
                          help="Estimated days to get this invoice approved (Random Forest Regressor)")

            wf = res["workflow_status"]
            if wf in ["Payment Blocked","Escalation Required","On Hold - Duplicate Review"]:
                st.error(f"⛔ {wf}")
            elif wf == "Ready for Approval":
                st.success(f"✅ {wf}")
            else:
                st.warning(f"⚠️ {wf}")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Detected Exceptions**")
                for exc in res["exception_types"]:
                    (st.success if exc=="No major exception detected" else st.warning)(exc)
            with col_b:
                st.markdown("**Three-Way Match**")
                (st.success if res["three_way_match_status"]=="Passed" else st.warning)(res["three_way_match_status"])
                st.markdown("**Recommended Action**")
                st.info(res["recommended_action"])

            st.markdown("**Manager Explanation**")
            st.info(res["genai_summary"])

            report = (
                f"OpsMind AI — Agent Decision Report\n\n"
                f"Risk: {res['risk_level']} | Fraud Score: {res['fraud_score']}/100 | Priority: {res['action_priority']}\n"
                f"Workflow: {res['workflow_status']} | Team: {res['assigned_team']}\n"
                f"3-Way Match: {res['three_way_match_status']}\n"
                f"Exceptions: {', '.join(res['exception_types'])}\n\n"
                f"Recommended Action:\n{res['recommended_action']}\n\n"
                f"Manager Explanation:\n{res['genai_summary']}"
            )
            st.download_button("⬇️ Download Report", data=report,
                               file_name="agent_report.txt", mime="text/plain")

            # AUTO EMAIL GENERATOR
            st.divider()
            st.markdown("### ✉️ Auto Email Generator")
            st.caption("OpsMind AI detects who needs to be notified based on the exceptions and generates all relevant emails automatically.")

            # Determine which emails to generate based on exceptions
            exceptions = res["exception_types"]
            emails_needed = []
            if "Missing purchase order" in exceptions:
                emails_needed.append(("📧 Vendor — Missing Purchase Order", "vendor_po"))
            if "Potential duplicate invoice" in exceptions:
                emails_needed.append(("📧 Vendor — Duplicate Investigation", "vendor_dup"))
            if res["risk_level"] == "High" or res["fraud_score"] >= 60:
                emails_needed.append(("📧 Manager — Escalation Alert", "manager_esc"))
            if "SLA breached" in exceptions or "Approval delay" in exceptions:
                emails_needed.append(("📧 AP Team — SLA Breach", "ap_sla"))
            if not emails_needed:
                emails_needed.append(("📧 AP Team — Invoice Approved", "ap_approved"))

            def _get_template(etype):
                exc_text = ", ".join(exceptions)
                if etype == "vendor_po":
                    return (
                        f"Subject: Action Required — Missing Purchase Order for Invoice [Invoice Ref]\n\n"
                        f"Dear [Vendor Name],\n\n"
                        f"We are writing regarding invoice [Invoice Ref] for ${inv['amount']:,.2f}.\n\n"
                        f"Our Accounts Payable system has flagged this invoice because no linked Purchase Order (PO) was found. "
                        f"As per our procurement policy, all invoices must reference a valid PO before payment can be processed.\n\n"
                        f"Please provide the correct PO reference at your earliest convenience. "
                        f"Payment will be on hold until this is resolved.\n\n"
                        f"Best regards,\nAccounts Payable Team"
                    )
                if etype == "vendor_dup":
                    return (
                        f"Subject: Invoice On Hold — Possible Duplicate Detected\n\n"
                        f"Dear [Vendor Name],\n\n"
                        f"Invoice [Invoice Ref] for ${inv['amount']:,.2f} has been placed on hold "
                        f"as our system has identified it as a potential duplicate submission.\n\n"
                        f"Please confirm whether this invoice has been previously submitted or paid. "
                        f"If this is a new invoice, kindly provide supporting documentation.\n\n"
                        f"Best regards,\nAccounts Payable Team"
                    )
                if etype == "manager_esc":
                    return (
                        f"Subject: ESCALATION — High Risk Invoice Requires Immediate Review\n\n"
                        f"Dear [Manager Name],\n\n"
                        f"OpsMind AI has flagged an invoice requiring your immediate attention.\n\n"
                        f"Risk Level: {res['risk_level']} | Fraud Score: {res['fraud_score']}/100\n"
                        f"Amount: ${inv['amount']:,.2f}\n"
                        f"Detected Issues: {exc_text}\n"
                        f"Recommended Action: {res['recommended_action']}\n\n"
                        f"Please review and approve or block the payment in the OpsMind dashboard.\n\n"
                        f"Best regards,\nOpsMind AI — Risk Monitoring"
                    )
                if etype == "ap_sla":
                    return (
                        f"Subject: SLA Breach Alert — Invoice Approval Overdue\n\n"
                        f"Dear AP Team,\n\n"
                        f"Invoice [Invoice Ref] for ${inv['amount']:,.2f} has exceeded the SLA threshold "
                        f"({inv['approval_days']} days in approval).\n\n"
                        f"Please prioritise this case to avoid further delays and contractual penalties.\n\n"
                        f"Best regards,\nOpsMind AI — AP Operations Monitor"
                    )
                return (
                    f"Subject: Invoice Cleared — Approved for Payment\n\n"
                    f"Dear AP Team,\n\n"
                    f"Invoice [Invoice Ref] for ${inv['amount']:,.2f} has passed all risk checks "
                    f"with no major exceptions detected. It may proceed through the standard payment workflow.\n\n"
                    f"Best regards,\nOpsMind AI"
                )

            st.info(f"**{len(emails_needed)} email(s) auto-detected** based on the invoice exceptions above.")

            all_drafts = []
            for label, etype in emails_needed:
                with st.expander(label, expanded=True):
                    template = _get_template(etype)

                    if assistant.has_api_key():
                        with st.spinner("Drafting..."):
                            try:
                                prompt = (
                                    f"You are an AP Operations specialist. Improve this email draft — "
                                    f"keep it professional, concise, and actionable. "
                                    f"Invoice amount: ${inv['amount']:,.2f} | Risk: {res['risk_level']} | "
                                    f"Fraud score: {res['fraud_score']}/100 | "
                                    f"Exceptions: {', '.join(exceptions)}\n\n"
                                    f"Draft to improve:\n{template}\n\n"
                                    f"Return only the improved email (Subject + Body). "
                                    f"Use placeholders [Vendor Name], [Invoice Ref] where needed."
                                )
                                if assistant.anthropic_client:
                                    msg = assistant.anthropic_client.messages.create(
                                        model="claude-haiku-4-5-20251001", max_tokens=350,
                                        messages=[{"role":"user","content":prompt}])
                                    draft = msg.content[0].text
                                else:
                                    from groq import Groq
                                    g = Groq(api_key=assistant.groq_api_key)
                                    c = g.chat.completions.create(
                                        model="llama-3.3-70b-versatile",
                                        messages=[{"role":"user","content":prompt}],
                                        max_tokens=350)
                                    draft = c.choices[0].message.content
                            except Exception:
                                draft = template
                    else:
                        draft = template

                    st.text_area("", value=draft, height=220, key=f"email_{etype}")
                    all_drafts.append(f"=== {label} ===\n\n{draft}")

            if all_drafts:
                st.download_button("⬇️ Download All Email Drafts",
                                   data="\n\n\n".join(all_drafts),
                                   file_name="email_drafts.txt",
                                   mime="text/plain")

    else:
        st.markdown("## 📦 Batch Invoice Analyzer")
        st.caption("Analyze an entire invoice portfolio at once — ranked by fraud score.")

        use_cur = st.checkbox("Use current filtered dataset", value=True)
        up_batch = None
        if not use_cur:
            up_batch = st.file_uploader("Upload CSV", type=["csv"])

        if st.button("▶  Run Batch Analysis", type="primary", use_container_width=True):
            batch = filtered_df.copy() if use_cur else (pd.read_csv(up_batch) if up_batch else None)
            if batch is None:
                st.warning("Please upload a CSV or use current dataset.")
            else:
                prog = st.progress(0)
                txt  = st.empty()
                def _cb(i,t): prog.progress(i/t); txt.text(f"Analyzing {i}/{t}...")
                with st.spinner("Running batch analysis..."):
                    results = analyzer.analyze_batch(batch, _cb)
                    summary = analyzer.get_batch_summary(results)
                prog.empty(); txt.empty()

                b1,b2,b3,b4 = st.columns(4)
                b1.metric("Analyzed",    summary["total_analyzed"])
                b2.metric("High Risk",   summary["high_risk_count"])
                b3.metric("Critical",    summary["critical_count"])
                b4.metric("Avg Fraud",   f"{summary['avg_fraud_score']}/100")
                st.caption(f"Top risk vendor: **{summary['top_risk_vendor']}** | Top risk dept: **{summary['top_risk_dept']}** | {summary['pct_high_risk']}% high risk")

                show = results[["invoice_id","vendor","department","amount",
                                 "risk_level","fraud_score","action_priority","exceptions"]].head(25).copy()
                show["amount"] = show["amount"].apply(lambda x: f"${x:,.0f}")
                st.dataframe(show, use_container_width=True)
                st.download_button("⬇️ Download Full Report",
                                   data=results.to_csv(index=False).encode(),
                                   file_name="batch_report.csv", mime="text/csv")


# ==================================================
# PAGE: GENAI ASSISTANT
# ==================================================
elif page == "💬  GenAI Assistant":
    asst_sub = st.radio("", ["💬 Chat Assistant", "🔍 Natural Language SQL"],
                         horizontal=True, label_visibility="collapsed")

    hr = filtered_df[filtered_df["risk_level"]=="High"]
    top_dept   = hr["department"].value_counts().idxmax() if not hr.empty else "N/A"
    top_vendor = hr["vendor"].value_counts().idxmax()     if not hr.empty else "N/A"
    mq = len(filtered_df[filtered_df["next_best_action"]=="Send for manual review"]) if "next_best_action" in filtered_df.columns else 0
    eq = len(filtered_df[filtered_df["next_best_action"]=="Escalate to manager"])    if "next_best_action" in filtered_df.columns else 0
    ctx = (
        f"Total invoices: {total_invoices}\nTotal amount: ${total_amount:,.2f}\n"
        f"High-risk invoices: {high_risk_count}\nSLA breaches: {sla_breaches}\n"
        f"Missing purchase order cases: {missing_po}\nDuplicate invoice flags: {dup_count}\n"
        f"Manual review queue: {mq}\nEscalations: {eq}\n"
        f"Top high-risk department: {top_dept}\nTop high-risk vendor: {top_vendor}"
    )

    if asst_sub == "💬 Chat Assistant":
        st.markdown("## 💬 GenAI Assistant")
        st.caption("Ask any business question about the invoice portfolio. Works with or without an API key.")

        user_q = st.text_area(
            "Ask a question about the invoice portfolio",
            placeholder="e.g. What should the AP manager focus on? / How many high-risk invoices are there? / Summarize the dashboard...",
            height=100
        )

        st.caption("Quick examples:")
        ex_cols = st.columns(4)
        examples = [
            "Summarize the dashboard",
            "What should the manager focus on?",
            "Which vendor is most risky?",
            "Recommend an action plan"
        ]
        for i, ex in enumerate(examples):
            if ex_cols[i].button(ex, key=f"ex_{i}", use_container_width=True):
                user_q = ex

        if user_q and user_q.strip():
            corrected, fixes = assistant.correct_query(user_q)
            if fixes:
                fix_str = "  |  ".join(f"~~{o}~~ → **{c}**" for o,c in fixes.items())
                st.caption(f"🔤 Possible typos: {fix_str}")
                if st.checkbox(f'Use corrected: *"{corrected}"*', value=True):
                    user_q = corrected

        if st.button("▶  Ask Assistant", type="primary", use_container_width=True):
            if not user_q or not user_q.strip():
                st.warning("Please write a question first.")
            else:
                with st.spinner("Generating response..."):
                    response = assistant.generate_response(user_q.strip(), ctx)
                st.markdown("### Response")
                st.info(response)
                if not assistant.has_api_key():
                    st.caption("ℹ️ Response from dashboard data. Add API key for richer AI responses.")
                st.download_button("⬇️ Download Response",
                                   data=f"Q: {user_q}\n\nA: {response}",
                                   file_name="assistant_response.txt", mime="text/plain")

    else:
        st.markdown("## 🔍 Natural Language SQL")
        st.caption("Ask a question in plain English — OpsMind converts it to SQL and runs it on the invoice database.")

        _schema = ("invoices (invoice_id, vendor, department, invoice_type, amount, "
                   "approval_days, payment_terms_days, has_purchase_order, duplicate_flag, "
                   "vendor_delay_history, sla_breached, approval_delay_flag, risk_level, "
                   "risk_reason, next_best_action)")

        nl_opts = [
            "Show all high risk invoices from Finance department",
            "Which vendors have SLA breaches?",
            "Total invoice amount per department",
            "Invoices with missing purchase order above $10000",
            "Top 5 vendors by average invoice amount",
            "Custom question"
        ]
        nl_sel = st.selectbox("Choose an example or write your own", nl_opts)
        nl_cus = ""
        if nl_sel == "Custom question":
            nl_cus = st.text_input("Your question", placeholder="e.g. How many duplicate invoices per vendor?")
        nl_q = nl_cus.strip() if nl_cus.strip() else nl_sel

        if st.button("▶  Run Query", type="primary", use_container_width=True):
            if not assistant.has_api_key():
                st.warning("Natural Language SQL needs an API key. Get one free at console.groq.com")
            else:
                with st.spinner("Converting to SQL and running..."):
                    try:
                        pr = (
                            f"Convert to a valid SQLite SELECT query.\n"
                            f"Table: {_schema}\n"
                            f"Question: {nl_q}\n\n"
                            f"Important rules:\n"
                            f"- Return ONLY the raw SQL query, no explanation, no markdown\n"
                            f"- SELECT statements only, LIMIT 50\n"
                            f"- Boolean columns (sla_breached, duplicate_flag, has_purchase_order, "
                            f"vendor_delay_history, approval_delay_flag) are stored as integers: 1=True, 0=False\n"
                            f"- Example: WHERE sla_breached = 1  (NOT 'True' or TRUE)\n\n"
                            f"SQL:"
                        )
                        anthropic_key = os.environ.get("ANTHROPIC_API_KEY","")
                        groq_key      = os.environ.get("GROQ_API_KEY","")

                        if anthropic_key and anthropic_key not in ("vendose_api_key_ketu",""):
                            import anthropic as _ant
                            _c = _ant.Anthropic(api_key=anthropic_key)
                            msg = _c.messages.create(
                                model="claude-haiku-4-5-20251001", max_tokens=200,
                                messages=[{"role":"user","content":pr}])
                            sql = msg.content[0].text.strip()
                        elif groq_key and groq_key not in ("vendose_api_key_ketu","vendos_ketu_kodin_nga_groq",""):
                            from groq import Groq
                            g = Groq(api_key=groq_key)
                            c = g.chat.completions.create(model="llama-3.3-70b-versatile",
                                    messages=[{"role":"user","content":pr}], max_tokens=200)
                            sql = c.choices[0].message.content.strip()
                        else:
                            st.warning("API key not found. Enter it in the sidebar and click Apply.")
                            st.stop()

                        # Clean up SQL — remove markdown, extra text
                        import re as _re
                        sql = _re.sub(r"```sql|```", "", sql).strip()
                        # Extract only the SELECT statement if there's extra text
                        match = _re.search(r"(SELECT\b.*)", sql, _re.IGNORECASE | _re.DOTALL)
                        if match:
                            sql = match.group(1).strip()
                        # Remove anything after semicolon on last line
                        sql = sql.split(";")[0].strip()

                        if not sql.upper().startswith("SELECT"):
                            st.error("Only SELECT queries allowed.")
                        else:
                            conn = sqlite3.connect(DATABASE_PATH)
                            qr   = pd.read_sql_query(sql, conn)
                            conn.close()
                            st.success(f"Query returned {len(qr)} rows.")
                            with st.expander("Generated SQL", expanded=True):
                                st.code(sql, language="sql")
                            st.dataframe(qr, use_container_width=True)
                            st.download_button("⬇️ Download Results",
                                               data=qr.to_csv(index=False).encode(),
                                               file_name="query_results.csv", mime="text/csv")
                    except Exception as ex:
                        st.error(f"Query failed: {ex}")


# ==================================================
# PAGE: DEEP ANALYTICS
# ==================================================
elif page == "📈  Deep Analytics":
    analytics_sub = st.radio("", ["🏢 Vendor Scorecard", "🧩 Segmentation"],
                              horizontal=True, label_visibility="collapsed")

    if analytics_sub == "🏢 Vendor Scorecard":
        st.markdown("## 🏢 Vendor Risk Scorecard")
        st.caption("Vendor-level risk profiling: high-risk rate, SLA compliance, duplicate rate.")
        sc = (filtered_df.groupby("vendor").agg(
            total    =("invoice_id","count"),
            amount   =("amount","sum"),
            high_r   =("risk_level", lambda x:(x=="High").sum()),
            sla_b    =("sla_breached","sum"),
            dup      =("duplicate_flag","sum"),
            miss_po  =("has_purchase_order", lambda x:(x==0).sum()),
            avg_appr =("approval_days","mean")
        ).reset_index())
        sc["high_%"]  = (sc["high_r"] / sc["total"] * 100).round(1)
        sc["sla_%"]   = (sc["sla_b"]  / sc["total"] * 100).round(1)
        sc["dup_%"]   = (sc["dup"]    / sc["total"] * 100).round(1)
        sc["avg_appr"]= sc["avg_appr"].round(1)
        sc = sc.sort_values("high_%", ascending=False)

        v1,v2 = st.columns(2)
        with v1:
            fig = px.bar(sc.head(10), x="vendor", y="high_%", color="high_%",
                         text="high_%", color_continuous_scale="Reds",
                         title="Top 10 Vendors — High-Risk Rate (%)")
            fig.update_layout(template="plotly_dark", height=370,
                              margin=dict(l=10,r=10,t=40,b=80))
            fig.update_traces(texttemplate="%{text}%")
            st.plotly_chart(fig, use_container_width=True)
        with v2:
            fig2 = px.scatter(sc, x="high_%", y="sla_%", size="total",
                              color="high_%", color_continuous_scale="Reds",
                              hover_name="vendor", text="vendor",
                              title="Risk vs SLA Breach Rate (bubble = invoice volume)")
            fig2.update_layout(template="plotly_dark", height=370,
                               margin=dict(l=10,r=10,t=40,b=10))
            fig2.update_traces(textposition="top center", textfont_size=9)
            st.plotly_chart(fig2, use_container_width=True)

        disp = sc[["vendor","total","high_%","sla_%","dup_%","miss_po","avg_appr"]].copy()
        disp.columns = ["Vendor","Invoices","High Risk %","SLA Breach %","Duplicate %","Missing PO","Avg Approval Days"]
        st.dataframe(disp, use_container_width=True)
        st.download_button("⬇️ Download Vendor Scorecard",
                           data=sc.to_csv(index=False).encode(),
                           file_name="vendor_scorecard.csv", mime="text/csv")

    else:
        st.markdown("## 🧩 Invoice Segmentation")
        st.caption("KMeans clustering groups invoices into behavioural segments — unsupervised ML, no risk label used.")
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA

            feats = [c for c in ["amount","approval_days","has_purchase_order",
                                  "duplicate_flag","vendor_delay_history","sla_breached"]
                     if c in filtered_df.columns]
            X  = filtered_df[feats].fillna(0)
            Xs = StandardScaler().fit_transform(X)
            k  = st.slider("Number of segments", 2, 5, 4)
            lb = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Xs)
            pc = PCA(n_components=2, random_state=42).fit_transform(Xs)

            cdf = filtered_df.copy()
            cdf["seg"] = lb.astype(str)
            cdf["px"],cdf["py"] = pc[:,0], pc[:,1]

            prof = cdf.groupby("seg").agg(
                count=("seg","count"),
                avg_amt=("amount","mean"),
                avg_appr=("approval_days","mean"),
                hr_pct=("risk_level", lambda x:round((x=="High").mean()*100,1)),
                sla_pct=("sla_breached", lambda x:round(x.mean()*100,1))
            ).reset_index()

            def _name(r):
                if r.hr_pct>=50: return f"Segment {r.seg} — Critical Risk"
                if r.sla_pct>=40: return f"Segment {r.seg} — SLA Watch"
                if r.avg_amt > filtered_df["amount"].quantile(0.75): return f"Segment {r.seg} — High Value"
                return f"Segment {r.seg} — Standard Flow"

            prof["label"] = prof.apply(_name, axis=1)
            cdf["label"]  = cdf["seg"].map(dict(zip(prof["seg"],prof["label"])))

            s1,s2 = st.columns(2)
            with s1:
                fig = px.scatter(cdf, x="px", y="py", color="label",
                                 hover_data=["vendor","amount","risk_level"] if "vendor" in cdf.columns else None,
                                 title="Invoice Segments (PCA projection)", opacity=0.75)
                fig.update_layout(template="plotly_dark", height=400,
                                  margin=dict(l=10,r=10,t=40,b=10),
                                  legend=dict(orientation="h",yanchor="bottom",y=-0.4))
                fig.update_traces(marker=dict(size=6))
                st.plotly_chart(fig, use_container_width=True)
            with s2:
                fig2 = px.bar(prof, x="label", y="count", color="hr_pct",
                              text="count", color_continuous_scale="RdYlGn_r",
                              title="Segment Size & High-Risk Rate (%)")
                fig2.update_layout(template="plotly_dark", height=400,
                                   margin=dict(l=10,r=10,t=40,b=80),
                                   xaxis_tickangle=-15)
                st.plotly_chart(fig2, use_container_width=True)

            dp = prof[["label","count","avg_amt","avg_appr","hr_pct","sla_pct"]].copy()
            dp.columns = ["Segment","Invoices","Avg Amount","Avg Approval Days","High Risk %","SLA %"]
            dp["Avg Amount"] = dp["Avg Amount"].apply(lambda x:f"${x:,.0f}")
            st.dataframe(dp, use_container_width=True)
        except Exception as e:
            st.info(f"Segmentation unavailable: {e}")


# ==================================================
# PAGE: DATA & REPORTS
# ==================================================
elif page == "📋  Data & Reports":
    rep_sub = st.radio("", ["🤖 Explainable AI", "📄 Export"],
                        horizontal=True, label_visibility="collapsed")

    if rep_sub == "🤖 Explainable AI":
        st.markdown("## 🤖 Explainable AI — Feature Importance")
        st.caption("Which invoice signals drive the ML risk classification model the most.")
        try:
            with open("models/risk_model.pkl","rb") as f: mdl = pickle.load(f)
            fnames_raw   = ["amount","approval_days","payment_terms_days","has_purchase_order",
                            "duplicate_flag","vendor_delay_history","sla_breached","approval_delay_flag"]
            fnames_clean = ["Invoice Amount","Approval Days","Payment Terms","Has Purchase Order",
                            "Duplicate Flag","Vendor Delay History","SLA Breached","Approval Delay"]
            idf = pd.DataFrame({"feature":fnames_clean,"importance":mdl.feature_importances_})
            idf = idf.sort_values("importance",ascending=False)
            fig = px.bar(idf, x="importance", y="feature", orientation="h",
                         color="importance", color_continuous_scale="Blues",
                         title="Feature Importance — What drives invoice risk?", text="importance")
            fig.update_traces(texttemplate="%{text:.3f}")
            fig.update_layout(template="plotly_dark", height=420,
                              margin=dict(l=10,r=10,t=40,b=10),
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

            top = idf.iloc[0]["feature"]
            second = idf.iloc[1]["feature"]
            st.info(
                f"**Key insight:** The most influential factor in predicting invoice risk is "
                f"**{top}** ({idf.iloc[0]['importance']:.1%}), followed by **{second}** "
                f"({idf.iloc[1]['importance']:.1%}). "
                f"This means the ML model pays most attention to vendor behaviour and approval timing "
                f"when classifying invoice risk."
            )
        except Exception:
            st.info("Model not found. Run: py src/train_models.py")

    else:
        st.markdown("## 📄 Reports & Export")
        if st.button("▶  Generate AI Executive Summary", type="primary"):
            with st.spinner("Generating..."):
                s = generate_business_summary(filtered_df)
            st.markdown(s)
            st.download_button("⬇️ Download Summary", data=s,
                               file_name="executive_summary.txt", mime="text/plain")
        st.divider()
        st.markdown("### Full Invoice Dataset")
        vcols = [c for c in ["invoice_id","vendor","department","invoice_type","amount",
                              "approval_days","risk_level","sla_breached","next_best_action"]
                 if c in filtered_df.columns]
        st.dataframe(filtered_df[vcols].head(50), use_container_width=True)
        st.download_button("⬇️ Download Dataset (CSV)",
                           data=filtered_df.to_csv(index=False).encode(),
                           file_name="invoice_data.csv", mime="text/csv")
