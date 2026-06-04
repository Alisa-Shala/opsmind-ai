# OpsMind AI — Technical Report

**Project:** Agentic AI System for Invoice Risk Management
**Domain:** Accounts Payable Operations
**Stack:** Python, Scikit-learn, Streamlit, SQLite, Groq API, RAG

---

## 1. Problem Statement

Companies like Genpact process thousands of vendor invoices every month on behalf of enterprise clients. The manual review process is:

- **Slow** — AP analysts spend hours reviewing each invoice
- **Error-prone** — duplicate payments, missing documents, and SLA breaches go undetected
- **Expensive** — manual labor costs increase with invoice volume
- **Reactive** — problems are discovered after payment, not before

**OpsMind AI solves this by automating invoice risk detection, exception handling, and manager communication — reducing manual effort and preventing financial losses.**

---

## 2. System Architecture

```
Raw Invoice Data
      ↓
Data Pipeline (generate → clean → load to SQLite)
      ↓
Machine Learning Models (Classifier + Regressor)
      ↓
AI Agent (fraud scoring + exception detection + tool use)
      ↓
GenAI Layer (RAG + Groq API + NL2SQL)
      ↓
Streamlit Dashboard (deployed on Streamlit Cloud)
```

---

## 3. Data Pipeline

### 3.1 Dataset Generation
- **File:** `src/generate_dataset.py`
- **Output:** 1,000 synthetic invoices with realistic distributions
- **Columns:** invoice_id, vendor, department, amount, approval_days, has_purchase_order, duplicate_flag, vendor_delay_history, sla_breached, risk_level
- **Risk labeling:** Business rules assign Low/Medium/High based on duplicate flags, missing PO, vendor history, and SLA compliance

### 3.2 Data Cleaning
- **File:** `src/data_cleaning.py`
- **Operations:** Convert dates to datetime, encode booleans as integers (0/1), add derived features (approval_delay_flag, days_until_due, amount_category)
- **Output:** `data/processed/clean_invoices.csv`

### 3.3 Database Loading
- **File:** `src/load_to_database.py`
- **Database:** SQLite (`database/opsmind.db`)
- **Purpose:** Enables SQL queries, NL2SQL execution, and dashboard data retrieval

---

## 4. Machine Learning Models

### 4.1 Risk Classifier (Supervised ML)
- **Algorithm:** Random Forest Classifier
- **Input features:** amount, approval_days, payment_terms_days, has_purchase_order, duplicate_flag, vendor_delay_history, sla_breached, approval_delay_flag
- **Target:** risk_level (Low / Medium / High)
- **Train/Test split:** 80% / 20%
- **Accuracy:** 98.5%
- **Saved as:** `models/risk_model.pkl`

**Classification Report:**

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| High | 1.00 | 0.86 | 0.93 |
| Low | 1.00 | 1.00 | 1.00 |
| Medium | 0.95 | 1.00 | 0.97 |

### 4.2 Approval Time Predictor (Supervised ML — Regression)
- **Algorithm:** Random Forest Regressor
- **Input features:** amount, payment_terms_days, has_purchase_order, duplicate_flag, vendor_delay_history, sla_breached
- **Target:** approval_days (number of days)
- **MAE:** 2.46 days (mean absolute error)
- **Saved as:** `models/approval_predictor.pkl`

### 4.3 Invoice Segmentation (Unsupervised ML)
- **Algorithm:** KMeans Clustering + PCA
- **Input features:** amount, approval_days, has_purchase_order, duplicate_flag, vendor_delay_history, sla_breached
- **Number of clusters:** 2-5 (user-configurable)
- **Visualization:** PCA reduces 6 dimensions to 2D scatter plot
- **Output segments:** Standard Flow, SLA Watch, Critical Risk, High Value

---

## 5. AI Agent

**File:** `src/agent.py` — Class: `OpsMindAgent`

The AI Agent processes each invoice through a multi-step decision pipeline:

1. **Risk Prediction** — Random Forest Classifier predicts Low/Medium/High
2. **Fraud Score Calculation** — weighted scoring system (0-100) based on risk signals
3. **Exception Detection** — identifies: duplicate invoice, missing PO, SLA breach, approval delay, vendor delay history, high amount
4. **Three-Way Matching** — validates Purchase Order, Goods Receipt, and Invoice alignment
5. **Workflow Routing** — assigns invoice to the correct team (AP Analyst, Procurement, Finance Controller, Risk/Compliance)
6. **Action Priority** — sets Critical / High / Medium / Low priority
7. **Manager Explanation** — Groq API generates a professional explanation
8. **Auto Email Generation** — drafts vendor and manager emails based on detected exceptions

### 5.1 Agentic Tool Use
When Agentic Deep Analysis is enabled, Claude API autonomously calls tools in sequence:
- `detect_exceptions(invoice_data)`
- `calculate_fraud_score(invoice_data)`
- `run_three_way_match(invoice_data)`
- `get_ap_policy(topic)`

This implements a genuine agentic loop — the model decides which tools to call and in what order.

---

## 6. GenAI Layer

### 6.1 RAG — Retrieval Augmented Generation
- **File:** `src/rag_retriever.py`
- **Method:** TF-IDF vectorization + cosine similarity
- **Knowledge base:** 6 AP policy documents (duplicate invoice policy, missing PO policy, SLA policy, three-way matching policy, risk scoring policy, general AP policy)
- **Process:** Query → TF-IDF retrieval → top 3 chunks → sent as context to LLM

### 6.2 GenAI Assistant
- **File:** `src/genai_assistant.py`
- **Primary:** Groq API — Llama 3.3 70B
- **Secondary:** Anthropic API — Claude Haiku
- **Fallback:** Smart rule-based responses using dashboard data (works without API key)
- **Safety:** Keyword-based guardrails block harmful or out-of-scope requests
- **Autocorrect:** pyspellchecker detects and suggests corrections for typos

### 6.3 Natural Language SQL
- User types a plain English question
- Groq API generates a SQLite SELECT query
- Safety check blocks non-SELECT statements
- Query executes on `database/opsmind.db`
- Results displayed as interactive table

---

## 7. Dashboard

**File:** `src/app.py` — Built with Streamlit

### Pages:
1. **Overview** — KPI cards, AP Health Score, Risk Heatmap, Risk Trend by Month, Priority Queue, Payment Due Monitor
2. **AI Agent** — Single invoice analysis + Batch analyzer
3. **GenAI Assistant** — Chat interface + Natural Language SQL
4. **Deep Analytics** — Vendor Risk Scorecard + Invoice Segmentation
5. **Data & Reports** — Explainable AI + Export

### AP Health Score Formula:
```
Health Score = Risk Score × 0.35
             + Compliance Score × 0.25
             + Efficiency Score × 0.25
             + Integrity Score × 0.15
```
Where:
- Risk Score = 100 - (high_risk_count / total × 250)
- Compliance Score = (1 - missing_po / total) × 100
- Efficiency Score = (1 - sla_breaches / total) × 100
- Integrity Score = (1 - duplicates / total) × 100

---

## 8. Results

| Metric | Value |
|---|---|
| Risk Classification Accuracy | 98.5% |
| Approval Time MAE | 2.46 days |
| Invoice Portfolio Size | 1,000 invoices |
| Knowledge Base Documents | 6 AP policy documents |
| ML Techniques Used | 3 (Classifier, Regressor, KMeans) |
| AI Techniques Used | 5 (Supervised ML, Unsupervised ML, RAG, GenAI, NL2SQL) |

---

## 9. Deployment

- **Platform:** Streamlit Community Cloud
- **Live URL:** https://opsmind-ai.streamlit.app
- **Repository:** https://github.com/Alisa-Shala/opsmind-ai
- **API Keys:** Stored as Streamlit Secrets (not in codebase)

---

## 10. Future Work

- **ERP/SAP Integration** — connect to real enterprise invoice systems
- **Real-time ingestion** — process live invoices as they arrive
- **Email integration** — send drafted emails directly via SMTP
- **User authentication** — role-based access for AP analysts and managers
- **Vector embeddings** — upgrade RAG from TF-IDF to ChromaDB/FAISS for better retrieval
- **Multi-language support** — dashboard in multiple languages
- **LangGraph integration** — more sophisticated multi-agent workflows

---

*OpsMind AI — Built for Genpact AI Engineering Internship*
