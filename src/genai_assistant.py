import os
import re
from dotenv import load_dotenv
from rag_retriever import OpsMindRAGRetriever

load_dotenv()


class OpsMindGenAIAssistant:

    SYSTEM_PROMPT = """You are OpsMind AI, a professional GenAI Business Copilot for Accounts Payable invoice operations.

Answer only questions related to invoice risk, fraud detection, AP workflows, SLA breaches,
missing purchase orders, duplicate invoices, SQL analytics, dashboard insights, and manager recommendations.

Rules:
- Use only the provided dashboard context and RAG policy context.
- Never invent numbers or facts not in the context.
- Be concise, professional, and give manager-ready recommendations.
- Refuse harmful or completely unrelated requests politely."""

    BLOCKED = ["kill", "harm", "hurt", "suicide", "weapon", "bomb", "hack",
               "steal", "password", "malware", "exploit", "illegal", "violence"]

    INVOICE_KEYWORDS = [
        "invoice", "invoices", "risk", "fraud", "sla", "breach", "purchase order",
        "po", "missing", "duplicate", "vendor", "manager", "dashboard", "summary",
        "analytics", "sql", "department", "amount", "payment", "approval", "ap",
        "accounts payable", "workflow", "priority", "escalation", "manual review",
        "three-way", "procurement", "finance", "operations", "recommendation",
        "action", "executive", "root cause", "explain", "report", "insight",
        "how many", "total", "cases", "queue", "policy", "high risk", "low risk",
        "medium risk", "control", "validation", "cost", "overdue", "delay",
        "exception", "flag", "block", "hold", "approve", "reject", "process"
    ]

    EXAMPLES = [
        "How many high-risk invoices are there?",
        "What should the AP manager focus on first?",
        "Summarize the current invoice portfolio",
        "Which vendor has the highest risk rate?",
        "How many SLA breaches do we have?",
    ]

    def __init__(self):
        self.retriever = OpsMindRAGRetriever()
        self.anthropic_client = None
        self.groq_api_key = os.getenv("GROQ_API_KEY")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key and api_key not in ("vendose_api_key_ketu", ""):
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                pass

        if self.groq_api_key in ("vendose_api_key_ketu", "vendos_ketu_kodin_nga_groq", ""):
            self.groq_api_key = None

    def has_api_key(self):
        return self.anthropic_client is not None or bool(self.groq_api_key)

    def correct_query(self, text):
        try:
            from spellchecker import SpellChecker
            spell = SpellChecker()
            domain = {"sla", "po", "ap", "kpi", "erp", "sap", "opsmind",
                      "invoice", "invoices", "vendor", "procurement", "escalation"}
            words, corrections, corrected = text.split(), {}, []
            for w in words:
                clean = w.strip(".,?!\"'").lower()
                if clean in domain or len(clean) <= 2:
                    corrected.append(w)
                    continue
                if spell.unknown([clean]):
                    s = spell.correction(clean)
                    if s and s != clean:
                        corrections[w] = s
                        corrected.append(s)
                        continue
                corrected.append(w)
            return " ".join(corrected), corrections
        except Exception:
            return text, {}

    def generate_response(self, question, dashboard_context, agent_context=None):
        safety = self._check_safety(question)
        if not safety["allowed"]:
            return safety["message"]

        rag_context = self.retriever.format_context(self.retriever.retrieve(question, top_k=3))
        prompt = self._build_prompt(question, dashboard_context, rag_context, agent_context)

        if self.anthropic_client:
            try:
                return self._call_claude(prompt)
            except Exception:
                pass

        if self.groq_api_key:
            try:
                return self._call_groq(prompt)
            except Exception:
                pass

        # Smart fallback — works without any API key
        return self._smart_fallback(question, dashboard_context)

    def _build_prompt(self, question, dashboard_context, rag_context, agent_context):
        return (
            f"User question:\n{question}\n\n"
            f"Dashboard context:\n{dashboard_context}\n\n"
            f"Retrieved policy context (RAG):\n{rag_context}\n\n"
            f"AI Agent context:\n{agent_context or 'Not provided.'}\n\n"
            "Generate a clear, professional, manager-ready answer."
        )

    def _call_claude(self, prompt):
        msg = self.anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text

    def _call_groq(self, prompt):
        from groq import Groq
        client = Groq(api_key=self.groq_api_key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, max_tokens=600
        )
        return resp.choices[0].message.content

    def _smart_fallback(self, question, dashboard_context):
        """Answer using dashboard numbers without an LLM."""
        def _get(label):
            m = re.search(rf"{label}:\s*([^\n]+)", dashboard_context, re.IGNORECASE)
            return m.group(1).strip() if m else "N/A"

        total       = _get("Total invoices")
        amount      = _get("Total amount")
        high_risk   = _get("High-risk invoices")
        sla         = _get("SLA breaches")
        missing_po  = _get("Missing purchase order cases")
        duplicates  = _get("Duplicate invoice flags")
        manual_q    = _get("Manual review queue")
        escalations = _get("Escalations")
        top_dept    = _get("Top high-risk department")
        top_vendor  = _get("Top high-risk vendor")

        q = question.lower()

        if any(w in q for w in ["how many", "number", "count", "total"]):
            if "high" in q and "risk" in q:
                return f"There are **{high_risk} high-risk invoices** in the current dashboard view."
            if "sla" in q:
                return f"There are **{sla} SLA breach cases** in the current dashboard."
            if "missing" in q or "purchase order" in q or " po " in q:
                return f"There are **{missing_po} invoices missing a Purchase Order**."
            if "duplicate" in q:
                return f"There are **{duplicates} potential duplicate invoices** flagged in the dashboard."
            if "escalat" in q:
                return f"There are **{escalations} escalation cases** requiring manager attention."
            if "manual" in q or "review" in q:
                return f"There are **{manual_q} invoices** in the manual review queue."
            return (
                f"The dashboard contains **{total} invoices** with a total value of **{amount}**. "
                f"Of these, **{high_risk}** are high-risk, **{sla}** have SLA breaches, "
                f"**{duplicates}** are potential duplicates, and **{missing_po}** are missing a Purchase Order."
            )

        if any(w in q for w in ["summary", "summarize", "executive", "overview"]):
            return (
                f"**OpsMind AI — Invoice Portfolio Summary**\n\n"
                f"The selected portfolio contains **{total} invoices** with a total value of **{amount}**.\n\n"
                f"**Key risks:**\n"
                f"- {high_risk} high-risk invoices\n"
                f"- {sla} SLA breaches\n"
                f"- {duplicates} potential duplicate invoices\n"
                f"- {missing_po} missing Purchase Orders\n\n"
                f"**Priority areas:** {top_dept} department and vendor {top_vendor} show the highest risk concentration.\n\n"
                f"**Recommended actions:** Review high-risk invoices manually, investigate duplicates before payment, "
                f"escalate SLA-breached cases, and request PO documentation for {missing_po} pending invoices."
            )

        if any(w in q for w in ["focus", "priority", "manager", "action", "recommend"]):
            return (
                f"**AP Manager Priority Recommendations:**\n\n"
                f"1. **Immediate:** Review {high_risk} high-risk invoices — block payment on any with fraud score above 80.\n"
                f"2. **Urgent:** Investigate {duplicates} duplicate flags before processing payment.\n"
                f"3. **High priority:** Escalate {sla} SLA-breached invoices to the responsible manager.\n"
                f"4. **Standard:** Request Purchase Order for {missing_po} invoices before approval.\n"
                f"5. **Monitor:** {top_vendor} is the top high-risk vendor — review their invoice history.\n\n"
                f"Manual review queue: **{manual_q}** | Escalation queue: **{escalations}**"
            )

        if "vendor" in q and ("risk" in q or "high" in q or "top" in q):
            return (
                f"The top high-risk vendor in the current dashboard is **{top_vendor}**. "
                f"This vendor is associated with the highest number of high-risk invoices. "
                f"Recommended action: place additional scrutiny on all incoming invoices from this vendor "
                f"and review their delay and exception history."
            )

        if "department" in q:
            return (
                f"The department with the highest number of high-risk invoices is **{top_dept}**. "
                f"The AP manager should prioritise reviewing invoices from this department, "
                f"particularly those with missing purchase orders or SLA breaches."
            )

        if "duplicate" in q:
            return (
                f"There are **{duplicates} potential duplicate invoices** in the current view. "
                f"Duplicate invoices are a major risk because they can result in double payment to the same vendor. "
                f"All {duplicates} should be placed on hold and investigated by the AP Analyst before any payment is released."
            )

        if "sla" in q:
            return (
                f"**{sla} invoices** have breached the SLA approval threshold. "
                f"SLA breach means the invoice took longer than the allowed approval period. "
                f"These cases should be escalated to the Operations Manager immediately "
                f"to avoid further delays and contractual penalties."
            )

        if "purchase order" in q or " po " in q or "missing" in q:
            return (
                f"**{missing_po} invoices** are missing a Purchase Order (PO). "
                f"A missing PO means the company cannot confirm whether the purchase was authorised before the invoice arrived. "
                f"These invoices must not be approved for payment until the Procurement team provides a valid PO reference."
            )

        if "root cause" in q or "why" in q:
            return (
                f"The main root causes of invoice risk in the current portfolio are:\n\n"
                f"1. **Duplicate submissions** ({duplicates} cases) — vendors submitting the same invoice more than once\n"
                f"2. **Missing Purchase Orders** ({missing_po} cases) — invoices arriving without prior authorisation\n"
                f"3. **SLA breaches** ({sla} cases) — approval process taking longer than allowed\n"
                f"4. **Vendor delay history** — repeat patterns from specific vendors like {top_vendor}\n\n"
                f"Focus controls on purchase order validation and duplicate checks."
            )

        # Generic fallback
        return (
            f"Based on the current dashboard, the portfolio has **{total} invoices** (total value: {amount}), "
            f"with **{high_risk} high-risk**, **{sla} SLA breaches**, **{duplicates} duplicate flags**, "
            f"and **{missing_po} missing PO cases**. "
            f"The top risk department is **{top_dept}** and top risk vendor is **{top_vendor}**. "
            f"For more detailed analysis, enable the GenAI assistant by adding an API key in the sidebar."
        )

    def _check_safety(self, question):
        q = question.lower().strip()
        for w in self.BLOCKED:
            if w in q:
                return {"allowed": False, "message": "I can only help with professional invoice operations topics."}
        if not any(w in q for w in self.INVOICE_KEYWORDS):
            examples = "\n".join(f"• {e}" for e in self.EXAMPLES)
            return {"allowed": False, "message": (
                f"I'm specialised in invoice risk management and Accounts Payable operations.\n\n"
                f"Here are some questions I can answer:\n{examples}"
            )}
        return {"allowed": True, "message": ""}
