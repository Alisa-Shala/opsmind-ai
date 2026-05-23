import os
import pandas as pd
import pickle
from dotenv import load_dotenv

load_dotenv()


class OpsMindAgent:

    TOOLS = [
        {
            "name": "detect_exceptions",
            "description": "Detect all compliance and risk exceptions present in the invoice based on its fields.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "has_purchase_order": {"type": "integer", "description": "1 if PO exists, 0 if missing"},
                    "duplicate_flag": {"type": "integer", "description": "1 if potential duplicate, 0 otherwise"},
                    "sla_breached": {"type": "integer", "description": "1 if SLA was breached, 0 otherwise"},
                    "approval_delay_flag": {"type": "integer", "description": "1 if approval was delayed, 0 otherwise"},
                    "vendor_delay_history": {"type": "integer", "description": "1 if vendor has delay history, 0 otherwise"},
                    "amount": {"type": "number", "description": "Invoice amount in USD"}
                },
                "required": ["has_purchase_order", "duplicate_flag", "sla_breached", "approval_delay_flag", "vendor_delay_history", "amount"]
            }
        },
        {
            "name": "calculate_fraud_score",
            "description": "Calculate a fraud risk score from 0 to 100 based on invoice risk signals.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "has_purchase_order": {"type": "integer"},
                    "duplicate_flag": {"type": "integer"},
                    "sla_breached": {"type": "integer"},
                    "approval_delay_flag": {"type": "integer"},
                    "vendor_delay_history": {"type": "integer"},
                    "amount": {"type": "number"}
                },
                "required": ["has_purchase_order", "duplicate_flag", "sla_breached", "approval_delay_flag", "vendor_delay_history", "amount"]
            }
        },
        {
            "name": "run_three_way_match",
            "description": "Run three-way matching between Purchase Order, Goods Receipt, and Invoice to check for discrepancies.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "has_purchase_order": {"type": "integer"},
                    "duplicate_flag": {"type": "integer"},
                    "amount": {"type": "number"}
                },
                "required": ["has_purchase_order", "duplicate_flag", "amount"]
            }
        },
        {
            "name": "get_ap_policy",
            "description": "Retrieve the relevant Accounts Payable policy for a specific risk topic to ground the recommendation.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Risk topic to look up. Examples: 'duplicate invoice', 'missing purchase order', 'sla breach', 'three-way matching', 'risk scoring'"
                    }
                },
                "required": ["topic"]
            }
        }
    ]

    def __init__(self):
        with open("models/risk_model.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open("models/label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)

        self.anthropic_client = None
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                pass

    # --------------------------------------------------
    # PUBLIC: Standard ML-based analysis
    # --------------------------------------------------
    def analyze_invoice(self, invoice_data):
        input_df = pd.DataFrame([{
            "amount": invoice_data["amount"],
            "approval_days": invoice_data["approval_days"],
            "payment_terms_days": invoice_data["payment_terms_days"],
            "has_purchase_order": invoice_data["has_purchase_order"],
            "duplicate_flag": invoice_data["duplicate_flag"],
            "vendor_delay_history": invoice_data["vendor_delay_history"],
            "sla_breached": invoice_data["sla_breached"],
            "approval_delay_flag": invoice_data["approval_delay_flag"]
        }])

        prediction = self.model.predict(input_df)
        risk_level = self.label_encoder.inverse_transform(prediction)[0]

        exceptions       = self.detect_exceptions(invoice_data)
        fraud_score      = self.calculate_fraud_score(invoice_data)
        three_way_status = self.three_way_match(invoice_data)
        recommended      = self.recommend_action(risk_level, fraud_score, exceptions, three_way_status)
        assigned_team    = self.assign_team(risk_level, fraud_score, exceptions, three_way_status)
        workflow_status  = self.get_workflow_status(fraud_score, exceptions, three_way_status, risk_level)
        action_priority  = self.get_action_priority(fraud_score, risk_level, exceptions)
        vendor_message   = self.generate_vendor_or_ap_message(exceptions, recommended, assigned_team)
        explanation      = self.generate_manager_explanation(
            invoice_data, risk_level, fraud_score, exceptions,
            three_way_status, recommended, assigned_team, workflow_status, action_priority
        )

        return {
            "risk_level": risk_level,
            "fraud_score": fraud_score,
            "exception_types": exceptions,
            "three_way_match_status": three_way_status,
            "recommended_action": recommended,
            "assigned_team": assigned_team,
            "workflow_status": workflow_status,
            "action_priority": action_priority,
            "vendor_message": vendor_message,
            "genai_summary": explanation
        }

    # --------------------------------------------------
    # PUBLIC: Agentic analysis with Claude tool use
    # --------------------------------------------------
    def analyze_invoice_agentic(self, invoice_data):
        """
        Full agentic analysis: Claude autonomously calls tools step by step
        to detect exceptions, score fraud, run three-way matching, and
        retrieve policy — then synthesises a grounded recommendation.
        Falls back to standard ML analysis if Claude API is unavailable.
        """
        if not self.anthropic_client:
            return self.analyze_invoice(invoice_data)

        ml_result = self.analyze_invoice(invoice_data)

        system = (
            "You are an expert Accounts Payable AI agent. "
            "Analyze the given invoice by calling the available tools in the right order. "
            "First detect exceptions, then calculate the fraud score, run three-way matching, "
            "and retrieve any relevant AP policy for the detected risk areas. "
            "After gathering all tool results, write a concise professional recommendation "
            "for the AP manager — max 5 sentences, no bullet points."
        )

        user_msg = (
            f"Analyze this invoice:\n"
            f"- Amount: ${invoice_data['amount']:,.2f}\n"
            f"- Approval Days: {invoice_data['approval_days']}\n"
            f"- Payment Terms: {invoice_data['payment_terms_days']} days\n"
            f"- Has Purchase Order: {'Yes' if invoice_data['has_purchase_order'] else 'No'}\n"
            f"- Duplicate Flag: {'Yes' if invoice_data['duplicate_flag'] else 'No'}\n"
            f"- Vendor Delay History: {'Yes' if invoice_data['vendor_delay_history'] else 'No'}\n"
            f"- SLA Breached: {'Yes' if invoice_data['sla_breached'] else 'No'}\n"
            f"- ML Risk Level (pre-computed): {ml_result['risk_level']}\n\n"
            "Use all relevant tools, then write the final manager recommendation."
        )

        messages = [{"role": "user", "content": user_msg}]

        try:
            for _ in range(8):
                response = self.anthropic_client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=1024,
                    system=system,
                    tools=self.TOOLS,
                    messages=messages
                )

                if response.stop_reason == "end_turn":
                    for block in response.content:
                        if hasattr(block, "text") and block.text.strip():
                            ml_result["genai_summary"] = block.text.strip()
                    break

                if response.stop_reason == "tool_use":
                    messages.append({"role": "assistant", "content": response.content})
                    tool_results = []
                    for block in response.content:
                        if block.type == "tool_use":
                            result = self._execute_tool(block.name, block.input)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": str(result)
                            })
                    messages.append({"role": "user", "content": tool_results})

        except Exception:
            pass

        return ml_result

    def _execute_tool(self, tool_name, tool_input):
        if tool_name == "detect_exceptions":
            return self.detect_exceptions(tool_input)
        if tool_name == "calculate_fraud_score":
            return {"fraud_score": self.calculate_fraud_score(tool_input)}
        if tool_name == "run_three_way_match":
            return {"result": self.three_way_match(tool_input)}
        if tool_name == "get_ap_policy":
            try:
                from rag_retriever import OpsMindRAGRetriever
                retriever = OpsMindRAGRetriever()
                chunks = retriever.retrieve(tool_input.get("topic", ""), top_k=2)
                return retriever.format_context(chunks)
            except Exception:
                return "Policy retrieval unavailable."
        return "Unknown tool."

    # --------------------------------------------------
    # Core logic methods
    # --------------------------------------------------
    def detect_exceptions(self, invoice_data):
        exceptions = []
        if invoice_data.get("duplicate_flag", 0) == 1:
            exceptions.append("Potential duplicate invoice")
        if invoice_data.get("has_purchase_order", 1) == 0:
            exceptions.append("Missing purchase order")
        if invoice_data.get("sla_breached", 0) == 1:
            exceptions.append("SLA breached")
        if invoice_data.get("approval_delay_flag", 0) == 1:
            exceptions.append("Approval delay")
        if invoice_data.get("vendor_delay_history", 0) == 1:
            exceptions.append("Vendor delay history")
        if invoice_data.get("amount", 0) > 15000:
            exceptions.append("High invoice amount")
        if not exceptions:
            exceptions.append("No major exception detected")
        return exceptions

    def calculate_fraud_score(self, invoice_data):
        score = 0
        if invoice_data.get("duplicate_flag", 0) == 1:
            score += 35
        if invoice_data.get("has_purchase_order", 1) == 0:
            score += 20
        if invoice_data.get("amount", 0) > 15000:
            score += 15
        if invoice_data.get("vendor_delay_history", 0) == 1:
            score += 10
        if invoice_data.get("sla_breached", 0) == 1:
            score += 10
        if invoice_data.get("approval_delay_flag", 0) == 1:
            score += 10
        return min(score, 100)

    def three_way_match(self, invoice_data):
        if invoice_data.get("has_purchase_order", 1) == 0:
            return "Failed - Missing purchase order"
        if invoice_data.get("duplicate_flag", 0) == 1:
            return "Failed - Possible duplicate invoice"
        if invoice_data.get("amount", 0) > 20000:
            return "Review Required - Amount variance detected"
        return "Passed"

    def recommend_action(self, risk_level, fraud_score, exceptions, three_way_status):
        if fraud_score >= 80:
            return "Block payment and escalate to AP manager"
        if "Potential duplicate invoice" in exceptions:
            return "Hold invoice and perform duplicate investigation"
        if "Missing purchase order" in exceptions:
            return "Request purchase order before approval"
        if three_way_status.startswith("Failed"):
            return "Send invoice to exception handling queue"
        if risk_level == "High":
            return "Send invoice for manual review"
        if risk_level == "Medium":
            return "Request additional validation"
        return "Approve invoice for payment"

    def assign_team(self, risk_level, fraud_score, exceptions, three_way_status):
        if fraud_score >= 80:
            return "AP Manager + Risk/Compliance Team"
        if "Potential duplicate invoice" in exceptions:
            return "Accounts Payable Analyst"
        if "Missing purchase order" in exceptions:
            return "Procurement Team"
        if "SLA breached" in exceptions or "Approval delay" in exceptions:
            return "Operations Manager"
        if three_way_status.startswith("Review Required"):
            return "Finance Controller"
        if risk_level == "Medium":
            return "AP Analyst"
        return "Standard AP Processing"

    def get_workflow_status(self, fraud_score, exceptions, three_way_status, risk_level):
        if fraud_score >= 80:
            return "Payment Blocked"
        if "Potential duplicate invoice" in exceptions:
            return "On Hold - Duplicate Review"
        if "Missing purchase order" in exceptions:
            return "Pending Purchase Order"
        if "SLA breached" in exceptions:
            return "Escalation Required"
        if three_way_status.startswith("Review Required"):
            return "Pending Amount Validation"
        if risk_level == "Medium":
            return "Pending Additional Validation"
        return "Ready for Approval"

    def get_action_priority(self, fraud_score, risk_level, exceptions):
        if fraud_score >= 80:
            return "Critical"
        if risk_level == "High":
            return "High"
        if "Missing purchase order" in exceptions or "SLA breached" in exceptions:
            return "Medium"
        if risk_level == "Medium":
            return "Medium"
        return "Low"

    def generate_vendor_or_ap_message(self, exceptions, recommended_action, assigned_team):
        if "Potential duplicate invoice" in exceptions:
            return (
                "This invoice has been placed on hold because it may be a duplicate. "
                "Please verify whether this invoice has already been submitted or paid."
            )
        if "Missing purchase order" in exceptions:
            return (
                "This invoice cannot proceed to payment because the Purchase Order is missing. "
                "Please provide the correct PO reference or request Procurement validation."
            )
        if "SLA breached" in exceptions:
            return (
                "This invoice has breached the expected approval SLA. "
                "The case should be reviewed by the responsible manager to avoid further delays."
            )
        if "High invoice amount" in exceptions:
            return (
                "This invoice amount is above the standard review threshold. "
                "Please ensure the amount is validated before payment approval."
            )
        return "No major exception detected. The invoice may continue through the standard approval workflow."

    def generate_manager_explanation(
        self, invoice_data, risk_level, fraud_score, exceptions,
        three_way_status, recommended_action, assigned_team,
        workflow_status, action_priority
    ):
        if self.anthropic_client:
            try:
                return self._generate_with_claude(
                    invoice_data, risk_level, fraud_score, exceptions,
                    three_way_status, recommended_action, assigned_team,
                    workflow_status, action_priority
                )
            except Exception:
                pass
        return self._generate_rule_based(
            risk_level, fraud_score, exceptions, three_way_status,
            recommended_action, assigned_team, workflow_status, action_priority
        )

    def _generate_with_claude(
        self, invoice_data, risk_level, fraud_score, exceptions,
        three_way_status, recommended_action, assigned_team,
        workflow_status, action_priority
    ):
        prompt = (
            f"You are an expert Accounts Payable AI analyst. Write a concise, professional "
            f"manager-ready explanation for the following invoice analysis. Be specific and actionable. Max 4 sentences.\n\n"
            f"Invoice: ${invoice_data['amount']:,.2f} | Approval Days: {invoice_data['approval_days']} | "
            f"PO: {'Yes' if invoice_data['has_purchase_order'] else 'No'} | "
            f"Duplicate: {'Yes' if invoice_data['duplicate_flag'] else 'No'} | "
            f"Vendor Delay: {'Yes' if invoice_data['vendor_delay_history'] else 'No'} | "
            f"SLA Breached: {'Yes' if invoice_data['sla_breached'] else 'No'}\n\n"
            f"Decision: Risk={risk_level} | Fraud Score={fraud_score}/100 | "
            f"Exceptions={', '.join(exceptions)} | 3-Way Match={three_way_status} | "
            f"Status={workflow_status} | Priority={action_priority} | Team={assigned_team} | "
            f"Action={recommended_action}\n\nWrite the manager explanation:"
        )
        msg = self.anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text

    def _generate_rule_based(
        self, risk_level, fraud_score, exceptions, three_way_status,
        recommended_action, assigned_team, workflow_status, action_priority
    ):
        exp = ", ".join(exceptions)
        base = (
            f"This invoice is classified as {risk_level} risk with a fraud score of {fraud_score}/100. "
            f"Detected exceptions: {exp}. Three-way match: {three_way_status}. "
            f"Workflow status: '{workflow_status}' ({action_priority} priority) — assigned to {assigned_team}. "
            f"Recommended action: {recommended_action}."
        )
        if fraud_score >= 80:
            return base + " Payment must be blocked immediately pending risk review."
        if risk_level == "High" or "Potential duplicate invoice" in exceptions:
            return base + " Manual review is required before any payment is processed."
        if risk_level == "Medium":
            return base + " Additional validation is needed before this invoice proceeds."
        return base + " No critical risk detected — invoice may proceed through standard workflow."
