import pandas as pd


class OpsMindBatchAnalyzer:
    """
    Batch invoice risk analyzer.
    Runs the OpsMind AI Agent across an entire invoice dataset
    and returns a ranked risk report sorted by fraud score.
    """

    REQUIRED_COLUMNS = [
        "amount", "approval_days", "payment_terms_days",
        "has_purchase_order", "duplicate_flag",
        "vendor_delay_history", "sla_breached"
    ]

    def __init__(self, agent):
        self.agent = agent

    def validate_dataframe(self, df):
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        return missing

    def analyze_batch(self, df, progress_callback=None):
        """
        Analyze every row in df with the AI agent.
        progress_callback(i, total) is called after each invoice if provided.
        Returns a DataFrame ranked by fraud_score descending.
        """
        records = []
        total = len(df)

        for i, (_, row) in enumerate(df.iterrows()):
            approval_days = int(row.get("approval_days", 0))
            invoice_data = {
                "amount":               float(row.get("amount", 0)),
                "approval_days":        approval_days,
                "payment_terms_days":   int(row.get("payment_terms_days", 30)),
                "has_purchase_order":   int(row.get("has_purchase_order", 1)),
                "duplicate_flag":       int(row.get("duplicate_flag", 0)),
                "vendor_delay_history": int(row.get("vendor_delay_history", 0)),
                "sla_breached":         int(row.get("sla_breached", 0)),
                "approval_delay_flag":  1 if approval_days > 10 else 0
            }

            result = self.agent.analyze_invoice(invoice_data)

            records.append({
                "invoice_id":          row.get("invoice_id", f"INV-{i+1:04d}"),
                "vendor":              row.get("vendor", "Unknown"),
                "department":          row.get("department", "Unknown"),
                "amount":              invoice_data["amount"],
                "risk_level":          result["risk_level"],
                "fraud_score":         result["fraud_score"],
                "action_priority":     result["action_priority"],
                "workflow_status":     result["workflow_status"],
                "assigned_team":       result["assigned_team"],
                "recommended_action":  result["recommended_action"],
                "exceptions":          " | ".join(result["exception_types"]),
                "three_way_match":     result["three_way_match_status"]
            })

            if progress_callback:
                progress_callback(i + 1, total)

        result_df = pd.DataFrame(records)
        result_df = result_df.sort_values("fraud_score", ascending=False).reset_index(drop=True)
        return result_df

    def get_batch_summary(self, result_df):
        total = len(result_df)
        high_risk = len(result_df[result_df["risk_level"] == "High"])
        critical = len(result_df[result_df["action_priority"] == "Critical"])
        avg_fraud = result_df["fraud_score"].mean()
        top_vendor = (
            result_df[result_df["risk_level"] == "High"]["vendor"].value_counts().idxmax()
            if high_risk > 0 else "N/A"
        )
        top_dept = (
            result_df[result_df["risk_level"] == "High"]["department"].value_counts().idxmax()
            if high_risk > 0 else "N/A"
        )
        return {
            "total_analyzed":    total,
            "high_risk_count":   high_risk,
            "critical_count":    critical,
            "avg_fraud_score":   round(avg_fraud, 1),
            "top_risk_vendor":   top_vendor,
            "top_risk_dept":     top_dept,
            "pct_high_risk":     round(high_risk / total * 100, 1) if total > 0 else 0
        }
