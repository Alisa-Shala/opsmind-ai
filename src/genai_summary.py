import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def generate_business_summary(df=None):
    """
    Generate an executive business summary.
    Uses Claude API when available, otherwise generates a structured text summary.
    """
    if df is None:
        df = pd.read_csv("data/processed/clean_invoices.csv")

    stats = _compute_stats(df)
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            return _generate_with_claude(client, stats)
        except Exception:
            pass

    return _generate_text_summary(stats)


def _compute_stats(df):
    high_risk_df = df[df["risk_level"] == "High"]
    return {
        "total_invoices":      len(df),
        "total_amount":        df["amount"].sum(),
        "avg_approval_days":   df["approval_days"].mean(),
        "high_risk_count":     len(high_risk_df),
        "medium_risk_count":   len(df[df["risk_level"] == "Medium"]),
        "low_risk_count":      len(df[df["risk_level"] == "Low"]),
        "sla_breaches":        int(df["sla_breached"].sum()),
        "duplicate_flags":     int(df["duplicate_flag"].sum()),
        "missing_po":          int((df["has_purchase_order"] == 0).sum()),
        "top_risk_department": high_risk_df["department"].value_counts().idxmax() if len(high_risk_df) > 0 else "N/A",
        "top_risk_vendor":     high_risk_df["vendor"].value_counts().idxmax() if len(high_risk_df) > 0 else "N/A"
    }


def _generate_with_claude(client, stats):
    prompt = (
        "You are an expert Accounts Payable analyst at a large enterprise. "
        "Write a concise, professional executive summary for the AP manager based on the following invoice portfolio statistics. "
        "Include key risks, operational insights, and 3 specific recommended actions. "
        "Use professional business language. Max 200 words.\n\n"
        f"Portfolio Statistics:\n"
        f"- Total invoices: {stats['total_invoices']}\n"
        f"- Total value: ${stats['total_amount']:,.2f}\n"
        f"- Average approval time: {stats['avg_approval_days']:.1f} days\n"
        f"- High risk: {stats['high_risk_count']} | Medium: {stats['medium_risk_count']} | Low: {stats['low_risk_count']}\n"
        f"- SLA breaches: {stats['sla_breaches']}\n"
        f"- Duplicate flags: {stats['duplicate_flags']}\n"
        f"- Missing purchase orders: {stats['missing_po']}\n"
        f"- Top risk department: {stats['top_risk_department']}\n"
        f"- Top risk vendor: {stats['top_risk_vendor']}\n\n"
        "Write the executive summary now:"
    )
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text


def _generate_text_summary(stats):
    return (
        f"OpsMind AI — Executive Business Summary\n\n"
        f"The invoice portfolio contains {stats['total_invoices']} invoices "
        f"with a total processed value of ${stats['total_amount']:,.2f}. "
        f"Average approval time is {stats['avg_approval_days']:.1f} days.\n\n"
        f"Risk Distribution:\n"
        f"  High Risk: {stats['high_risk_count']} invoices\n"
        f"  Medium Risk: {stats['medium_risk_count']} invoices\n"
        f"  Low Risk: {stats['low_risk_count']} invoices\n\n"
        f"Operational Alerts:\n"
        f"  {stats['sla_breaches']} SLA breaches detected\n"
        f"  {stats['duplicate_flags']} potential duplicate invoices\n"
        f"  {stats['missing_po']} invoices missing purchase orders\n"
        f"  Highest-risk department: {stats['top_risk_department']}\n"
        f"  Highest-risk vendor: {stats['top_risk_vendor']}\n\n"
        f"Recommended Actions:\n"
        f"  1. Prioritize manual review of {stats['high_risk_count']} high-risk invoices.\n"
        f"  2. Investigate {stats['duplicate_flags']} duplicate flags before payment.\n"
        f"  3. Request purchase order documentation for {stats['missing_po']} pending invoices."
    )


if __name__ == "__main__":
    print(generate_business_summary())
