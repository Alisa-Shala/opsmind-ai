import pandas as pd
import random
from datetime import datetime, timedelta
import os

random.seed(42)

vendors = [
    "TechNova Solutions",
    "CloudBridge Ltd",
    "DataCore Systems",
    "FinEdge Services",
    "OfficePlus Group",
    "NetSecure Inc",
    "GreenLogix",
    "Alpha Consulting"
]

departments = [
    "Finance",
    "IT",
    "Procurement",
    "Operations",
    "HR",
    "Risk Management"
]

invoice_types = [
    "Software License",
    "Consulting Service",
    "Cloud Infrastructure",
    "Office Supplies",
    "Security Service",
    "Logistics"
]


def generate_invoice_data(num_records=1000):
    data = []

    for i in range(1, num_records + 1):
        invoice_id   = f"INV-{i:05d}"
        vendor       = random.choice(vendors)
        department   = random.choice(departments)
        invoice_type = random.choice(invoice_types)

        amount             = round(random.uniform(500, 30000), 2)
        approval_days      = random.randint(1, 15)
        payment_terms_days = random.choice([15, 30, 45, 60])

        submitted_date = datetime(2025, 1, 1) + timedelta(days=random.randint(0, 330))
        due_date       = submitted_date + timedelta(days=payment_terms_days)
        approved_date  = submitted_date + timedelta(days=approval_days)

        # Realistic rates
        has_purchase_order   = random.random() > 0.10   # 10% missing PO
        duplicate_flag       = random.random() < 0.06   # 6% duplicates
        vendor_delay_history = random.random() < 0.15   # 15% vendor delay history

        # SLA breached only if approval takes more than 12 days
        sla_breached = approval_days > 12

        # Risk logic
        if duplicate_flag:
            risk_level  = "High"
            risk_reason = "Duplicate invoice detected"
        elif amount > 20000 and not has_purchase_order:
            risk_level  = "High"
            risk_reason = "High amount with missing purchase order"
        elif vendor_delay_history and sla_breached:
            risk_level  = "High"
            risk_reason = "Vendor delay history with SLA breach"
        elif not has_purchase_order or vendor_delay_history:
            risk_level  = "Medium"
            risk_reason = "Missing purchase order" if not has_purchase_order else "Vendor has previous delays"
        elif sla_breached:
            risk_level  = "Medium"
            risk_reason = "SLA approval deadline exceeded"
        else:
            risk_level  = "Low"
            risk_reason = "No risk detected"

        if risk_level == "High":
            next_best_action = "Send for manual review" if not duplicate_flag else "Escalate to manager"
        elif risk_level == "Medium":
            next_best_action = "Request missing information" if not has_purchase_order else "Request additional validation"
        elif sla_breached:
            next_best_action = "Escalate to manager"
        else:
            next_best_action = "Approve for payment"

        data.append({
            "invoice_id":          invoice_id,
            "vendor":              vendor,
            "department":          department,
            "invoice_type":        invoice_type,
            "amount":              amount,
            "submitted_date":      submitted_date.date(),
            "due_date":            due_date.date(),
            "approved_date":       approved_date.date(),
            "approval_days":       approval_days,
            "payment_terms_days":  payment_terms_days,
            "has_purchase_order":  has_purchase_order,
            "duplicate_flag":      duplicate_flag,
            "vendor_delay_history":vendor_delay_history,
            "sla_breached":        sla_breached,
            "risk_level":          risk_level,
            "risk_reason":         risk_reason,
            "next_best_action":    next_best_action
        })

    return pd.DataFrame(data)


if __name__ == "__main__":
    df = generate_invoice_data(1000)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/invoices.csv", index=False)

    high  = len(df[df["risk_level"]=="High"])
    med   = len(df[df["risk_level"]=="Medium"])
    low   = len(df[df["risk_level"]=="Low"])
    sla   = int(df["sla_breached"].sum())
    dup   = int(df["duplicate_flag"].sum())
    mpo   = int((df["has_purchase_order"]==False).sum())

    print(f"Dataset generated: {len(df)} invoices")
    print(f"  High Risk:    {high} ({high/len(df)*100:.1f}%)")
    print(f"  Medium Risk:  {med} ({med/len(df)*100:.1f}%)")
    print(f"  Low Risk:     {low} ({low/len(df)*100:.1f}%)")
    print(f"  SLA Breaches: {sla} ({sla/len(df)*100:.1f}%)")
    print(f"  Duplicates:   {dup} ({dup/len(df)*100:.1f}%)")
    print(f"  Missing PO:   {mpo} ({mpo/len(df)*100:.1f}%)")
