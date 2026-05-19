import pandas as pd
import os


def clean_invoice_data():
    df = pd.read_csv("data/raw/invoices.csv")

    # Convert date columns to datetime
    date_columns = ["submitted_date", "due_date", "approved_date"]

    for col in date_columns:
        df[col] = pd.to_datetime(df[col])

    # Create extra useful columns
    df["days_until_due"] = (df["due_date"] - df["submitted_date"]).dt.days
    df["approval_delay_flag"] = df["approval_days"].apply(
        lambda x: True if x > 10 else False
    )

    df["amount_category"] = df["amount"].apply(categorize_amount)

    # Make boolean columns easier for ML later
    boolean_columns = [
        "has_purchase_order",
        "duplicate_flag",
        "vendor_delay_history",
        "sla_breached",
        "approval_delay_flag"
    ]

    for col in boolean_columns:
        df[col] = df[col].astype(int)

    os.makedirs("data/processed", exist_ok=True)

    df.to_csv("data/processed/clean_invoices.csv", index=False)

    print("Clean data saved successfully!")
    print(df.head())
    print("Rows:", len(df))
    print("Columns:", len(df.columns))


def categorize_amount(amount):
    if amount < 1000:
        return "Small"
    elif amount < 10000:
        return "Medium"
    else:
        return "Large"


if __name__ == "__main__":
    clean_invoice_data()