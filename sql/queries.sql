import sqlite3
import pandas as pd


DATABASE_PATH = "database/opsmind.db"


def run_query(query, title):
    connection = sqlite3.connect(DATABASE_PATH)

    df = pd.read_sql_query(query, connection)

    connection.close()

    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print(df.head(10))


def main():
    queries = {
        "Portfolio Overview": """
            SELECT
                COUNT(*) AS total_invoices,
                ROUND(SUM(amount), 2) AS total_invoice_amount,
                ROUND(AVG(amount), 2) AS average_invoice_amount,
                ROUND(AVG(approval_days), 2) AS average_approval_days
            FROM invoices;
        """,

        "Risk Level Distribution": """
            SELECT
                risk_level,
                COUNT(*) AS invoice_count,
                ROUND(SUM(amount), 2) AS total_amount
            FROM invoices
            GROUP BY risk_level
            ORDER BY invoice_count DESC;
        """,

        "High-Risk Invoices by Department": """
            SELECT
                department,
                COUNT(*) AS high_risk_count,
                ROUND(SUM(amount), 2) AS high_risk_amount
            FROM invoices
            WHERE risk_level = 'High'
            GROUP BY department
            ORDER BY high_risk_count DESC;
        """,

        "SLA Breaches by Department": """
            SELECT
                department,
                COUNT(*) AS sla_breach_count,
                ROUND(AVG(approval_days), 2) AS avg_approval_days
            FROM invoices
            WHERE sla_breached = 1
            GROUP BY department
            ORDER BY sla_breach_count DESC;
        """,

        "Top High-Risk Vendors": """
            SELECT
                vendor,
                COUNT(*) AS high_risk_invoice_count,
                ROUND(SUM(amount), 2) AS high_risk_amount
            FROM invoices
            WHERE risk_level = 'High'
            GROUP BY vendor
            ORDER BY high_risk_invoice_count DESC
            LIMIT 10;
        """,

        "Duplicate Invoice Monitoring": """
            SELECT
                vendor,
                COUNT(*) AS duplicate_invoice_flags,
                ROUND(SUM(amount), 2) AS duplicate_invoice_amount
            FROM invoices
            WHERE duplicate_flag = 1
            GROUP BY vendor
            ORDER BY duplicate_invoice_flags DESC;
        """,

        "Missing PO Cases": """
            SELECT
                department,
                COUNT(*) AS missing_po_count,
                ROUND(SUM(amount), 2) AS missing_po_amount
            FROM invoices
            WHERE has_purchase_order = 0
            GROUP BY department
            ORDER BY missing_po_count DESC;
        """
    }

    for title, query in queries.items():
        run_query(query, title)


if __name__ == "__main__":
    main()