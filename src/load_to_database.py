import os
import sqlite3
import pandas as pd


DATABASE_PATH = "database/opsmind.db"
INPUT_PATH = "data/processed/clean_invoices.csv"


def load_invoices_to_database():
    """
    Loads the cleaned invoice dataset into a SQLite database.
    The Streamlit dashboard will read data directly from this database.
    """

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Cleaned dataset not found at {INPUT_PATH}. "
            "Run: py src\\data_cleaning.py first."
        )

    os.makedirs("database", exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    connection = sqlite3.connect(DATABASE_PATH)

    df.to_sql(
        name="invoices",
        con=connection,
        if_exists="replace",
        index=False
    )

    connection.close()

    print("Clean invoice data loaded successfully into SQLite database.")
    print(f"Database path: {DATABASE_PATH}")
    print("Table created/replaced: invoices")
    print(f"Total rows loaded: {len(df)}")
    print(f"Total columns loaded: {len(df.columns)}")


if __name__ == "__main__":
    load_invoices_to_database()