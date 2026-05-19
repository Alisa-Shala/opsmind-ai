import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error


FEATURES = [
    "amount", "approval_days", "payment_terms_days",
    "has_purchase_order", "duplicate_flag",
    "vendor_delay_history", "sla_breached", "approval_delay_flag"
]


def train_risk_model():
    df = pd.read_csv("data/processed/clean_invoices.csv")

    X = df[FEATURES]
    y = df["risk_level"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Risk model trained! Accuracy:", round(accuracy, 4))
    print(classification_report(y_test, predictions, target_names=label_encoder.classes_))

    os.makedirs("models", exist_ok=True)
    with open("models/risk_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    print("Saved: models/risk_model.pkl")


def train_approval_predictor():
    df = pd.read_csv("data/processed/clean_invoices.csv")

    reg_features = [
        "amount", "payment_terms_days", "has_purchase_order",
        "duplicate_flag", "vendor_delay_history", "sla_breached"
    ]

    X = df[reg_features]
    y = df["approval_days"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    print(f"Approval predictor trained! MAE: {mae:.2f} days")

    with open("models/approval_predictor.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/approval_predictor_features.pkl", "wb") as f:
        pickle.dump(reg_features, f)
    print("Saved: models/approval_predictor.pkl")


if __name__ == "__main__":
    train_risk_model()
    print()
    train_approval_predictor()
