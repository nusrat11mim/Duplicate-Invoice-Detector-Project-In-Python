import random
from datetime import date, timedelta

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from features import pair_features

random.seed(7)

VENDORS = [
    "Acme Supplies Ltd", "Bright Star Logistics", "Crestview Consulting",
    "Delta Office Solutions", "Everest Freight Co", "Falcon Hardware",
    "Golden Gate Marketing", "Horizon IT Services", "Ivory Paper Mills",
    "Jupiter Cleaning Co", "Zenith Packaging", "Northwind Traders",
]

DESCRIPTIONS = [
    "Monthly service fee", "Office supplies", "Freight & handling",
    "Consulting services", "Software license", "Equipment rental",
    "Maintenance contract", "Raw materials", "IT support", "Cleaning services",
]


def random_invoice():
    return {
        "invoice_number": f"INV-{random.randint(10000, 99999)}",
        "vendor_name": random.choice(VENDORS),
        "amount": round(random.uniform(50, 9000), 2),
        "invoice_date": (date(2024, 1, 1) + timedelta(days=random.randint(0, 300))).isoformat(),
        "description": random.choice(DESCRIPTIONS),
    }


def make_duplicate_of(inv):
    """Create a realistic near-duplicate (or exact duplicate) of inv."""
    dup = dict(inv)
    mode = random.choice(["exact", "typo_vendor", "rounded_amount", "date_drift", "combo"])

    if mode in ("typo_vendor", "combo"):
        name = dup["vendor_name"]
        variants = [name.replace("Ltd", "Ltd."), name.lower(), name + " ", name[:-1]]
        dup["vendor_name"] = random.choice(variants)

    if mode in ("rounded_amount", "combo"):
        dup["amount"] = round(dup["amount"] + random.choice([0, 0.01, -0.01, 1.0, -1.0]), 2)

    if mode in ("date_drift", "combo"):
        d = date.fromisoformat(dup["invoice_date"]) + timedelta(days=random.randint(0, 6))
        dup["invoice_date"] = d.isoformat()

    return dup


def build_training_set(n_pairs=4000):
    X, y = [], []
    for _ in range(n_pairs // 2):
        # Positive example: an invoice and its near/exact duplicate
        base = random_invoice()
        dup = make_duplicate_of(base)
        X.append(pair_features(base, dup))
        y.append(1)

        # Negative example: two unrelated random invoices
        a, b = random_invoice(), random_invoice()
        X.append(pair_features(a, b))
        y.append(0)

    return X, y


def main(n_pairs=4000, model_path="invoice_dup_model.joblib", log=print):
    log("Building synthetic labeled training pairs...")
    X, y = build_training_set(n_pairs=n_pairs)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    log("Training RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    log("\nValidation performance on held-out synthetic pairs:")
    log(classification_report(y_test, y_pred, target_names=["not_duplicate", "duplicate"]))

    joblib.dump(clf, model_path)
    log(f"Saved trained model to {model_path}")
    return model_path


if __name__ == "__main__":
    main()
