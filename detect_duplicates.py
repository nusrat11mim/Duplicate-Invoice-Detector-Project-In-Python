import sys
import pandas as pd
import joblib

from features import pair_features, FEATURE_NAMES

DUPLICATE_THRESHOLD = 0.6   # probability >= this -> flagged as likely duplicate
REVIEW_THRESHOLD = 0.35     # probability >= this (but below duplicate) -> "needs review"


def load_invoices(path="invoices.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"invoice_id", "invoice_number", "vendor_name", "amount", "invoice_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"invoices.csv is missing required columns: {missing}")
    if "description" not in df.columns:
        df["description"] = ""
    return df


def candidate_pairs(df: pd.DataFrame):
    """
    Blocking step: group invoices by (first letter of vendor, amount
    bucket) so we only score plausible pairs instead of every combination.
    """
    df = df.copy()
    df["_vendor_key"] = df["vendor_name"].astype(str).str.strip().str.lower().str[0]
    df["_amount_bucket"] = (df["amount"].astype(float) // 50).astype(int)  # $50 buckets

    pairs = []
    grouped = df.groupby(["_vendor_key"])
    for _, group in grouped:
        # within a vendor-letter block, only compare invoices whose amount
        # buckets are within 1 of each other (catches rounding differences)
        records = group.to_dict("records")
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                a, b = records[i], records[j]
                if abs(a["_amount_bucket"] - b["_amount_bucket"]) <= 1:
                    pairs.append((a, b))
    return pairs


def score_pairs(pairs, model, review_threshold=REVIEW_THRESHOLD, duplicate_threshold=DUPLICATE_THRESHOLD):
    """
    Scores every candidate pair with the model and returns the ones at or
    above review_threshold, each tagged with a verdict based on
    duplicate_threshold. Kept as separate parameters (rather than only the
    module constants) so callers -- like the GUI -- can re-run with
    different threshold values without retraining or reloading the model.
    """
    results = []
    if not pairs:
        return results
    feature_rows = [pair_features(a, b) for a, b in pairs]
    probs = model.predict_proba(feature_rows)[:, 1]  # probability of class "1" = duplicate

    for (a, b), prob in zip(pairs, probs):
        if prob >= review_threshold:
            results.append({
                "invoice_id_a": a["invoice_id"],
                "invoice_number_a": a["invoice_number"],
                "vendor_a": a["vendor_name"],
                "amount_a": a["amount"],
                "date_a": a["invoice_date"],
                "invoice_id_b": b["invoice_id"],
                "invoice_number_b": b["invoice_number"],
                "vendor_b": b["vendor_name"],
                "amount_b": b["amount"],
                "date_b": b["invoice_date"],
                "duplicate_probability": round(float(prob), 4),
                "verdict": "DUPLICATE" if prob >= duplicate_threshold else "NEEDS_REVIEW",
            })
    results.sort(key=lambda r: r["duplicate_probability"], reverse=True)
    return results


def run_detection(invoices_path="invoices.csv", model_path="invoice_dup_model.joblib",
                   duplicate_threshold=DUPLICATE_THRESHOLD, review_threshold=REVIEW_THRESHOLD,
                   report_path="duplicate_report.csv", log=print):
    """
    Core pipeline, reused by both the CLI (main()) and the GUI (gui.py):
    load invoices -> block into candidate pairs -> score with the model ->
    save + return the report. `log` is a callable used for progress
    messages so the GUI can route them into its own log widget instead of
    stdout.

    Raises FileNotFoundError if the model hasn't been trained yet.
    """
    model = joblib.load(model_path)  # raises FileNotFoundError if missing

    df = load_invoices(invoices_path)
    log(f"Loaded {len(df)} invoices from {invoices_path}")

    pairs = candidate_pairs(df)
    log(f"Generated {len(pairs)} candidate pairs after blocking (instead of "
        f"{len(df) * (len(df) - 1) // 2} total possible pairs)")

    results = score_pairs(pairs, model, review_threshold=review_threshold,
                           duplicate_threshold=duplicate_threshold)
    report_df = pd.DataFrame(results)
    report_df.to_csv(report_path, index=False)

    n_dup = sum(1 for r in results if r["verdict"] == "DUPLICATE")
    n_review = sum(1 for r in results if r["verdict"] == "NEEDS_REVIEW")

    log(f"\n=== Duplicate Invoice Detection Report ===")
    log(f"Likely duplicates found : {n_dup}")
    log(f"Needs manual review     : {n_review}")
    log(f"Full report saved to {report_path}\n")

    if results:
        log("Top matches:")
        for r in results[:10]:
            log(f"  [{r['verdict']:>12}] ({r['duplicate_probability']:.2%}) "
                f"#{r['invoice_id_a']} '{r['vendor_a']}' ${r['amount_a']} on {r['date_a']}  "
                f"<->  #{r['invoice_id_b']} '{r['vendor_b']}' ${r['amount_b']} on {r['date_b']}")

    return report_df


def main():
    try:
        report_df = run_detection()
    except FileNotFoundError:
        print("No trained model found. Run train_model.py first (main.py does this automatically).")
        sys.exit(1)
    return report_df


if __name__ == "__main__":
    main()
