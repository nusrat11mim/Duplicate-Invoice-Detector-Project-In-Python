from datetime import date
from rapidfuzz import fuzz


def _to_date(value):
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def pair_features(inv_a: dict, inv_b: dict) -> list:
    """
    Returns a list of 6 numeric features describing how similar two
    invoice records are. Order must match FEATURE_NAMES below.
    """
    vendor_ratio = fuzz.token_sort_ratio(str(inv_a["vendor_name"]), str(inv_b["vendor_name"])) / 100.0
    invnum_ratio = fuzz.ratio(str(inv_a["invoice_number"]), str(inv_b["invoice_number"])) / 100.0
    desc_ratio = fuzz.token_sort_ratio(str(inv_a.get("description", "")), str(inv_b.get("description", ""))) / 100.0

    amt_a, amt_b = float(inv_a["amount"]), float(inv_b["amount"])
    max_amt = max(abs(amt_a), abs(amt_b), 1e-6)
    amount_diff_pct = abs(amt_a - amt_b) / max_amt
    exact_amount_match = 1.0 if abs(amt_a - amt_b) < 0.005 else 0.0

    date_diff_days = abs((_to_date(inv_a["invoice_date"]) - _to_date(inv_b["invoice_date"])).days)
    # squashed so 0 days -> 1.0 closeness, large gaps -> near 0
    date_closeness = 1.0 / (1.0 + date_diff_days)

    return [vendor_ratio, invnum_ratio, desc_ratio, amount_diff_pct, exact_amount_match, date_closeness]


FEATURE_NAMES = [
    "vendor_similarity",
    "invoice_number_similarity",
    "description_similarity",
    "amount_diff_pct",
    "exact_amount_match",
    "date_closeness",
]
