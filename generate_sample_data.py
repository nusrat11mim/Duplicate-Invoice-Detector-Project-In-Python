import random
import csv
from datetime import date, timedelta

random.seed(42)

VENDORS = [
    "Acme Supplies Ltd", "Bright Star Logistics", "Crestview Consulting",
    "Delta Office Solutions", "Everest Freight Co", "Falcon Hardware",
    "Golden Gate Marketing", "Horizon IT Services", "Ivory Paper Mills",
    "Jupiter Cleaning Co",
]

DESCRIPTIONS = [
    "Monthly service fee", "Office supplies", "Freight & handling",
    "Consulting services", "Software license", "Equipment rental",
    "Maintenance contract", "Raw materials", "IT support", "Cleaning services",
]


def random_date(start_year=2024):
    start = date(start_year, 1, 1)
    return start + timedelta(days=random.randint(0, 300))


def typo_vendor(name):
    """Introduce a small, realistic typo into a vendor name."""
    variants = [
        name.replace("Ltd", "Ltd."),
        name.replace("Co", "Co."),
        name.lower(),
        name + " ",
        name.replace("a", "a ", 1),
        name[:-1] if len(name) > 5 else name,
    ]
    return random.choice(variants)


def build_invoices(n_base=8, n_exact_dupes=1, n_near_dupes=1):
    rows = []
    invoice_id = 1000

    # --- base, unique invoices ---
    for _ in range(n_base):
        invoice_id += 1
        vendor = random.choice(VENDORS)
        amount = round(random.uniform(50, 9000), 2)
        rows.append({
            "invoice_id": invoice_id,
            "invoice_number": f"INV-{random.randint(10000, 99999)}",
            "vendor_name": vendor,
            "amount": amount,
            "invoice_date": random_date().isoformat(),
            "description": random.choice(DESCRIPTIONS),
        })

    # --- exact duplicates (same everything, re-submitted invoice) ---
    for _ in range(n_exact_dupes):
        original = random.choice(rows).copy()
        invoice_id += 1
        original["invoice_id"] = invoice_id
        rows.append(original)

    # --- near duplicates (typo'd vendor / rounded amount / date drift) ---
    for _ in range(n_near_dupes):
        original = random.choice(rows[:n_base]).copy()
        invoice_id += 1
        original["invoice_id"] = invoice_id
        original["vendor_name"] = typo_vendor(original["vendor_name"])
        # amount sometimes rounded, sometimes identical
        if random.random() < 0.6:
            original["amount"] = round(original["amount"] + random.choice([0, 0.01, -0.01, 1.0]), 2)
        # date drifts by a few days
        d = date.fromisoformat(original["invoice_date"]) + timedelta(days=random.randint(0, 5))
        original["invoice_date"] = d.isoformat()
        rows.append(original)

    random.shuffle(rows)
    return rows


def main(output_path="invoices.csv", n_base=8, n_exact_dupes=1, n_near_dupes=1, log=print):
    rows = build_invoices(n_base=n_base, n_exact_dupes=n_exact_dupes, n_near_dupes=n_near_dupes)
    fieldnames = ["invoice_id", "invoice_number", "vendor_name", "amount", "invoice_date", "description"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log(f"Generated {output_path} with {len(rows)} invoices "
        f"(includes deliberate exact + near duplicates).")
    return output_path


if __name__ == "__main__":
    main()
