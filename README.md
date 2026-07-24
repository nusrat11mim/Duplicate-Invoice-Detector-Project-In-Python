# Duplicate Invoice Detector (AI-powered)

Detects **exact and near-duplicate invoices** (typo'd vendor names, rounded
amounts, dates a few days apart) using a machine-learning classifier
(RandomForest) trained on invoice-pair similarity features.

Comes with both a **command-line pipeline** (`main.py`) and a **desktop GUI**
(`gui.py`) built on the same code.

## Project structure
```
duplicate_invoice_detector/
├── main.py                  <- CLI entry point
├── gui.py                    <- Desktop GUI entry point (Tkinter)
├── generate_sample_data.py  <- creates sample invoices.csv (with seeded duplicates)
├── features.py               <- turns an invoice PAIR into ML features
├── train_model.py            <- trains + saves the RandomForest model
├── detect_duplicates.py      <- blocking + scoring + report generation
└── requirements.txt
```

## How to run in VS Code

1. Open this folder in VS Code: `File > Open Folder...`
2. Open a terminal in VS Code: `` Ctrl+` `` (backtick)
3. (Recommended) create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Run it:
   - **GUI** (recommended for exploring results interactively):
     ```
     python gui.py
     ```
   - **Command line**:
     ```
     python main.py
     ```

### Using the GUI

`gui.py` opens a window with:
- File pickers for the invoices CSV, model file, and report output path
  (defaults match the CLI: `invoices.csv`, `invoice_dup_model.joblib`,
  `duplicate_report.csv`).
- Three buttons matching the three pipeline steps — **Generate Sample
  Data**, **Train Model**, **Run Detection** — plus a **Run All Steps**
  button that does all three in order (same as `python main.py`).
- A results table (color-coded: red = duplicate, yellow = needs review)
  that you can sort by clicking any column header.
- Two sliders for the Duplicate / Needs-review probability thresholds.
  Dragging them re-filters and re-labels the table **instantly**, without
  retraining or re-scoring — detection only re-runs when you click
  **Run Detection** again.
- An **Export Shown Results...** button to save whatever's currently
  in the table (after your threshold adjustments) to a CSV of your choice.
- A log panel at the bottom showing the same progress messages the CLI
  prints to the terminal.

All the heavy work (training, scoring) runs on a background thread so the
window stays responsive.

> **Note:** Tkinter ships with the standard Python installer on Windows and
> macOS. On Linux, if `python gui.py` complains it can't find `tkinter`,
> install it via your package manager, e.g. `sudo apt install python3-tk`
> (Debian/Ubuntu) or the equivalent for your distro.

### Command-line pipeline

Running `python main.py` will, on first run:
- generate `invoices.csv` (150 sample invoices with deliberate duplicates)
- train the ML model and save `invoice_dup_model.joblib`
- run detection and save `duplicate_report.csv`, printing a summary in the terminal

## Using your own data

Replace `invoices.csv` with your own export containing these columns:

| column          | required | example                |
|-----------------|----------|------------------------|
| invoice_id      | yes      | 1001                   |
| invoice_number  | yes      | INV-48213              |
| vendor_name     | yes      | Acme Supplies Ltd      |
| amount          | yes      | 1245.50                |
| invoice_date    | yes      | 2024-03-15 (YYYY-MM-DD)|
| description     | no       | Office supplies        |

Then just re-run:
```
python detect_duplicates.py
```
(no need to regenerate sample data or retrain — the trained model is reusable)

## How the detection works

1. **Blocking** — instead of comparing every invoice to every other invoice
   (slow and noisy at scale), invoices are grouped by vendor initial + a
   $50 amount bucket. Only invoices in the same/adjacent bucket are compared.
2. **Feature extraction** (`features.py`) — each candidate pair gets scored on:
   - vendor name similarity (fuzzy text matching)
   - invoice number similarity
   - description similarity
   - amount difference %
   - exact amount match flag
   - date closeness
3. **ML scoring** — a RandomForestClassifier (trained on thousands of
   synthetic labeled examples in `train_model.py`) outputs a duplicate
   probability for each pair.
4. **Verdict thresholds** (tune in `detect_duplicates.py`):
   - `>= 0.60` → `DUPLICATE`
   - `>= 0.35` → `NEEDS_REVIEW`
   - below → not flagged

## Tuning

- Change `DUPLICATE_THRESHOLD` / `REVIEW_THRESHOLD` in `detect_duplicates.py`
  to be stricter or looser.
- Change `n_pairs` in `train_model.py` for a bigger/smaller training set.
- Adjust the `$50` amount bucket size in `detect_duplicates.py`'s
  `candidate_pairs()` if your invoice amounts are much larger/smaller.
