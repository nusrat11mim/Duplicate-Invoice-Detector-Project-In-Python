import os
import queue
import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox

import joblib
import pandas as pd

import generate_sample_data
import train_model
import detect_duplicates as dd

APP_TITLE = "Duplicate Invoice Detector"

# ---------------------------------------------------------------- palette
NAVY = "#1E2761"
NAVY_DARK = "#141A47"
ICE = "#CADCFC"
ICE_LIGHT = "#EAF0FB"
PAGE_BG = "#F2F4F9"
CARD_BG = "#FFFFFF"
TEXT_DARK = "#22283C"
TEXT_MUTED = "#5B6472"
GOLD = "#E3A857"
GOLD_HOVER = "#EDC079"
RED_BG = "#F8D7DA"
RED_FG = "#8B2E3B"
AMBER_BG = "#FFF3CD"
AMBER_FG = "#8A6100"

RESULT_COLUMNS = [
    ("verdict", "Verdict", 100),
    ("duplicate_probability", "Prob.", 70),
    ("invoice_id_a", "ID A", 60),
    ("vendor_a", "Vendor A", 160),
    ("amount_a", "Amount A", 80),
    ("date_a", "Date A", 90),
    ("invoice_id_b", "ID B", 60),
    ("vendor_b", "Vendor B", 160),
    ("amount_b", "Amount B", 80),
    ("date_b", "Date B", 90),
]


class DuplicateInvoiceGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1220x780")
        self.minsize(980, 620)
        self.configure(background=PAGE_BG)

        # state shared between the worker thread and the UI thread
        self.msg_queue = queue.Queue()
        self.invoices_path = tk.StringVar(value=os.path.abspath("invoices.csv"))
        self.model_path = tk.StringVar(value=os.path.abspath("invoice_dup_model.joblib"))
        self.report_path = tk.StringVar(value=os.path.abspath("duplicate_report.csv"))

        self.duplicate_threshold = tk.DoubleVar(value=dd.DUPLICATE_THRESHOLD)
        self.review_threshold = tk.DoubleVar(value=dd.REVIEW_THRESHOLD)
        self.invoice_count = tk.IntVar(value=10)  # total sample invoices to generate

        self.all_results = []      # every scored pair (>= review threshold used at run time)
        self.model = None          # cached trained model, so re-scoring doesn't reload from disk

        self._setup_style()
        self._build_layout()
        self._poll_queue()

    # -------------------------------------------------------------- style
    def _setup_style(self):
        available = set(tkfont.families())
        body_font = "Segoe UI" if "Segoe UI" in available else "Helvetica"
        self.font_body = (body_font, 10)
        self.font_body_bold = (body_font, 10, "bold")
        self.font_small = (body_font, 9)
        self.font_mono = "Consolas" if "Consolas" in available else "Courier New"

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", background=PAGE_BG, foreground=TEXT_DARK, font=self.font_body)
        style.configure("TFrame", background=PAGE_BG)
        style.configure("Card.TFrame", background=CARD_BG)
        style.configure("Header.TFrame", background=NAVY)
        style.configure("Footer.TFrame", background=NAVY_DARK)

        style.configure("TLabel", background=PAGE_BG, foreground=TEXT_DARK, font=self.font_body)
        style.configure("Card.TLabel", background=CARD_BG, foreground=TEXT_DARK, font=self.font_body)
        style.configure("Muted.Card.TLabel", background=CARD_BG, foreground=TEXT_MUTED, font=self.font_small)
        style.configure("CardTitle.TLabel", background=CARD_BG, foreground=NAVY,
                         font=(body_font, 12, "bold"))
        style.configure("Header.TLabel", background=NAVY, foreground="white",
                         font=(body_font, 18, "bold"))
        style.configure("HeaderSub.TLabel", background=NAVY, foreground=ICE, font=self.font_small)
        style.configure("Footer.TLabel", background=NAVY_DARK, foreground=ICE, font=self.font_small)
        style.configure("Summary.TLabel", background=CARD_BG, foreground=NAVY, font=self.font_body_bold)

        style.configure("TButton", font=self.font_small, padding=(10, 6),
                         background=ICE_LIGHT, foreground=NAVY, borderwidth=0)
        style.map("TButton",
                  background=[("active", ICE), ("disabled", "#EDEFF4")],
                  foreground=[("disabled", "#A7ADBB")])

        style.configure("Accent.TButton", font=(body_font, 10, "bold"), padding=(14, 8),
                         background=GOLD, foreground=NAVY_DARK, borderwidth=0)
        style.map("Accent.TButton",
                  background=[("active", GOLD_HOVER), ("disabled", "#E7DAC3")],
                  foreground=[("disabled", "#A7A090")])

        style.configure("Secondary.TButton", font=self.font_small, padding=(10, 6),
                         background=NAVY, foreground="white", borderwidth=0)
        style.map("Secondary.TButton",
                  background=[("active", NAVY_DARK), ("disabled", "#B9BEC9")])

        style.configure("TEntry", fieldbackground="white", foreground=TEXT_DARK,
                         padding=5, borderwidth=1, relief="solid")
        style.configure("TSpinbox", fieldbackground="white", foreground=TEXT_DARK, padding=4)

        style.configure("Horizontal.TScale", background=CARD_BG, troughcolor=ICE_LIGHT)

        style.configure("Treeview", background="white", fieldbackground="white",
                         foreground=TEXT_DARK, rowheight=27, font=self.font_small, borderwidth=0)
        style.configure("Treeview.Heading", background=NAVY, foreground="white",
                         font=self.font_body_bold, padding=(6, 6), relief="flat")
        style.map("Treeview.Heading", background=[("active", NAVY_DARK)])
        style.map("Treeview", background=[("selected", ICE)], foreground=[("selected", NAVY_DARK)])

    # ------------------------------------------------------------------ UI
    def _build_layout(self):
        self._build_menu()
        self._build_header()

        body = ttk.Frame(self, style="TFrame")
        body.pack(fill="both", expand=True, padx=16, pady=(12, 8))

        self._build_card(body, "Pipeline", self._fill_pipeline_card, pady=(0, 10))
        self._build_card(body, "Verdict Thresholds  \u2014  drag to re-score instantly, no retraining",
                          self._fill_thresholds_card, pady=(0, 10))

        self._build_card(body, "Results", self._fill_results_card, pady=(0, 10), expand=True)

        self._build_card(body, "Log", self._fill_log_card, pady=(0, 0), fixed_height=150)

        self._build_footer()

    def _build_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="File Locations...", command=self._open_file_locations_dialog)
        menubar.add_cascade(label="Settings", menu=file_menu)
        self.config(menu=menubar)

    def _open_file_locations_dialog(self):
        dialog = tk.Toplevel(self)
        dialog.title("File Locations")
        dialog.configure(background=CARD_BG)
        dialog.transient(self)
        dialog.geometry("680x230")
        dialog.resizable(False, False)
        content = ttk.Frame(dialog, style="Card.TFrame")
        content.pack(fill="both", expand=True, padx=18, pady=18)
        self._fill_files_card(content)
        ttk.Button(content, text="Close", command=dialog.destroy).pack(anchor="e", pady=(10, 0))
        dialog.grab_set()

    def _build_header(self):
        header = ttk.Frame(self, style="Header.TFrame")
        header.pack(fill="x")
        inner = ttk.Frame(header, style="Header.TFrame")
        inner.pack(fill="x", padx=20, pady=14)
        ttk.Label(inner, text="\U0001F9FE  Duplicate Invoice Detector", style="Header.TLabel").pack(anchor="w")
        ttk.Label(inner, text="AI-powered detection of exact & near-duplicate invoices",
                  style="HeaderSub.TLabel").pack(anchor="w", pady=(2, 0))

    def _build_footer(self):
        footer = ttk.Frame(self, style="Footer.TFrame")
        footer.pack(fill="x", side="bottom")
        self.status_var = tk.StringVar(value="\u25CF Ready")
        self.status_label = ttk.Label(footer, textvariable=self.status_var, style="Footer.TLabel")
        self.status_label.pack(side="left", padx=16, pady=6)

    def _build_card(self, parent, title_text, fill_fn, pady=(0, 10), expand=False, fixed_height=None):
        card = tk.Frame(parent, background=CARD_BG, highlightthickness=1,
                         highlightbackground="#E1E6F0", highlightcolor="#E1E6F0", bd=0)
        card.pack(fill="both" if expand else "x", expand=expand, pady=pady)
        if fixed_height:
            card.pack_propagate(False)
            card.configure(height=fixed_height)

        title_row = ttk.Frame(card, style="Card.TFrame")
        title_row.pack(fill="x", padx=16, pady=(12, 6))
        ttk.Label(title_row, text=title_text, style="CardTitle.TLabel").pack(anchor="w")

        if fill_fn is not None:
            content = ttk.Frame(card, style="Card.TFrame")
            content.pack(fill="both" if expand else "x", expand=expand, padx=16, pady=(0, 14))
            fill_fn(content)
        return card

    def _fill_files_card(self, parent):
        self._build_file_row(parent, "Invoices CSV", self.invoices_path, self._browse_invoices)
        self._build_file_row(parent, "Model file", self.model_path, self._browse_model)
        self._build_file_row(parent, "Report output", self.report_path, self._browse_report)

    def _fill_pipeline_card(self, parent):
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x")
        ttk.Button(row, text="1. Generate Sample Data",
                   command=self.on_generate_data).pack(side="left")
        ttk.Label(row, text="Count:", style="Card.TLabel").pack(side="left", padx=(10, 4))
        ttk.Spinbox(row, from_=5, to=100000, width=7,
                    textvariable=self.invoice_count).pack(side="left")
        ttk.Button(row, text="2. Train Model",
                   command=self.on_train_model).pack(side="left", padx=(10, 0))
        ttk.Button(row, text="3. Run Detection",
                   command=self.on_run_detection).pack(side="left", padx=(10, 0))
        ttk.Button(row, text="Run All Steps", style="Accent.TButton",
                   command=self.on_run_all).pack(side="left", padx=(16, 0))
        ttk.Button(row, text="Export Shown Results...", style="Secondary.TButton",
                   command=self.on_export).pack(side="right")

    def _fill_thresholds_card(self, parent):
        self._build_threshold_row(parent, "Duplicate threshold", self.duplicate_threshold, RED_FG)
        self._build_threshold_row(parent, "Needs-review threshold", self.review_threshold, AMBER_FG)

    def _fill_results_card(self, content):
        self.summary_var = tk.StringVar(value="No results yet.")
        ttk.Label(content, textvariable=self.summary_var, style="Summary.TLabel").pack(anchor="w", pady=(0, 8))

        table_frame = ttk.Frame(content, style="Card.TFrame")
        table_frame.pack(fill="both", expand=True)

        columns = [c[0] for c in RESULT_COLUMNS]
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")
        for key, label, width in RESULT_COLUMNS:
            self.tree.heading(key, text=label, command=lambda k=key: self._sort_by(k))
            self.tree.column(key, width=width, anchor="center")

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        self.tree.tag_configure("DUPLICATE", background=RED_BG, foreground=RED_FG)
        self.tree.tag_configure("NEEDS_REVIEW", background=AMBER_BG, foreground=AMBER_FG)

    def _fill_log_card(self, parent):
        self.log_text = tk.Text(parent, state="disabled", wrap="word", background=NAVY_DARK,
                                 foreground=ICE, insertbackground=ICE, relief="flat",
                                 font=(self.font_mono, 9), padx=10, pady=8)
        log_vsb = ttk.Scrollbar(parent, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_vsb.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_vsb.pack(side="right", fill="y")

    def _build_file_row(self, parent, label, var, browse_cmd):
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label, width=14, style="Card.TLabel").pack(side="left")
        ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(row, text="Browse...", command=browse_cmd).pack(side="left")

    def _build_threshold_row(self, parent, label, var, accent_color):
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=6)
        ttk.Label(row, text=label, width=22, style="Card.TLabel").pack(side="left")
        value_lbl = tk.Label(row, text=f"{var.get():.2f}", width=5, background=CARD_BG,
                              foreground=accent_color, font=self.font_body_bold)

        def on_change(_evt=None):
            value_lbl.configure(text=f"{var.get():.2f}")
            self._rescore_and_render()

        scale = ttk.Scale(row, from_=0.0, to=1.0, variable=var, orient="horizontal",
                           command=lambda _v: on_change())
        scale.pack(side="left", fill="x", expand=True, padx=10)
        value_lbl.pack(side="left")

    # ------------------------------------------------------------ browsing
    def _browse_invoices(self):
        path = filedialog.askopenfilename(title="Select invoices CSV",
                                           filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.invoices_path.set(path)

    def _browse_model(self):
        path = filedialog.asksaveasfilename(title="Model file", defaultextension=".joblib",
                                             filetypes=[("Joblib model", "*.joblib")])
        if path:
            self.model_path.set(path)

    def _browse_report(self):
        path = filedialog.asksaveasfilename(title="Report output CSV", defaultextension=".csv",
                                             filetypes=[("CSV files", "*.csv")])
        if path:
            self.report_path.set(path)

    # ------------------------------------------------------------- actions
    def on_generate_data(self):
        self._run_in_background(self._task_generate_data)

    def on_train_model(self):
        self._run_in_background(self._task_train_model)

    def on_run_detection(self):
        self._run_in_background(self._task_run_detection)

    def on_run_all(self):
        def all_steps():
            self._task_generate_data()
            self._task_train_model()
            self._task_run_detection()
        self._run_in_background(all_steps)

    def on_export(self):
        if not self.all_results:
            messagebox.showinfo(APP_TITLE, "No results to export yet -- run detection first.")
            return
        path = filedialog.asksaveasfilename(title="Export results", defaultextension=".csv",
                                             filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        pd.DataFrame(self._visible_rows()).to_csv(path, index=False)
        messagebox.showinfo(APP_TITLE, f"Exported {len(self._visible_rows())} rows to:\n{path}")

    # --------------------------------------------------------- background
    def _run_in_background(self, fn):
        self._set_busy(True)
        thread = threading.Thread(target=self._safe_run, args=(fn,), daemon=True)
        thread.start()

    def _safe_run(self, fn):
        try:
            fn()
        except FileNotFoundError as e:
            self.msg_queue.put(("error", f"File not found: {e}"))
        except Exception as e:  # noqa: BLE001 -- surface any failure to the GUI instead of crashing the thread
            self.msg_queue.put(("error", f"{type(e).__name__}: {e}"))
        finally:
            self.msg_queue.put(("done", None))

    def _task_generate_data(self):
        path = self.invoices_path.get()
        total = max(3, self.invoice_count.get())
        # keep roughly the original 80% base / 8% exact-dup / 12% near-dup
        # split, but always generate at least one of each so small counts
        # still contain something for the detector to find.
        n_exact = max(1, round(total * 0.08))
        n_near = max(1, round(total * 0.12))
        n_base = max(1, total - n_exact - n_near)
        generate_sample_data.main(output_path=path, n_base=n_base,
                                   n_exact_dupes=n_exact, n_near_dupes=n_near, log=self._log)

    def _task_train_model(self):
        path = self.model_path.get()
        train_model.main(model_path=path, log=self._log)

    def _task_run_detection(self):
        model = joblib.load(self.model_path.get())
        df = dd.load_invoices(self.invoices_path.get())
        self._log(f"Loaded {len(df)} invoices from {self.invoices_path.get()}")

        pairs = dd.candidate_pairs(df)
        self._log(f"Generated {len(pairs)} candidate pairs after blocking (instead of "
                  f"{len(df) * (len(df) - 1) // 2} total possible pairs)")

        # score at review_threshold = 0 so every pair is retained; the GUI
        # applies the actual thresholds live when rendering, so dragging
        # the sliders afterwards never needs to reload the model or re-run
        # feature extraction.
        self.model = model
        self.all_results = dd.score_pairs(pairs, model, review_threshold=0.0,
                                           duplicate_threshold=self.duplicate_threshold.get())
        self.msg_queue.put(("render", None))
        self._log(f"Scored {len(self.all_results)} pairs. Adjust thresholds above to filter the table.")

    # ------------------------------------------------------------ queue
    def _log(self, message):
        self.msg_queue.put(("log", str(message)))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "error":
                    self._append_log(f"ERROR: {payload}")
                    messagebox.showerror(APP_TITLE, payload)
                elif kind == "render":
                    self._rescore_and_render()
                elif kind == "done":
                    self._set_busy(False)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _append_log(self, message):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_busy(self, busy):
        self.status_var.set("\u25CF Working..." if busy else "\u25CF Ready")
        state = "disabled" if busy else "normal"
        for child in self.winfo_children():
            self._set_state_recursive(child, state)

    def _set_state_recursive(self, widget, state):
        if isinstance(widget, (ttk.Button, ttk.Entry, ttk.Scale, ttk.Spinbox)):
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass
        for child in widget.winfo_children():
            self._set_state_recursive(child, state)

    # ------------------------------------------------------------ render
    def _visible_rows(self):
        dup_th = self.duplicate_threshold.get()
        rev_th = self.review_threshold.get()
        rows = []
        for r in self.all_results:
            if r["duplicate_probability"] < rev_th:
                continue
            row = dict(r)
            row["verdict"] = "DUPLICATE" if r["duplicate_probability"] >= dup_th else "NEEDS_REVIEW"
            rows.append(row)
        rows.sort(key=lambda r: r["duplicate_probability"], reverse=True)
        return rows

    def _rescore_and_render(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        rows = self._visible_rows()
        for r in rows:
            values = [r.get(key, "") for key, _, _ in RESULT_COLUMNS]
            values[1] = f"{r['duplicate_probability']:.2%}"
            self.tree.insert("", "end", values=values, tags=(r["verdict"],))

        n_dup = sum(1 for r in rows if r["verdict"] == "DUPLICATE")
        n_review = sum(1 for r in rows if r["verdict"] == "NEEDS_REVIEW")
        self.summary_var.set(f"Showing {len(rows)} pairs   \u2022   Duplicate: {n_dup}   \u2022   Needs review: {n_review}")

    def _sort_by(self, key):
        rows = self._visible_rows()
        reverse = getattr(self, "_last_sort", None) == (key, False)
        rows.sort(key=lambda r: r.get(key, ""), reverse=reverse)
        self._last_sort = (key, reverse)
        for item in self.tree.get_children():
            self.tree.delete(item)
        for r in rows:
            values = [r.get(k, "") for k, _, _ in RESULT_COLUMNS]
            values[1] = f"{r['duplicate_probability']:.2%}"
            self.tree.insert("", "end", values=values, tags=(r["verdict"],))


def main():
    app = DuplicateInvoiceGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
