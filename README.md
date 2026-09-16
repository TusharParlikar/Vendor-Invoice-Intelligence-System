# Vendor-Invoice-Intelligence-System
# Vendor Invoice Intelligence System

End-to-end machine learning system for vendor invoice cost forecasting and audit-risk triage. Built as a production-style project: SQL data layer → modular Python pipeline → saved model artifacts → inference layer → Streamlit web application.

**Two prediction modules:**

| Module | Type | Output |
|---|---|---|
| Freight Cost Prediction | Regression | Expected freight cost for an invoice |
| Invoice Risk Flagging | Binary Classification | Auto-approve vs. route for manual review |

---

## Table of Contents

- [Business Problem](#business-problem)
- [System Architecture](#system-architecture)
- [Data Source](#data-source)
- [Feature Engineering](#feature-engineering)
- [Modeling & Results](#modeling--results)
- [Application](#application)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Known Limitations](#known-limitations)
- [Tech Stack](#tech-stack)
- [Author](#author)

---

## Business Problem

**Invoice cost leakage and audit risk management.**

Finance teams reviewing vendor invoices face two recurring problems:

1. **Freight cost is unpredictable.** Freight is a non-trivial component of landed cost. Without an expected value, teams cannot budget accurately, estimate margin, or negotiate rates with vendors from a position of data.

2. **Manual invoice review does not scale.** Reviewing every invoice by hand is slow and gets slower as transaction volume grows. Abnormal freight charges, price deviations, and delivery delays usually indicate data-entry errors, disputes, or compliance risk — but they are buried among invoices that are perfectly fine.

This system addresses both: it predicts expected freight cost per invoice, and it flags only the invoices that actually need a human. Low-risk invoices are cleared for auto-approval, so reviewer attention concentrates where it adds value.

---

## System Architecture

```
SQLite (inventory.db)
        │
        ▼
  SQL aggregation + joins  ──►  Feature engineering
        │
        ▼
  EDA · correlation analysis · Welch's t-test feature screening
        │
        ├──────────────────────────────┐
        ▼                              ▼
  Regression pipeline           Classification pipeline
  (freight cost)                (invoice risk)
        │                              │
        ▼                              ▼
  Model comparison             Model comparison + GridSearchCV
        │                              │
        ▼                              ▼
  predict_freight_model.pkl     predict_flag_invoice.pkl + scaler.pkl
        │                              │
        └──────────────┬───────────────┘
                       ▼
              Inference layer (inference/)
                       │
                       ▼
              Streamlit portal (app.py)
```

Training is script-driven, not notebook-driven. Notebooks are exploratory only; every step that survived exploration was rewritten as a function and chained into `train.py`, so the pipeline can be re-run on a schedule for retraining.

---

## Data Source

Relational SQLite database (`data/inventory.db`) with the following tables:

| Table | Contents |
|---|---|
| `purchases` | Line-item purchase records — inventory ID, brand, vendor, PO number, PO date, receiving date, invoice date, purchase price, quantity, dollars |
| `purchase_prices` | Per-item purchase pricing by vendor |
| `vendor_invoice` | Vendor-generated invoice per PO — quantity, dollars, freight, invoice date, pay date, approval status |
| `begin_inventory` | Inventory position at start of period |
| `end_inventory` | Inventory position at end of period |

The two modeling tables are `vendor_invoice` (what the vendor billed) and `purchases` (what was actually ordered and received). The gap between them is the core signal for risk flagging.

---

## Feature Engineering

### Freight Cost Prediction

Direct features from `vendor_invoice`:

- `Quantity` — units on the invoice
- `Dollars` — invoice amount

Correlation with freight: quantity ≈ 0.94, dollars ≈ 0.96. The two predictors are themselves highly collinear (≈ 0.99), so the final model uses `Dollars` alone with no measurable loss in fit.

**Supporting analysis:** a derived `freight_per_unit` column split at the 25th and 75th quantiles of quantity shows bulk buyers pay materially less per unit in freight — roughly **$0.04/unit at high volume vs. $0.09/unit at low volume**. Outliers were retained; high-quantity vendors sit on the same regression line and represent real bulk behaviour, not data errors.

### Invoice Risk Flagging

Built by aggregating `purchases` to PO level and left-joining `vendor_invoice` on `PONumber`.

| Feature | Source | Meaning |
|---|---|---|
| `total_item_quantity` | `SUM(purchases.Quantity)` | Units actually ordered |
| `total_item_dollars` | `SUM(purchases.Dollars)` | Order value on the company side |
| `invoice_quantity` | `vendor_invoice.Quantity` | Units the vendor billed for |
| `invoice_dollars` | `vendor_invoice.Dollars` | Amount the vendor billed |
| `freight` | `vendor_invoice.Freight` | Freight charged |

**Target label** (`flag_invoice`): an invoice is flagged when the invoice-level total does not reconcile against the item-level total, or when the average receiving delay for that PO exceeds 10 days.

Class balance: 3,693 normal / 1,850 flagged — imbalanced but not severely.

**Feature screening.** Two passes removed noise:

1. **Welch's t-test** (`ttest_ind`, `equal_var=False`) comparing flagged vs. normal group means for every candidate metric. `days_to_pay` (35.42 vs. 35.49) and `total_brands` (42 vs. 40) returned p > 0.05 and were dropped as non-discriminative.
2. **Random Forest feature importance** ranked `total_brands` and `days_po_to_invoice` lowest; removing them lifted F1 by roughly one point.

`avg_receiving_delay` was excluded from the feature set deliberately — it feeds the labelling rule and is not reliably known at invoice-arrival time.

---

## Modeling & Results

### Regression — Freight Cost

Three candidates compared on an 80/20 split (`random_state=42`):

| Model | R² |
|---|---|
| **Linear Regression** | **0.97** |
| Random Forest Regressor | 0.963 |
| Decision Tree Regressor | 0.937 |

Linear Regression also produced the lowest MAE and RMSE. Depth tuning on the tree models (`max_depth` 2 → 5) never closed the gap: the relationship between invoice value and freight is close to linear, so the simplest model wins. **Selected: Linear Regression.**

Metrics: MAE, RMSE, R².

### Classification — Invoice Risk

| Model | Accuracy |
|---|---|
| Logistic Regression | ~0.66 |
| Decision Tree Classifier | ~0.81 |
| Random Forest Classifier | ~0.87 |
| **Random Forest + GridSearchCV** | **0.89** |

Scaling was tested both ways — `StandardScaler` and `MinMaxScaler` produced effectively identical results; StandardScaler was kept.

**Hyperparameter tuning:** `GridSearchCV`, 5-fold, scored on **F1** (precision/recall balance matters more than raw accuracy given the class imbalance). 216 candidate configurations.

Best parameters: `n_estimators=300`, `criterion='gini'`, `max_depth=None`, `min_samples_split=5`, `min_samples_leaf=1`.

**Confusion matrix improvement:** false positives on normal invoices dropped from **20/725 to 12/725** after tuning — a direct reduction in unnecessary manual review.

Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix.

---

## Application

`app.py` — a Streamlit portal, **Vendor Invoice Intelligence Portal**, with a sidebar radio selector that switches between the two modules.

**Freight Cost Prediction** — numeric input form → returns predicted freight cost as a metric card.

**Invoice Risk Flagging** — five-field numeric input form → returns either `Invoice requires manual approval` or `Invoice is safe for auto-approval`.

Both modules call the `inference/` layer, which loads the saved `.pkl` artifacts. The app contains no training code and no model logic of its own.

> Add screenshots to `images/` and link them here — recruiters spend very little time on a repo, and a screenshot does more than a paragraph.

---

## Project Structure

```
vendor-invoice-intelligence/
├── data/
│   └── inventory.db
├── notebooks/
│   ├── predicting_freight_cost.ipynb
│   └── invoice_flagging.ipynb
├── freight_cost_prediction/
│   ├── data_preprocessing.py
│   ├── modeling_evaluation.py
│   └── train.py
├── invoice_flagging/
│   ├── data_preprocessing.py
│   ├── modeling_evaluation.py
│   └── train.py
├── inference/
│   ├── predict_freight.py
│   └── predict_invoice_flag.py
├── models/
│   ├── predict_freight_model.pkl
│   ├── predict_flag_invoice.pkl
│   └── scaler.pkl
├── images/
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How to Run

**1. Clone and install**

```bash
git clone https://github.com/<your-username>/vendor-invoice-intelligence.git
cd vendor-invoice-intelligence
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Place the database**

Put `inventory.db` in `data/`.

**3. Train both models**

```bash
python freight_cost_prediction/train.py
python invoice_flagging/train.py
```

Artifacts are written to `models/`.

**4. Verify inference**

```bash
python inference/predict_freight.py
python inference/predict_invoice_flag.py
```

**5. Launch the app**

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Known Limitations

Stated openly, because these are the first things a reviewer will probe:

- **Label leakage.** The risk label is rule-derived, and one of its rules (invoice total vs. item total mismatch) uses the same columns that appear as model features. The classifier is therefore partly re-learning a rule it could execute directly. This is acceptable as a bootstrapping step to get a supervised dataset off the ground, but the honest next step is labels from actual reviewer decisions, or unsupervised clustering to discover risk groups rather than assert them.
- **No logging.** Pipeline steps are not instrumented. Adding Python `logging` to each stage is the obvious next improvement for tracking which step ran and where a run failed.
- **Basic deployment.** Models are saved locally as pickles and consumed directly by Streamlit. A FastAPI service in front of the artifacts, containerisation, and cloud hosting would be the production path.
- **No drift monitoring.** Retraining is manual (re-run `train.py`). There is no automated check on whether model performance degrades over time.
- **Single data snapshot.** Trained on one historical extract with no temporal validation split, so performance on genuinely future invoices is untested.

---

## Tech Stack

`Python` · `SQLite` · `SQL` · `pandas` · `NumPy` · `scikit-learn` · `SciPy` · `Matplotlib` · `Seaborn` · `joblib` · `Streamlit`

---

## Author

**<Your Name>**

- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: [your-linkedin](https://linkedin.com/in/your-linkedin)
- Email: your.email@example.com
