# Vendor Invoice Intelligence System

An end-to-end machine learning project for predicting vendor freight costs and identifying invoices that may need manual review.

The project has two parts:

| Module                  | Type           | What it does                       |
| ----------------------- | -------------- | ---------------------------------- |
| Freight Cost Prediction | Regression     | Predicts the expected freight cost |
| Invoice Risk Flagging   | Classification | Flags invoices for manual review   |

---

## Business Problem

When companies receive a large number of vendor invoices, checking everything manually can take a lot of time.

Two common problems are:

* Freight costs can vary depending on the invoice value and quantity.
* Some invoices may have mismatched amounts or unusual delays and need extra checking.

I built this project to handle both problems using machine learning.

The first model predicts the expected freight cost, while the second model identifies invoices that may need manual approval.

---

## System Architecture

```text
SQLite Database
      ↓
SQL Queries + Data Processing
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Saved Models
      ↓
Inference
      ↓
Streamlit App
```

The training code is separated from the Streamlit application.

The notebooks were mainly used for exploration and testing. The final preprocessing, training, and evaluation steps were moved into Python scripts so the models can be trained again when needed.

---

## Data

The project uses a SQLite database:

```text
data/inventory.db
```

Main tables used:

| Table             | Description                        |
| ----------------- | ---------------------------------- |
| `purchases`       | Purchase and receiving information |
| `purchase_prices` | Purchase prices by vendor          |
| `vendor_invoice`  | Vendor invoice information         |
| `begin_inventory` | Beginning inventory                |
| `end_inventory`   | Ending inventory                   |

For invoice risk prediction, I mainly used `purchases` and `vendor_invoice`.

The purchase data represents what the company ordered, while the invoice data represents what the vendor billed.

---

## Feature Engineering

### Freight Cost Prediction

I initially looked at:

* `Quantity`
* `Dollars`

Both had a strong correlation with freight cost.

However, `Quantity` and `Dollars` were also highly correlated with each other, so I used **`Dollars`** as the main feature for the final model.

I also looked at freight cost per unit and found that larger purchases generally had a lower freight cost per unit.

---

### Invoice Risk Flagging

Purchase data was grouped by `PONumber` and then joined with the vendor invoice data.

The main features were:

* `total_item_quantity`
* `total_item_dollars`
* `invoice_quantity`
* `invoice_dollars`
* `freight`

An invoice was marked as risky when the invoice did not match the purchase information or when the receiving delay was more than 10 days.

The dataset contained:

* **3,693 normal invoices**
* **1,850 flagged invoices**

I also used statistical testing and Random Forest feature importance to remove features that were not useful.

---

## Model Results

### Freight Cost Prediction

I compared three models:

| Model                 |       R² |
| --------------------- | -------: |
| **Linear Regression** | **0.97** |
| Random Forest         |    0.963 |
| Decision Tree         |    0.937 |

Linear Regression performed the best, so I selected it for the final model.

The model was evaluated using:

* MAE
* RMSE
* R²

---

### Invoice Risk Flagging

I compared:

| Model                   |  Accuracy |
| ----------------------- | --------: |
| Logistic Regression     |     ~0.66 |
| Decision Tree           |     ~0.81 |
| Random Forest           |     ~0.87 |
| **Tuned Random Forest** | **~0.89** |

For the Random Forest, I used `GridSearchCV` with 5-fold cross-validation and optimized for F1 score.

The final model used:

```text
n_estimators = 300
criterion = gini
max_depth = None
min_samples_split = 5
min_samples_leaf = 1
```

After tuning, false positives on normal invoices decreased from **20 to 12 out of 725**.

---

## Streamlit Application

The project includes a simple Streamlit application called:

**Vendor Invoice Intelligence Portal**

It has two options.

### Freight Cost Prediction

Enter the invoice information and the application predicts the expected freight cost.

### Invoice Risk Flagging

Enter the invoice details and the application returns:

```text
Invoice requires manual approval
```

or

```text
Invoice is safe for auto-approval
```

The application loads the saved models through the `inference/` folder. Training is not performed inside the application.

---

## Project Structure

```text
vendor-invoice-intelligence/
│
├── data/
│   └── inventory.db
│
├── notebooks/
│   ├── predicting_freight_cost.ipynb
│   └── invoice_flagging.ipynb
│
├── freight_cost_prediction/
│   ├── data_preprocessing.py
│   ├── modeling_evaluation.py
│   └── train.py
│
├── invoice_flagging/
│   ├── data_preprocessing.py
│   ├── modeling_evaluation.py
│   └── train.py
│
├── inference/
│   ├── predict_freight.py
│   └── predict_invoice_flag.py
│
├── models/
│   ├── predict_freight_model.pkl
│   ├── predict_flag_invoice.pkl
│   └── scaler.pkl
│
├── images/
├── app.py
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Clone the project

```bash
git clone https://github.com/<your-username>/vendor-invoice-intelligence.git
cd vendor-invoice-intelligence
```

### 2. Create environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

### 4. Add database

Put `inventory.db` inside:

```text
data/
```

### 5. Train the models

```bash
python freight_cost_prediction/train.py
python invoice_flagging/train.py
```

### 6. Run the application

```bash
streamlit run app.py
```

The application will open at:

```text
http://localhost:8501
```

---

## Limitations

* The invoice-risk labels are generated using predefined rules, so there is some label leakage.
* The models were trained on one historical dataset.
* No temporal validation was performed.
* There is currently no model-drift monitoring.
* Models are stored locally using `joblib`.
* Logging and automated retraining can be added later.

---

## Tech Stack

`Python` · `SQL` · `SQLite` · `Pandas` · `NumPy` · `Scikit-learn` · `SciPy` · `Matplotlib` · `Seaborn` · `Joblib` · `Streamlit`

---



Email: `your.email@example.com`
