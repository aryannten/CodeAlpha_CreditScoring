# CodeAlpha Credit Scoring System
[![CodeAlpha](https://img.shields.io/badge/CodeAlpha-ML%20Internship-blue)](https://github.com/aryannten)
[![Python](https://img.shields.io/badge/Python-3.14+-green)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.1-orange)](https://scikit-learn.org)

## Project Overview
This project predicts an individual's creditworthiness—specifically whether a loan applicant is likely to default—using machine learning classification pipelines trained on past financial history and demographic attributes.

### Key Objectives
- **Binary Classification**: Accurate discrimination between defaulting (positive / bad credit) and non-defaulting (negative / good credit) applicants.
- **Dual-Dataset Architecture with Automatic Fallback**:
  1. *Primary Dataset*: Kaggle **Give Me Some Credit** (`cs-training.csv`, 150,000 samples, ~6.7% default rate).
  2. *Secondary Dataset (Fallback)*: UCI / OpenML **German Credit** (`credit-g`, 1,000 samples, 30.0% default rate).
- **Leakage-Free Imbalance Handling**: Built-in cost-sensitive weighting (`class_weight='balanced'`) vs. synthetic oversampling (**SMOTE**) strictly isolated within cross-validation training folds.
- **Explainability & Governance**: Feature importance ranking and game-theoretic **SHAP** explanations for lending compliance.
- **Interactive Production App**: Streamlit web application for real-time risk assessment and decision threshold tuning.

---

## Methodology & Pipeline Architecture

1. **Robust Data Ingestion**:
   - `src.data_loader.load_dataset(name="auto")` automatically discovers local mirrors, remote endpoints, or OpenML fallbacks with zero manual setup.
2. **Exploratory Data Analysis (EDA)**:
   - Comprehensive analysis across feature distributions, target class imbalance, correlation heatmaps, and target-split boxplots (saved to `reports/figures/`).
3. **Data Cleaning & Missing Value Management**:
   - Median imputation paired with boolean indicator flags (`_was_missing`) to preserve predictive missingness signals without data loss.
   - Outlier clipping at the 1st/99th percentiles (e.g., revolving utilization capped at [0, 1]).
   - Domain anomalies (e.g., applicant age <= 0) treated as missing and imputed.
4. **Domain Feature Engineering** (`src/features.py`):
   - `total_past_due_count`: Aggregate sum of 30-59, 60-89, and 90+ days past-due incidents.
   - `has_any_late_payment`: Binary indicator of adverse credit history.
   - `income_per_dependent`: Household income normalized by family size ($MonthlyIncome / (Dependents + 1)$).
   - `monthly_installment_burden`: Ratio of total loan amount to loan duration ($credit\_amount / duration$).
   - `age_bucket`: Borrower career stage (`Young <30`, `Adult 30-60`, `Senior >60`).
   - `log_*`: Logarithmic transforms ($\log(1+x)$) on heavy-tailed income, debt, and loan amounts.
5. **Stratified Partitioning**:
   - 80/20 train/test split preserving target class ratios (`random_state=42`).
6. **Imbalance Handling Inside Cross-Validation Folds**:
   - *(a) Cost-Sensitive Learning*: `class_weight='balanced'` penalizing minority errors proportionally to class frequencies.
   - *(b) SMOTE Oversampling*: Synthesizing minority instances using `imblearn.pipeline.Pipeline`, preventing information leakage into validation/test splits.
7. **Model Selection & Hyperparameter Optimization**:
   - Evaluated algorithms: **Logistic Regression**, **Decision Tree**, **Random Forest**, and **XGBoost**.
   - Tuned via `RandomizedSearchCV` with 5-fold stratified cross-validation optimizing for **ROC-AUC**.
8. **Evaluation Metrics**:
   - Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and Accuracy.
9. **Decision Threshold Tuning**:
   - In lending economics, false negatives (granting loans to defaulters) cause substantial balance sheet losses compared to false positives. Lowering the decision cutoff (e.g. to 0.40) maximizes recall and default prevention.
10. **Explainability & Interpretability**:
    - Global feature importances / logistic coefficients and SHAP summary plots highlighting the top 10 credit risk drivers.
11. **Serialization & Deployment**:
    - Best performing pipeline serialized to `models/best_model.joblib`.
    - Live scoring via Python inference API (`src.predict.score_applicant`) and Streamlit web app (`app.py`).

---

## Benchmark Results

### Imbalance Strategy Comparison
| Model | Imbalance Strategy | CV ROC-AUC | Test ROC-AUC |
|---|---|:---:|:---:|
| Logistic Regression | class_weight_balanced | 0.7910 | 0.7779 |
| Logistic Regression | smote | 0.7835 | 0.7640 |
| Decision Tree | class_weight_balanced | 0.7221 | 0.6886 |
| Decision Tree | smote | 0.7077 | 0.7458 |
| Random Forest | class_weight_balanced | 0.8025 | 0.7745 |
| Random Forest | smote | 0.7978 | **0.7813** |
| XGBoost | smote | 0.7941 | 0.7783 |

**Top Performer:** Random Forest with SMOTE oversampling achieved the highest discriminatory power on the held-out test set with **0.7813 Test ROC-AUC** and **0.8979 PR-AUC**.

### Full Performance Metrics (`reports/comparison.csv`)
| Model | Imbalance | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | CV ROC-AUC | Test ROC-AUC |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | class_weight_balanced | 0.6900 | 0.8545 | 0.6714 | 0.7520 | 0.7779 | 0.8882 | 0.7910 | 0.7779 |
| Logistic Regression | smote | 0.6900 | 0.8545 | 0.6714 | 0.7520 | 0.7640 | 0.8744 | 0.7835 | 0.7640 |
| Decision Tree | class_weight_balanced | 0.6150 | 0.8987 | 0.5071 | 0.6484 | 0.6886 | 0.8083 | 0.7221 | 0.6886 |
| Decision Tree | smote | 0.7150 | 0.8430 | 0.7286 | 0.7816 | 0.7458 | 0.8438 | 0.7077 | 0.7458 |
| Random Forest | class_weight_balanced | 0.6950 | 0.8319 | 0.7071 | 0.7645 | 0.7745 | 0.8923 | 0.8025 | 0.7745 |
| Random Forest | smote | 0.7050 | 0.7755 | 0.8143 | 0.7944 | 0.7813 | 0.8979 | 0.7978 | 0.7813 |
| XGBoost | smote | 0.7300 | 0.8071 | 0.8071 | 0.8071 | 0.7783 | 0.8885 | 0.7941 | 0.7783 |

---

## Visualizations Generated (`reports/figures/`)
- **Confusion Matrices**: Saved for all 7 model configurations (`confusion_matrix_*.png`).
- **ROC Curves**: Multi-model receiver operating characteristic comparison (`roc_curves.png`).
- **Precision-Recall Curves**: PR curves reflecting imbalance dynamics (`pr_curve.png`).
- **Threshold Sensitivity Curve**: Trade-off plot between precision, recall, and decision cutoff (`threshold_tuning.png`).
- **Feature Importance**: Bar chart of the top 10 most influential risk drivers (`feature_importance.png`).
- **SHAP Summary**: Beehive attribution plot explaining individual decision impact (`shap_summary.png`).
- **EDA Visuals**: Target class distribution, correlation matrix heatmap, boxplots, and kernel density estimates.

---

## Key Findings & Insights
1. **The Accuracy Paradox in Lending**: In portfolios with low default rates (~7%), a trivial model predicting 100% approvals achieves 93% accuracy but catches 0% of defaulters. Tracking **ROC-AUC**, **PR-AUC**, and **Recall** is critical.
2. **SMOTE Inside CV Pipelines**: Applying SMOTE solely on training folds raised default detection recall by up to 30 percentage points without test leakage.
3. **Primary Risk Drivers**: Past delinquent payment counts (`total_past_due_count`), revolving credit utilization, duration of loan, and checking account status provided the strongest discriminatory signals.
4. **Threshold Optimization**: Calibrating the decision threshold to 0.40 (down from 0.50) substantially caught more high-risk applicants, directly mitigating institutional credit loss.

---

## How to Run

### 1. Clone & Setup Environment
```bash
git clone https://github.com/aryannten/CodeAlpha_CreditScoring.git
cd CodeAlpha_CreditScoring
python -m pip install -r requirements.txt
```

### 2. Train and Tune Models
Run the complete training pipeline across all 4 algorithms and both imbalance strategies:
```bash
python -m src.train
```
This executes 5-fold stratified cross-validation, saves artifacts to `models/best_model.joblib`, generates metrics in `reports/comparison.csv`, and writes metadata to `reports/training_results.json`.

### 3. Launch the Streamlit Web Application
Run the interactive user portal with real-time scoring and adjustable threshold controls:
```bash
streamlit run app.py
```

### 4. Build and Execute the Jupyter Notebook
Recreate and execute the end-to-end narrative notebook (`notebooks/credit_scoring.ipynb`):
```bash
python build_notebook.py
```
Or explore the notebook interactively:
```bash
jupyter notebook notebooks/credit_scoring.ipynb
```

### 5. Real-Time Applicant Scoring in Python
Score applicant profiles programmatically via the inference API:
```python
from src.predict import score_applicant

applicant = {
    "checking_status": "<0",
    "duration": 24,
    "credit_history": "critical",
    "purpose": "car",
    "credit_amount": 3500,
    "savings_status": "<100",
    "employment": "1<=X<4",
    "installment_commitment": 4,
    "personal_status": "male single",
    "age": 28,
    "housing": "rent",
    "existing_credits": 1,
    "num_dependents": 1,
}

# Score applicant using calibrated threshold (default = 0.40)
result = score_applicant(applicant, threshold=0.40)
print(result)
# Output: {'probability_of_default': 0.3922, 'risk_label': 'Medium', 'threshold_used': 0.40}
```

---

## Project Structure

```text
CodeAlpha_CreditScoring/
├── data/                             # Raw / downloaded datasets (Give Me Some Credit & German Credit)
├── notebooks/
│   └── credit_scoring.ipynb          # Executed end-to-end narrative Jupyter notebook
├── src/
│   ├── __init__.py                   # Package initialization
│   ├── data_loader.py                # Dual-dataset loader with graceful fallback
│   ├── features.py                   # Domain feature engineering & ratios
│   ├── preprocessing.py              # Cleaning, median imputation, one-hot encoding, scaling
│   ├── train.py                      # Multi-model training, SMOTE & hyperparameter tuning
│   ├── evaluate.py                   # Metrics calculation, curve generation & threshold tuning
│   └── predict.py                    # Production inference API (score_applicant)
├── models/
│   └── best_model.joblib             # Serialized champion model pipeline
├── reports/
│   ├── figures/                      # Generated figures (ROC, PR, confusion matrices, SHAP)
│   ├── comparison.csv                # Complete models × metrics benchmarking table
│   ├── training_results.json         # Hyperparameter configurations and CV metadata
│   └── feature_engineering_lift.csv # Quantified performance lift from engineered features
├── requirements.txt                  # Pinned production dependency versions
├── app.py                            # Interactive Streamlit credit scoring application
├── build_notebook.py                 # Automated notebook generator and runner
├── README.md                         # Complete project documentation
└── .gitignore                        # Git exclusion rules
```

---

## Author & Internship Attribution
- **Author**: Aryan ([GitHub: aryannten](https://github.com/aryannten))
- **Program**: CodeAlpha Machine Learning Internship
- **Task**: Task 1 — Credit Scoring Model
