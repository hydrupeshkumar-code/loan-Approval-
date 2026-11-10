# Loan Approval Prediction System Using Machine Learning

A small, reproducible classifier that estimates whether a home-loan application resembles applications that were approved in a public historical table. It also includes a separate EMI calculator for home/mortgage and personal-loan scenarios using dated, published Indian lender rate ranges. It is built for a college evaluation, a technical demo, a GitHub portfolio, and machine-learning interviews.

The model is not a lender's decision system: it cannot tell an applicant whether a real lender will approve them or what rate they will be offered. The EMI calculator is a fixed-rate scenario, not an offer or eligibility check.

**Live app:** https://loan-approval-hydrupesh.streamlit.app/

## Team

| Name | Roll number |
| --- | --- |
| Rupesh Kumar | 24STUCH010715 |
| Yarram Rohit Reddy | 24STUCHH010037 |

## At a glance

| | |
| --- | --- |
| **Question** | How does an application compare with historical home-loan approval labels? |
| **Type of problem** | Binary classification (Approved = 1, Rejected = 0) |
| **Data** | 614 real loan applications, 13 columns (Analytics Vidhya "Dream Housing Finance") |
| **Models compared** | Baseline (always approve), Logistic Regression, Decision Tree, Random Forest, Extra Trees |
| **Winner** | Random Forest (100 trees, max depth 12, min 5 rows per leaf) |
| **Test accuracy** | **85.4%** (105 of 123 unseen applications correct), vs 69% for the baseline |
| **Most important feature** | Credit history: 79.6% approved with good history vs 7.9% without |
| **Demo** | Streamlit web app: historical-label estimate, explanations, lender-rate sources and EMI scenarios |

## How the code works (walkthrough)

The project runs as a pipeline. Each file does one job:

| Step | File | What it does |
| --- | --- | --- |
| 1. Get data | `src/loan_approval/download.py` | Downloads the CSV, checks it is real data (not an error page), and saves a SHA-256 checksum so the file can be verified. |
| 2. Check data | `src/loan_approval/data_validation.py` | Checks columns, data types, missing values, duplicates and class balance, then writes `reports/data_validation_report.md`. |
| 3. Explore | `src/loan_approval/eda.py` | Makes the EDA charts (histograms, box plots, approval rate by category, correlation heatmap). |
| 4. Clean + features | `src/loan_approval/preprocessing.py` | Normalises labels (`Y`/`yes` becomes `Yes`), turns impossible values (negative income) into missing, and adds `TotalIncome`, `LoanToIncome` and `AmountPerMonth`. Fills missing values (median/mode), one-hot encodes categories, and scales numbers for logistic regression. Everything sits inside **one scikit-learn Pipeline**. |
| 5. Train + choose | `src/loan_approval/train.py` | 80/20 stratified split, then 5-fold cross-validated hyperparameter search for each model. Picks the best model by CV accuracy and tunes the approval threshold, using **training data only**. |
| 6. Evaluate | `src/loan_approval/evaluate.py` | Scores the chosen model **once** on the held-out 20%: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC and PR curves. |
| 7. Explain | `src/loan_approval/explainability.py` | Permutation importance (which fields matter), a per-applicant "what moved this prediction" explanation, and a gender fairness check. |
| 8. Predict | `src/loan_approval/predict.py` | Loads the saved model and scores new applicants. It never retrains. |
| 9. Web app | `app.py` | Streamlit dashboard with an applicant form, prediction, metrics, charts and batch CSV upload. |
| 10. Check | `scripts/robustness_checks.py` | Nested CV, 30 random splits, stronger models, calibration and the credit-history rule. Writes `reports/robustness_report.md`. |
| Tests | `tests/` | Pytest coverage for cleaning, validation, training, prediction, decision thresholds, review flags, SHAP and EMI calculations. |

**Why a single Pipeline matters.** Missing-value filling and scaling are learned from the training rows only. When the test set or a new applicant passes through, it gets the training medians and scales. This prevents *data leakage* (information from the test set sneaking into training).

**Why a threshold of 0.36 instead of 0.5?** The model outputs a score for the historical approval label. On training cross-validation, classifying scores of 36% or above as approved-like gave the highest accuracy, so that cut-off was frozen before the test set was touched. It is not a lender's underwriting threshold.

## Demo script (2 minutes)

1. `streamlit run app.py`
2. On **Historical estimate**, keep the *Typical applicant* example and click **Estimate historical pattern**. Review the score and reasons; this is not an approval.
3. Pick *Poor credit history* and estimate again. The "Why this result?" bars show how the model's historical-label score changes.
4. Open **EMI calculator**, choose a loan type and lender, then enter an amount and term to see a fixed-rate repayment scenario and its source range.
5. Open **Model performance** for the test accuracy, confusion matrix and ROC/PR curves.
6. Open the **Model insights** tab: the model comparison table and the feature-importance chart.
7. Optional: the **Batch prediction** tab scores the sample file in one click.

## Problem statement

Lenders record applicant income, loan size, credit history, and a few application fields, then approve or reject the loan. This project asks a narrow question: given those fields, how well can a classical model rank and label the historical decision?

The positive class is **Approved** (`Loan_Status = Y`). A false positive is a historically rejected application the model would approve. A false negative is a historically approved application the model would reject. The labels are past decisions, not proof that those decisions were correct or fair.

## Objectives

- Download a real labelled loan table without a manual upload.
- Validate it and record where it came from.
- Compare a majority-class baseline with logistic regression and tree ensembles.
- Choose the model on training cross-validation only, then score it once on a held-out test set.
- Save the full preprocess-and-predict pipeline and expose it in a Streamlit app.

## Dataset

| Item | Value |
| --- | --- |
| Name | Analytics Vidhya Loan Prediction (Dream Housing Finance) |
| Original page | https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/ |
| File used | https://raw.githubusercontent.com/shrikant-temburwar/Loan-Prediction-Dataset/master/train.csv |
| Provider | Public GitHub mirror of that practice problem |
| Rows × columns | 614 × 13 |
| Target | `Loan_Status` (Y approved, N rejected) |
| Download date | 2026-11-05 |
| SHA-256 | `8c2cc67a13279ec746b15f76f14e8503154274ef7f221d2437033b4407c80ca8` |

Columns: `Loan_ID`, `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, `Property_Area`, `Loan_Status`.

`LoanAmount` is stored in thousands. `Credit_History` is 1 when the recorded history meets the guidelines and 0 when it does not.

**Licence.** The GitHub mirror has no licence file (the GitHub license API returns 404). Analytics Vidhya publishes this table for its practice problem. This project uses it for coursework and portfolios. Check Analytics Vidhya's current terms before any commercial use. The raw CSV is gitignored so the mirror is not republished from this repo. UCI Statlog German Credit was not used: it is a different credit-risk table.

Class balance on the full file: **422 approved (68.7%)** and **192 rejected (31.3%)**. No duplicate rows. The largest missing field is `Credit_History` (50 rows, 8.1%).

## Download

From the project folder:

```powershell
python scripts/download_data.py
```

The script tries three public raw URLs of the same practice-problem table, rejects HTML error pages, checks that the file has a loan-status target and the applicant columns, and writes `data/raw/loan_prediction.csv` plus `data/source_manifest.json`. If that local file already matches the manifest checksum, it does not download again.

If GitHub is blocked, save this file yourself as `data/raw/loan_prediction.csv`:

```text
https://raw.githubusercontent.com/shrikant-temburwar/Loan-Prediction-Dataset/master/train.csv
```

## Features

The model does **not** use `Loan_ID` or `Gender`. Gender is kept only for a test-set audit. Removing it does not remove bias.

Engineered columns, fit with no target statistics:

- **TotalIncome** — applicant plus co-applicant income. One missing income does not wipe out the other; if both are missing the total stays missing.
- **LoanToIncome** — `LoanAmount / TotalIncome`. Zero or missing income becomes missing, not infinity.
- **AmountPerMonth** — `LoanAmount / Loan_Amount_Term`, in the dataset's loan-amount units. A zero or missing term becomes missing.

Median imputation, most-frequent imputation, and scaling are inside a scikit-learn pipeline fit on the training split only. One-hot encoding uses `handle_unknown="ignore"`. Logistic regression scales numeric columns. Tree models do not. Class weights are not rebalanced in full training (the models are ranked by accuracy, and reweighting would trade accuracy for recall on the rejected class); instead the most-frequent dummy baseline shows what "always approve" scores, and the decision threshold is tuned on training predictions. SMOTE is not used.

## Technologies

Python 3.10+, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, joblib, Streamlit, pytest, requests.

## Algorithms

1. `DummyClassifier` (most frequent class) — the baseline.
2. Logistic regression — grid search on `C`.
3. Decision tree — grid search on depth and leaf size.
4. Random forest — randomized search.
5. Extra trees — randomized search.

Selection metric: **mean stratified 5-fold accuracy on the training set**. The decision threshold also maximises accuracy on training out-of-fold probabilities. The test set is not used for either choice. A most-frequent dummy is kept so the accuracy has to beat always predicting approval (about 69%).

## Project architecture

```text
loan-approval-prediction/
├── app.py
├── README.md
├── requirements.txt
├── pyproject.toml
├── data/                  raw CSV, manifest, processed table, sample batch
├── notebooks/01_exploratory_data_analysis.ipynb
├── src/loan_approval/     load, validate, preprocess, train, explain, predict
├── scripts/download_data.py
├── scripts/run_eda.py
├── artifacts/             best_model.joblib, model_metadata.json
├── reports/               EDA, comparison, curves, fairness
└── tests/
```

`python -m src.loan_approval.train` runs the full fit. `app.py` only loads the saved pipeline.

## EDA

Computed from the 614-row file. Charts are in `reports/`.

- Credit history dominates the raw approval rates: **79.6%** approved when credit history is 1 (475 rows) and **7.9%** when it is 0 (89 rows).
- Property area: Semiurban 76.8% versus Rural 61.5%.
- Education, marriage, and dependents move approval by about 9 to 15 percentage points.
- Self-employment barely moves it (68.6% versus 68.3%). Gender on the raw table is close (Male 69.3%, Female 67.0%).
- Applicant income has 50 IQR outliers. They are kept. Extreme incomes can be real applications.
- Household total-income medians are close (5439 approved, 5290 rejected). Credit history, not income level, is the obvious split in this table.

## Actual results

Training date: 2026-11-05. Random state 42. Split 80:20 stratified. Training rows: 337 approved and 154 rejected. Test rows: 85 approved and 38 rejected. No rows were dropped for a bad target or an exact duplicate.

### Training cross-validation (model choice)

| Model | CV accuracy |
| --- | --- |
| Random forest | 0.798 |
| Extra trees | 0.798 |
| Logistic regression | 0.794 |
| Decision tree | 0.790 |
| Dummy (most frequent) | 0.686 |

**Selected model: random forest** (`n_estimators=100`, `max_depth=12`, `min_samples_leaf=5`). The forest won the tie with extra trees on training ROC-AUC. These CV scores are a little optimistic because the same folds chose the hyperparameters. Quote the test accuracy below.

### Held-out test set

Threshold **0.36**.

| Metric | Value |
| --- | --- |
| Accuracy | 0.854 |
| Precision (Approved) | 0.832 |
| Recall (Approved) | 0.988 |
| F1 (Approved) | 0.903 |
| Balanced accuracy | 0.770 |
| ROC-AUC | 0.854 |
| Precision-recall AUC | 0.891 |

Confusion matrix (rows are the historical label, columns are the prediction), order Rejected then Approved:

|  | Predicted rejected | Predicted approved |
| --- | --- | --- |
| Historical rejected | 21 | 17 |
| Historical approved | 1 | 84 |

That is 105 correct predictions out of 123 test applications (85.4%). All 17 false positives have `Credit_History = 1`, and all 21 correctly rejected applications have `Credit_History = 0`. In practice the model rejects almost only on credit history. It does not reach 90%. On this same split, picking the cutoff with the test labels in hand still stayed under 89%, and an unweighted logistic regression, extra trees, and gradient boosting did not clear 90% either. An unconstrained forest can score 100% on the training rows and still miss the test rows. Reporting that training score as the accuracy would be wrong.

Permutation importance on the training set, measured as the drop in ROC-AUC when a column is shuffled, is led by **Credit_History** (about 0.20), then **ApplicantIncome** and **LoanAmount** (about 0.14). Those income bars also include `TotalIncome`, `LoanToIncome`, and `AmountPerMonth`, which are built from the income and loan fields inside the pipeline. Importance is association, not a cause of approval.

## Decision modes and review flags

The model gives a probability; the cut-off turns it into a decision. Both cut-offs are chosen on training out-of-fold predictions only.

| Mode | Cut-off | Accuracy | Precision | Risky applications caught | Good applicants wrongly rejected |
| --- | --- | --- | --- | --- | --- |
| Standard (maximises accuracy) | 0.36 | 85.4% | 83.2% | 21 of 38 | 1 of 85 |
| Cautious (maximises balanced accuracy) | 0.71 | 76.4% | 91.2% | 32 of 38 | 23 of 85 |

The "cautious" name describes a statistical threshold that labels more historical rejections as rejected-like; it is not a lender policy. The app has a switch in the sidebar and a cut-off slider on the Model performance tab.

Every prediction also gets a `Review_Note` when a human should double-check it: credit history missing (the pipeline fills it with the most common value), probability within 5 points of the cut-off, or a number outside the range seen in training.

These threshold modes only classify historical labels in a coursework dataset. They must not be used to approve or reject a real applicant.

## Interest-rate and EMI scenarios

The **EMI calculator** is separate from the classifier. It uses the standard fixed-rate amortization formula

`EMI = P × r × (1+r)^n / ((1+r)^n − 1)`

where `P` is principal, `r` is the annual rate divided by 12 and 100, and `n` is the number of monthly payments. It reports estimated EMI, total interest and total repayment. Processing fees, insurance, taxes, and floating-rate changes are excluded.

The app's advertised rate ranges are from lender pages checked on **10 Nov 2026**; they are dated snapshots, not a live feed. The bank pages are human-readable and do not provide a documented rate API. Verify the source pages for current details:

| Product | Lender | Published range shown in app | Source |
| --- | --- | --- | --- |
| Home / mortgage | ICICI Bank, standard home loan | 8.75%–10.05% p.a.; spans borrower/loan-size categories, valid through 30 Nov 2026 | [Official rates page](https://www.icici.bank.in/personal-banking/loans/home-loan/interest-rates) |
| Personal loan | ICICI Bank | 10.25%–16.50% p.a. | [Official rates page](https://www.icici.bank.in/personal-banking/loans/personal-loan/personal-loan-interest-rates) |
| Personal loan | HDFC Bank, salaried rack rate | 9.99%–24.00% p.a. | [Official rates page](https://www.hdfc.bank.in/personal-loan/interest-rates-and-charges) |

These are not individualized rates. A lender may quote a different rate based on credit history, income, employment, loan amount, term and other eligibility criteria. The current model has no interest-rate labels and is trained only on historical home-loan decisions; it does not predict personal-loan approval or an applicant's offered rate.

## Robustness checks

`python scripts/robustness_checks.py` (about 2 minutes) writes `reports/robustness_report.md`:

- **Nested cross-validation:** 80.6% ± 2.4%. This is the unflattered estimate; the 85.4% test score is on the lucky side of it.
- **30 random 80/20 splits:** 77.2% to 87.0%, mean 81.5%. None reached 90%.
- **Stronger models under the same rules:** gradient boosting 84.6%, HistGradientBoosting 85.4%, logistic regression 86.2%. The best accuracy any of them reaches even with the cut-off picked on the test labels is 88.6%.
- **One-line rule** "approve if credit history is not 0": 85.4%, the same as the model.
- **Calibration:** Brier score 0.124; sigmoid calibration only improves it to 0.117, so the model is kept as it is.
- **Missing credit history:** 50 applications, 74% of them approved, close to the overall 69%, so filling with the most common value is reasonable.

## Extra experiment: a larger real dataset (HMDA)

To check whether more data helps, the same method was run on **HMDA**, the US public mortgage records published by the CFPB: all 2023 applications in Delaware that ended in a decision (30,784 rows, 73.9% approved). `python experiments/hmda_experiment.py --download` fetches the data (about 12 MB, not committed) and writes `reports/hmda_experiment.md`.

Same rules as the main project: 80/20 stratified split, models and cut-off chosen on training cross-validation only, test set used once.

| | Main project | HMDA experiment |
| --- | --- | --- |
| Rows | 614 | 30,784 |
| Best model | Random Forest | HistGradientBoosting |
| Baseline (always approve) | 69% | 73.9% |
| Test accuracy | 85.4% | 86.4% |
| Precision | 83.2% | 86.6% |
| ROC-AUC | 0.854 | 0.889 |
| Rejections caught (standard / cautious) | 21 / 32 of 38 | 927 / 1,180 of 1,608 |
| Most important field | Credit history | Debt-to-income ratio |

- Cross-validation (86.3%) and test (86.4%) agree, so with 6,157 test rows the estimate is far more stable than on 123.
- The model relies on debt-to-income, loan purpose and loan-to-value, the factors real lenders check.
- **Leakage precautions:** fields filled in after the decision (interest rate, costs, denial reasons, purchaser) are excluded, and race, ethnicity, sex and age are not model inputs. Loan-to-value is missing more often for denied applications, so the model was refit on the 26,743 complete rows: 87.0% accuracy, ROC-AUC 0.885. The result does not come from missing values.
- Even 30,000 real decisions do not reach 90%: HMDA has no credit score, and lender decisions depend on things no dataset records.

## Explainability

- Global: permutation importance of the input fields, plus the forest's own transformed-column importances (`reports/feature_importance.md`).
- **SHAP (main explanation):** for each prediction, SHAP splits the probability into one contribution per field. They add up exactly: the average applicant starts at 68.3%, and the contributions take it to this applicant's probability. For the typical applicant, credit history adds +9.3 points and property area +4.2, ending at 93.5%. With poor credit history, credit history alone takes away 60.1 points, ending at 12.4%. One-hot columns are summed back into their original field. The global SHAP chart (`reports/shap_importance.png`) ranks credit history first (0.145), then property area (0.031) and applicant income (0.024), which agrees with permutation importance.
- **What-if:** the app also swaps one field at a time for the training median or mode and reports how the approval probability moves.

## Fairness

On the test set the model predicted approval for 72% of women (25 rows) and 84% of men (96 rows). False-positive rates were 0.36 and 0.46. False-negative rates were about 0 and 0.01. Two test rows have no recorded gender; that rate is too small to read. Mean applicant income in the cleaned table is about 4643 for women and 5446 for men, so a model that never sees gender can still score groups differently. This is a description of one small test split, not a finding of discrimination and not a certificate of fairness. Details: `reports/fairness_report.md`.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Execution

```powershell
python scripts/download_data.py
python -m src.loan_approval.train
python scripts/run_eda.py
python scripts/robustness_checks.py
python experiments/hmda_experiment.py --download
pytest -q
streamlit run app.py
```

Run the commands from `loan-approval-prediction`. If the CSV or the model is missing, the loader prints the command that creates it.

The notebook `notebooks/01_exploratory_data_analysis.ipynb` calls the same EDA functions. `python scripts/run_eda.py` is the command that writes the report.

## Screenshots

The dashboard title is **LOAN PLANNING & DECISION SUPPORT**. After `streamlit run app.py`, capture:

1. The overview and applicant form.
2. A historical-label estimate, including its score and disclaimer.
3. The test metrics and confusion matrix.
4. The model-comparison table.

Those screens depend on the local Streamlit session, so they are not stored as fake images in this repo. The charts the training and EDA scripts write are in `reports/`.

## Limitations

- 614 practice-problem rows, not a live loan book.
- The historical label may already be biased. The model learns that label.
- Held-out accuracy is 85.4% on one small split, not a guarantee of real-world precision. Credit history already separates most applications, and the remaining cases overlap. A deeper tree memorizes the training set instead of fixing those cases.
- Cross-validation scores used for the search are slightly optimistic. Nested CV (80.6%) gives the honest estimate.
- The dataset has no debts, savings or a modern credit score, so the model leans almost entirely on credit history. Changing the score threshold changes historical-label errors; it does not improve the training data.
- Probabilities are only roughly calibrated (see the robustness report).
- The app does not monitor incoming data beyond the out-of-range flag.
- Published rate ranges are a dated snapshot and must be checked on the lender pages. The EMI output is not an interest-rate prediction or lender offer.
- The data mirror has no SPDX licence.

## Future improvements

- Collect richer, newer data (debts, savings, credit score, employment) — the main lever for accuracy.
- Replace the cautious cut-off with a cost matrix once a real lender states its costs.
- Train the app's model on a large real dataset with a credit score (the HMDA experiment shows the method scales).
- Repeat the fairness audit on a larger book before anyone treats a gap as stable.

## Ethical considerations

Do not use this app to approve or reject a real applicant or treat the EMI estimate as a lender offer. Gender is excluded from the inputs and is still a poor fairness guarantee, because marital status, dependents, income, property area, and the old labels can proxy for protected characteristics. The app labels model outputs as historical estimates. Uploaded batch files are not written to disk by the app.
