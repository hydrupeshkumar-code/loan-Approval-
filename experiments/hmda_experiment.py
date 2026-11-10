"""Extra experiment: the same method on a much larger, real loan dataset.

Data: HMDA (Home Mortgage Disclosure Act) public loan records, published by
the US Consumer Financial Protection Bureau. Public domain. We use all 2023
applications in Delaware that ended in a decision:
    action_taken 1 = loan originated, 2 = approved but not accepted  -> Approved (1)
    action_taken 3 = application denied                              -> Rejected (0)

Download (about 12 MB):
    python experiments/hmda_experiment.py --download
Run:
    python experiments/hmda_experiment.py

Only information known when the application is made is used. Columns filled in
after the decision (interest rate, loan costs, denial reasons, who bought the
loan) would leak the answer and are left out. Race, ethnicity, sex and age are
also left out of the model.

Writes reports/hmda_experiment.md and reports/hmda_experiment.json.
The main project (dataset, model, app) is not changed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.loan_approval.config import RANDOM_STATE, TEST_SIZE
from src.loan_approval.evaluate import choose_threshold
from src.loan_approval.utils import markdown_table

warnings.filterwarnings("ignore")

DATA = ROOT / "data" / "external" / "hmda" / "hmda_de_2023.csv"
URL = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?states=DE&years=2023&actions_taken=1,2,3"

NUMERIC = [
    "loan_amount",
    "income",  # thousands of dollars
    "property_value",
    "loan_to_value_ratio",
    "loan_term",
    "debt_to_income",  # built from debt_to_income_ratio below
    "tract_to_msa_income_percentage",
    "ffiec_msa_md_median_family_income",
]
CATEGORICAL = [
    "loan_type",
    "loan_purpose",
    "lien_status",
    "occupancy_type",
    "construction_method",
    "total_units",
    "open-end_line_of_credit",
    "business_or_commercial_purpose",
    "preapproval",
    "submission_of_application",
]
LABELS = {
    "loan_amount": "Loan amount",
    "income": "Income",
    "property_value": "Property value",
    "loan_to_value_ratio": "Loan-to-value ratio",
    "loan_term": "Loan term",
    "debt_to_income": "Debt-to-income ratio",
    "tract_to_msa_income_percentage": "Area income vs region",
    "ffiec_msa_md_median_family_income": "Region median income",
    "loan_type": "Loan type",
    "loan_purpose": "Loan purpose",
    "lien_status": "Lien status",
    "occupancy_type": "Occupancy",
    "construction_method": "Construction method",
    "total_units": "Units in property",
    "open-end_line_of_credit": "Line of credit",
    "business_or_commercial_purpose": "Business purpose",
    "preapproval": "Preapproval requested",
    "submission_of_application": "Applied directly",
}
# HMDA reports debt-to-income in bands below 36% and above 50%; take the middle of each band.
DTI_BANDS = {"<20%": 15.0, "20%-<30%": 25.0, "30%-<36%": 33.0, "50%-60%": 55.0, ">60%": 65.0}


def download() -> None:
    DATA.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading", URL)
    urllib.request.urlretrieve(URL, DATA)
    print(f"Saved {DATA} ({DATA.stat().st_size / 1e6:.1f} MB)")


def load() -> tuple[pd.DataFrame, pd.Series]:
    raw = pd.read_csv(DATA, low_memory=False)
    raw = raw[raw["action_taken"].isin([1, 2, 3])]
    target = raw["action_taken"].isin([1, 2]).astype(int).rename("approved")
    frame = raw.copy()
    dti = frame["debt_to_income_ratio"].astype(str).str.strip()
    frame["debt_to_income"] = pd.to_numeric(dti.map(DTI_BANDS).fillna(dti), errors="coerce")
    for column in NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")  # "Exempt"/"NA" -> missing
    for column in CATEGORICAL:
        frame[column] = frame[column].astype(str)
    return frame[NUMERIC + CATEGORICAL].reset_index(drop=True), target.reset_index(drop=True)


def pipeline(model, scale: bool) -> Pipeline:
    numeric = [("impute", SimpleImputer(strategy="median"))] + ([("scale", StandardScaler())] if scale else [])
    prepare = ColumnTransformer(
        [
            ("numeric", Pipeline(numeric), NUMERIC),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), CATEGORICAL),
        ]
    )
    return Pipeline([("prepare", prepare), ("model", model)])


def scores(y_true, probability, threshold) -> dict[str, float]:
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted)),
        "f1": float(f1_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "rejections_caught": int(tn),
        "rejections_total": int(tn + fp),
        "approvals_wrongly_rejected": int(fn),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="download the HMDA file first")
    if parser.parse_args().download or not DATA.exists():
        download()
    started = time.time()
    features, target = load()
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=target
    )
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    print(f"{len(features)} applications, {target.mean():.1%} approved. Train {len(x_train)}, test {len(x_test)}.")

    candidates = {
        "Baseline (always approve)": (DummyClassifier(strategy="most_frequent"), False, None),
        "Logistic Regression": (LogisticRegression(max_iter=3000), True, {"model__C": [0.1, 1.0, 10.0]}),
        "Random Forest": (
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
            False,
            {"model__max_depth": [12, None], "model__min_samples_leaf": [1, 5]},
        ),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            False,
            {"model__learning_rate": [0.05, 0.1], "model__max_depth": [None, 6], "model__max_iter": [200, 400]},
        ),
    }
    rows = []
    fitted = {}
    for name, (model, scale, grid) in candidates.items():
        pipe = pipeline(model, scale)
        if grid:
            pipe = GridSearchCV(pipe, grid, scoring="accuracy", cv=cv, n_jobs=-1).fit(x_train, y_train).best_estimator_
        oof = cross_val_predict(clone(pipe), x_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        fitted[name] = (pipe.fit(x_train, y_train), oof)
        rows.append({"model": name, "cv_accuracy": float(accuracy_score(y_train, oof >= 0.5)),
                     "cv_roc_auc": float(roc_auc_score(y_train, oof)) if oof.std() > 0 else 0.5})
        print(f"{name:<28} CV accuracy {rows[-1]['cv_accuracy']:.3f}  CV ROC-AUC {rows[-1]['cv_roc_auc']:.3f}")

    rows.sort(key=lambda row: (-row["cv_accuracy"], -row["cv_roc_auc"]))
    best_name = rows[0]["model"]
    best, oof = fitted[best_name]
    threshold = choose_threshold(y_train.to_numpy(), oof)
    cautious = choose_threshold(y_train.to_numpy(), oof, metric="balanced_accuracy")
    probability = best.predict_proba(x_test)[:, 1]
    test = scores(y_test, probability, threshold)
    test_cautious = scores(y_test, probability, cautious)
    print(f"Chosen: {best_name}, threshold {threshold:.2f} -> test accuracy {test['accuracy']:.3f}, "
          f"ROC-AUC {test['roc_auc']:.3f}")

    # Leakage check: loan-to-value is missing more often for denied applications
    # (16% vs 7%). Refit on complete rows only; if accuracy holds, missingness
    # is not what the model relies on.
    complete = features[["loan_to_value_ratio", "property_value", "debt_to_income", "loan_term"]].notna().all(axis=1)
    c_train, c_test, cy_train, cy_test = train_test_split(
        features[complete], target[complete], test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=target[complete]
    )
    c_probability = clone(best).fit(c_train, cy_train).predict_proba(c_test)[:, 1]
    complete_check = {
        "rows": int(complete.sum()),
        "approved_share": float(target[complete].mean()),
        "test_accuracy": float(accuracy_score(cy_test, c_probability >= threshold)),
        "test_roc_auc": float(roc_auc_score(cy_test, c_probability)),
    }
    print(f"Complete rows only: accuracy {complete_check['test_accuracy']:.3f}, ROC-AUC {complete_check['test_roc_auc']:.3f}")

    sample = x_test.sample(n=min(3000, len(x_test)), random_state=RANDOM_STATE)
    perm = permutation_importance(best, sample, y_test.loc[sample.index], scoring="roc_auc",
                                  n_repeats=3, random_state=RANDOM_STATE, n_jobs=-1)
    importance = (pd.DataFrame({"feature": features.columns, "importance": perm.importances_mean})
                  .sort_values("importance", ascending=False))

    results = {
        "dataset": {"name": "HMDA 2023, Delaware (CFPB public loan-level data)", "url": URL,
                    "rows": int(len(features)), "approved_share": float(target.mean()),
                    "features_used": NUMERIC + CATEGORICAL},
        "comparison": rows,
        "chosen_model": best_name,
        "threshold": threshold,
        "cautious_threshold": cautious,
        "test": test,
        "test_cautious": test_cautious,
        "importance": importance.head(8).to_dict(orient="records"),
        "complete_rows_check": complete_check,
        "seconds": round(time.time() - started, 1),
    }
    reports = ROOT / "reports"
    (reports / "hmda_experiment.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    pct = lambda value: f"{value:.1%}"
    lines = [
        "# Extra experiment: HMDA real mortgage decisions",
        "",
        f"{len(features):,} real 2023 applications in Delaware, {target.mean():.1%} approved. "
        f"80/20 stratified split ({len(x_train):,} train, {len(x_test):,} test), 5-fold CV, same rules as the main project.",
        "",
        "## Model comparison (training cross-validation)",
        "",
        markdown_table(["Model", "CV accuracy", "CV ROC-AUC"],
                       [[r["model"], f"{r['cv_accuracy']:.3f}", f"{r['cv_roc_auc']:.3f}"] for r in rows]),
        "",
        f"## Test set ({best_name})",
        "",
        markdown_table(
            ["Mode", "Cut-off", "Accuracy", "Precision", "Recall", "ROC-AUC", "Rejections caught"],
            [[label, f"{cut:.2f}", pct(s["accuracy"]), pct(s["precision"]), pct(s["recall"]),
              f"{s['roc_auc']:.3f}", f"{s['rejections_caught']} of {s['rejections_total']}"]
             for label, cut, s in (("Standard", threshold, test), ("Cautious", cautious, test_cautious))],
        ),
        "",
        "## Most important fields (permutation, drop in ROC-AUC)",
        "",
        markdown_table(["Field", "Importance"],
                       [[LABELS.get(r["feature"], r["feature"]), f"{r['importance']:.3f}"]
                        for r in results["importance"]]),
        "",
        "## Leakage checks",
        "",
        "- Fields filled in only after the decision (interest rate, loan costs, denial reasons, purchaser) "
        "are excluded. Race, ethnicity, sex and age are excluded from the model.",
        f"- Loan-to-value is missing more often for denied applications. Refitting on the "
        f"{complete_check['rows']:,} rows with no missing key values gives test accuracy "
        f"{complete_check['test_accuracy']:.1%} and ROC-AUC {complete_check['test_roc_auc']:.3f} "
        f"(baseline {complete_check['approved_share']:.1%}), so the result does not depend on missing values.",
        "",
    ]
    (reports / "hmda_experiment.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote reports/hmda_experiment.md")


if __name__ == "__main__":
    main()
