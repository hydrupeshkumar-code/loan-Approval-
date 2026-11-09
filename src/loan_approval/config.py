"""Paths and column roles shared by training, the app, and the reports."""

from __future__ import annotations

from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Approved is the positive class. Y/N is the label used by the source table.
POSITIVE_LABEL = "Y"
NEGATIVE_LABEL = "N"
APPROVED_NAME = "Approved"
REJECTED_NAME = "Rejected"

TARGET_CANDIDATES = (
    "Loan_Status",
    "loan_status",
    "LoanStatus",
    "loan_approved",
)

RAW_NUMERIC = (
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
)
RAW_CATEGORICAL = (
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Credit_History",
    "Property_Area",
)
REQUIRED_INPUTS = RAW_NUMERIC + RAW_CATEGORICAL

# Gender stays out of the model. It is kept only for the fairness audit.
# Loan_ID is an identifier, not a predictor.
EXCLUDED_FEATURES = ("Gender",)
IDENTIFIER_CANDIDATES = ("Loan_ID", "LoanID", "loan_id")

ENGINEERED_FEATURES = (
    "TotalIncome",
    "LoanToIncome",
    "AmountPerMonth",
)

FEATURE_NOTES = {
    "TotalIncome": (
        "ApplicantIncome plus CoapplicantIncome. The instalment is paid from "
        "the household, not from one income field alone. If both incomes are "
        "missing the total stays missing. If only one is missing, the known "
        "income is kept and the missing one counts as zero for this sum."
    ),
    "LoanToIncome": (
        "LoanAmount divided by TotalIncome. It measures how large the request "
        "is relative to combined income. A zero or missing total becomes "
        "missing instead of infinity."
    ),
    "AmountPerMonth": (
        "LoanAmount divided by Loan_Amount_Term. In the source table LoanAmount "
        "is stored in thousands, so this is a rough monthly burden in those "
        "units, not a currency instalment. A zero or missing term becomes missing."
    ),
}

FIELD_LABELS = {
    "ApplicantIncome": "Applicant income",
    "CoapplicantIncome": "Co-applicant income",
    "LoanAmount": "Loan amount (thousands)",
    "Loan_Amount_Term": "Loan term (months)",
    "Credit_History": "Credit history",
    "Education": "Education",
    "Self_Employed": "Self-employed",
    "Dependents": "Dependents",
    "Property_Area": "Property area",
    "Married": "Married",
}

FIELD_HELP = {
    "ApplicantIncome": "Applicant income in the dataset's currency units.",
    "CoapplicantIncome": "Co-applicant income. Use 0 when there is no co-applicant.",
    "LoanAmount": "Requested amount. This dataset stores loan amount in thousands.",
    "Loan_Amount_Term": "Term in months. 360 months is 30 years.",
    "Credit_History": "1 means the history meets the recorded guidelines. 0 means it does not.",
    "Education": "Education group recorded on the application.",
    "Self_Employed": "Whether the applicant is self-employed.",
    "Dependents": "Number of dependents. 3+ means three or more.",
    "Property_Area": "Where the property is located.",
    "Married": "Whether the applicant is married.",
}

CREDIT_LABELS = {
    "1": "1 — meets credit guidelines",
    "0": "0 — does not meet credit guidelines",
}

SELECTION_METRIC = "accuracy"
SELECTION_REASON = (
    "Models are ranked by mean stratified cross-validation accuracy on the "
    "training set, because that is the score being asked for. The cutoff is "
    "also chosen on training out-of-fold predictions to maximise accuracy. "
    "The test set is scored once, after both choices. A most-frequent dummy "
    "is included so a high accuracy still has to beat always predicting approval."
)

CAUSATION_NOTE = (
    "Feature importance and the one-row explanation show association inside "
    "this model. They do not show that a feature causes approval, and they "
    "do not show what a lender should do."
)


def find_project_root() -> Path:
    """Locate the repository root from the working directory or this file."""
    starts = [Path.cwd().resolve(), Path(__file__).resolve().parent]
    seen: set[Path] = set()
    for start in starts:
        for candidate in (start, *start.parents):
            if candidate in seen:
                continue
            seen.add(candidate)
            package = candidate / "src" / "loan_approval"
            if (candidate / "pyproject.toml").is_file() and package.is_dir():
                return candidate
    raise RuntimeError(
        "Could not find the project root. Run commands from loan-approval-prediction."
    )


class Paths:
    """Configurable locations. Tests pass their own output directory."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or find_project_root()
        self.data = self.root / "data"
        self.raw_dir = self.data / "raw"
        self.processed = self.data / "processed"
        self.sample = self.data / "sample"
        self.raw_csv = self.raw_dir / "loan_prediction.csv"
        self.manifest = self.data / "source_manifest.json"
        self.artifacts = self.root / "artifacts"
        self.model = self.artifacts / "best_model.joblib"
        self.metadata = self.artifacts / "model_metadata.json"
        self.reports = self.root / "reports"

    def ensure(self) -> None:
        for folder in (
            self.raw_dir,
            self.processed,
            self.sample,
            self.artifacts,
            self.reports,
        ):
            folder.mkdir(parents=True, exist_ok=True)
