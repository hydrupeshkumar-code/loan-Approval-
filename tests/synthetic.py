"""Synthetic applications for tests only. Not used as the training dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_loan_frame(n: int = 160, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    credit = rng.integers(0, 2, n)
    income = rng.integers(1500, 14000, n)
    coapplicant = rng.integers(0, 4000, n)
    loan = rng.integers(40, 350, n)
    noise = rng.normal(0, 0.35, n)
    score = credit * 1.5 - (loan / np.maximum(income / 1000, 0.1)) * 0.2 + noise
    status = np.where(score > 0.15, "Y", "N")
    status[0] = "Y"
    if n > 1:
        status[1] = "N"
    gender = rng.choice(["Male", "Female"], n)
    frame = pd.DataFrame(
        {
            "Loan_ID": [f"LP{index:05d}" for index in range(n)],
            "Gender": gender,
            "Married": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["0", "1", "2", "3+"], n),
            "Education": rng.choice(["Graduate", "Not Graduate"], n),
            "Self_Employed": rng.choice(["Yes", "No"], n),
            "ApplicantIncome": income,
            "CoapplicantIncome": coapplicant,
            "LoanAmount": loan.astype(float),
            "Loan_Amount_Term": rng.choice([120, 180, 240, 360], n).astype(float),
            "Credit_History": credit.astype(float),
            "Property_Area": rng.choice(["Urban", "Semiurban", "Rural"], n),
            "Loan_Status": status,
        }
    )
    frame.loc[0:2, "Gender"] = np.nan
    frame.loc[3:5, "Self_Employed"] = np.nan
    frame.loc[6:8, "LoanAmount"] = np.nan
    return frame
