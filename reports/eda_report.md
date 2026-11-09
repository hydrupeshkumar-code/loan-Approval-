# Exploratory data analysis

Every number in this file was calculated from the dataset on disk.

## Dataset overview

- Records: **614**
- Columns: **13**
- Duplicate records: **0**
- Target column: **Loan_Status**

### Data types

| Column | Type |
| --- | --- |
| Loan_ID | object |
| Gender | object |
| Married | object |
| Dependents | object |
| Education | object |
| Self_Employed | object |
| ApplicantIncome | int64 |
| CoapplicantIncome | float64 |
| LoanAmount | float64 |
| Loan_Amount_Term | float64 |
| Credit_History | float64 |
| Property_Area | object |
| Loan_Status | object |

### Numeric summary

| index | count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ApplicantIncome | 614.00 | 5403.46 | 6109.04 | 150.00 | 2877.50 | 3812.50 | 5795.00 | 81000.00 |
| CoapplicantIncome | 614.00 | 1621.25 | 2926.25 | 0.00 | 0.00 | 1188.50 | 2297.25 | 41667.00 |
| LoanAmount | 592.00 | 146.41 | 85.59 | 9.00 | 100.00 | 128.00 | 168.00 | 700.00 |
| Loan_Amount_Term | 600.00 | 342.00 | 65.12 | 12.00 | 360.00 | 360.00 | 360.00 | 480.00 |
| Credit_History | 564.00 | 0.84 | 0.36 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### Missing values

| Column | Missing | Percent |
| --- | --- | --- |
| Gender | 13 | 2.1% |
| Married | 3 | 0.5% |
| Dependents | 15 | 2.4% |
| Self_Employed | 32 | 5.2% |
| LoanAmount | 22 | 3.6% |
| Loan_Amount_Term | 14 | 2.3% |
| Credit_History | 50 | 8.1% |

## Target distribution

- Approved (Y): **422** (68.7% of labelled rows)
- Rejected (N): **192** (31.3% of labelled rows)

The classes are uneven. A model that always predicts approval can look accurate and still be useless, which is why the training script compares models with ROC-AUC and reports a most-frequent dummy baseline.

## Univariate notes

- ApplicantIncome: median 3812.5, mean 5403.5, IQR outliers 50 of 614 non-missing values. Outliers are kept. Extreme incomes and loan amounts can be real applications.
- CoapplicantIncome: median 1188.5, mean 1621.2, IQR outliers 18 of 614 non-missing values. Outliers are kept. Extreme incomes and loan amounts can be real applications.
- LoanAmount: median 128.0, mean 146.4, IQR outliers 39 of 592 non-missing values. Outliers are kept. Extreme incomes and loan amounts can be real applications.
- Loan_Amount_Term: median 360.0, mean 342.0, IQR outliers 88 of 600 non-missing values. Outliers are kept. Extreme incomes and loan amounts can be real applications.

## Relationship with approval

- Credit_History: approval is 79.6% for 1.0 (475 rows) and 7.9% for 0.0 (89 rows), a gap of 71.7%.
- Education: approval is 70.8% for Graduate (480 rows) and 61.2% for Not Graduate (134 rows), a gap of 9.6%.
- Self_Employed: approval is 68.6% for No (500 rows) and 68.3% for Yes (82 rows), a gap of 0.3%.
- Dependents: approval is 75.2% for 2 (101 rows) and 64.7% for 1 (102 rows), a gap of 10.5%.
- Property_Area: approval is 76.8% for Semiurban (233 rows) and 61.5% for Rural (179 rows), a gap of 15.4%.
- Married: approval is 71.6% for Yes (398 rows) and 62.9% for No (213 rows), a gap of 8.7%.
- Gender: approval is 69.3% for Male (489 rows) and 67.0% for Female (112 rows), a gap of 2.4%.

## Engineered features (descriptive only)

- TotalIncome: median 5439.000 among approved and 5289.500 among rejected labelled rows.
- LoanToIncome: median 0.024 among approved and 0.024 among rejected labelled rows.
- AmountPerMonth: median 0.361 among approved and 0.375 among rejected labelled rows.

These medians motivate keeping the three features. They are not a test-set result, and they are computed before any model is fit.

## Data quality

See `data_validation_report.md` for invalid values and unexpected labels. Validation does not delete rows.
