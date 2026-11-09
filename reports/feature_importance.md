# Feature importance

Feature importance and the one-row explanation show association inside this model. They do not show that a feature causes approval, and they do not show what a lender should do.

Permutation importance shuffles one input column at a time and measures the drop in training ROC-AUC. Income columns also feed TotalIncome, LoanToIncome, and AmountPerMonth, so their bars include those engineered uses.

| Feature | Mean AUC drop | Std |
| --- | --- | --- |
| Credit_History | 0.2017 | 0.0112 |
| ApplicantIncome | 0.1403 | 0.0048 |
| LoanAmount | 0.1399 | 0.0130 |
| CoapplicantIncome | 0.1031 | 0.0053 |
| Property_Area | 0.0334 | 0.0034 |
| Married | 0.0163 | 0.0034 |
| Education | 0.0140 | 0.0016 |
| Dependents | 0.0137 | 0.0025 |
| Loan_Amount_Term | 0.0109 | 0.0021 |
| Self_Employed | 0.0029 | 0.0005 |

## Model-internal weights

Tree importances and absolute logistic coefficients are on the transformed columns (scaled numbers and one-hot categories). A one-hot column can look small because the signal is split across levels.

| Transformed feature | Importance |
| --- | --- |
| Credit_History_1 | 0.2191 |
| Credit_History_0 | 0.2145 |
| TotalIncome | 0.0910 |
| LoanToIncome | 0.0806 |
| ApplicantIncome | 0.0782 |
| AmountPerMonth | 0.0743 |
| LoanAmount | 0.0698 |
| CoapplicantIncome | 0.0442 |
| Property_Area_Semiurban | 0.0196 |
| Property_Area_Rural | 0.0134 |
| Dependents_0 | 0.0123 |
| Married_Yes | 0.0105 |
| Property_Area_Urban | 0.0102 |
| Education_Graduate | 0.0098 |
| Education_Not Graduate | 0.0092 |
