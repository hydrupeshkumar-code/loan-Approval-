# Extra experiment: HMDA real mortgage decisions

30,784 real 2023 applications in Delaware, 73.9% approved. 80/20 stratified split (24,627 train, 6,157 test), 5-fold CV, same rules as the main project.

## Model comparison (training cross-validation)

| Model | CV accuracy | CV ROC-AUC |
| --- | --- | --- |
| HistGradientBoosting | 0.863 | 0.886 |
| Random Forest | 0.862 | 0.882 |
| Logistic Regression | 0.799 | 0.799 |
| Baseline (always approve) | 0.739 | 0.500 |

## Test set (HistGradientBoosting)

| Mode | Cut-off | Accuracy | Precision | Recall | ROC-AUC | Rejections caught |
| --- | --- | --- | --- | --- | --- | --- |
| Standard | 0.50 | 86.4% | 86.6% | 96.5% | 0.889 | 927 of 1608 |
| Cautious | 0.71 | 83.7% | 90.3% | 87.3% | 0.889 | 1180 of 1608 |

## Most important fields (permutation, drop in ROC-AUC)

| Field | Importance |
| --- | --- |
| Debt-to-income ratio | 0.132 |
| Loan purpose | 0.073 |
| Loan-to-value ratio | 0.041 |
| Preapproval requested | 0.021 |
| Income | 0.014 |
| Construction method | 0.012 |
| Property value | 0.012 |
| Line of credit | 0.009 |

## Leakage checks

- Fields filled in only after the decision (interest rate, loan costs, denial reasons, purchaser) are excluded. Race, ethnicity, sex and age are excluded from the model.
- Loan-to-value is missing more often for denied applications. Refitting on the 26,743 rows with no missing key values gives test accuracy 87.0% and ROC-AUC 0.885 (baseline 77.3%), so the result does not depend on missing values.
