# Final evaluation

Model: **RandomForestClassifier**
Decision threshold: **0.36** (chosen to maximise accuracy on training out-of-fold predictions, then frozen).

Excluded 0 rows with a missing or unrecognised loan status and 0 exact duplicate rows before the split. 614 rows remained. Validation itself did not delete anything.

## Class counts after the split

| Split | Approved | Rejected |
| --- | --- | --- |
| Train | 337 | 154 |
| Test | 85 | 38 |

## Test metrics

| Metric | Value |
| --- | --- |
| Accuracy | 0.854 |
| Precision (Approved) | 0.832 |
| Recall (Approved) | 0.988 |
| F1 (Approved) | 0.903 |
| Balanced accuracy | 0.770 |
| ROC-AUC | 0.854 |
| Precision-recall AUC | 0.891 |

## Errors

- True approved: 84
- True rejected: 21
- False positives: 17. The historical label was rejected, and the model would approve.
- False negatives: 1. The historical label was approved, and the model would reject.

A false positive is the more expensive lending mistake if the historical rejection was a real default risk. A false negative turns away an applicant the historical process approved. The labels are past decisions, not proof that the decision was fair or correct.

## Classification report

```
              precision    recall  f1-score   support

    Rejected      0.955     0.553     0.700        38
    Approved      0.832     0.988     0.903        85

    accuracy                          0.854       123
   macro avg      0.893     0.770     0.802       123
weighted avg      0.870     0.854     0.840       123
```
