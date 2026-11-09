# Fairness notes

Gender is not a model input. Removing it does not remove bias. Marital status, dependents, income, and property area can still carry information about gender, and the historical labels can themselves be biased.

Group rates below are descriptive error counts on the held-out test set. They were not used to pick the model. Small groups move around a lot; a gap here is not proof of discrimination, and no gap is not proof of fairness.

| Group | Rows | Historical approval | Predicted approval | FPR | FNR | Note |
| --- | --- | --- | --- | --- | --- | --- |
| Female | 25 | 0.560 | 0.720 | 0.364 | 0.000 |  |
| Male | 96 | 0.729 | 0.844 | 0.462 | 0.014 |  |
| Missing | 2 | 0.500 | 1.000 | 1.000 | 0.000 | unstable (fewer than 20 test rows) |
| All test rows | 123 | 0.691 | 0.821 | 0.447 | 0.012 |  |

## Descriptive association in the cleaned table

Mean applicant income by gender (not used as a training target):

| Gender | Mean applicant income |
| --- | --- |
| Female | 4643 |
| Male | 5446 |
| Missing | 10334 |

An income gap can make a model that never sees gender still score groups differently.
