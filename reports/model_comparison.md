# Model comparison

Scores are stratified cross-validation on the training set.
The test set was not used to choose a model or a parameter.

Models are ranked by mean stratified cross-validation accuracy on the training set, because that is the score being asked for. The cutoff is also chosen on training out-of-fold predictions to maximise accuracy. The test set is scored once, after both choices. A most-frequent dummy is included so a high accuracy still has to beat always predicting approval.

| Model | CV ROC-AUC | CV ROC-AUC std | CV F1 | CV accuracy |
| --- | --- | --- | --- | --- |
| RandomForestClassifier | 0.775 | 0.082 | 0.870 | 0.798 |
| ExtraTreesClassifier | 0.719 | 0.086 | 0.870 | 0.798 |
| LogisticRegression | 0.730 | 0.092 | 0.867 | 0.794 |
| DecisionTreeClassifier | 0.685 | 0.084 | 0.864 | 0.790 |
| DummyClassifier | 0.500 | 0.000 | 0.814 | 0.686 |

Selected model: **RandomForestClassifier**

Parameters:

```json
{
  "model__n_estimators": 100,
  "model__min_samples_leaf": 5,
  "model__max_depth": 12
}
```

The cross-validation score of the winner is a little optimistic because the same folds were used to pick its hyperparameters. The test metrics are the ones to quote.
