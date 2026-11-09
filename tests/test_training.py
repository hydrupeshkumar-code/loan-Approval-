"""Fast training covers comparison, cross-validation, and saved files."""

from __future__ import annotations


def test_training_selects_on_cv_and_saves(trained) -> None:
    paths, metadata = trained
    assert (paths.model).exists()
    assert paths.metadata.exists()
    names = [row["model"] for row in metadata["comparison"]]
    assert names == [
        "DummyClassifier",
        "LogisticRegression",
        "DecisionTreeClassifier",
        "RandomForestClassifier",
        "ExtraTreesClassifier",
    ] or set(names) == {
        "DummyClassifier",
        "LogisticRegression",
        "DecisionTreeClassifier",
        "RandomForestClassifier",
        "ExtraTreesClassifier",
    }
    assert metadata["selection_metric"] == "accuracy"
    assert metadata["model_name"] == metadata["comparison"][0]["model"]
    assert "test_metrics" not in metadata["comparison"][0]
    metrics = metadata["test_metrics"]
    for key in ("accuracy", "precision", "recall", "f1", "balanced_accuracy", "roc_auc", "average_precision"):
        assert 0.0 <= float(metrics[key]) <= 1.0
    assert metadata["cv_folds"] >= 2
    assert (paths.reports / "model_comparison.md").exists()
    assert (paths.reports / "final_evaluation.md").exists()
    assert "Gender" not in metadata["raw_features"]
