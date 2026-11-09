"""Metrics and charts for the locked test evaluation. No model selection here."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.loan_approval.config import APPROVED_NAME, REJECTED_NAME, SELECTION_REASON
from src.loan_approval.utils import fmt, markdown_table


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, object]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "confusion_matrix": matrix.tolist(),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray, metric: str = "accuracy") -> float:
    """Pick a cutoff on training out-of-fold scores. Never pass test rows.

    metric="accuracy" gives the standard cut-off. metric="balanced_accuracy"
    weights both classes equally, so it rejects more risky applications
    (the "cautious" cut-off).
    """
    scorer = {"accuracy": accuracy_score, "balanced_accuracy": balanced_accuracy_score}[metric]
    best_threshold = 0.5
    best_score = -1.0
    for step in range(5, 96):
        threshold = step / 100
        predictions = (probabilities >= threshold).astype(int)
        score = float(scorer(y_true, predictions))
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(metrics: dict[str, object], path: Path) -> None:
    matrix = np.array(metrics["confusion_matrix"], dtype=float)
    fig, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        xticklabels=[REJECTED_NAME, APPROVED_NAME],
        yticklabels=[REJECTED_NAME, APPROVED_NAME],
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Historical label")
    axis.set_title("Test confusion matrix")
    _save(fig, path)


def plot_roc(y_true: np.ndarray, y_proba: np.ndarray, path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, axis = plt.subplots(figsize=(5, 4))
    axis.plot(fpr, tpr, color="#1B4F72", label=f"ROC-AUC {roc_auc_score(y_true, y_proba):.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#7F8C8D")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("Test ROC curve")
    axis.legend(loc="lower right")
    _save(fig, path)


def plot_precision_recall(y_true: np.ndarray, y_proba: np.ndarray, path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig, axis = plt.subplots(figsize=(5, 4))
    axis.plot(recall, precision, color="#148F77")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Test precision-recall curve")
    _save(fig, path)


def plot_comparison(comparison: list[dict[str, object]], path: Path) -> None:
    frame = pd.DataFrame(comparison).sort_values("cv_accuracy_mean")
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.barh(frame["model"], frame["cv_accuracy_mean"], color="#1B4F72")
    axis.set_xlabel("Mean training accuracy")
    axis.set_xlim(0, 1)
    axis.set_title("Model comparison (cross-validation)")
    _save(fig, path)


def plot_importance(table: pd.DataFrame, path: Path, title: str, value_column: str = "importance_mean") -> None:
    show = table.head(12).iloc[::-1]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.barh(show["feature"], show[value_column], color="#1B4F72")
    axis.set_title(title)
    axis.set_xlabel(value_column.replace("_", " "))
    _save(fig, path)


def write_model_reports(
    *,
    comparison: list[dict[str, object]],
    best_model: str,
    best_params: dict[str, object],
    threshold: float,
    metrics: dict[str, object],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_distribution: dict[str, dict[str, int]],
    rows_note: str,
    report_dir: Path,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    comparison_rows = [
        [
            row["model"],
            fmt(float(row["cv_roc_auc_mean"])),
            fmt(float(row["cv_roc_auc_std"])),
            fmt(float(row["cv_f1_mean"])),
            fmt(float(row["cv_accuracy_mean"])),
        ]
        for row in comparison
    ]
    comparison_md = "\n".join(
        [
            "# Model comparison",
            "",
            "Scores are stratified cross-validation on the training set.",
            "The test set was not used to choose a model or a parameter.",
            "",
            SELECTION_REASON,
            "",
            markdown_table(
                ["Model", "CV ROC-AUC", "CV ROC-AUC std", "CV F1", "CV accuracy"],
                comparison_rows,
            ),
            "",
            f"Selected model: **{best_model}**",
            "",
            "Parameters:",
            "",
            "```json",
            json.dumps(best_params, indent=2),
            "```",
            "",
            "The cross-validation score of the winner is a little optimistic because "
            "the same folds were used to pick its hyperparameters. The test metrics "
            "are the ones to quote.",
            "",
        ]
    )
    (report_dir / "model_comparison.md").write_text(comparison_md, encoding="utf-8")

    report_text = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[REJECTED_NAME, APPROVED_NAME],
        digits=3,
        zero_division=0,
    )
    (report_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    final = "\n".join(
        [
            "# Final evaluation",
            "",
            f"Model: **{best_model}**",
            f"Decision threshold: **{threshold:.2f}** (chosen to maximise accuracy on training out-of-fold predictions, then frozen).",
            "",
            rows_note,
            "",
            "## Class counts after the split",
            "",
            markdown_table(
                ["Split", "Approved", "Rejected"],
                [
                    ["Train", class_distribution["train"]["approved"], class_distribution["train"]["rejected"]],
                    ["Test", class_distribution["test"]["approved"], class_distribution["test"]["rejected"]],
                ],
            ),
            "",
            "## Test metrics",
            "",
            markdown_table(
                ["Metric", "Value"],
                [
                    ["Accuracy", fmt(float(metrics["accuracy"]))],
                    ["Precision (Approved)", fmt(float(metrics["precision"]))],
                    ["Recall (Approved)", fmt(float(metrics["recall"]))],
                    ["F1 (Approved)", fmt(float(metrics["f1"]))],
                    ["Balanced accuracy", fmt(float(metrics["balanced_accuracy"]))],
                    ["ROC-AUC", fmt(float(metrics["roc_auc"]))],
                    ["Precision-recall AUC", fmt(float(metrics["average_precision"]))],
                ],
            ),
            "",
            "## Errors",
            "",
            f"- True approved: {metrics['true_positive']}",
            f"- True rejected: {metrics['true_negative']}",
            f"- False positives: {metrics['false_positive']}. "
            "The historical label was rejected, and the model would approve.",
            f"- False negatives: {metrics['false_negative']}. "
            "The historical label was approved, and the model would reject.",
            "",
            "A false positive is the more expensive lending mistake if the historical "
            "rejection was a real default risk. A false negative turns away an applicant "
            "the historical process approved. The labels are past decisions, not proof "
            "that the decision was fair or correct.",
            "",
            "## Classification report",
            "",
            "```",
            report_text.rstrip(),
            "```",
            "",
        ]
    )
    (report_dir / "final_evaluation.md").write_text(final, encoding="utf-8")
    (report_dir / "final_evaluation.json").write_text(
        json.dumps(
            {
                "best_model": best_model,
                "threshold": threshold,
                "test_metrics": metrics,
                "class_distribution": class_distribution,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
