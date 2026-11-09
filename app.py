"""Streamlit dashboard. Reads the saved pipeline and does not retrain."""

from __future__ import annotations

import html
import json

import pandas as pd
import streamlit as st

from src.loan_approval.config import CREDIT_LABELS, FIELD_HELP, FIELD_LABELS, Paths
from src.loan_approval.explainability import local_explanation, make_shap_explainer, shap_explanation
from src.loan_approval.predict import DISCLAIMER, load_model, review_notes, score_frame
from src.loan_approval.utils import ModelNotFoundError

st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    #MainMenu, footer, .stAppDeployButton {visibility: hidden;}
    .block-container {padding-top: 1.6rem;}
    .hero {background: linear-gradient(120deg, #1B4F72 0%, #148F77 100%); color: #fff;
           padding: 1.6rem 1.8rem; border-radius: 16px; margin-bottom: 1.2rem;
           box-shadow: 0 6px 20px rgba(27, 79, 114, 0.18);}
    .hero h1 {color: #fff; margin: 0; font-size: 2rem; letter-spacing: 0.5px;}
    .hero p {color: #E8F6F3; margin: 0.35rem 0 1rem 0; font-size: 1.02rem;}
    .chips {display: flex; flex-wrap: wrap; gap: 0.6rem;}
    .chip {background: rgba(255, 255, 255, 0.16); border: 1px solid rgba(255, 255, 255, 0.28);
           border-radius: 999px; padding: 0.3rem 0.85rem; font-size: 0.9rem; color: #fff;}
    .chip b {color: #fff;}
    .result {border-radius: 14px; padding: 1.2rem 1.4rem; margin-bottom: 0.8rem; color: #fff;}
    .result.ok {background: linear-gradient(120deg, #1E8449, #27AE60);}
    .result.no {background: linear-gradient(120deg, #922B21, #CB4335);}
    .result .label {font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9;}
    .result .status {font-size: 2.1rem; font-weight: 700; line-height: 1.2;}
    .result .prob {font-size: 1rem; opacity: 0.95;}
    .gauge {position: relative; height: 16px; background: #E5E8EC; border-radius: 999px; margin: 0.4rem 0 0.2rem 0;}
    .gauge .fill {height: 100%; border-radius: 999px;}
    .gauge .mark {position: absolute; top: -5px; width: 3px; height: 26px; background: #1C2833; border-radius: 2px;}
    .gauge-legend {position: relative; display: flex; justify-content: space-between; font-size: 0.78rem;
           color: #566573; margin-bottom: 0.9rem;}
    .gauge-legend .cut {position: absolute; transform: translateX(-50%); font-weight: 600; color: #1C2833;}
    .why {display: grid; grid-template-columns: 14rem 1fr 4.2rem; align-items: center;
           gap: 0.5rem; margin: 0.35rem 0; font-size: 0.92rem;}
    .why .track {background: #EEF1F4; border-radius: 6px; height: 12px;}
    .why .bar {height: 12px; border-radius: 6px;}
    .why .val {text-align: right; font-weight: 600;}
    .note {background: #FEF9E7; border-left: 4px solid #B7950B; padding: 0.7rem 1rem;
           border-radius: 6px; font-size: 0.9rem; color: #4D3F0A;}
    .muted {color: #566573; font-size: 0.9rem;}
    div[data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #E1E7EE; border-radius: 12px;
           padding: 0.8rem 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def cached_model():
    return load_model()


@st.cache_data
def read_manifest(path_text: str) -> dict:
    return json.loads(open(path_text, encoding="utf-8").read())


PATHS = Paths()

try:
    pipeline, metadata = cached_model()
    model_error = None
except ModelNotFoundError as exc:
    pipeline, metadata = None, None
    model_error = str(exc)

manifest = read_manifest(str(PATHS.manifest)) if PATHS.manifest.exists() else {}
metrics = (metadata or {}).get("test_metrics") or {}
standard_threshold = float((metadata or {}).get("threshold", 0.5))
cautious_threshold = (metadata or {}).get("cautious_threshold")
cautious_metrics = (metadata or {}).get("cautious_test_metrics") or {}
MODEL_NAMES = {
    "RandomForestClassifier": "Random Forest",
    "ExtraTreesClassifier": "Extra Trees",
    "LogisticRegression": "Logistic Regression",
    "DecisionTreeClassifier": "Decision Tree",
    "DummyClassifier": "Baseline (always approve)",
}
model_label = MODEL_NAMES.get((metadata or {}).get("model_name", ""), (metadata or {}).get("model_name", "n/a"))
rows_total = (manifest.get("dimensions") or {}).get("rows", 614)


def pct(value: object) -> str:
    return f"{float(value):.1%}"


# ---------------------------------------------------------------- header
chips = ""
if metrics:
    chips = (
        f"<span class='chip'>Model: <b>{html.escape(model_label)}</b></span>"
        f"<span class='chip'>Test accuracy: <b>{pct(metrics['accuracy'])}</b></span>"
        f"<span class='chip'>ROC-AUC: <b>{float(metrics['roc_auc']):.3f}</b></span>"
        f"<span class='chip'>Trained on: <b>{rows_total} real applications</b></span>"
    )
st.markdown(
    f"""
    <div class="hero">
      <h1>🏦 LOAN APPROVAL PREDICTION SYSTEM</h1>
      <p>Enter an applicant's details and a machine-learning model estimates whether the home loan is likely to be approved.</p>
      <div class="chips">{chips}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### How to use")
    st.markdown(
        "1. Pick an **example applicant** or type your own values.\n"
        "2. Click **Predict loan status**.\n"
        "3. Read the result, the probability and **why** the model decided that.\n"
        "4. Explore the other tabs for model performance and insights."
    )
    st.divider()
    st.markdown("### Decision mode")
    modes = {"Standard": standard_threshold}
    if cautious_threshold is not None:
        modes["Cautious"] = float(cautious_threshold)
    mode = st.radio(
        "Decision mode",
        list(modes),
        format_func=lambda name: f"{name} (cut-off {modes[name]:.0%})",
        label_visibility="collapsed",
        help="Standard gives the best overall accuracy. Cautious approves fewer people, so it catches "
        "more of the applications lenders rejected, at the cost of some accuracy.",
    )
    threshold = modes[mode]
    active_metrics = cautious_metrics if mode == "Cautious" and cautious_metrics else metrics
    if active_metrics:
        st.caption(
            f"On the test set: accuracy {pct(active_metrics['accuracy'])}, precision "
            f"{pct(active_metrics['precision'])}, risky applications caught "
            f"{active_metrics['true_negative']} of {active_metrics['true_negative'] + active_metrics['false_positive']}."
        )
    st.divider()
    st.markdown("### Model at a glance")
    if metrics:
        st.markdown(
            f"- **Algorithm:** {model_label}\n"
            f"- **Test accuracy:** {pct(metrics['accuracy'])} (baseline 69%)\n"
            f"- **Approval cut-off:** {threshold:.0%} ({mode.lower()} mode)\n"
            "- **Most important field:** credit history"
        )
    st.divider()
    st.caption(DISCLAIMER)

if model_error:
    st.error(model_error)
    st.stop()

tab_predict, tab_perf, tab_insights, tab_batch, tab_about = st.tabs(
    ["🔮 Predict", "📊 Model performance", "🧠 Model insights", "📁 Batch prediction", "ℹ️ About the project"]
)

# ---------------------------------------------------------------- prediction tab
schema = {field["name"]: field for field in metadata["input_schema"]}
reference = metadata.get("reference_values") or {}
credit_options = schema["Credit_History"]["options"]
credit_shown = [CREDIT_LABELS.get(option, option) for option in credit_options]

@st.cache_resource
def cached_explainer():
    return make_shap_explainer(pipeline)


SHAP_LABELS = {
    **FIELD_LABELS,
    "LoanAmount": "Loan amount",
    "TotalIncome": "Total income",
    "LoanToIncome": "Loan ÷ income",
    "AmountPerMonth": "Loan per month",
}


def shap_label(name: str) -> str:
    return SHAP_LABELS.get(name, name)


def shap_value(name: str, value: object) -> str:
    if name == "Credit_History":
        return {"1": "good", "0": "poor"}.get(str(value), "missing")
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "missing"
    if isinstance(value, (int, float)):
        return f"{float(value):,.0f}" if abs(float(value)) >= 10 else f"{float(value):.3f}"
    return str(value)


def contribution_bars(rows: list[tuple[str, float]]) -> str:
    biggest = max((abs(change) for _, change in rows), default=0) or 1.0
    bars = ""
    for label, change in rows:
        tone = "#27AE60" if change >= 0 else "#CB4335"
        bars += (
            f"<div class='why'><span>{html.escape(label)}</span>"
            f"<div class='track'><div class='bar' style='width:{100 * abs(change) / biggest:.0f}%; "
            f"background:{tone};'></div></div>"
            f"<span class='val' style='color:{tone};'>{change * 100:+.1f}</span></div>"
        )
    return bars


EXAMPLES: dict[str, dict[str, object]] = {
    "Typical applicant": dict(reference),
    "Poor credit history": {**reference, "Credit_History": "0"},
    "Borderline: large loan, low income": {
        **reference,
        "ApplicantIncome": 1800.0,
        "CoapplicantIncome": 0.0,
        "LoanAmount": 450.0,
        "Married": "No",
        "Dependents": "2",
        "Education": "Not Graduate",
        "Property_Area": "Rural",
    },
}


def _state_value(name: str, value: object) -> object:
    if name == "Credit_History":
        return CREDIT_LABELS.get(str(value), str(value))
    if schema[name]["kind"] == "number":
        return float(value)
    return str(value)


def apply_example() -> None:
    for name, value in EXAMPLES[st.session_state["example"]].items():
        if name in schema:
            st.session_state[f"in_{name}"] = _state_value(name, value)
    st.session_state.pop("result", None)


for name in schema:
    st.session_state.setdefault(f"in_{name}", _state_value(name, reference.get(name, "")))


def number_field(name: str) -> None:
    st.number_input(FIELD_LABELS.get(name, name), min_value=0.0, step=100.0 if "Income" in name else 1.0,
                    key=f"in_{name}", help=FIELD_HELP.get(name))


def choice_field(name: str, horizontal: bool = False) -> None:
    if name == "Credit_History":
        st.radio(FIELD_LABELS[name], credit_shown, key=f"in_{name}", help=FIELD_HELP.get(name))
    elif horizontal:
        st.radio(FIELD_LABELS.get(name, name), schema[name]["options"], key=f"in_{name}",
                 help=FIELD_HELP.get(name), horizontal=True)
    else:
        st.selectbox(FIELD_LABELS.get(name, name), schema[name]["options"], key=f"in_{name}",
                     help=FIELD_HELP.get(name))


with tab_predict:
    form_col, result_col = st.columns([1.35, 1], gap="large")
    with form_col:
        st.selectbox(
            "Start from an example applicant",
            list(EXAMPLES),
            key="example",
            on_change=apply_example,
            help="Fills the form with a ready-made applicant. You can still change any field.",
        )
        with st.form("applicant"):
            with st.container(border=True):
                st.markdown("**👤 Personal details**")
                a, b = st.columns(2)
                with a:
                    choice_field("Married", horizontal=True)
                    choice_field("Education")
                with b:
                    choice_field("Dependents", horizontal=True)
                    choice_field("Self_Employed", horizontal=True)
            with st.container(border=True):
                st.markdown("**💰 Income**")
                a, b = st.columns(2)
                with a:
                    number_field("ApplicantIncome")
                with b:
                    number_field("CoapplicantIncome")
            with st.container(border=True):
                st.markdown("**🏠 Loan and credit**")
                a, b = st.columns(2)
                with a:
                    number_field("LoanAmount")
                    choice_field("Property_Area")
                with b:
                    number_field("Loan_Amount_Term")
                choice_field("Credit_History")
            submitted = st.form_submit_button("🔍 Predict loan status", type="primary", width="stretch")

    if submitted:
        values: dict[str, object] = {}
        for name in schema:
            raw = st.session_state[f"in_{name}"]
            values[name] = credit_options[credit_shown.index(raw)] if name == "Credit_History" else raw
        frame = pd.DataFrame([values])
        scored, _ = score_frame(frame, pipeline=pipeline, metadata=metadata)
        explanation = local_explanation(pipeline, frame, reference)
        st.session_state["result"] = {
            "probability": float(scored["Approval_Probability"].iloc[0]),
            "values": values,
            "explanation": [item for item in explanation if abs(float(item["probability_change"])) >= 0.0005],
            "shap": shap_explanation(pipeline, cached_explainer(), frame),
        }

    with result_col:
        result = st.session_state.get("result")
        if result:
            # Status and review notes follow the decision mode chosen in the sidebar.
            result["status"] = "Approved" if result["probability"] >= threshold else "Rejected"
            result["notes"] = review_notes(
                pd.DataFrame([result["values"]]), [result["probability"]], threshold, metadata
            )[0]
        if not result:
            with st.container(border=True):
                st.markdown("#### Result")
                st.markdown(
                    "<p class='muted'>Fill in the form and click <b>Predict loan status</b>. "
                    "The decision, the approval probability and the main reasons will appear here.</p>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<p class='muted'>Tip for the demo: choose <b>Poor credit history</b> from the examples "
                    "to see how strongly credit history drives the decision.</p>",
                    unsafe_allow_html=True,
                )
        else:
            probability = result["probability"]
            approved = result["status"] == "Approved"
            css = "ok" if approved else "no"
            icon = "✅" if approved else "❌"
            st.markdown(
                f"""
                <div class="result {css}">
                  <div class="label">Predicted loan status</div>
                  <div class="status">{icon} {result['status']}</div>
                  <div class="prob">Estimated probability of approval: <b>{probability:.1%}</b>
                    · {mode.lower()} mode</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if result["notes"]:
                st.warning(
                    "**Needs manual review:** " + result["notes"] + ". "
                    "Treat this prediction with extra care.",
                    icon="🔎",
                )
            colour = "#27AE60" if approved else "#CB4335"
            st.markdown(
                f"""
                <div class="gauge">
                  <div class="fill" style="width:{probability * 100:.1f}%; background:{colour};"></div>
                  <div class="mark" style="left:calc({threshold * 100:.0f}% - 1px);" title="cut-off"></div>
                </div>
                <div class="gauge-legend"><span>0% · reject</span>
                  <span class="cut" style="left:{threshold * 100:.0f}%;">▲ cut-off {threshold:.0%}</span>
                  <span>approve · 100%</span></div>
                """,
                unsafe_allow_html=True,
            )

            values = result["values"]
            total = float(values["ApplicantIncome"]) + float(values["CoapplicantIncome"])
            m1, m2 = st.columns(2)
            m1.metric("Household income", f"{total:,.0f}", help="Applicant plus co-applicant income.")
            ratio = float(values["LoanAmount"]) / total if total else float("nan")
            m2.metric("Loan ÷ income", "n/a" if total == 0 else f"{ratio:.3f}",
                      help="Loan amount (thousands) divided by household income. Lower means a lighter burden.")

            st.markdown("#### Why this result?")
            shap_result = result.get("shap")
            if shap_result:
                st.markdown(
                    f"<p class='muted'><b>SHAP explanation.</b> The average applicant starts at "
                    f"<b>{shap_result['base_value']:.1%}</b>. Each field then pushes the probability up (green) "
                    f"or down (red), ending at <b>{probability:.1%}</b> for this applicant.</p>",
                    unsafe_allow_html=True,
                )
                shap_items = shap_result["items"]
                shown_items = shap_items[:6]
                rest = sum(item["contribution"] for item in shap_items[6:])
                rows = [(f"{shap_label(i['feature'])} = {shap_value(i['feature'], i['value'])}", i["contribution"])
                        for i in shown_items]
                if abs(rest) >= 0.0005:
                    rows.append((f"{len(shap_items) - 6} other fields", rest))
                st.markdown(contribution_bars(rows), unsafe_allow_html=True)
                st.caption("Values are percentage points of approval probability. They add up exactly from the "
                           "average to this applicant's probability.")
            items = result["explanation"][:5]
            with st.expander("What if? Swap one field for a typical value"):
                if not items:
                    st.info(
                        "Every field is at (or close to) the typical training value, so swapping any one field "
                        "changes nothing. Try the **Poor credit history** example to see a large effect."
                    )
                else:
                    st.markdown(
                        "<p class='muted'>How much the probability would change if only that field were "
                        "replaced by a typical applicant's value.</p>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        contribution_bars([(FIELD_LABELS.get(i["feature"], i["feature"]), float(i["probability_change"]))
                                           for i in items]),
                        unsafe_allow_html=True,
                    )
            st.markdown(f"<div class='note'>⚠️ {DISCLAIMER}</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- performance tab
with tab_perf:
    test_rows = sum(sum(row) for row in metrics["confusion_matrix"])
    st.markdown(
        f"Scores of the **{model_label}** on **{test_rows} test applications it never saw during training**, "
        f"using the approval cut-off of {threshold:.0%}."
    )
    cards = st.columns(5)
    explained = [
        ("Accuracy", "accuracy", "Share of all test applications predicted correctly."),
        ("Precision", "precision", "Of the applications the model approved, the share that were really approved."),
        ("Recall", "recall", "Of the really approved applications, the share the model also approved."),
        ("F1 score", "f1", "Balance of precision and recall."),
        ("ROC-AUC", "roc_auc", "How well the probabilities rank approved above rejected (1.0 is perfect, 0.5 is guessing)."),
    ]
    for card, (label, key, help_text) in zip(cards, explained):
        value = float(metrics[key])
        card.metric(label, f"{value:.1%}" if key != "roc_auc" else f"{value:.3f}", help=help_text)

    st.markdown("")
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.markdown("#### Confusion matrix")
        tn, fp = metrics["confusion_matrix"][0]
        fn, tp = metrics["confusion_matrix"][1]
        matrix = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=["Actually rejected", "Actually approved"],
            columns=["Predicted rejected", "Predicted approved"],
        )
        st.dataframe(matrix, width="stretch")
        st.markdown(
            f"- ✅ **{tn + tp}** of {tn + fp + fn + tp} correct ({pct(metrics['accuracy'])})\n"
            f"- ⚠️ **{fp}** rejected applications the model would approve (false positives)\n"
            f"- ⚠️ **{fn}** approved application(s) the model would reject (false negatives)\n"
            "- All false positives had a good credit history: the model rejects almost only on credit history."
        )
    with right:
        image = PATHS.reports / "confusion_matrix.png"
        if image.exists():
            st.image(str(image), caption="Confusion matrix (test set)", width="stretch")

    a, b = st.columns(2, gap="large")
    for column, name, caption in (
        (a, "roc_curve.png", "ROC curve: higher and further left is better"),
        (b, "precision_recall_curve.png", "Precision-recall curve"),
    ):
        image = PATHS.reports / name
        if image.exists():
            column.image(str(image), caption=caption, width="stretch")

    st.markdown("#### Standard vs cautious decision mode")
    if cautious_metrics:
        def mode_row(name: str, cut: float, scores: dict) -> dict:
            return {
                "Mode": name,
                "Cut-off": f"{cut:.0%}",
                "Accuracy": pct(scores["accuracy"]),
                "Precision": pct(scores["precision"]),
                "Recall": pct(scores["recall"]),
                "Risky applications caught": f"{scores['true_negative']} of {scores['true_negative'] + scores['false_positive']}",
                "Good applications wrongly rejected": scores["false_negative"],
            }

        st.dataframe(
            pd.DataFrame([
                mode_row("Standard (best accuracy)", standard_threshold, metrics),
                mode_row("Cautious (balanced accuracy)", float(cautious_threshold), cautious_metrics),
            ]),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Both cut-offs were chosen on training data only. Cautious mode catches more of the applications "
            "lenders rejected, which matters because a wrong approval usually costs a bank more than a wrong "
            "rejection, but it also turns away more good applicants."
        )

    predictions_file = PATHS.reports / "test_predictions.csv"
    if predictions_file.exists():
        st.markdown("#### Try your own cut-off")
        test_scores = pd.read_csv(predictions_file)
        chosen = st.slider("Approve when probability is at least", 0.05, 0.95, float(threshold), 0.01)
        predicted = test_scores["approval_probability"] >= chosen
        actual = test_scores["actual_approved"] == 1
        tp, fp = int((predicted & actual).sum()), int((predicted & ~actual).sum())
        tn, fn = int((~predicted & ~actual).sum()), int((~predicted & actual).sum())
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Accuracy", pct((tp + tn) / len(test_scores)))
        s2.metric("Precision", pct(tp / (tp + fp)) if tp + fp else "n/a")
        s3.metric("Risky caught", f"{tn} of {tn + fp}")
        s4.metric("Good wrongly rejected", f"{fn} of {fn + tp}")
        st.caption("Computed on the 123 test applications. This explorer is for understanding the trade-off; "
                   "the two saved modes were chosen without looking at the test set.")

# ---------------------------------------------------------------- insights tab
with tab_insights:
    st.markdown("#### Which model won?")
    comparison = pd.DataFrame(metadata["comparison"])
    table = pd.DataFrame(
        {
            "Model": comparison["model"].map(lambda name: MODEL_NAMES.get(name, name)),
            "CV accuracy": comparison["cv_accuracy_mean"],
            "CV ROC-AUC": comparison["cv_roc_auc_mean"],
            "CV F1": comparison["cv_f1_mean"],
        }
    )
    table.insert(0, "Rank", range(1, len(table) + 1))
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "CV accuracy": st.column_config.ProgressColumn("CV accuracy", format="%.3f", min_value=0.5, max_value=1.0),
            "CV ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
            "CV F1": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.caption(
        "5-fold cross-validation on the training set only. The test set was not used to pick the model. "
        "Every real model beats the always-approve baseline by 10 to 11 points."
    )

    st.markdown("#### Which fields matter most?")
    importance_file = PATHS.reports / "permutation_importance.csv"
    if importance_file.exists():
        importance = pd.read_csv(importance_file)
        importance["Field"] = importance["feature"].map(lambda name: FIELD_LABELS.get(name, name))
        st.bar_chart(
            importance.rename(columns={"importance_mean": "Drop in ROC-AUC when shuffled"}),
            x="Field",
            y="Drop in ROC-AUC when shuffled",
            sort="-Drop in ROC-AUC when shuffled",
            horizontal=True,
            color="#1B4F72",
            height=360,
        )
        st.caption(
            "Permutation importance: shuffle one field and measure how much the model's ROC-AUC drops. "
            "Credit history matters most. Importance shows association, not cause."
        )
    shap_file = PATHS.reports / "shap_importance.csv"
    if shap_file.exists():
        shap_table = pd.read_csv(shap_file)
        shap_table["Field"] = shap_table["feature"].map(shap_label)
        st.bar_chart(
            shap_table.rename(columns={"importance_mean": "Mean |SHAP| (probability points)"}),
            x="Field",
            y="Mean |SHAP| (probability points)",
            sort="-Mean |SHAP| (probability points)",
            horizontal=True,
            color="#148F77",
            height=380,
        )
        st.caption(
            "SHAP importance: the average size of each field's contribution to the approval probability, over "
            "the training applicants. Two different methods agree: credit history matters most."
        )

    robustness_file = PATHS.reports / "robustness_report.json"
    if robustness_file.exists():
        st.markdown("#### How far can the results be trusted?")
        checks = json.loads(robustness_file.read_text(encoding="utf-8"))
        splits = checks["random_splits"]
        nested = checks["nested_cv_accuracy"]
        best_other = max(row["test_accuracy"] for row in checks["stronger_models"].values())
        cheat = max(row["best_accuracy_if_cut_off_picked_on_test"] for row in checks["stronger_models"].values())
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Nested CV accuracy", f"{nested['mean']:.1%}", help="Tuning repeated inside each fold: an unflattered estimate.")
        r2.metric("30 random splits", f"{splits['min']:.0%}–{splits['max']:.0%}", help=f"Mean {splits['mean']:.1%}. Shows how much luck moves the test score.")
        r3.metric("Best stronger model", f"{best_other:.1%}", help="Gradient boosting, HistGradientBoosting and logistic regression under the same fair rules.")
        r4.metric("Ceiling even if cheating", f"{cheat:.1%}", help="Best accuracy if the cut-off were picked using the test answers.")
        rule = checks["credit_history_rule"]
        calibration = checks["calibration"]
        st.caption(
            f"A one-line rule 'approve if credit history is good' scores {rule['test_accuracy']:.1%}, the same as the model. "
            f"Calibration: Brier score {calibration['brier_raw']:.3f} (calibrating only reaches "
            f"{calibration['brier_sigmoid_calibrated']:.3f}). Full details: reports/robustness_report.md."
        )

    with st.expander("Limitations of this model"):
        st.markdown(
            "- The table holds 614 historical applications from a practice problem, not a live loan book.\n"
            "- Gender is excluded from the model. That does not remove bias, because other fields and the "
            "old labels can still carry it.\n"
            "- In practice the model rejects almost only on credit history, so it approves some applicants "
            "that lenders rejected. Cautious mode reduces this at the cost of accuracy.\n"
            "- The dataset has no debts, savings or credit score, so about 85% accuracy is the ceiling here.\n"
            "- Cross-validation scores used to pick the model are slightly optimistic. Quote the test metrics."
        )

# ---------------------------------------------------------------- batch tab
with tab_batch:
    st.markdown("Score many applicants at once from a CSV file with the same columns as the training data.")
    st.caption("The uploaded file stays in this browser session. The app does not write it to disk.")
    upload = st.file_uploader("Applicant CSV", type=["csv"])
    use_sample = st.button("Or try the sample file", disabled=not (PATHS.sample / "sample_applicants.csv").exists())
    batch = None
    if upload is not None:
        batch = pd.read_csv(upload)
    elif use_sample:
        batch = pd.read_csv(PATHS.sample / "sample_applicants.csv")
    if batch is not None:
        try:
            scored, _metadata = score_frame(batch, pipeline=pipeline, metadata=metadata, threshold=threshold)
        except (ValueError, ModelNotFoundError) as exc:
            st.error(str(exc))
        else:
            approved = int((scored["Predicted_Status"] == "Approved").sum())
            flagged = int((scored["Review_Note"] != "").sum())
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Applicants scored", len(scored))
            c2.metric("Predicted approved", approved)
            c3.metric("Predicted rejected", len(scored) - approved)
            c4.metric("Need manual review", flagged, help="Missing credit history, close to the cut-off, "
                      "or a number outside the range seen in training.")
            st.caption(f"Using the {mode.lower()} decision mode (cut-off {threshold:.0%}).")
            st.dataframe(
                scored.head(50),
                width="stretch",
                column_config={
                    "Approval_Probability": st.column_config.ProgressColumn(
                        "Approval probability", format="%.2f", min_value=0.0, max_value=1.0
                    )
                },
            )
            st.download_button(
                "⬇️ Download predictions",
                data=scored.to_csv(index=False).encode("utf-8"),
                file_name="loan_predictions.csv",
                mime="text/csv",
            )

# ---------------------------------------------------------------- about tab
with tab_about:
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Problem")
        st.write(
            "Lenders record applicant income, loan size, credit history and a few personal details, then "
            "approve or reject the loan. This project asks: given those fields, how well can a machine-learning "
            "model predict the historical decision? It is a coursework model, not a lender's decision system."
        )
        st.markdown("#### How it works")
        st.markdown(
            "1. Download and validate the public dataset.\n"
            "2. Clean fields and add income and loan-burden features.\n"
            "3. Train five models with 5-fold cross-validation on 80% of the data.\n"
            "4. Pick the best by cross-validated accuracy and tune the approval cut-off.\n"
            "5. Score the chosen model once on the untouched 20% test set.\n"
            "6. Save the full pipeline and use it in this app."
        )
    with right:
        st.markdown("#### Dataset")
        dataset = metadata.get("dataset") or {}
        source_url = manifest.get("original_source_url") or dataset.get("original_source_url")
        st.markdown(
            f"- **Name:** {manifest.get('dataset_name') or dataset.get('name') or 'n/a'}\n"
            f"- **Records:** {rows_total} applications, "
            f"{(manifest.get('dimensions') or {}).get('columns', 13)} columns\n"
            f"- **Target:** {manifest.get('target_column') or metadata.get('target_column') or 'Loan_Status'} "
            "(Y approved, N rejected)\n"
            "- **Class balance:** 422 approved (68.7%), 192 rejected (31.3%)\n"
            f"- **Downloaded:** {manifest.get('download_date') or dataset.get('download_date') or 'n/a'}"
            + (f"\n- **Source:** [Analytics Vidhya practice problem]({source_url})" if source_url else "")
        )
        st.markdown("#### Technology")
        st.markdown("Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · joblib · Streamlit · pytest")
    licence = manifest.get("licence") or dataset.get("licence")
    if licence:
        st.caption(licence)
