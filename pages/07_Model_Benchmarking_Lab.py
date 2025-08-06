import streamlit as st
import pandas as pd
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from reportlab.pdfgen import canvas
import tempfile
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
from sklearn.calibration import calibration_curve

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import plotly.graph_objects as go

st.set_page_config(page_title="Model Benchmarking Lab", layout="wide")
st.title("🧪 Benchmarking Lab & Advanced Reporting")

st.markdown("""
Upload a classification dataset and benchmark up to **3 models side-by-side**.  
Includes SHAP explainability, radar metrics, training time, ROC/AUC, calibration curves, and PDF export.
""")

uploaded_file = st.file_uploader("Upload your classification CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Data loaded: {df.shape[0]} rows × {df.shape[1]} columns.")
    st.dataframe(df.head(10), use_container_width=True)

    # Exclude unreliable targets
    excluded_targets = [
        "progression_status", "study_is_benchmark",
        "study_is_validation", "patient_is_benchmark"
    ]
    cat_targets = [
        col for col in df.columns
        if df[col].nunique() <= 10
        and df[col].dtype != 'float'
        and col not in excluded_targets
    ]

    st.subheader("Step 1: Select Target & Features")
    target_col = st.selectbox("Target column", options=cat_targets)
    all_features = [col for col in df.columns if col != target_col]
    feature_cols = st.multiselect("Feature columns", options=all_features, default=all_features[:20])

    if target_col and feature_cols:
        if len(feature_cols) > 20:
            st.warning("Too many features selected. Limiting to first 20.")
            feature_cols = feature_cols[:20]

        X = df[feature_cols].copy()
        y = df[target_col].astype(str)

        for col in X.select_dtypes('object').columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        y_enc = LabelEncoder().fit_transform(y)
        X = X.fillna(X.mean())

        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.3, random_state=42)

        model_options = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier()
        }

        selected_models = st.multiselect("Select models (max 3)", list(model_options.keys()), default=list(model_options.keys())[:2])
        if len(selected_models) > 3:
            st.warning("Max 3 models allowed for performance reasons.")
            selected_models = selected_models[:3]

        auto_metric = st.radio("Auto-select best model by:", ["F1 Score", "AUC"])
        results, train_times, roc_data = [], {}, {}

        @st.cache_resource(show_spinner=False)
        def train_model(name, X_train, y_train):
            model = model_options[name]
            model.fit(X_train, y_train)
            return model

        with st.spinner("Training and benchmarking models..."):
            for model_name in selected_models:
                start_time = time.time()
                model = train_model(model_name, X_train, y_train)
                end_time = time.time()
                train_times[model_name] = round(end_time - start_time, 2)

                y_pred = model.predict(X_test)
                y_score = model.predict_proba(X_test)[:, 1] if (
                    hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2
                ) else y_pred

                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                try:
                    auc = roc_auc_score(y_test, y_score) if len(np.unique(y_test)) == 2 else np.nan
                except:
                    auc = np.nan

                results.append({
                    "Model": model_name,
                    "Accuracy": round(acc, 3),
                    "F1 Score": round(f1, 3),
                    "AUC": round(auc, 3) if not np.isnan(auc) else "N/A",
                    "Training Time (s)": train_times[model_name]
                })

                if len(np.unique(y_test)) == 2:
                    fpr, tpr, _ = roc_curve(y_test, y_score)
                    roc_data[model_name] = (fpr, tpr)
                else:
                    roc_data[model_name] = None

        # Leaderboard
        st.subheader("Step 2: Model Leaderboard")
        leaderboard_df = pd.DataFrame(results).sort_values(by=auto_metric, ascending=False)
        st.dataframe(leaderboard_df, use_container_width=True)

        top_model = leaderboard_df.iloc[0]["Model"]
        top_model_instance = train_model(top_model, X_train, y_train)
        y_pred_top = top_model_instance.predict(X_test)

        # Confusion Matrix
        st.subheader("Step 3: Confusion Matrix – Best Model")
        cm = confusion_matrix(y_test, y_pred_top)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix: {top_model}")
        st.pyplot(fig)

        # ROC Curve (binary only)
        if len(np.unique(y_test)) == 2:
            st.subheader("Step 4: ROC Curve Comparison")
            fig_roc, ax_roc = plt.subplots()
            for name, roc in roc_data.items():
                if roc:
                    fpr, tpr = roc
                    ax_roc.plot(fpr, tpr, label=name)
            ax_roc.plot([0, 1], [0, 1], 'k--')
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("ROC Curve")
            ax_roc.legend()
            st.pyplot(fig_roc)

        # Radar Chart
        st.subheader("Step 5: Radar Chart – Metric Profile")
        radar_fig = go.Figure()
        for r in results:
            radar_fig.add_trace(go.Scatterpolar(
                r=[r["Accuracy"], r["F1 Score"], r["AUC"] if isinstance(r["AUC"], float) else 0],
                theta=["Accuracy", "F1 Score", "AUC"],
                fill='toself',
                name=r["Model"]
            ))
        radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
        st.plotly_chart(radar_fig, use_container_width=True)

        # Calibration Curve
        if len(np.unique(y_test)) == 2:
            st.subheader("Step 6: Calibration Curve")
            fig_cal, ax_cal = plt.subplots()
            for model_name in selected_models:
                model = train_model(model_name, X_train, y_train)
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:, 1]
                    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
                    ax_cal.plot(prob_pred, prob_true, marker='o', label=model_name)
            ax_cal.plot([0, 1], [0, 1], 'k--')
            ax_cal.set_xlabel("Predicted Probability")
            ax_cal.set_ylabel("Observed Proportion")
            ax_cal.legend()
            st.pyplot(fig_cal)

        # SHAP Explainability (Tree Models)
        for model_name in selected_models:
            model = train_model(model_name, X_train, y_train)
            if "Tree" in model_name or "Forest" in model_name or "Boosting" in model_name:
                try:
                    st.subheader(f"SHAP Summary – {model_name}")
                    explainer = shap.Explainer(model, X_test)
                    shap_values = explainer(X_test)
                    shap.plots.beeswarm(shap_values, max_display=10, show=False)
                    st.pyplot(bbox_inches='tight')
                except Exception as e:
                    st.warning(f"SHAP not available for {model_name}: {e}")

        # PDF Export
        st.subheader("Step 7: Export to PDF")
        if st.button("Generate PDF Report"):
            tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
            c = canvas.Canvas(tmp_path)
            c.setFont("Helvetica", 12)
            c.drawString(50, 800, "LungGuard: Model Benchmarking Report")
            y = 780
            c.drawString(50, y, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
            y -= 30
            for r in leaderboard_df.to_dict("records"):
                c.drawString(50, y, f"Model: {r['Model']}")
                y -= 20
                c.drawString(70, y, f"Accuracy: {r['Accuracy']}, F1: {r['F1 Score']}, AUC: {r['AUC']}, Time: {r['Training Time (s)']}s")
                y -= 30
            c.drawString(50, y, f"Auto-selected Top Model: {top_model}")
            c.save()
            with open(tmp_path, "rb") as f:
                st.download_button("Download PDF Report", data=f, file_name="benchmarking_report.pdf", mime="application/pdf")
            os.remove(tmp_path)

else:
    st.info("Upload a classification dataset to begin benchmarking.")

st.caption("© 2025 Benchmarking Lab | MSc Applied AI and Data Science, Solent University")
