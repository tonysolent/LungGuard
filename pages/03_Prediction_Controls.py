import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Prediction Controls & What-If Analysis", layout="wide")
st.title("⚙️ Prediction Controls & What-If Analysis")

st.markdown("""
Use this advanced page to experiment with:
- **Threshold for High Probability:** Set what probability you consider as "high risk" for lung cancer, with detailed explanation on hover.
- **Prediction Rate Increase Factor:** Explore how artificially increasing or decreasing prediction confidence affects results (experimental).
- **Upload all patient data and run a prediction with your selected settings.**
""")

# --- Threshold Slider with Hover Explanation ---
threshold = st.slider(
    "Threshold for High Probability (default: 0.5)",
    min_value=0.1, max_value=0.99, value=0.5, step=0.01,
    help=(
        "This slider sets the minimum probability above which the model will label a case as 'high risk of lung cancer'.\n"
        "• Lowering this value makes the model more sensitive (flags more patients as high risk).\n"
        "• Increasing this value makes the model more specific (flags only the most confident cases as high risk).\n"
        "Choose this setting based on your risk tolerance and the clinical scenario."
    ),
)
st.caption(f"Current threshold: `{threshold:.2f}`")

# --- Prediction Rate Increase Slider with Hover Explanation ---
pred_boost = st.slider(
    "Prediction Rate Increase Factor (experimental)",
    min_value=0.5, max_value=2.0, value=1.0, step=0.01,
    help=(
        "This advanced, experimental setting multiplies the model's risk probability by the chosen factor before making the high-risk decision.\n"
        "• Use values > 1.0 to simulate a model that makes higher predictions (more aggressive, more positive results).\n"
        "• Use values < 1.0 to simulate a model that is less confident (more conservative, fewer positives).\n"
        "Set to 1.0 for normal use. Intended for scenario testing, research, and robustness experiments only."
    ),
)
st.caption(f"Current boost factor: `{pred_boost:.2f}`")

st.divider()

# --- Patient Data Uploads ---
st.header("Step 1: Upload Patient Data")
uploaded_image = st.file_uploader(
    "Upload Chest X-ray (PNG/JPG)", 
    type=["png", "jpg", "jpeg"], 
    help="Upload a chest X-ray image file. Required for multimodal prediction."
)
uploaded_tabular = st.file_uploader(
    "Upload Tabular Data (CSV)", 
    type=["csv"], 
    help="Upload patient tabular data (as a .csv file) containing risk factors and clinical details."
)
clinical_note = st.text_area(
    "Paste Clinical Text (Doctor's Note)", 
    "", 
    help="Paste or type the clinical note (free text) that the doctor has written about the case."
)

if st.button("Predict Lung Cancer Risk", use_container_width=True, type="primary"):
    if not uploaded_image or not uploaded_tabular or not clinical_note:
        st.warning("Please provide all three inputs: X-ray image, tabular CSV, and clinical note.")
    else:
        with st.spinner("Running multimodal fusion prediction..."):
            # TODO: Replace the below dummy logic with your real model!
            np.random.seed(42)
            raw_pred_prob = np.random.uniform(0, 1)  # Placeholder for your model's output

            # Apply prediction rate boost
            pred_prob = np.clip(raw_pred_prob * pred_boost, 0, 1)

            is_high_risk = pred_prob >= threshold

        st.success(f"**Predicted Probability of Lung Cancer:** `{pred_prob:.2f}` (with boost factor `{pred_boost:.2f}`)")
        st.markdown(
            f"**Risk Category:** {'🟥 HIGH RISK' if is_high_risk else '🟩 LOW RISK'} (using threshold `{threshold:.2f}`)"
        )

st.markdown("---")
st.info(
    "All advanced controls, uploads, and predictions are available **only on this page**. "
    "Use the hover (mouse over the sliders) to see detailed explanations for each setting."
)

st.caption("© 2025 LungGuard | MSc Applied AI and Data Science, Solent University | For research and educational use only.")
