import streamlit as st

st.set_page_config(page_title="Help & User Guide", layout="wide")
st.title("📖 LungGuard Help & User Guide")

st.markdown("""
Welcome to **LungGuard – A Multimodal AI System for Early Lung Cancer Detection**.

This page provides step-by-step guidance for using each feature in the app. It is designed to support non-technical users, MSc examiners, clinicians, and stakeholders.
""")

st.header("System Overview")
st.markdown("""
LungGuard enables:
- Uploading clinical patient data (image, text, tabular)
- Running real-time AI predictions
- Benchmarking and explaining multiple models
- Exploring fairness, bias, and societal impact
- Producing exportable results and ethical audit trails
""")

st.header("🗂️ Feature-by-Feature Guide")

# 1
st.subheader("1️⃣ Grad-CAM Overlay (Visual Explainability)")
st.markdown("""
This page lets you upload a chest X-ray and generate a clinical-quality Grad-CAM, Grad-CAM++, or EigenCAM overlay.

**Purpose:**  
- To reveal which regions of the X-ray most influenced your AI model’s decision (explainable AI).

**How to use:**
- Upload a chest X-ray in JPG, PNG, TIFF, or BMP format.
- Select the explainability method:  
    - Grad-CAM: Standard clinical saliency map  
    - Grad-CAM++: Finer, more detailed focus  
    - EigenCAM: Advanced spectral focus
- Adjust overlay opacity (0–100%) for best visibility of anatomy vs. heatmap.
- Click “Generate Grad-CAM Overlay” to view results.
- Download overlay as PNG (for reports) or DICOM (for PACS/clinical systems, research use only).

**Interpretation:**  
- Red/yellow highlights show “attention” areas for the model’s predicted class.
- Blue/cold regions are less influential in the decision.

**Ethical Note:**  
- Overlays are for research/education, not for clinical diagnosis.
""")

# 2
st.subheader("2️⃣Grad-CAM Core Logic (Technical Reference)")
st.markdown("""
This technical reference page provides insights into how Grad-CAM overlays are generated for explainable AI:

- **Model loading**: Uses a custom-trained convolutional network for X-ray analysis.
- **Image preprocessing**: Ensures 8-bit/16-bit grayscale or RGB X-rays are safely converted for clinical use.
- **Feature map extraction**: Grad-CAM hooks into the last (or user-selected) convolutional layer to calculate a saliency map.
- **Overlay creation**: The heatmap is upsampled to the X-ray resolution and blended using transparent colormaps.
- **Download/Export**: Results can be saved as PNG for dissertation submission, or as DICOM for integration into clinical imaging workflows (advanced).

**Best Practice:**  
- Always interpret overlays alongside the original image.
- Use this logic to understand model focus, detect bias, or validate clinical relevance.

For technical details or to customise the Grad-CAM pipeline, see the comments and docstrings in the actual page code.
""")



# 2
st.subheader("3️⃣ Data Exploration EDA")
st.markdown("""
Upload a CSV to explore:
- Dataset shape, column types
- Missing values
- Categorical imbalance
- Outlier detection

Tooltips help explain the importance of each element for medical ML readiness.
""")

# 1
st.subheader("4️⃣  Prediction Controls")
st.markdown("""
Upload image/tabular/text data to get early lung cancer risk predictions.

Features:
- Adjustable probability threshold slider  
- Prediction tuning slider to simulate sensitivity boosts  
- Tooltip hovers to explain controls  
- SHAP and Grad-CAM integrated (Explainable AI)
""")


# 3
st.subheader("5️⃣ Visualisation Gallery")
st.markdown("""
Generate interactive visuals:
- Heatmaps (correlation)
- Radar charts (multivariate profiles)
- Box plots, pie charts, treemaps

Dropdowns dynamically filter valid column options. No-code users can explore relationships visually.
""")

# 4
st.subheader("6️⃣ Model Playground")
st.markdown("""
Experiment with:
- Regression: Linear, Decision Tree  
- Classification: Logistic, Decision Tree  
- Clustering: K-Means  

Includes auto encoding, validation splitting, performance metrics, confusion matrix, and ROC.
""")

# 5
st.subheader("7️⃣ Time Series Forecasting")
st.markdown("""
For longitudinal clinical or imaging data:
- Choose valid datetime and numeric value columns  
- Visualise trends and seasonality (via decomposition)  
- Forecast using ARIMA (configurable p, d, q)  
""")

# 6
st.subheader("8️⃣ Societal Impact Simulation")
st.markdown("""
This page explores how AI affects people and systems:
- Patient stories and scenario walkthroughs  
- Clinical workflow mapping (PlantUML diagram)  
- Stakeholder perspective matrix (e.g. radiologist, NHS, policymakers)

Supports ethical and policy justification in MSc dissertations and real-world trials.
""")

# 7
st.subheader("9️⃣ Model Benchmarking Lab")
st.markdown("""
Upload a labelled dataset and compare up to 3 classifiers.

Outputs include:
- Accuracy, F1 Score, AUC
- Radar plots for model profiling  
- Calibration curves  
- SHAP summary plots (tree models)  
- ROC curve (binary targets)  
- PDF export for submission-ready appendices  
- Auto-selection of best model by AUC or F1
""")

# 8
st.subheader("🔟 Deployment and Professional Audit")
st.markdown("""
Explore how LungGuard is production-ready:
- System architecture diagram  
- Component responsibilities  
- Technical risk checklist  
- Transparency and fairness compliance  
- Downloadable audit sheet (coming soon)
""")

# 9
st.subheader("1️⃣1️⃣ AI Ethics and Risk Assessment")
st.markdown("""
- BCS Code of Conduct summary  
- Interactive quiz on AI bias and fairness  
- Dataset scanning for:
  - Class imbalance  
  - Privacy-sensitive columns  
  - Missing values  
- Simulates real-world ethical compliance like NICE/DHSC frameworks
""")

# 10
st.subheader("1️⃣2️⃣ AI and Data Science Glossary")
st.markdown("""
Search or browse technical concepts:
- 50+ glossary entries
- Definitions + module mapping (e.g. COM727, COM724)  
- Usage examples  
- Future: term submission for crowdsourced glossary building  
""")

# 11
st.subheader("📖 Help and User Guide")
st.markdown("""
This guide outlines every LungGuard feature and how to use it.

Use it as:
- A navigation starter  
- A viva support document  
- A user-facing instruction page  
""")

st.header("💡 Best Practice Tips")
st.markdown("""
- ✅ Upload one CSV per page to avoid memory overload  
- 🔍 Hover tooltips guide users through unfamiliar terms  
- ⬇️ Use export buttons for PDF reporting  
- 🚫 Do not upload real patient IDs — use anonymised or synthetic datasets only
""")

st.header("🎓 Project Metadata")
st.markdown("""
- **Project Title**: LungGuard – A Multimodal AI System for Early Lung Cancer Detection  
- **Student**: Tony Livins  
- **Student ID**: 15735320           
- **Supervisor**: Dr. Kashif Talpur (Lecturer in AI & Data Science, Solent University)  
- **Course**: MSc Applied AI & Data Science  
- **Module Code**: COM726 – MSc Dissertation  
- **Institution**: Solent University  
- **Modules Integrated**:
  - COM727 – Introduction to AI  
  - COM724 – Applied AI in Business  
  - COM725 – Data Analytics and Visualisation  
  - COM731 – Programming for Problem Solving  
""")

st.caption("© 2025 LungGuard | MSc Applied AI and Data Science | Solent University | Developed by Tony Livins.")
