# 🫁 LungGuard – A Multimodal Deep Learning System for Early Lung Cancer Detection

**LungGuard** is an MSc Applied AI & Data Science dissertation project that develops a **multimodal AI system** integrating:

- **Chest X‑ray images** (image modality)
- **Structured patient data** (tabular modality)
- **Clinical text notes** (NLP modality)

to enable **early lung cancer detection** with **explainable AI**.

---

## 🌟 Key Features

- **Multimodal Fusion Architectures**
  - Early, Late, and Hybrid fusion of image, tabular, and text data
- **Explainable AI (XAI)**
  - Grad‑CAM for visual interpretability of chest X‑rays
  - SHAP for tabular and textual clinical data
- **Streamlit Web Application**
  - Interactive dashboard for clinicians and researchers
  - Prediction controls and “what‑if” scenario testing
- **Time Series & EDA Modules**
  - Built‑in pages for dataset exploration and trend forecasting
- **Research‑Grade Model Training**
  - Trained using PyTorch with GPU support
  - Models stored via Git LFS to handle large checkpoints

---

## 📂 Repository Structure

```
LungGuard/
│
├── LungGuard_Home.py           # Main Streamlit homepage
├── 01_GradCAM_Overlay.py       # Grad-CAM visual explainability
├── 02_GradCAM_Core_Logic.py    # Core Grad-CAM logic & DICOM export
├── 03_Data_Exploration_EDA.py  # Automated EDA
├── 03_Prediction_Controls.py   # Threshold tuning & prediction
├── 03_Visualisation_Gallery.py # Interactive visualisation gallery
├── 04_Model_Playground.py      # Regression/Classification/Clustering sandbox
├── 05_Time_Series_Forecasting.py # ARIMA forecasting tool
│
├── models/                     # Model definitions
│   ├── image_model.py
│   ├── tabular_model.py
│   ├── text_model.py
│   └── fusion_model.py
│
├── checkpoints/                # Trained model weights (via Git LFS)
├── utils/                       # Explainability and helper scripts
├── master_table.csv             # Multimodal metadata table
└── README.md
```

---

## ⚡ Installation

1. **Clone Repository**
```bash
git clone git@github.com:tonysolent/LungGuard.git
cd LungGuard
```

2. **Create Virtual Environment & Install Dependencies**
```bash
python -m venv lungguard_env
.\lungguard_env\Scripts\activate  # On Windows
pip install -r requirements.txt
```

3. **Run Streamlit App**
```bash
streamlit run LungGuard_Home.py
```

---

## 🧠 Research Context

This project is part of the **MSc Applied AI & Data Science** dissertation (COM726) at **Solent University**, aiming for **early-stage lung cancer detection** using **multimodal fusion** of:

- **CNN (Image)**
- **MLP (Tabular)**
- **Transformer/Bio_ClinicalBERT (Text)**

with **explainable AI** to support **clinician trust** and **ethical AI deployment**.

---

## 📊 Evaluation

- **Metrics:** Accuracy, Recall, Precision, F1‑Score, AUC  
- **Explainability:** Grad‑CAM (image), SHAP (tabular/text)  
- **Model Fusion:** Early, Late, and Hybrid evaluated

---

## ⚖️ License

For **academic and research purposes only**.  
All datasets used are **publicly available** and processed for **educational use** in compliance with ethical AI research standards.

---

## 🙌 Acknowledgements

- **PyTorch** and **HuggingFace Transformers** for model development  
- **Streamlit** for building the interactive web dashboard  
- **Open-source chest X‑ray datasets** such as MIMIC‑CXR and LIDC‑IDRI  
- **Solent University MSc Applied AI & Data Science** program for guidance

---

© 2025 Tony Livins | MSc Applied AI & Data Science Dissertation Project | Supervisor- Dr. Kashif Talpur

