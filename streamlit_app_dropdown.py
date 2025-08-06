
import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
import os
import pandas as pd

from models.image_model import ImageFeatureExtractor
from models.tabular_model import TabularFeatureExtractor
from models.text_model import TextFeatureExtractor
from models.fusion_model import FusionClassifier
from utils.explainability import run_shap_tabular
from torch.utils.data import TensorDataset, DataLoader

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FUSION_TYPE = "early"
CHECKPOINT_DIR = "checkpoints"
TOKENIZER_NAME = "emilyalsentzer/Bio_ClinicalBERT"
IMG_SIZE = 224
TABULAR_FIELDS = [
    'boxes_count', 'extra_boxes_count', 'locations', 'prior_study', 'progression_status',
    'study_is_benchmark', 'study_is_validation', 'split', 'patient_is_benchmark',
    'PatientAge', 'PatientSex_DICOM', 'PatientID', 'PatientBirth',
    'StudyDate', 'StudyDate_DICOM', 'label_group', 'Year', 'prior_imageID'
]
LABEL_MAP = {"male": 1, "female": 0, "m": 1, "f": 0, "M": 1, "F": 0}

# Load tokenizer and transform
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
img_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# Load master table for dropdown
master_df = pd.read_csv("master_table.csv").dropna(subset=TABULAR_FIELDS)
master_df = master_df.reset_index(drop=True)
dropdown_options = [
    f"PatientID: {row['PatientID']} | Age: {row['PatientAge']} | Sex: {row['PatientSex_DICOM']} | Date: {row['StudyDate']}"
    for _, row in master_df.iterrows()
]

# --- Load Models ---
def load_models():
    image_model = ImageFeatureExtractor(pretrained=False, out_dim=512).to(DEVICE)
    tabular_model = TabularFeatureExtractor(input_dim=len(TABULAR_FIELDS), out_dim=128).to(DEVICE)
    text_model = TextFeatureExtractor(model_name=TOKENIZER_NAME, out_dim=128).to(DEVICE)
    fusion_model = FusionClassifier(
        image_feature_dim=512, tabular_feature_dim=128, text_feature_dim=128,
        fusion_type=FUSION_TYPE, num_classes=1
    ).to(DEVICE)

    image_model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, f"image_model_{FUSION_TYPE}.pt"), map_location=DEVICE))
    tabular_model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, f"tabular_model_{FUSION_TYPE}.pt"), map_location=DEVICE))
    text_model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, f"text_model_{FUSION_TYPE}.pt"), map_location=DEVICE))
    fusion_model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, f"fusion_model_{FUSION_TYPE}.pt"), map_location=DEVICE))

    image_model.eval()
    tabular_model.eval()
    text_model.eval()
    fusion_model.eval()
    return image_model, tabular_model, text_model, fusion_model

# --- Inference ---
def predict(image, tabular_dict, text, models):
    image_model, tabular_model, text_model, fusion_model = models
    image_tensor = img_transform(image).unsqueeze(0).to(DEVICE)

    tab_values = []
    for field in TABULAR_FIELDS:
        val = tabular_dict.get(field, 0)
        if field == 'PatientSex_DICOM':
            val = LABEL_MAP.get(str(val).strip().lower(), 0)
        tab_values.append(val)
    tab_tensor = torch.tensor([tab_values], dtype=torch.float32).to(DEVICE)

    encoded = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    text_input_ids = encoded['input_ids'].to(DEVICE)
    text_attention_mask = encoded['attention_mask'].to(DEVICE)

    with torch.no_grad():
        img_feat = image_model(image_tensor)
        tab_feat = tabular_model(tab_tensor)
        txt_feat = text_model(text_input_ids, text_attention_mask)
        output = fusion_model(img_feat, tab_feat, txt_feat)
        prob = torch.sigmoid(output).item()
    return prob

# --- UI ---
st.title("🩺 LungGuard – Lung Cancer Risk Predictor")

image_file = st.file_uploader("Upload Chest X-ray Image (.png/.jpg)", type=["png", "jpg"])
selected_patient = st.selectbox("🔍 Select Patient from CSV", options=dropdown_options)
selected_index = dropdown_options.index(selected_patient)
selected_row = master_df.iloc[selected_index]

clinical_text = st.text_area("📝 Clinical Notes", "Multiple nodules seen in upper lobe...", height=150)
if st.button("🔍 Predict"):
    if image_file:
        image = Image.open(image_file).convert("RGB")
        tabular_input = {k: selected_row[k] for k in TABULAR_FIELDS}
        tabular_input["PatientSex_DICOM"] = selected_row["PatientSex_DICOM"]

        with st.spinner("Running prediction..."):
            models = load_models()
            prob = predict(image, tabular_input, clinical_text, models)

        st.success(f"🧠 Predicted Lung Cancer Probability: **{prob:.2%}**")
        if prob > 0.5:
            st.error("⚠️ High Risk Detected")
        else:
            st.info("🟢 Low Risk")

        # --- SHAP Tabular Explanation ---
        st.subheader("📊 SHAP Explanation (Tabular Features)")
        try:
            tab_values = []
            for field in TABULAR_FIELDS:
                val = tabular_input.get(field, 0)
                if field == 'PatientSex_DICOM':
                    val = LABEL_MAP.get(str(val).strip().lower(), 0)
                tab_values.append(val)
            tab_tensor = torch.tensor([tab_values], dtype=torch.float32).to(DEVICE)
            dummy_loader = DataLoader(TensorDataset(tab_tensor), batch_size=1)

            run_shap_tabular(
                tabular_model=models[1],
                dataloader=dummy_loader,
                device=DEVICE,
                fusion_type=FUSION_TYPE,
                tabular_feature_names=TABULAR_FIELDS.copy(),
                save_dir="results"
            )
            shap_path = f"results/shap_{FUSION_TYPE}/tabular_summary.png"
            st.image(shap_path, caption="SHAP Feature Importance", use_column_width=True)

        except Exception as e:
            st.error(f"❌ SHAP failed: {e}")
    else:
        st.warning("📷 Please upload a chest X-ray image.")
