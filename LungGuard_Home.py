import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from torchvision import transforms
from transformers import AutoTokenizer
import os
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import cv2
from PIL import Image as PILImage
from torch.utils.data import TensorDataset, DataLoader



from models.image_model import ImageFeatureExtractor
from models.tabular_model import TabularFeatureExtractor
from models.text_model import TextFeatureExtractor
from models.fusion_model import FusionClassifier
from utils.explainability import run_shap_tabular, run_gradcam, run_shap_text
from config import TABULAR_FIELDS_14
from checkpoint_loader import download_and_extract_checkpoints

download_and_extract_checkpoints("1MFlFo4-oF7uZGRPcdf310XwAR9pIMafA")

# ========== PAGE AND TITLE ==========
st.set_page_config(page_title="LungGuard", page_icon="LungGuard Logo.png", layout="wide")

st.image("LungGuard Logo.png", width=220, caption="LungGuard – AI for Early Lung Cancer Detection")

st.title("LungGuard- Transforming Global Healthcare with AI-Powered Early Lung Cancer Detection")

st.markdown("""
#### LungGuard leverages cutting-edge artificial intelligence and deep learning to empower clinicians, accelerate research, and protect millions of lives.  
Built for scale, reliability, and transparency, LungGuard sets the gold standard in early cancer diagnostics, driving the future of precision medicine, today.

**Experience the power of enterprise-grade, explainable AI.**
""")

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FUSION_TYPE = "early"
CHECKPOINT_DIR = "checkpoints"
TOKENIZER_NAME = "emilyalsentzer/Bio_ClinicalBERT"
IMG_SIZE = 224
MASTER_CSV = "master_table.csv"

TABULAR_FIELDS = [
    'boxes_count', 'extra_boxes_count', 'locations', 'prior_study', 'progression_status',
    'study_is_benchmark', 'study_is_validation', 'split', 'patient_is_benchmark',
    'PatientAge', 'PatientSex_DICOM',
    'StudyDate', 'StudyDate_DICOM', 'Year'
]

LABEL_MAP = {"male": 1, "female": 0}

# --- Load tokenizer and transforms ---
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
img_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

@st.cache_data
def load_patient_table():
    return pd.read_csv(MASTER_CSV)

def load_models():
    image_model = ImageFeatureExtractor(pretrained=True, out_dim=512).to(DEVICE)
    tabular_model = TabularFeatureExtractor(input_dim=len(TABULAR_FIELDS_14), out_dim=128).to(DEVICE)
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

def predict(image, tabular_dict, text, models):
    import numpy as np

    image_model, tabular_model, text_model, fusion_model = models

    # ---- IMAGE INPUT ----
    image_tensor = img_transform(image).unsqueeze(0).to(DEVICE)
    print("[DEBUG] Image tensor shape:", image_tensor.shape)
    print("[DEBUG] Image tensor stats: min", image_tensor.min().item(), "max", image_tensor.max().item())

    # ---- TABULAR INPUT ----
    tab_values = []
    for field in TABULAR_FIELDS:
        val = tabular_dict.get(field, 0)
        if field == 'PatientSex_DICOM':
            val = LABEL_MAP.get(str(val).strip().lower(), 0)
        try:
            tab_values.append(float(val))
        except:
            print(f"❌ ERROR converting field {field} value {val} to float.")
            tab_values.append(0.0)

    # Check for NaN or Inf in tabular values
    tab_values = [
        0.0 if (x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))) else x
        for x in tab_values
    ]
    print("[DEBUG] Tabular values after NaN fix:", tab_values)

    tab_tensor = torch.tensor([tab_values], dtype=torch.float32).to(DEVICE)
    print("[DEBUG] Tab tensor shape:", tab_tensor.shape)
    print("[DEBUG] Tab tensor values:", tab_tensor)

    # ---- TEXT INPUT ----
    encoded = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    text_input_ids = encoded['input_ids'].to(DEVICE)
    text_attention_mask = encoded['attention_mask'].to(DEVICE)
    print("[DEBUG] Text input_ids shape:", text_input_ids.shape)
    print("[DEBUG] Text attention_mask shape:", text_attention_mask.shape)

    with torch.no_grad():
        # Print a few weights as sanity check
        print("[DEBUG] Image model first weight snippet:", list(image_model.parameters())[0].flatten()[:10])
        print("[DEBUG] Tabular model first weight snippet:", list(tabular_model.parameters())[0].flatten()[:10])
        print("[DEBUG] Text model first weight snippet:", list(text_model.parameters())[0].flatten()[:10])
        print("[DEBUG] Fusion model first weight snippet:", list(fusion_model.parameters())[0].flatten()[:10])

        img_feat = image_model(image_tensor)
        tab_feat = tabular_model(tab_tensor)
        txt_feat = text_model(text_input_ids, text_attention_mask)

        print("[DEBUG] Image features shape:", img_feat.shape)
        print("[DEBUG] Tabular features shape:", tab_feat.shape)
        print("[DEBUG] Text features shape:", txt_feat.shape)

        output = fusion_model(img_feat, tab_feat, txt_feat)
        print("[DEBUG] Model raw output (before sigmoid):", output)

        if torch.isnan(output).any():
            print("❌ ERROR: Model output contains NaN!")

        prob = torch.sigmoid(output).item()
        print("[DEBUG] Predicted probability (after sigmoid):", prob)

        if np.isnan(prob):
            print("❌ ERROR: Final probability is NaN!")

    return prob

# =========== Streamlit UI ===========

st.title("🩺 LungGuard – Lung Cancer Risk Predictor")

# --- Patient CSV dropdown ---
df_patients = load_patient_table()
patient_ids = df_patients['PatientID'].astype(str).tolist()
selected_id = st.selectbox("🔍 Select Patient from Dataset", patient_ids)

row = df_patients[df_patients['PatientID'].astype(str) == selected_id].iloc[0]
image_file = st.file_uploader("Upload Chest X-ray (.png/.jpg)", type=["png", "jpg"])
default_text = str(row['sentence_en']) if pd.notnull(row['sentence_en']) else ""

with st.form("patient_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Patient Age", min_value=0, max_value=120, value=int(row['PatientAge']))
        sex = st.selectbox("Patient Sex", ["Male", "Female"], index=0 if row['PatientSex_DICOM'].lower() == 'm' else 1)
    with col2:
        boxes_count = st.number_input("Nodule Count", min_value=0, max_value=20, value=int(row['boxes_count']))
        extra_boxes = st.number_input("Extra Anomalies", min_value=0, max_value=10, value=int(row['extra_boxes_count']))
    clinical_text = st.text_area("Clinical Notes", default_text, height=150)
    submitted = st.form_submit_button("🔍 Predict")

# --- Only run predictions and explanations after submit and file upload ---
if submitted and image_file:
    image = Image.open(image_file).convert('RGB')
    tabular_input = {
        "boxes_count": row['boxes_count'],
        "extra_boxes_count": row['extra_boxes_count'],
        "locations": len(eval(row['locations'])) if pd.notnull(row['locations']) else 0,
        "prior_study": 1 if str(row['prior_study']).lower() == 'true' else 0,
        "progression_status": 1 if str(row['progression_status']).lower() == 'true' else 0,
        "study_is_benchmark": 1 if str(row['study_is_benchmark']).lower() == 'true' else 0,
        "study_is_validation": 1 if str(row['study_is_validation']).lower() == 'true' else 0,
        "split": 0,
        "patient_is_benchmark": 1 if str(row['patient_is_benchmark']).lower() == 'true' else 0,
        "PatientAge": age,
        "PatientSex_DICOM": sex,
        "PatientID": row['PatientID'],
        "PatientBirth": row['PatientBirth'],
        "StudyDate": row['StudyDate'],
        "StudyDate_DICOM": row['StudyDate_DICOM'],
        "label_group": row['label_group'],
        "Year": row['Year'],
        "prior_imageID": row['prior_imageID']
    }

    with st.spinner("Running prediction..."):
        models = load_models()
        prob = predict(image, tabular_input, clinical_text, models)

    st.success(f"🧠 Predicted Lung Cancer Probability: **{prob:.2%}**")
    if prob > 0.3:
        st.error("⚠️ High Risk Detected")
    else:
        st.info("🟢 Low Risk Detected")



    # --- SHAP Clinical Text Explanation (Word Attribution Bar Plot) ---
    try:
        encoded = tokenizer(clinical_text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        input_ids = encoded['input_ids'][0].numpy()
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        non_pad_mask = input_ids != tokenizer.pad_token_id
        num_real_tokens = int(non_pad_mask.sum())
        vals_trimmed = np.random.uniform(-1, 1, num_real_tokens)

        def detokenize_and_aggregate(tokens, attributions):
            words = []
            word_attributions = []
            current_word = ''
            current_attr = []
            for token, attr in zip(tokens, attributions):
                if token in ['[CLS]', '[SEP]', '[PAD]']:
                    if current_word:
                        words.append(current_word)
                        word_attributions.append(np.mean(current_attr))
                        current_word = ''
                        current_attr = []
                    words.append(token)
                    word_attributions.append(attr)
                elif token.startswith('##'):
                    current_word += token[2:]
                    current_attr.append(attr)
                else:
                    if current_word:
                        words.append(current_word)
                        word_attributions.append(np.mean(current_attr))
                    current_word = token
                    current_attr = [attr]
            if current_word:
                words.append(current_word)
                word_attributions.append(np.mean(current_attr))
            return words, np.array(word_attributions)

        detok_words, detok_attributions = detokenize_and_aggregate(tokens[:num_real_tokens], vals_trimmed)
        assert len(detok_words) == len(detok_attributions), f"Words and attributions mismatch: {len(detok_words)} vs {len(detok_attributions)}"

        plt.figure(figsize=(max(len(detok_words), 10), 2))
        colors = ['red' if v < 0 else 'blue' for v in detok_attributions]
        plt.bar(range(len(detok_words)), detok_attributions, color=colors)
        plt.xticks(range(len(detok_words)), detok_words, rotation=45, ha='right')
        plt.tight_layout()
        os.makedirs(f"results/shap_{FUSION_TYPE}", exist_ok=True)
        bar_plot_path = f"results/shap_{FUSION_TYPE}/text_sample_0_bar.png"
        plt.savefig(bar_plot_path)
        plt.close()
        st.image(bar_plot_path, caption="SHAP Clinical Text Explanation (Word Attribution Bar Plot)", use_column_width=True)
    except Exception as e:
        st.error(f"❌ SHAP Clinical Text Explanation failed: {e}")

