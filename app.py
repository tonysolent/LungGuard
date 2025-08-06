import os
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from models.image_model import ImageFeatureExtractor
from models.tabular_model import TabularFeatureExtractor
from models.text_model import TextFeatureExtractor
from models.fusion_model import FusionClassifier
from utils.explainability import run_gradcam, run_shap_tabular, run_shap_text

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINTS_DIR = "checkpoints"
FUSION_TYPES = ['early', 'late', 'hybrid']
TOKENIZER_NAME = "emilyalsentzer/Bio_ClinicalBERT"
IMAGE_SIZE = 224

tabular_features = [
    'boxes_count', 'extra_boxes_count', 'locations', 'prior_study', 'progression_status',
    'study_is_benchmark', 'study_is_validation', 'split', 'patient_is_benchmark',
    'PatientAge', 'PatientSex_DICOM', 'PatientID', 'PatientBirth', 'StudyDate',
    'StudyDate_DICOM', 'label_group', 'Year', 'prior_imageID'
]

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

@st.cache_resource
def load_models(fusion_type, tabular_dim):
    image_model = ImageFeatureExtractor(pretrained=False, out_dim=512).to(DEVICE)
    tabular_model = TabularFeatureExtractor(input_dim=tabular_dim, out_dim=128).to(DEVICE)
    text_model = TextFeatureExtractor(model_name=TOKENIZER_NAME, out_dim=128).to(DEVICE)
    fusion_model = FusionClassifier(
        image_feature_dim=512, tabular_feature_dim=128, text_feature_dim=128,
        fusion_type=fusion_type, num_classes=1).to(DEVICE)

    prefix = f"{CHECKPOINTS_DIR}/{fusion_type}"
    image_model.load_state_dict(torch.load(f"{prefix}_image_model.pt", map_location=DEVICE))
    tabular_model.load_state_dict(torch.load(f"{prefix}_tabular_model.pt", map_location=DEVICE))
    text_model.load_state_dict(torch.load(f"{prefix}_text_model.pt", map_location=DEVICE))
    fusion_model.load_state_dict(torch.load(f"{prefix}_fusion_model.pt", map_location=DEVICE))

    image_model.eval()
    tabular_model.eval()
    text_model.eval()
    fusion_model.eval()

    return image_model, tabular_model, text_model, fusion_model

def preprocess_image(image_file):
    image = Image.open(image_file).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def preprocess_tabular(user_inputs, feature_cols):
    vals = [float(user_inputs[col]) for col in feature_cols]
    tensor = torch.tensor(vals, dtype=torch.float32).unsqueeze(0)
    return tensor.to(DEVICE)

def preprocess_text(text):
    encoded = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    return encoded['input_ids'].to(DEVICE), encoded['attention_mask'].to(DEVICE)

def predict(image_tensor, tabular_tensor, text_input_ids, text_attention_mask,
            image_model, tabular_model, text_model, fusion_model):
    with torch.no_grad():
        img_feat = image_model(image_tensor)
        tab_feat = tabular_model(tabular_tensor)
        txt_feat = text_model(text_input_ids, text_attention_mask)
        out = fusion_model(img_feat, tab_feat, txt_feat)
        prob = torch.sigmoid(out).item()
        return prob

st.title("🩺 LungGuard: Early Lung Cancer Detection")
st.markdown("Upload patient data to predict risk of lung cancer using our multimodal AI model.")

fusion_type = st.selectbox("Select Fusion Type", FUSION_TYPES)

uploaded_image = st.file_uploader("Upload Chest X-ray Image", type=["png", "jpg"])
uploaded_text = st.text_area("Enter Clinical Notes (English)")

tabular_inputs = {}
for col in tabular_features:
    tabular_inputs[col] = st.number_input(f"{col}", min_value=0.0)

if st.button("Predict"):
    if not uploaded_image:
        st.warning("Please upload a chest X-ray image.")
    elif not uploaded_text.strip():
        st.warning("Please enter clinical notes.")
    else:
        try:
            st.info("Running prediction...")
            image_tensor = preprocess_image(uploaded_image)
            tabular_tensor = preprocess_tabular(tabular_inputs, tabular_features)
            text_input_ids, text_attention_mask = preprocess_text(uploaded_text)

            image_model, tabular_model, text_model, fusion_model = load_models(fusion_type, len(tabular_features))
            prob = predict(image_tensor, tabular_tensor, text_input_ids, text_attention_mask,
                           image_model, tabular_model, text_model, fusion_model)

            st.success(f"🧠 Lung Cancer Probability: **{prob*100:.2f}%**")
            if prob >= 0.5:
                st.error("⚠️ High Risk Detected – Further investigation recommended.")
            else:
                st.info("✅ Low Risk Detected – Continue monitoring.")

            # EXPLAINABILITY

            # Create dummy dataloader for Grad-CAM
            dummy_dataloader = [{
                'image': image_tensor,
                'tabular': tabular_tensor,
                'text_input_ids': text_input_ids,
                'text_attention_mask': text_attention_mask
            }]

            explain_dir = os.path.join("results", f"gradcam_{fusion_type}")
            os.makedirs(explain_dir, exist_ok=True)
            if fusion_type in ['early', 'hybrid']:
                st.subheader("🧬 Grad-CAM (Image)")
                run_gradcam(image_model, fusion_model, dummy_dataloader, DEVICE, fusion_type=fusion_type, save_dir="results")
                st.image(os.path.join(explain_dir, "sample_0.png"), caption="Grad-CAM Example")

            explain_dir = os.path.join("results", f"shap_{fusion_type}")
            os.makedirs(explain_dir, exist_ok=True)
            st.subheader("📊 SHAP Explanations")
            run_shap_tabular(tabular_model, dummy_dataloader, DEVICE, fusion_type, tabular_features, save_dir="results")
            st.image(os.path.join(explain_dir, "tabular_summary.png"), caption="SHAP Tabular Summary")
            run_shap_text(text_model, dummy_dataloader, DEVICE, fusion_type, tokenizer, save_dir="results")
            st.image(os.path.join(explain_dir, "text_sample_0.png"), caption="SHAP Text Explanation")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
