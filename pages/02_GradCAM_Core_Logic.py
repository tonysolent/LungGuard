import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
from models.image_model import ImageFeatureExtractor 
from utils.explainability import run_gradcam, run_gradcam_pp, run_eigencam

# ------- MODEL LOADING -------
@st.cache_resource
def load_model():
    # Loads the trained image model for prediction and Grad-CAM.
    model = ImageFeatureExtractor(pretrained=True, out_dim=512)
    model.load_state_dict(torch.load("checkpoints/image_model_early.pt", map_location="cpu"))
    model.eval()
    model.to("cpu")
    return model

model = load_model()
device = "cpu"

# ------- IMAGE HANDLING -------
def safe_open_xray(file):
    """
    Loads and normalizes 8/16-bit medical X-ray files, ensuring output is display- and model-ready.
    """
    raw_img = Image.open(file)
    img_arr = np.array(raw_img)
    if img_arr.ndim == 2 and img_arr.max() > 255:
        img_arr = (255 * (img_arr / img_arr.max())).astype(np.uint8)
    img = Image.fromarray(img_arr).convert("RGB")
    return img

def preprocess_xray(img, size=(224,224)):
    """Resizes the image for model input (default 224x224)."""
    img = img.resize(size)
    return img

def get_tensor(img):
    """
    Converts image to PyTorch tensor, using medical-neutral normalization (mean/std=0.5).
    """
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])
    return transform(img).unsqueeze(0)

# ------- OVERLAY FUNCTION -------
def make_overlay(orig_img, cam, alpha=0.35, blur=True):
    """
    Creates a clinical-quality Grad-CAM overlay with optional Gaussian blur and user-chosen transparency.
    """
    orig_np = np.array(orig_img)
    cam_resized = cv2.resize(cam, (orig_np.shape[1], orig_np.shape[0]), interpolation=cv2.INTER_LINEAR)
    if blur:
        cam_resized = cv2.GaussianBlur(cam_resized, (13, 13), 0)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32)
    overlay = cv2.addWeighted(heatmap, alpha, orig_np.astype(np.float32), 1 - alpha, 0)
    return np.uint8(overlay)

# ------- DICOM EXPORT FUNCTION -------
def save_as_dicom(overlay_img, filename="gradcam_overlay.dcm"):
    """
    Exports the Grad-CAM overlay as a DICOM file for clinical/PACS compatibility (for research use).
    """
    import pydicom
    from pydicom.dataset import FileDataset
    import datetime
    tempfile = f"./{filename}"
    pil_img = Image.fromarray(overlay_img)
    arr = np.array(pil_img)
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ds = FileDataset(tempfile, {}, preamble=b"\0" * 128)
    ds.Modality = "OT"
    ds.ContentDate = str(datetime.date.today()).replace('-', '')
    ds.ContentTime = str(datetime.datetime.now().time()).replace(':', '').split('.')[0]
    ds.PatientName = "LungGuard^Test"
    ds.Rows, ds.Columns = arr.shape[0], arr.shape[1]
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = arr.tobytes()
    ds.save_as(tempfile)
    return tempfile

# ------- MODEL PREDICTION -------
def predict_image(img, model, device="cpu"):
    """
    Makes a prediction and returns both the predicted label and its probability.
    """
    from torchvision import transforms
    model.eval()
    model.to(device)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output)
        prob_scalar = prob.view(-1)[0].item()
        pred_label = int(prob_scalar >= 0.5)
    return pred_label, prob_scalar

def get_class_labels():
    """
    Returns the label dictionary for model outputs (edit if you add classes).
    """
    return {0: "No Cancer", 1: "Lung Cancer"}

# ======= STREAMLIT UI =======
st.title("LungGuard: Grad-CAM Visual Explainability")
st.write("""
This interactive tool lets you upload a chest X-ray and instantly generate a Grad-CAM, Grad-CAM++, or EigenCAM heatmap overlay.
These overlays reveal which parts of the image most influenced your AI model's decision, supporting explainable AI in clinical workflows.
""")

uploaded_file = st.file_uploader(
    "Upload Chest X-ray (.jpg, .png, .tif, .bmp)",
    type=['jpg', 'jpeg', 'png', 'tif', 'bmp'],
    help="Supported formats: JPG, PNG, TIFF, BMP. Handles 8/16-bit grayscale or color X-rays. "
)

if uploaded_file:
    img = safe_open_xray(uploaded_file)
    orig_img = preprocess_xray(img)
    img_tensor = get_tensor(orig_img).to(device)

    # --- Grad-CAM Variant Selection ---
    cam_method = st.selectbox(
        "Choose explainability method:",
        ["Grad-CAM", "Grad-CAM++", "EigenCAM"],
        help="Select Grad-CAM (standard), Grad-CAM++ (more detail), or EigenCAM (advanced) for explainability overlay."
    )
    opacity = st.slider(
        "Overlay Opacity (%)",
        min_value=10, max_value=90, value=35,
        help="Controls transparency of the Grad-CAM overlay (lower = more X-ray visible, higher = more heatmap visible)."
    )

    # --- Model Prediction ---
    class_labels = get_class_labels()
    pred_label, pred_prob = predict_image(orig_img, model, device=device)
    st.markdown(
        f"**Prediction:** `{class_labels.get(pred_label, 'Unknown')}` &nbsp;&nbsp; **Probability:** `{pred_prob*100:.1f}%`",
        help="The model's prediction and its probability/confidence."
    )

    # --- Compute Grad-CAM ---
    if st.button("Generate Grad-CAM Overlay", help="Click to compute the Grad-CAM overlay using your selected method."):
        with st.spinner("Computing explanation..."):
            if cam_method == "Grad-CAM":
                cam = run_gradcam(model, img_tensor, device=device, return_numpy=True)
            elif cam_method == "Grad-CAM++":
                cam = run_gradcam_pp(model, img_tensor, device=device, return_numpy=True)
            elif cam_method == "EigenCAM":
                cam = run_eigencam(model, img_tensor, device=device, return_numpy=True)
            else:
                st.error("Unknown CAM method selected.")
                st.stop()
            overlay = make_overlay(orig_img, cam, alpha=opacity/100, blur=True)

        # --- Display side-by-side ---
        col1, col2 = st.columns(2)
        with col1:
            st.image(orig_img, caption="Original X-ray", use_column_width=True)
        with col2:
            st.image(overlay, caption=f"{cam_method} Overlay", use_column_width=True)

        # --- Download Overlay as PNG ---
        overlay_img_pil = Image.fromarray(overlay)
        import io
        with io.BytesIO() as output:
            overlay_img_pil.save(output, format="PNG")
            png_bytes = output.getvalue()
        st.download_button(
            label="Download Overlay as PNG",
            data=png_bytes,
            file_name="lungguard_gradcam_overlay.png",
            mime="image/png",
            help="Download the Grad-CAM overlay as a PNG file for sharing or publication."
        )

        # --- Advanced: Download as DICOM ---
        if st.checkbox(
            "Export overlay as DICOM (advanced, for PACS/clinical use)",
            help="Export overlay as a DICOM file for research integration into PACS/radiology systems."
        ):
            dicom_path = save_as_dicom(overlay, filename="lungguard_gradcam.dcm")
            with open(dicom_path, "rb") as f:
                st.download_button(
                    label="Download DICOM",
                    data=f,
                    file_name="lungguard_gradcam.dcm",
                    mime="application/dicom",
                    help="Download the overlay as a DICOM file (experimental)."
                )

st.info(
    "This tool supports research, education, and explainability in AI for medical imaging. "
    "DICOM export is for testing; overlays are not for primary clinical use without validation.",
    icon="ℹ️"
)

st.caption("© 2025 | GradCAM Logic | MSc Applied AI and Data Science, Solent University.")