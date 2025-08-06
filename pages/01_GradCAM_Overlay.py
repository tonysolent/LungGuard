import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
from torch.nn import functional as F

# 🛠 Page setup
st.set_page_config(page_title="Grad-CAM X-ray Overlay", layout="wide")
st.title("🔬 Grad-CAM Overlay – Explainable AI for Chest X-rays")

st.markdown("""
Upload a chest X-ray image to generate a Grad-CAM activation map from a CNN.  
The output overlay is visualised as a transparent `jet` colormap aligned with the original X-ray.
""")

# 📂 Output directory
os.makedirs("gradcam_outputs", exist_ok=True)

# ✅ Pretrained CNN model (or replace with your own fine-tuned)
model = models.resnet18(pretrained=True)
model.eval()

# 🎯 Target final convolution layer for Grad-CAM
target_layer = model.layer4[1].conv2

# 🧠 Grad-CAM hooks
gradients = None
activations = None

def save_gradient_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

def save_activation_hook(module, input, output):
    global activations
    activations = output

target_layer.register_forward_hook(save_activation_hook)
target_layer.register_full_backward_hook(save_gradient_hook)

# 🔁 Preprocessing for X-ray images (medical-safe, avoids over-normalisation)
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Neutral scaling
    ])
    return transform(image).unsqueeze(0)

# 🔥 Grad-CAM generator
def generate_gradcam(input_tensor, class_idx=None):
    output = model(input_tensor)
    if class_idx is None:
        class_idx = output.argmax().item()

    model.zero_grad()
    target = output[0, class_idx]
    target.backward()

    pooled_grad = torch.mean(gradients, dim=[0, 2, 3])
    cam = torch.zeros(activations.shape[2:], dtype=torch.float32)

    for i, w in enumerate(pooled_grad):
        cam += w * activations[0, i, :, :]

    cam = np.maximum(cam.detach().numpy(), 0)
    cam = cam - np.min(cam)
    cam = cam / np.max(cam) if np.max(cam) > 0 else cam
    return cam

# 🎨 Overlay Grad-CAM on original X-ray
def overlay_heatmap_on_xray(original_img, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
    orig = np.array(original_img.convert("RGB"))
    cam_resized = cv2.resize(cam, (orig.shape[1], orig.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(alpha * heatmap + (1 - alpha) * orig)
    return overlay

# 📤 Upload and process image
uploaded_file = st.file_uploader("Upload a Chest X-ray Image (JPG or PNG)", type=["jpg", "png"])
if uploaded_file:
    # ⛑️ Safe image loading for clinical-grade X-rays
    raw_img = Image.open(uploaded_file)
    img_arr = np.array(raw_img)

    # If grayscale and 16-bit or high range → normalize to 8-bit
    if img_arr.ndim == 2 and img_arr.max() > 255:
        img_arr = (255 * (img_arr / img_arr.max())).astype(np.uint8)

    # Final image conversion for display + model
    image = Image.fromarray(img_arr).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)


    # 🔁 Preprocess and run Grad-CAM
    input_tensor = preprocess(image)
    cam = generate_gradcam(input_tensor)
    overlay = overlay_heatmap_on_xray(image.convert("RGB"), cam)

    # 🖼️ Show side-by-side
    st.subheader("🔍 Grad-CAM Overlay")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original X-ray", use_column_width=True)
    with col2:
        st.image(overlay, caption="Grad-CAM Overlay", use_column_width=True)

    # 💾 Save overlay to disk
    save_path = os.path.join("gradcam_outputs", "gradcam_overlay.png")
    Image.fromarray(overlay).save(save_path)
    st.success(f"Overlay saved to: {save_path}")
    with open(save_path, "rb") as f:
        st.download_button("📥 Download Overlay", data=f, file_name="gradcam_overlay.png", mime="image/png")

else:
    st.info("Upload a high-quality chest X-ray image to begin visualising explainable AI overlays.")

st.caption("© 2025 | GradCAM Overlay | MSc Applied AI and Data Science, Solent University.")
