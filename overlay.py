# utils/overlay.py

import numpy as np
import cv2

def show_gradcam_overlay_on_xray(
    image_pil,
    gradcam_mask,
    alpha=0.4,
    colormap=cv2.COLORMAP_JET,
    caption="Grad-CAM Overlay"
):
    """
    Args:
        image_pil: PIL.Image, original chest X-ray (RGB or L)
        gradcam_mask: 2D numpy array, normalized [0,1], same size or will be resized
        alpha: float, transparency of heatmap overlay (0=transparent, 1=opaque)
        colormap: OpenCV colormap, e.g. cv2.COLORMAP_JET
    Returns:
        overlay: np.array, RGB image ready for display (same shape as input image)
    """
    # Convert X-ray to RGB numpy
    xray = np.array(image_pil.convert("RGB"))
    # Resize gradcam to match xray
    gradcam_resized = cv2.resize(gradcam_mask, (xray.shape[1], xray.shape[0]), interpolation=cv2.INTER_LINEAR)
    gradcam_norm = np.uint8(255 * gradcam_resized)
    # Colorize Grad-CAM mask
    heatmap = cv2.applyColorMap(gradcam_norm, colormap)
    # Overlay heatmap and original image
    overlay = cv2.addWeighted(heatmap, alpha, xray, 1 - alpha, 0)
    return overlay
