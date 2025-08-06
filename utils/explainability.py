# explainability.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import shap

def run_gradcam(
    image_model,
    image_tensor,
    save_path=None,        # Optional: provide a file path to save PNG heatmap
    device='cpu',
    return_numpy=False     # True: return 2D numpy array for overlay
):
    """
    Runs Grad-CAM on the input image_tensor using the given model.
    Returns: (optionally) the Grad-CAM mask as a normalized [0,1] NumPy array.
    Optionally saves a heatmap visualization if save_path is provided.
    """
    image_model.eval()
    image_tensor = image_tensor.to(device)
    image_tensor.requires_grad_()

    # Automatically find the last Conv2d layer for Grad-CAM
    target_layer = None
    for module in image_model.modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module

    assert target_layer is not None, "No Conv2d layer found for Grad-CAM."

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    hook_forward = target_layer.register_forward_hook(forward_hook)
    hook_backward = target_layer.register_backward_hook(backward_hook)

    # Forward and backward pass
    output = image_model(image_tensor)
    class_idx = output.squeeze().argmax().item() if output.shape[-1] > 1 else 0
    output[0, class_idx].backward()

    acts = activations[0].squeeze(0)       # [C, H, W]
    grads = gradients[0].squeeze(0)        # [C, H, W]
    weights = grads.mean(dim=(1, 2))       # [C]
    cam = (weights[:, None, None] * acts).sum(0)
    cam = cam.detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()  # Normalize to [0,1]

    # Clean up hooks
    hook_forward.remove()
    hook_backward.remove()

    # Optionally save as PNG
    if save_path:
        plt.imshow(cam, cmap='jet')
        plt.axis('off')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    if return_numpy:
        return cam  # 2D normalized NumPy array

    # By default: returns nothing if not saving and not returning numpy

def run_shap_tabular(tabular_model, dataloader, device, fusion_type, tabular_feature_names, save_dir):
    """
    Runs SHAP explainability for tabular models and saves a summary plot.
    """
    tabular_model.eval()
    batch = next(iter(dataloader))[0].to(device)
    batch.requires_grad = True

    def model_forward(x):
        with torch.no_grad():
            return tabular_model(torch.tensor(x, dtype=torch.float32, device=device)).cpu().numpy()

    explainer = shap.Explainer(model_forward, batch.cpu().numpy())
    shap_values = explainer(batch.cpu().numpy())

    out_dir = f"{save_dir}/shap_{fusion_type}"
    os.makedirs(out_dir, exist_ok=True)
    shap.summary_plot(shap_values.values, batch.cpu().numpy(), feature_names=tabular_feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/tabular_summary.png", bbox_inches='tight', pad_inches=0)
    plt.close()

def run_shap_text(
    text_model,
    dataloader,
    device,
    fusion_type,
    tokenizer,
    save_dir="results"
):
    """
    Runs SHAP explainability for text models and returns tokens + SHAP values.
    """
    batch = next(iter(dataloader))

    input_ids = batch['text_input_ids'].to(device)
    if input_ids.ndim == 3:
        input_ids = input_ids.squeeze(1)
    elif input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    if 'text_attention_mask' in batch:
        attention_mask = batch['text_attention_mask'].to(device)
        if attention_mask.ndim == 3:
            attention_mask = attention_mask.squeeze(1)
        elif attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
    else:
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

    text_model.eval()

    def model_forward(input_ids_array):
        input_ids_tensor = torch.tensor(input_ids_array, dtype=torch.long, device=device)
        if input_ids_tensor.ndim == 1:
            input_ids_tensor = input_ids_tensor.unsqueeze(0)
        attention_mask_tensor = (input_ids_tensor != tokenizer.pad_token_id).long()
        with torch.no_grad():
            outputs = text_model(input_ids_tensor, attention_mask_tensor)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            return outputs.cpu().numpy()

    background_input_ids = input_ids[:1].cpu().numpy()

    explainer = shap.KernelExplainer(model_forward, background_input_ids)

    test_input_ids = input_ids.cpu().numpy()
    shap_values = explainer.shap_values(test_input_ids, nsamples=50)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
    input_ids_row = input_ids[0].cpu().numpy()
    non_pad_mask = input_ids_row != tokenizer.pad_token_id
    num_real_tokens = non_pad_mask.sum()
    vals_trimmed = np.diagonal(shap_values[0, :num_real_tokens, :num_real_tokens])

    return tokens[:num_real_tokens], vals_trimmed

def run_gradcam_pp(
    image_model,
    image_tensor,
    save_path=None,
    device='cpu',
    return_numpy=False
):
    """
    Runs Grad-CAM++ on the input image_tensor using the given model.
    Returns: (optionally) the Grad-CAM++ mask as a normalized [0,1] NumPy array.
    Optionally saves a heatmap visualization if save_path is provided.
    """
    image_model.eval()
    image_tensor = image_tensor.to(device)
    image_tensor.requires_grad_()

    # Automatically find the last Conv2d layer for Grad-CAM++
    target_layer = None
    for module in image_model.modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module

    assert target_layer is not None, "No Conv2d layer found for Grad-CAM++."

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    hook_forward = target_layer.register_forward_hook(forward_hook)
    hook_backward = target_layer.register_backward_hook(backward_hook)

    # Forward and backward pass
    output = image_model(image_tensor)
    class_idx = output.squeeze().argmax().item() if output.shape[-1] > 1 else 0
    score = output[0, class_idx]
    image_model.zero_grad()
    score.backward(retain_graph=True)

    acts = activations[0].squeeze(0)       # [C, H, W]
    grads = gradients[0].squeeze(0)        # [C, H, W]
    grads_2 = grads ** 2
    grads_3 = grads ** 3

    alpha_num = grads_2
    alpha_denom = 2 * grads_2 + acts * grads_3
    alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
    alphas = alpha_num / (alpha_denom + 1e-7)
    weights = (alphas * torch.relu(grads)).sum(dim=(1, 2))
    cam = (weights[:, None, None] * acts).sum(0)
    cam = cam.detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    hook_forward.remove()
    hook_backward.remove()

    # Optionally save as PNG
    if save_path:
        plt.imshow(cam, cmap='jet')
        plt.axis('off')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    if return_numpy:
        return cam


def run_eigencam(
    image_model,
    image_tensor,
    save_path=None,
    device='cpu',
    return_numpy=False
):
    """
    Runs EigenCAM on the input image_tensor using the given model.
    Returns: (optionally) the EigenCAM mask as a normalized [0,1] NumPy array.
    Optionally saves a heatmap visualization if save_path is provided.
    """
    import numpy.linalg as LA

    image_model.eval()
    image_tensor = image_tensor.to(device)
    image_tensor.requires_grad_(False)

    # Automatically find the last Conv2d layer for EigenCAM
    target_layer = None
    for module in image_model.modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module

    assert target_layer is not None, "No Conv2d layer found for EigenCAM."

    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    hook_forward = target_layer.register_forward_hook(forward_hook)

    # Forward pass
    _ = image_model(image_tensor)
    acts = activations[0].squeeze(0).detach().cpu().numpy()  # [C, H, W]
    # Flatten each feature map to (C, H*W)
    acts_flat = acts.reshape(acts.shape[0], -1)
    # Compute covariance
    cam = np.dot(acts_flat, acts_flat.T)
    eigvals, eigvecs = LA.eigh(cam)
    principal_eigvec = eigvecs[:, -1]
    principal_component = np.dot(principal_eigvec, acts_flat)
    cam_map = principal_component.reshape(acts.shape[1], acts.shape[2])
    cam_map = np.maximum(cam_map, 0)
    if cam_map.max() > 0:
        cam_map = cam_map / cam_map.max()

    hook_forward.remove()

    if save_path:
        plt.imshow(cam_map, cmap='jet')
        plt.axis('off')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    if return_numpy:
        return cam_map