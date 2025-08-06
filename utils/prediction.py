import torch
from torchvision import transforms

def predict_image(img, model, device='cpu'):
    """
    Predicts the class and probability for a given PIL image using a PyTorch model.
    Returns (pred_label (int), pred_prob (float)).
    """
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1)
        pred_label = torch.argmax(prob, dim=1).item()
        pred_prob = prob[0, pred_label].item()
    return pred_label, pred_prob

def get_class_labels():
    """
    Returns a dictionary mapping class indices to labels.
    """
    return {
        0: "Normal",
        1: "Nodule",
        2: "Mass"
        # Extend as needed for your model
    }
