import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd

from utils.data_loader import create_lungguard_loader
from models.image_model import ImageFeatureExtractor
from models.tabular_model import TabularFeatureExtractor
from models.text_model import TextFeatureExtractor
from models.fusion_model import FusionClassifier
from utils.explainability import run_gradcam, run_shap_tabular, run_shap_text
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set all your paths here
image_dir = r"F:\AE2 Software\LungGuard\data\images"
tabular_path = r"F:\AE2 Software\LungGuard\data\master_table_100.csv"

text_col = "sentence_en"
label_col = "label"
tokenizer_name = "emilyalsentzer/Bio_ClinicalBERT"

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4
FUSION_TYPES = ['early', 'late', 'hybrid']

def get_tabular_dim(tabular_path, text_col, label_col):
    df = pd.read_csv(tabular_path)
    exclude = [text_col, label_col, 'PatientID', 'ImageID', 'StudyID']
    tabular_cols = [col for col in df.columns if col not in exclude]
    return len(tabular_cols)

def train_one_epoch(loader, image_model, tabular_model, text_model, fusion_model, optimizer, criterion):
    image_model.train()
    tabular_model.train()
    text_model.train()
    fusion_model.train()

    losses = []
    preds = []
    targets = []

    for batch_idx, batch in enumerate(loader):
        print(f"[TRAIN] Batch {batch_idx+1}/{len(loader)}", flush=True)

        images = batch['image'].to(DEVICE)
        tabular = batch['tabular'].to(DEVICE)
        input_ids = batch['text_input_ids'].to(DEVICE)
        attention_mask = batch['text_attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()

        img_feat = image_model(images)
        tab_feat = tabular_model(tabular)
        txt_feat = text_model(input_ids, attention_mask)

        outputs = fusion_model(img_feat, tab_feat, txt_feat)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        targets.extend(labels.detach().cpu().numpy())

    return np.mean(losses), np.vstack(preds), np.vstack(targets)


def eval_epoch(loader, image_model, tabular_model, text_model, fusion_model, criterion):
    image_model.eval()
    tabular_model.eval()
    text_model.eval()
    fusion_model.eval()

    losses = []
    preds = []
    targets = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            print(f"[VAL] Batch {batch_idx+1}/{len(loader)}", flush=True)

            images = batch['image'].to(DEVICE)
            tabular = batch['tabular'].to(DEVICE)
            input_ids = batch['text_input_ids'].to(DEVICE)
            attention_mask = batch['text_attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE).unsqueeze(1)

            img_feat = image_model(images)
            tab_feat = tabular_model(tabular)
            txt_feat = text_model(input_ids, attention_mask)

            outputs = fusion_model(img_feat, tab_feat, txt_feat)
            loss = criterion(outputs, labels)

            losses.append(loss.item())
            preds.extend(torch.sigmoid(outputs).cpu().numpy())
            targets.extend(labels.cpu().numpy())

    return np.mean(losses), np.vstack(preds), np.vstack(targets)



def compute_metrics(preds, targets, threshold=0.5):
    preds_binary = (preds >= threshold).astype(int)
    try:
        auc = roc_auc_score(targets, preds)
    except:
        auc = float('nan')
    try:
        f1 = f1_score(targets, preds_binary)
    except:
        f1 = float('nan')
    accuracy = accuracy_score(targets, preds_binary)
    cm = confusion_matrix(targets, preds_binary)
    return accuracy, auc, f1, cm


def run_pipeline(fusion_type, disable_eval=True):
    print(f"\n--- Fusion type: {fusion_type.upper()} ---")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    tabular_dim = get_tabular_dim(tabular_path, text_col, label_col)

    checkpoint_prefix = f"checkpoints/{fusion_type}"
    checkpoint_paths = {
        'image': f"{checkpoint_prefix}_image_model.pt",
        'tabular': f"{checkpoint_prefix}_tabular_model.pt",
        'text': f"{checkpoint_prefix}_text_model.pt",
        'fusion': f"{checkpoint_prefix}_fusion_model.pt",
    }

    image_model = ImageFeatureExtractor(pretrained=True, out_dim=512).to(DEVICE)
    tabular_model = TabularFeatureExtractor(input_dim=tabular_dim, out_dim=128).to(DEVICE)
    text_model = TextFeatureExtractor(model_name=tokenizer_name, out_dim=128).to(DEVICE)
    fusion_model = FusionClassifier(
        image_feature_dim=512, tabular_feature_dim=128, text_feature_dim=128,
        fusion_type=fusion_type, num_classes=1
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    # Skip training if all checkpoints exist
    if all(os.path.exists(p) for p in checkpoint_paths.values()):
        print(f"[✔] Found saved checkpoints for {fusion_type.upper()}. Skipping training...")

        image_model.load_state_dict(torch.load(checkpoint_paths['image'], map_location=DEVICE))
        tabular_model.load_state_dict(torch.load(checkpoint_paths['tabular'], map_location=DEVICE))
        text_model.load_state_dict(torch.load(checkpoint_paths['text'], map_location=DEVICE))
        fusion_model.load_state_dict(torch.load(checkpoint_paths['fusion'], map_location=DEVICE))

        if not disable_eval:
            val_loader = create_lungguard_loader(
                image_dir, tabular_path, text_col, label_col, tokenizer_name,
                batch_size=BATCH_SIZE, shuffle=False, num_workers=2, augment=False
            )
            val_loss, val_preds, val_targets = eval_epoch(
                val_loader, image_model, tabular_model, text_model, fusion_model, criterion
            )
            acc, auc, f1, cm = compute_metrics(val_preds, val_targets)
            print(f"[✔] Evaluation | Val Loss: {val_loss:.4f} | Acc: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}")
            np.save(f"results/preds_{fusion_type}.npy", val_preds)
            np.save(f"results/targets_{fusion_type}.npy", val_targets)
            pd.DataFrame(cm).to_csv(f"results/confusion_matrix_{fusion_type}.csv", index=False)

        return

    print(f"[⏳] Training {fusion_type.upper()} fusion model...")

    train_loader = create_lungguard_loader(
        image_dir, tabular_path, text_col, label_col, tokenizer_name,
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2, augment=True
    )
    val_loader = create_lungguard_loader(
        image_dir, tabular_path, text_col, label_col, tokenizer_name,
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2, augment=False
    )

    params = list(image_model.parameters()) + \
             list(tabular_model.parameters()) + \
             list(text_model.parameters()) + \
             list(fusion_model.parameters())

    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        train_loss, train_preds, train_targets = train_one_epoch(
            train_loader, image_model, tabular_model, text_model, fusion_model, optimizer, criterion
        )
        val_loss, val_preds, val_targets = eval_epoch(
            val_loader, image_model, tabular_model, text_model, fusion_model, criterion
        )
        acc, auc, f1, cm = compute_metrics(val_preds, val_targets)
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Acc: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}")

    # ✅ Save model checkpoints after final epoch
    torch.save(image_model.state_dict(), checkpoint_paths['image'])
    torch.save(tabular_model.state_dict(), checkpoint_paths['tabular'])
    torch.save(text_model.state_dict(), checkpoint_paths['text'])
    torch.save(fusion_model.state_dict(), checkpoint_paths['fusion'])

    # ✅ Save final results
    np.save(f"results/preds_{fusion_type}.npy", val_preds)
    np.save(f"results/targets_{fusion_type}.npy", val_targets)
    pd.DataFrame(cm).to_csv(f"results/confusion_matrix_{fusion_type}.csv", index=False)

    print(f"[✔] {fusion_type.upper()} fusion model and results saved.")
    print(f"--- {fusion_type.upper()} fusion finished ---")





    # Check if saved models exist
    if all(os.path.exists(p) for p in checkpoint_paths.values()):

        print(f"[✔] Found saved checkpoints for {fusion_type.upper()}. Skipping training...")

        image_model.load_state_dict(torch.load(image_path, map_location=DEVICE))
        tabular_model.load_state_dict(torch.load(tabular_path, map_location=DEVICE))
        text_model.load_state_dict(torch.load(text_path, map_location=DEVICE))
        fusion_model.load_state_dict(torch.load(fusion_path, map_location=DEVICE))
    else:
        print(f"[⏳] Training {fusion_type.upper()} fusion model...")

        for epoch in range(EPOCHS):
            train_loss, train_preds, train_targets = train_one_epoch(
                train_loader, image_model, tabular_model, text_model, fusion_model, optimizer, criterion
            )
            val_loss, val_preds, val_targets = eval_epoch(
                val_loader, image_model, tabular_model, text_model, fusion_model, criterion
            )

            acc, auc, f1, cm = compute_metrics(val_preds, val_targets)
            print(f"Epoch {epoch+1}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Acc: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}")

        # Save models
        torch.save(image_model.state_dict(), image_path)
        torch.save(tabular_model.state_dict(), tabular_path)
        torch.save(text_model.state_dict(), text_path)
        torch.save(fusion_model.state_dict(), fusion_path)
        print(f"[💾] Saved checkpoints for {fusion_type.upper()} fusion.")


        # Save best model/checkpointing (optional, can add later)

    # Save evaluation metrics
    np.save(f"results/preds_{fusion_type}.npy", val_preds)
    np.save(f"results/targets_{fusion_type}.npy", val_targets)

    # Save confusion matrix
    pd.DataFrame(cm).to_csv(f"results/confusion_matrix_{fusion_type}.csv", index=False)

    print(f"--- {fusion_type.upper()} fusion finished ---")

    # EXPLAINABILITY
    explain_dir = f"results/explain_{fusion_type}"
    os.makedirs(explain_dir, exist_ok=True)

    if fusion_type in ['early', 'hybrid']:
        generate_gradcam(
            image_model, fusion_model, val_loader, DEVICE,
            out_dir=os.path.join(explain_dir, "gradcam")
        )

    generate_shap_tabular(
        tabular_model, val_loader, DEVICE,
        out_dir=os.path.join(explain_dir, "shap_tabular")
    )

    generate_shap_text(
        text_model, val_loader, DEVICE,
        out_dir=os.path.join(explain_dir, "shap_text")
    )

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    for fusion_type in FUSION_TYPES:
        run_pipeline(fusion_type)


