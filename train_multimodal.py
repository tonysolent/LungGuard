import os
import pandas as pd
import numpy as np
import ast
from PIL import Image
import time
import multiprocessing

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

from config import TABULAR_FIELDS_14

# === CONFIG ===
MASTER_CSV = 'master_table_100.csv'
IMAGE_DIR = 'data/images'  # Adjust to your actual image folder
CHECKPOINT_DIR = 'checkpoints'
FUSION_TYPE = 'early'
N_SAMPLES = 100
BATCH_SIZE = 8
EPOCHS = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

LABEL_MAP = {"male": 1, "female": 0, "m": 1, "f": 0}

# === TEXT CONFIG ===
TOKENIZER_NAME = "emilyalsentzer/Bio_ClinicalBERT"
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
MAX_LEN = 128

# === IMAGE CONFIG ===
IMG_SIZE = 224
img_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# === LOAD MODELS ===
from models.image_model import ImageFeatureExtractor
from models.tabular_model import TabularFeatureExtractor
from models.text_model import TextFeatureExtractor
from models.fusion_model import FusionClassifier

# === DATA PREPARATION ===
df = pd.read_csv(MASTER_CSV)

def cancer_label(x):
    x = str(x).lower()
    return int(any(word in x for word in ["thickening", "effusion", "blunting", "mass", "tumor", "malig"]))

df['label'] = df['label_group'].apply(cancer_label)

def safe_len(val):
    try:
        return len(ast.literal_eval(val))
    except:
        return 0
df['locations'] = df['locations'].apply(safe_len)

bool_cols = ['prior_study', 'progression_status', 'study_is_benchmark', 'study_is_validation', 'patient_is_benchmark']
for col in bool_cols:
    if col in df:
        df[col] = df[col].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0}).fillna(0)

df['PatientSex_DICOM'] = df['PatientSex_DICOM'].map({'M': 1, 'F': 0, 'm': 1, 'f': 0}).fillna(0)
df['split'] = df['split'].map({'train': 1, 'test': 0}).fillna(0)

for col in TABULAR_FIELDS_14:
    if col not in df.columns:
        print(f"Column {col} not found in CSV. Adding column with zeros.")
        df[col] = 0

print("==== NaN counts for TABULAR_FIELDS ====")
print(df[TABULAR_FIELDS_14].isna().sum())
df[TABULAR_FIELDS_14] = df[TABULAR_FIELDS_14].fillna(0)

print("==== First 5 rows of tabular features (post-fix) ====")
print(df[TABULAR_FIELDS_14].head())

train_df = df[df['split'] == 1].reset_index(drop=True)
val_df = df[df['split'] == 0].reset_index(drop=True)

if N_SAMPLES < len(train_df):
    train_df = train_df.sample(n=N_SAMPLES, random_state=42).reset_index(drop=True)

tabular_train = train_df[TABULAR_FIELDS_14]
tabular_mean = tabular_train.mean()
tabular_std = tabular_train.std().replace(0, 1)

print("Tabular means:\n", tabular_mean)
print("Tabular stds:\n", tabular_std)

tabular_mean.to_csv('tabular_mean.csv')
tabular_std.to_csv('tabular_std.csv')

class MultimodalDataset(Dataset):
    def __init__(self, df, image_dir, mean, std, tabular_fields):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.mean = mean.values
        self.std = std.values
        self.tabular_fields = tabular_fields

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, str(row['ImageID']))
        image = Image.open(img_path).convert('RGB')
        image = img_transform(image)

        tab_values = []
        for field in self.tabular_fields:
            val = row[field]
            if field == 'PatientSex_DICOM':
                val = LABEL_MAP.get(str(val).strip().lower(), 0)
            try:
                tab_values.append(float(val))
            except:
                tab_values.append(0.0)

        tab_array = np.array(tab_values)
        tab_array = (tab_array - self.mean) / self.std
        tab_tensor = torch.tensor(tab_array, dtype=torch.float32)

        clinical_text = str(row['sentence_en']) if pd.notnull(row['sentence_en']) else ""
        encoded = tokenizer(clinical_text, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt')
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        label = float(row['label'])
        return image, tab_tensor, input_ids, attention_mask, torch.tensor(label, dtype=torch.float32)

train_dataset = MultimodalDataset(train_df, IMAGE_DIR, tabular_mean, tabular_std, TABULAR_FIELDS_14)
val_dataset = MultimodalDataset(val_df, IMAGE_DIR, tabular_mean, tabular_std, TABULAR_FIELDS_14)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

image_model = ImageFeatureExtractor(pretrained=True, out_dim=512).to(DEVICE)
tabular_model = TabularFeatureExtractor(input_dim=len(TABULAR_FIELDS_14), out_dim=128).to(DEVICE)
text_model = TextFeatureExtractor(model_name=TOKENIZER_NAME, out_dim=128).to(DEVICE)
fusion_model = FusionClassifier(
    image_feature_dim=512, tabular_feature_dim=128, text_feature_dim=128,
    fusion_type=FUSION_TYPE, num_classes=1
).to(DEVICE)

params = list(image_model.parameters()) + list(tabular_model.parameters()) + \
         list(text_model.parameters()) + list(fusion_model.parameters())
optimizer = optim.Adam(params, lr=2e-4)
criterion = nn.BCEWithLogitsLoss()

def evaluate(models, dataloader):
    image_model, tabular_model, text_model, fusion_model = models
    image_model.eval()
    tabular_model.eval()
    text_model.eval()
    fusion_model.eval()

    all_labels = []
    all_preds = []
    running_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            img, tab, ids, mask, label = [x.to(DEVICE) for x in batch]
            img_feat = image_model(img)
            tab_feat = tabular_model(tab)
            txt_feat = text_model(ids, mask)
            output = fusion_model(img_feat, tab_feat, txt_feat).squeeze(1)
            loss = criterion(output, label)
            running_loss += loss.item() * label.size(0)
            preds = torch.sigmoid(output).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(label.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_auc = roc_auc_score(all_labels, all_preds)
    epoch_acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    return epoch_loss, epoch_auc, epoch_acc

def main():
    print(f"Training on device: {DEVICE}")
    best_val_auc = 0.0
    patience = 5
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        image_model.train()
        tabular_model.train()
        text_model.train()
        fusion_model.train()

        running_loss = 0.0
        all_labels = []
        all_preds = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        for batch_idx, batch in enumerate(progress_bar, 1):
            img, tab, ids, mask, label = [x.to(DEVICE) for x in batch]

            optimizer.zero_grad()
            img_feat = image_model(img)
            tab_feat = tabular_model(tab)
            txt_feat = text_model(ids, mask)
            output = fusion_model(img_feat, tab_feat, txt_feat).squeeze(1)

            loss = criterion(output, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * label.size(0)
            preds = torch.sigmoid(output).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(label.cpu().numpy())

            if batch_idx % 10 == 0 or batch_idx == len(train_loader):
                avg_loss = running_loss / (batch_idx * BATCH_SIZE)
                progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_auc = roc_auc_score(all_labels, all_preds)
        epoch_acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))

        print(f"\nEpoch {epoch+1}/{EPOCHS} Train Loss: {epoch_loss:.4f} AUC: {epoch_auc:.4f} Acc: {epoch_acc:.4f}")

        val_loss, val_auc, val_acc = evaluate(
            (image_model, tabular_model, text_model, fusion_model), val_loader
        )
        print(f"Epoch {epoch+1}/{EPOCHS} Validation Loss: {val_loss:.4f} AUC: {val_auc:.4f} Acc: {val_acc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            torch.save(image_model.state_dict(), os.path.join(CHECKPOINT_DIR, f"image_model_{FUSION_TYPE}.pt"))
            torch.save(tabular_model.state_dict(), os.path.join(CHECKPOINT_DIR, f"tabular_model_{FUSION_TYPE}.pt"))
            torch.save(text_model.state_dict(), os.path.join(CHECKPOINT_DIR, f"text_model_{FUSION_TYPE}.pt"))
            torch.save(fusion_model.state_dict(), os.path.join(CHECKPOINT_DIR, f"fusion_model_{FUSION_TYPE}.pt"))
            print(f"✅ Saved best model checkpoint with val AUC: {best_val_auc:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs with no improvement.")
                break

    print("Training complete.")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
