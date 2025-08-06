import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import AutoTokenizer


class LungGuardDataset(Dataset):
    def __init__(
        self, 
        image_dir,          # Folder with X-ray images
        tabular_path,       # CSV file with patient features & label
        text_col,           # Name of text column in CSV
        label_col,          # Name of label column in CSV
        tokenizer_name,     # e.g., "emilyalsentzer/Bio_ClinicalBERT"
        max_text_len=128, 
        img_size=224, 
        augment=False
    ):
        self.df = pd.read_csv(tabular_path)
        self.image_dir = image_dir
        self.text_col = text_col
        self.label_col = label_col
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_text_len = max_text_len

        # Tabular columns (excluding IDs, label, text)
        exclude = [self.text_col, self.label_col, 'PatientID', 'ImageID', 'StudyID']
        self.tabular_cols = [col for col in self.df.columns if col not in exclude]

        # Image preprocessing
        tfms = [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
        if augment:
            tfms.insert(0, transforms.RandomHorizontalFlip())
        self.img_transform = transforms.Compose(tfms)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image
        img_path = os.path.join(self.image_dir, str(row['ImageID']))
        image = Image.open(img_path).convert('RGB')
        image = self.img_transform(image)

        # Tabular
        tabular_series = row[self.tabular_cols]
        tabular_values = pd.to_numeric(tabular_series, errors='coerce').fillna(0).to_numpy(dtype=np.float32)
        tabular = torch.tensor(tabular_values, dtype=torch.float32)

        # Text
        text = str(row[self.text_col])
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_len,
            return_tensors='pt'
        )
        text_input_ids = encoded['input_ids'].squeeze(0)
        text_attention_mask = encoded['attention_mask'].squeeze(0)

        # Label
        label_raw = row[self.label_col]
        label_map = {"negative": 0.0, "positive": 1.0, "no": 0.0, "yes": 1.0, "0": 0.0, "1": 1.0}
        label = torch.tensor(label_map.get(str(label_raw).strip().lower(), 0.0), dtype=torch.float32)

        return {
            'image': image,
            'tabular': tabular,
            'text_input_ids': text_input_ids,
            'text_attention_mask': text_attention_mask,
            'label': label
        }

# Utility: Create DataLoader
def create_lungguard_loader(
    image_dir, tabular_path, text_col, label_col, tokenizer_name,
    batch_size=32, shuffle=True, num_workers=4, augment=False
):
    dataset = LungGuardDataset(
        image_dir, tabular_path, text_col, label_col, tokenizer_name,
        augment=augment
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
