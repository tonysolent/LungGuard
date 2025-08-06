import os
import pandas as pd

# Paths
image_dir = r"F:\AE2 Software\LungGuard\data\images"
csv_path = r"F:\AE2 Software\LungGuard\data\master_table.csv"
output_csv = r"F:\AE2 Software\LungGuard\data\master_table_100.csv"

# Load CSV
df = pd.read_csv(csv_path)

# Get .png filenames from the image folder (without extension)
available_images = set(os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.png'))

# Filter rows where ImageID (excluding .png) is in available image names
df['ImageID_clean'] = df['ImageID'].str.replace('.png', '', regex=False)
matched_df = df[df['ImageID_clean'].isin(available_images)].copy()

# Optional: limit to 100 rows for fast testing
matched_df = matched_df.head(100)

# Save cleaned CSV
matched_df.drop(columns=['ImageID_clean'], inplace=True)
matched_df.to_csv(output_csv, index=False)

print(f"[✔] Saved filtered CSV with {len(matched_df)} matched rows → {output_csv}")

