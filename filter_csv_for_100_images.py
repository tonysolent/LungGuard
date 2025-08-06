import os
import pandas as pd

# Paths
image_dir = r"F:\AE2 Software\LungGuard\data\images"
csv_path = r"F:\AE2 Software\LungGuard\data\master_table.csv"
filtered_csv_path = r"F:\AE2 Software\LungGuard\data\master_table_100.csv"

# Get image IDs (strip .png)
image_ids = [f.replace(".png", "") for f in os.listdir(image_dir) if f.endswith(".png")]

# Load original CSV
df = pd.read_csv(csv_path)

# Filter rows where ImageID is in image_ids
filtered_df = df[df['ImageID'].astype(str).isin(image_ids)]

# Save filtered CSV
filtered_df.to_csv(filtered_csv_path, index=False)
print(f"[✔] Saved filtered CSV with {len(filtered_df)} rows → {filtered_csv_path}")
