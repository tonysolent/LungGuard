import os
import pandas as pd

image_dir = r"F:\AE2 Software\LungGuard\data\images"
csv_path = r"F:\AE2 Software\LungGuard\data\master_table.csv"

# Load full CSV
df = pd.read_csv(csv_path)
csv_ids = df['ImageID'].astype(str).tolist()

# Scan image directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
image_prefixes = [os.path.splitext(f)[0].split('_')[0] for f in image_files]

# Show sample values
print("\n📄 Sample ImageID from CSV:")
print(csv_ids[:10])

print("\n🖼️ Prefixes from image filenames:")
print(image_prefixes[:10])

# Check intersection
matched_ids = list(set(csv_ids) & set(image_prefixes))
print(f"\n✅ Matched {len(matched_ids)} ImageIDs")
print(matched_ids[:10])
