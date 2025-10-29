import pandas as pd

input_csv = "/home/yogeshwar/Yogesh-MTP/csv/Combined/combined.csv"
output_csv = "/home/yogeshwar/Yogesh-MTP/csv/Combined/combined.csv_cleaned.csv"  

# Columns to drop
cols_to_drop = [
    "Flow ID", "Src IP", "Dst IP", "Timestamp"
]

df = pd.read_csv(input_csv)

df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Save new CSV
df.to_csv(output_csv, index=False)

print(f"✅ Saved cleaned CSV to {output_csv} with {df.shape[1]} columns")
