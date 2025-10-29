import pandas as pd

INPUT_CSV = "/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/New/Trans_Updated2/Improved/transformer_predictions.csv"
OUTPUT_CSV = "final_predictions_binary.csv"

APT_STAGES = [
    "Reconnaissance",
    "Lateral Movement",
    "Establish Foothold",
    "Data Exfiltration"
]

df = pd.read_csv(INPUT_CSV)

required_cols = ["True_Stage", "Pred_Stage"]
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in {INPUT_CSV}. Found: {list(df.columns)}")

df["True_Stage"] = df["True_Stage"].astype(str).str.strip()
df["Pred_Stage"] = df["Pred_Stage"].astype(str).str.strip()

df["True_Stage"] = df["True_Stage"].apply(lambda x: "Malicious" if x in APT_STAGES else x)
df["Pred_Stage"] = df["Pred_Stage"].apply(lambda x: "Malicious" if x in APT_STAGES else x)

df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved updated CSV (APT stages replaced with 'Malicious'): {OUTPUT_CSV}")

print("\nUnique values after replacement:")
print("True_Stage:", df["True_Stage"].unique())
print("Pred_Stage:", df["Pred_Stage"].unique())

print("\Note: Only the APT stages (Reconnaissance, Lateral Movement, Establish Foothold, Data Exfiltration) "
      "were replaced by 'Malicious'.")
print("   'Benign' labels remain unchanged.")
