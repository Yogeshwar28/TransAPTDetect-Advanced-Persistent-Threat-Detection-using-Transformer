import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

INPUT_CSV = "final_predictions_binary.csv"  
OUTPUT_METRICS = "binary_metrics_results.csv"

df = pd.read_csv(INPUT_CSV)

required_cols = ["True_Stage", "Pred_Stage"]
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in {INPUT_CSV}. Found: {list(df.columns)}")

allowed_labels = {"Benign", "Malicious"}

valid_mask = df["True_Stage"].isin(allowed_labels) & df["Pred_Stage"].isin(allowed_labels)
valid_df = df[valid_mask]

if valid_df.empty:
    raise ValueError("No valid rows found with both True_Stage and Pred_Stage in {'Benign','Malicious'}")

y_true = valid_df["True_Stage"].values
y_pred = valid_df["Pred_Stage"].values

pos_label = "Malicious"   

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, pos_label=pos_label)
recall = recall_score(y_true, y_pred, pos_label=pos_label)  
f1 = f1_score(y_true, y_pred, pos_label=pos_label)

labels = ["Benign", "Malicious"]
cm = confusion_matrix(y_true, y_pred, labels=labels)

TN, FP, FN, TP = cm.ravel()

FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
FNR = FN / (FN + TP) if (FN + TP) > 0 else 0.0
TPR = recall 

print("\n=== Binary Classification Metrics (Benign vs. Malicious) ===")
print(f"Total samples: {len(valid_df)}")
print(f"Accuracy              : {accuracy:.4f}")
print(f"Precision             : {precision:.4f}")
print(f"Recall / TPR          : {recall:.4f}")
print(f"F1-Score              : {f1:.4f}")
print(f"False Positive Rate   : {FPR:.4f}")
print(f"False Negative Rate   : {FNR:.4f}")

print("\n=== Confusion Matrix (rows=True, cols=Pred) ===")
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print(cm_df)

metrics_dict = {
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall (TPR)": [recall],
    "F1-Score": [f1],
    "False Positive Rate (FPR)": [FPR],
    "False Negative Rate (FNR)": [FNR],
    "True Negatives (TN)": [int(TN)],
    "False Positives (FP)": [int(FP)],
    "False Negatives (FN)": [int(FN)],
    "True Positives (TP)": [int(TP)],
    "Total Samples": [len(valid_df)]
}

metrics_df = pd.DataFrame(metrics_dict)
metrics_df.to_csv(OUTPUT_METRICS, index=False)

print(f"\Metrics saved to '{OUTPUT_METRICS}'")
