import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

csv_path = "/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/New/Trans_Updated2/Improved/transformer_predictions.csv"  # <-- change this to your actual file
df = pd.read_csv(csv_path)

true_labels = df["True_Stage"]
pred_labels = df["Pred_Stage"]

classes = sorted(df["True_Stage"].unique())

cm = confusion_matrix(true_labels, pred_labels, labels=classes)

tpr_dict = {}   
fnr_dict = {}   
fpr_dict = {} 
precision_dict = {}
recall_dict = {}
f1_dict = {}

for i, cls in enumerate(classes):
    TP = cm[i, i]
    FN = sum(cm[i, :]) - TP
    FP = sum(cm[:, i]) - TP
    TN = cm.sum() - (TP + FN + FP)

    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1 = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) > 0 else 0.0
    FNR = 1 - Recall
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    precision_dict[cls] = round(Precision, 4)
    recall_dict[cls] = round(Recall, 4)
    f1_dict[cls] = round(F1, 4)
    fnr_dict[cls] = round(FNR, 4)
    fpr_dict[cls] = round(FPR, 4)
    tpr_dict[cls] = round(Recall, 4)

stagewise_df = pd.DataFrame({
    "Stage": classes,
    "Precision": [precision_dict[c] for c in classes],
    "Recall (TPR)": [recall_dict[c] for c in classes],
    "F1-Score": [f1_dict[c] for c in classes],
    "False Negative Rate (FNR)": [fnr_dict[c] for c in classes],
    "False Positive Rate (FPR)": [fpr_dict[c] for c in classes],
})

overall_metrics = {
    "Precision (Macro)": precision_score(true_labels, pred_labels, average="macro"),
    "Recall (Macro)": recall_score(true_labels, pred_labels, average="macro"),
    "F1-Score (Macro)": f1_score(true_labels, pred_labels, average="macro"),
    "Precision (Weighted)": precision_score(true_labels, pred_labels, average="weighted"),
    "Recall (Weighted)": recall_score(true_labels, pred_labels, average="weighted"),
    "F1-Score (Weighted)": f1_score(true_labels, pred_labels, average="weighted"),
}

TP = sum(cm[i, i] for i in range(len(classes)))
FN = sum(sum(cm[i, :]) - cm[i, i] for i in range(len(classes)))
FP = sum(sum(cm[:, i]) - cm[i, i] for i in range(len(classes)))
TN = cm.sum() - (TP + FN + FP)

overall_tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
overall_fnr = 1 - overall_tpr
overall_fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0

overall_metrics.update({
    "True Positive Rate (TPR)": overall_tpr,
    "False Negative Rate (FNR)": overall_fnr,
    "False Positive Rate (FPR)": overall_fpr
})

print("\n=== Confusion Matrix ===")
print(pd.DataFrame(cm, index=classes, columns=classes))

print("\n=== Stage-wise Metrics ===")
print(stagewise_df)

print("\n=== Overall Metrics ===")
for k, v in overall_metrics.items():
    print(f"{k}: {v:.4f}")

stagewise_df.to_csv("stagewise_metrics.csv", index=False)
pd.DataFrame([overall_metrics]).to_csv("overall_metrics.csv", index=False)

print("\Stage-wise metrics saved as 'stagewise_metrics.csv'")
print("Overall metrics saved as 'overall_metrics.csv'")
