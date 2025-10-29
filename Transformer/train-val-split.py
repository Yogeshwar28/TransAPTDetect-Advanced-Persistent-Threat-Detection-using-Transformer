import pandas as pd
from sklearn.model_selection import train_test_split

data_path = "/home/yogeshwar/Yogesh-MTP/csv/Combined/Improved/DAPT_Original_train_data.csv"
df = pd.read_csv(data_path)

X = df.drop(columns=["Stage"])
y = df["Stage"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,   
    random_state=42
)

train_df = pd.concat([X_train, y_train], axis=1)
val_df   = pd.concat([X_val, y_val], axis=1)

train_out = "/home/yogeshwar/Yogesh-MTP/csv/Combined/Improved/DAPT_Original_train_split.csv"
val_out   = "/home/yogeshwar/Yogesh-MTP/csv/Combined/Improved/DAPT_Original_val_split.csv"

train_df.to_csv(train_out, index=False)
val_df.to_csv(val_out, index=False)

print("Stratified train/validation split completed")
print(f"Training set saved to: {train_out}")
print(f"Validation set saved to: {val_out}")
print("\nClass distribution in Train:")
print(y_train.value_counts())
print("\nClass distribution in Validation:")
print(y_val.value_counts())
