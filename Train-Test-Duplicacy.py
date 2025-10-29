import pandas as pd


train_path = "/home/yogeshwar/Yogesh-MTP/csv/Combined/DAPT_train_data.csv"
test_path  = "/home/yogeshwar/Yogesh-MTP/csv/Combined/DAPT_test_data.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

test = test[train.columns]

merged = pd.merge(test.assign(_in_test=1), train.assign(_in_train=1), how='inner', on=list(train.columns))
print("Exact duplicate rows found between test and train:", len(merged))

if len(merged) > 0:
    print(merged.head())
