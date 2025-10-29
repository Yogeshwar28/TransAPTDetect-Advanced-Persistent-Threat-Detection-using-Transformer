# import pandas as pd

# df = pd.read_csv("/home/yogeshwar/Yogesh-MTP/csv/Combined/combined_cleaned_balanced_ENN_standard_scaled.csv")

# # Count total duplicates 
# total_dupes = df.duplicated().sum()
# print(f"Total duplicate rows in dataset: {total_dupes}")

# # Count duplicates per Stage
# # Keep Stage column safe
# if "Stage" not in df.columns:
#     raise ValueError("Column 'Stage' not found in the CSV!")

# dupe_mask = df.duplicated(keep='first')

# # Group by Stage and count how many duplicates fall into each Stage
# dupes_per_stage = df.loc[dupe_mask].groupby("Stage").size()

# print("\Duplicate counts per Stage:")
# print(dupes_per_stage)

# stage_counts = df["Stage"].value_counts()
# dupes_pct = (dupes_per_stage / stage_counts * 100).round(2)

# print("\nDuplicate percentage per Stage:")
# print(dupes_pct)
























#Checking for near-duplicate entries (e.g., 80% similar) in the dataset



import pandas as pd
import numpy as np
from itertools import combinations
from collections import Counter

df = pd.read_csv("/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/combined_balanced_hybrid_standard_scaled.csv")

features = df.drop(columns=["Stage"]).values
labels = df["Stage"].values

similarity_threshold = 0.8   
n_cols = features.shape[1]
min_matches = int(n_cols * similarity_threshold)

print(f"Total features: {n_cols}, Minimum matches for near-duplicate: {min_matches}")

near_duplicates = []
for i, j in combinations(range(len(df)), 2):
    matches = np.sum(features[i] == features[j])
    if matches >= min_matches:
        near_duplicates.append((i, j))

print(f"Total near-duplicate pairs found: {len(near_duplicates)}")

stage_counts = Counter()
for i, j in near_duplicates:
    stage_counts[labels[i]] += 1
    stage_counts[labels[j]] += 1

print("\nNear-duplicate counts per Stage:")
for stage, count in stage_counts.items():
    print(f"{stage}: {count}")
