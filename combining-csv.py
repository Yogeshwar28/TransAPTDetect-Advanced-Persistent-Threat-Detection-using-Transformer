import pandas as pd

config = [
    {"path": "/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/network-flows/Week5_Day6_06272021/net1013x_Flow_labeled.csv", "stage": "Data Exfiltration", "count": 1331},
    {"path": "/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/network-flows/Week5_Day1_06212021/netgw_Flow_labeled.csv", "stage": "Data Exfiltration", "count": 5},
    {"path": "/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/network-flows/Week5_Day1_06212021/net1013x_Flow_labeled.csv", "stage": "Data Exfiltration", "count": 441},
    {"path": "/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/network-flows/Week5_Day5-6_06252021-06262021/net1013x_Flow_labeled.csv", "stage": "Data Exfiltration", "count": 1299},
    {"path": "/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/network-flows/Week6_Day4_07012021/net1015x_Flow_labeled.csv", "stage": "Data Exfiltration", "count": 126},
    {"path": "/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/network-flows/Week6_Day4_07012021/net1013x_Flow_labeled.csv", "stage": "Data Exfiltration", "count": 228},
    {"path": "/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/network-flows/Week6_Day5_07022021_part2/net1015x_Flow_labeled.csv", "stage": "Data Exfiltration", "count": 183},
    {"path": "/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/network-flows/Week6_Day5_07022021_part2/net1013x_Flow_labeled.csv", "stage": "Data Exfiltration", "count": 258},
    {"path": "/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/network-flows/Week6_Day6-7_07032021-07042021/net1015x_Flow_labeled.csv", "stage": "Data Exfiltration", "count": 694},
    {"path": "/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/network-flows/Week6_Day6-7_07032021-07042021/net1013x_Flow_labeled.csv", "stage": "Data Exfiltration", "count": 694},
    {"path": "/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/network-flows/Week6_Day2_06292021/net1013x_Flow_labeled.csv", "stage": "Data Exfiltration", "count": 2253},
]

output_csv = "/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Data_Exfiltration/final_stage_entries.csv"

final_entries = []

for entry in config:
    path = entry["path"]
    stage = entry["stage"]
    count = entry["count"]

    try:
        header = pd.read_csv(path, nrows=0).columns
        if len(header) < 3:
            print(f"Skipping {path}, not enough columns.")
            continue

        stage_col = header[-3]

        try:
            df = pd.read_csv(path, on_bad_lines='skip', engine='python')
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue

        filtered = df[df[stage_col] == stage]

        if len(filtered) == 0:
            print(f"No entries for stage '{stage}' in {path}")
            continue

        if len(filtered) >= count:
            sampled = filtered.sample(n=count, random_state=42)
        else:
            sampled = filtered
            print(f" Only {len(sampled)} entries found for stage '{stage}' in {path} (requested {count})")

        final_entries.append(sampled)
        print(f"Collected {len(sampled)} entries of stage '{stage}' from {path}")

    except Exception as e:
        print(f"Error processing {path}: {e}")

# Save the final CSV
if final_entries:
    final_df = pd.concat(final_entries, ignore_index=True)
    final_df.to_csv(output_csv, index=False)
    print(f"\Final combined CSV saved as: {output_csv} ({len(final_df)} total rows)")
else:
    print("No data collected.")
