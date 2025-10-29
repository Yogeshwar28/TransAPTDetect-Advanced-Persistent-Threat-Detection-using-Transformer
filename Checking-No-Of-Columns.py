import csv

def check_csv_column_count(file_path, expected_columns=80):
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        all_ok = True
        for row_num, row in enumerate(reader, start=1):
            column_count = len(row)
            if column_count != expected_columns:
                print(f"Row {row_num} has {column_count} columns")
                all_ok = False
        if all_ok:
            print(f"All rows have exactly {expected_columns} columns.")

if __name__ == "__main__":
    file_path = "/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/combined_cleaned.csv"
    check_csv_column_count(file_path)
