import csv
import os

def combine_csv_files(file_paths, output_file):
    if not file_paths:
        print("No input files provided.")
        return

    with open(output_file, 'w', newline='') as out_f:
        writer = None

        for i, file_path in enumerate(file_paths):
            with open(file_path, 'r') as in_f:
                reader = csv.reader(in_f)
                header = next(reader) 

                if writer is None:
                    writer = csv.writer(out_f)
                    writer.writerow(header) 

                for row in reader:
                    writer.writerow(row)

    print(f"Combined {len(file_paths)} files into: {output_file}")


file_paths = [
    '/home/yogeshwar/Yogesh-MTP/csv/Benign/final_stage_data.csv',
    '/home/yogeshwar/Yogesh-MTP/csv/Reconnaissance/final_stage_data.csv',
    '/home/yogeshwar/Yogesh-MTP/csv/Foothold/final_stage_data.csv',
    '/home/yogeshwar/Yogesh-MTP/csv/Lateral_Movement/final_stage_data.csv',
    '/home/yogeshwar/Yogesh-MTP/csv/Data_Exfiltration/final_stage_data.csv',
]

output_file = '/home/yogeshwar/Yogesh-MTP/csv/Combined/combined.csv'

combine_csv_files(file_paths, output_file)
