import struct
import csv

def convert_gnh_to_csv(gnh_file_path, csv_file_path):
    with open(gnh_file_path, "rb") as file:
        binary_content = file.read()

    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Value'])  # Write header for CSV

        # Parse the binary data and extract information
        for i in range(0, len(binary_content), 4):
            if i + 4 > len(binary_content):
                break
            value = struct.unpack('i', binary_content[i:i + 4])[0]
            csv_writer.writerow([value])

    print(f"CSV file created: {csv_file_path}")

# Convert GNH to CSV
gnh_file_path = "task3.gnh"
csv_file_path = "output_from_gnh.csv"
convert_gnh_to_csv(gnh_file_path, csv_file_path)
