import csv
import svgwrite

def csv_to_svg(csv_file_path, svg_file_path):
    dwg = svgwrite.Drawing(svg_file_path, profile='tiny')

    with open(csv_file_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip header row

        last_value = None
        scaling_factor = 0.00001
        max_svg_value = 1000
        for row in csv_reader:
            value = int(row[0])

            if value > 100:  # Adjust the condition based on your data
                if last_value is not None:
                    x, y = last_value, value
                    x_scaled = min(x * scaling_factor, max_svg_value)
                    y_scaled = min(y * scaling_factor, max_svg_value)

                    if 0 <= x_scaled <= max_svg_value and 0 <= y_scaled <= max_svg_value:
                        print(f"Adding circle at x: {x_scaled}, y: {y_scaled}")
                        dwg.add(dwg.circle(center=(x_scaled, y_scaled), r=2, fill='blue'))
                    last_value = None
                else:
                    last_value = value

    dwg.save()

# Convert CSV to SVG
csv_file_path = "output_from_gnh.csv"
svg_file_path = "output_from_csv.svg"
csv_to_svg(csv_file_path, svg_file_path)

print(f"SVG file created from CSV: {svg_file_path}")
