import svgwrite
import struct

# Step 2: Open the .gnh file in binary mode
file_path = "task3.gnh"

with open(file_path, "rb") as file:
    binary_content = file.read()

# Print a portion of the binary content to inspect
print(binary_content[:1000])  # Limiting to the first 1000 bytes for readability

# Step 4: Create an SVG file with svgwrite
def create_svg(file_name):
    # Create an SVG drawing object
    dwg = svgwrite.Drawing(file_name, profile='tiny')

    # Example: Adding a circle (we can replace this with data from .gnh)
    dwg.add(dwg.circle(center=(100, 100), r=50, fill='red'))

    # Save the SVG file
    dwg.save()

# Path to save the SVG file
output_svg_path = "output_file.svg"
create_svg(output_svg_path)

print(f"SVG file created: {output_svg_path}")

# Step 5: Function to parse .gnh binary data and generate SVG
def parse_gnh_to_svg(gnh_file, svg_output_file):
    # Open the .gnh file in binary mode
    with open(gnh_file, "rb") as file:
        binary_content = file.read()

    # Create an SVG drawing object
    dwg = svgwrite.Drawing(svg_output_file, profile='tiny')

    # Track coordinates and text elements
    last_value = None
    current_text = ""
    text_list = []
    
    # Set initial positions for text
    text_x, text_y = 50, 50
    line_height = 20  # Space between text lines

    # Parse the binary data in chunks of 4 bytes (integer)
    for i in range(0, len(binary_content), 4):
        if i + 4 > len(binary_content):
            break

        # Unpack the current 4 bytes as an integer
        value = struct.unpack('i', binary_content[i:i+4])[0]

        # Filtering values in the expected range for coordinates/text
        if 32 <= value <= 126:  # ASCII printable characters range
            current_text += chr(value)
        else:
            if current_text:
                # When we hit a non-printable value, finalize the text and position it
                text_list.append(current_text)
                print(f"Adding text: {current_text}")
                dwg.add(dwg.text(current_text, insert=(text_x, text_y), font_size="16px", font_family="Arial", fill="black"))
                text_y += line_height
                current_text = ""  # Reset current_text for the next round

        # Collect values for circle drawing
        if value > 100:  # This condition should be refined based on your data specifics
            if last_value is not None:
                x, y = last_value, value
                print(f"Adding circle at x: {x}, y: {y}")
                dwg.add(dwg.circle(center=(x % 1000, y % 1000), r=5, fill='blue'))  # Scaling within 1000x1000 range
                last_value = None  # Reset the last_value after pairing
            else:
                # Store the current value as a potential x-coordinate
                last_value = value
        else:
            last_value = None

    # Save the generated SVG file
    dwg.save()

# Path to your .gnh file
gnh_file_path = "task3.gnh"  # Ensure this is the correct path to your .gnh file
# Path to save the output SVG file
output_svg_file_path = "output_from_gnh_refined.svg"

# Parse the .gnh file and generate the SVG
parse_gnh_to_svg(gnh_file_path, output_svg_file_path)

print(f"SVG file created from .gnh: {output_svg_file_path}")
