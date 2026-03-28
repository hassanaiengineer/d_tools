import svgwrite

def extract_shapes_from_binary(data):
    shapes = []
    
    # Assuming the binary data follows a specific structure; you need to decode it based on that structure
    # For demonstration, let’s assume we are reading shapes from binary data
    # This is a mock structure; replace with your actual extraction logic
    index = 0
    while index < len(data):
        shape_type = data[index]
        
        if shape_type == 1:  # Circle
            x = data[index + 1]
            y = data[index + 2]
            radius = data[index + 3]
            color = 'red'  # Replace with actual color extraction logic
            shapes.append({'type': 'circle', 'x': x, 'y': y, 'radius': radius, 'color': color})
            index += 4  # Move to the next shape (adjust based on your binary format)
        
        elif shape_type == 2:  # Rectangle
            x = data[index + 1]
            y = data[index + 2]
            width = data[index + 3]
            height = data[index + 4]
            color = 'blue'  # Replace with actual color extraction logic
            shapes.append({'type': 'rectangle', 'x': x, 'y': y, 'width': width, 'height': height, 'color': color})
            index += 5  # Move to the next shape
        
        elif shape_type == 3:  # Line
            x1 = data[index + 1]
            y1 = data[index + 2]
            x2 = data[index + 3]
            y2 = data[index + 4]
            color = 'green'  # Replace with actual color extraction logic
            shapes.append({'type': 'line', 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'color': color})
            index += 5  # Move to the next shape
        
        else:
            index += 1  # Move to the next byte (if the shape type is unknown)
    
    return shapes

def convert_binary_to_svg(binary_file_path, svg_file_path):
    # Open the binary file and read its contents
    with open(binary_file_path, 'rb') as bin_file:
        data = bin_file.read()

    # Extract shapes from binary data
    shapes = extract_shapes_from_binary(data)

    # Create SVG drawing
    dwg = svgwrite.Drawing(svg_file_path, profile='tiny', size=(600, 400))

    # Add shapes to the SVG drawing
    for shape in shapes:
        if shape['type'] == 'circle':
            dwg.add(dwg.circle(center=(shape['x'], shape['y']), r=shape['radius'], fill=shape['color'], stroke='black', stroke_width=2))
        elif shape['type'] == 'rectangle':
            dwg.add(dwg.rect(insert=(shape['x'], shape['y']), size=(shape['width'], shape['height']), fill=shape['color'], stroke='black', stroke_width=2))
        elif shape['type'] == 'line':
            dwg.add(dwg.line(start=(shape['x1'], shape['y1']), end=(shape['x2'], shape['y2']), stroke=shape['color'], stroke_width=2))

    # Save the SVG file
    dwg.save()

def main():
    binary_file_path = 'task3.gnh'  # Replace with your actual .gnh file path
    svg_file_path = 'output.svg'  # Output SVG file path
    convert_binary_to_svg(binary_file_path, svg_file_path)
    print(f'SVG file saved as: {svg_file_path}')

if __name__ == "__main__":
    main()
