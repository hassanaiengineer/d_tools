import os
import json
import ntpath
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pydantic import BaseModel

# Disable the pixel limit warning in Pillow
Image.MAX_IMAGE_PIXELS = None

OUTPUT_DIR = "template_1"  # Output directory
INPUT_DIR = "input"
SQUARE_TOLERANCE = 0.05  # 5% tolerance for square images

class Template(BaseModel):
    size: int
    orientation: str
    path: str
    dimensions: list
    box: list

    def get_dimensions_tuple(self) -> tuple:
        return tuple(map(int, self.dimensions[0].split(",")))

    def get_box_tuple(self) -> tuple:
        return tuple(map(int, self.box[0].split(",")))

def load_json_db(file_path: str):
    with open(file_path) as file:
        return json.load(file)

def get_file_name(path: str, extension: bool = True) -> str:
    if not extension:
        return str(ntpath.basename(path)).split(".")[0]
    return ntpath.basename(path)

def recursive_images_to_list(directory_path: str) -> list:
    image_paths = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                full_path = os.path.join(root, file)
                image_paths.append(full_path)
    return image_paths

def create_output_directory_structure(input_file: str, template_name: str) -> str:
    output_dir = os.path.join(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    base_name = get_file_name(input_file, extension=False)
    output_file_name = f"{template_name}-{base_name}.webp"
    return os.path.join(output_dir, output_file_name)

def determine_orientation(image_path: str) -> str:
    with Image.open(image_path) as img:
        width, height = img.size
        aspect_ratio = width / height

        if abs(aspect_ratio - 1) <= SQUARE_TOLERANCE:
            return "Square"
        elif width > height:
            return "Landscape"
        else:
            return "Portrait"

def find_matching_templates(db: list, orientation: str) -> list[Template]:
    matching_templates = [Template(**entry) for entry in db if entry["orientation"] == orientation]
    if not matching_templates:
        raise FileNotFoundError(f"No matching templates found for orientation '{orientation}'")
    return matching_templates

def resize_and_crop_image(image, box_width, box_height):
    """
    Resize the image to completely fill the target box while maintaining the aspect ratio.
    """
    img_width, img_height = image.size
    aspect_ratio = img_width / img_height
    box_aspect_ratio = box_width / box_height

    # Determine the resizing dimensions based on the aspect ratios
    if aspect_ratio > box_aspect_ratio:
        # Crop width
        new_height = box_height
        new_width = int(box_height * aspect_ratio)
    else:
        # Crop height
        new_width = box_width
        new_height = int(box_width / aspect_ratio)

    # Resize the image using LANCZOS resampling
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Calculate cropping coordinates to fit the resized image within the box
    left = (resized_image.width - box_width) // 2
    top = (resized_image.height - box_height) // 2
    right = left + box_width
    bottom = top + box_height

    # Crop the resized image to fit exactly within the target box
    cropped_image = resized_image.crop((left, top, right, bottom))

    return cropped_image

def process_product_image(template: Template, image_file: str) -> None:
    try:
        # Open template image
        template_image = Image.open(template.path).convert("RGB")

        # Open input image
        input_image = Image.open(image_file).convert("RGB")

        # Resize and crop the input image to fill the template box size
        box_width, box_height = template.get_dimensions_tuple()
        input_resized = resize_and_crop_image(input_image, box_width, box_height)

        # Get the box coordinates in the template
        box_x, box_y = template.get_box_tuple()

        # Paste the cropped input image onto the template at the specified box position
        template_image.paste(input_resized, (box_x, box_y))

        output_path = create_output_directory_structure(image_file, get_file_name(template.path, extension=False))

        # Save as WebP format with quality of 50%
        template_image.save(output_path, format='WEBP', quality=50)

    except Exception as ex:
        print(f"Error processing image {image_file}: {ex}")

if __name__ == "__main__":
    db = load_json_db("db.json")
    images_list = recursive_images_to_list(INPUT_DIR)

    tasks = []

    for image_file in images_list:
        orientation = determine_orientation(image_file)
        matching_templates = find_matching_templates(db, orientation)

        for template in matching_templates:
            tasks.append((template, image_file))

    with tqdm(total=len(tasks), ncols=80) as pbar:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_product_image, tmp, img_file) for tmp, img_file in tasks]

            for future in as_completed(futures):
                future.result()
                pbar.update(1)
