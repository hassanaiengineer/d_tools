import os
import json
import ntpath
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pydantic import BaseModel

# Disable the pixel limit warning in Pillow
Image.MAX_IMAGE_PIXELS = None

OUTPUT_DIR = "output"
INPUT_DIR = "input"
SQUARE_TOLERANCE = 0.05  # 5% tolerance for square images


class Template(BaseModel):
    size: int
    orientation: str
    path: str
    dimensions: list
    box: list
    images: list

    def get_dimensions_tuple_list(self) -> list[tuple]:
        return [tuple(map(int, dimension.split(","))) for dimension in self.dimensions]

    def get_boxes_tuple_list(self) -> list[tuple]:
        return [tuple(map(int, box.split(","))) for box in self.box]


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
            if file.endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(root, file)
                image_paths.append(full_path)
    return image_paths


def create_output_directory_structure(input_file: str, template_name: str) -> str:
    relative_path = os.path.relpath(input_file, INPUT_DIR)
    base_name = get_file_name(input_file, extension=False)
    
    output_dir = os.path.join(OUTPUT_DIR, os.path.dirname(relative_path))
    os.makedirs(output_dir, exist_ok=True)

    output_file_name = f"{template_name}-{base_name}.webp"
    return os.path.join(output_dir, output_file_name)


def determine_orientation(image_path: str) -> str:
    with Image.open(image_path) as img:
        width, height = img.size
        aspect_ratio = width / height

        if abs(aspect_ratio - 1) <= SQUARE_TOLERANCE:  # Square tolerance check
            return "Square"
        elif width > height:
            return "Landscape"
        else:
            return "Portrait"


def find_matching_template(db: list, orientation: str, base_name: str) -> Template:
    for entry in db:
        if entry["orientation"] == orientation and entry["path"].endswith(base_name):
            return Template(**entry)
    raise FileNotFoundError(f"No matching template found for orientation '{orientation}' with name '{base_name}'")


def process_product_image(template: Template, image_file: str) -> None:
    try:
        template_image = Image.open(template.path)
        
        process_image = Image.open(image_file)
        image_resized = process_image.resize(template.get_dimensions_tuple_list()[0])
        template_image.paste(image_resized, box=template.get_boxes_tuple_list()[0])

        # Convert RGBA to RGB if needed before saving as WebP
        if template_image.mode == 'RGBA':
            template_image = template_image.convert('RGB')

        output_path = create_output_directory_structure(image_file, get_file_name(template.path, extension=False))
        
        # Save as WebP format with quality of 50%
        template_image.save(output_path, format='WEBP', quality=50)

    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    db = load_json_db("db.json")
    images_list = recursive_images_to_list(INPUT_DIR)

    random = 0  # This flag can be used to toggle random vs chunking behavior
    size = 1
    sample_amount = 3

    tasks = []

    if random == 1:
        for obj in db:
            if not size == 0 and obj["size"] == size:
                tmp = Template(
                    images=get_random_combinations_of_images(
                        images_list, size, sample_amount
                    ),
                    **obj,
                )
                tasks.append(tmp)

            elif size == 0:
                tmp = Template(
                    images=get_random_combinations_of_images(
                        images_list, obj["size"], sample_amount
                    ),
                    **obj,
                )
                tasks.append(tmp)

    elif random == 0:
        for image_file in images_list:
            orientation = determine_orientation(image_file)
            base_name = get_file_name(image_file)
            matching_template = find_matching_template(db, orientation, base_name)
            tasks.append((matching_template, image_file))

    with tqdm(total=len(tasks), ncols=80) as pbar:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_product_image, tmp, img_file) for tmp, img_file in tasks]

            for future in as_completed(futures):
                future.result()
                pbar.update(1)
