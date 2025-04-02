import os
import math
from PIL import Image, ImageDraw, ImageFont

class NMFKImageAggregator:
    def __init__(self, base_directory, output_directory):
        self.base_directory = base_directory
        self.output_directory = os.path.join(base_directory, output_directory)
        os.makedirs(self.output_directory, exist_ok=True)
        self.final_images = []  # Collect all FINAL images here
        self.process_directories()
    
    def create_square_image_grid(self, images, image_width=400, image_height=350):
        if not images:
            return None
        images_count = len(images)
        columns = rows = math.ceil(math.sqrt(images_count))  
        grid_width = columns * image_width
        grid_height = rows * image_height
        grid_image = Image.new('RGB', (grid_width, grid_height), 'white')
        for index, image in enumerate(images):
            resized_image = image.resize((image_width, image_height), Image.LANCZOS)
            x_offset = (index % columns) * image_width
            y_offset = (index // columns) * image_height
            grid_image.paste(resized_image, (x_offset, y_offset))
        return grid_image

    def create_image_grid(self, images, columns=9, image_width=400, image_height=350):
        rows = (len(images) + columns - 1) // columns
        grid_width = columns * image_width
        grid_height = rows * image_height
        grid_image = Image.new('RGB', (grid_width, grid_height), 'white')
        for index, image in enumerate(images):
            resized_image = image.resize((image_width, image_height), Image.LANCZOS)
            x_offset = (index % columns) * image_width
            y_offset = (index // columns) * image_height
            grid_image.paste(resized_image, (x_offset, y_offset))
        return grid_image

    def annotate_image(self, image, text):
        font = ImageFont.truetype("DejaVuSans.ttf", 40)
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), text, fill='black', font=font)
        return image

    def extract_number(self, filename):
        parts = filename.split("_")
        if parts[0] == "k" and len(parts) > 1 and parts[1].isdigit():
            return int(parts[1])
        return float('inf')

    def custom_sort(self, filename):
        if "FINAL" in filename:
            return -2, filename
        if "cophenetic_coeff" in filename:
            return -1, filename
        return self.extract_number(filename), filename

    def process_directories(self):
        for subdir in os.listdir(self.base_directory):
            subdir_path = os.path.join(self.base_directory, subdir)
            if os.path.isdir(subdir_path) and subdir != "processed_images" and subdir != "post_process":
                for specific_dir in os.listdir(subdir_path):
                    specific_path = os.path.join(subdir_path, specific_dir)
                    if os.path.isdir(specific_path):
                        self.process_specific_directory(specific_path)

        if self.final_images:
            final_grid_image = self.create_square_image_grid(self.final_images)
            if final_grid_image is not None:
                final_output_path = os.path.join(self.output_directory, "FINAL_Combined.png")
                final_grid_image.save(final_output_path)

    def process_specific_directory(self, specific_path):
        images = []
        for file in sorted(os.listdir(specific_path), key=self.custom_sort):
            if file.endswith(".png") and "k_1_con" not in file:
                image_path = os.path.join(specific_path, file)
                image = Image.open(image_path)
                if "FINAL" in file:
                    self.final_images.append(image)  # Add to final_images list
                images.append(self.annotate_or_return_image(file, image))
            
        if images:
            grid_image = self.create_image_grid(images)
            output_path = os.path.join(self.output_directory, f"{os.path.basename(specific_path)}.png")
            grid_image.save(output_path)

    def annotate_or_return_image(self, filename, image):
        if "FINAL" not in filename and "cophenetic_coeff" not in filename and "k_1_" not in filename:
            text = f"{os.path.basename(filename)}, {os.path.basename(os.path.dirname(image.filename))}"
            return self.annotate_image(image, text[:4])
        return image