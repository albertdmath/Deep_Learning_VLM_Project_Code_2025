import os
import random
import math
import csv
from PIL import Image, ImageDraw

# === Configuration and Setup ===
OUTPUT_DIR = "synthetic_dataset"
NUM_IMAGES = 100
IMG_SIZE = 256
SHAPE_SIZE = 40
SHAPES = ["circle", "square", "triangle"]
COLORS = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 255, 0),
    "yellow": (255, 255, 0),
    "purple": (160, 32, 240)
}
os.makedirs(OUTPUT_DIR, exist_ok=True)


def draw_shape(draw, shape, color, center, size):
    """Draw one geometric shape with given center coordinates, color, and size."""
    x, y = center
    half = size // 2
    # 3 possible shapes:
    if shape == "circle":
        draw.ellipse([x - half, y - half, x + half, y + half], fill=color)
    elif shape == "square":
        draw.rectangle([x - half, y - half, x + half, y + half], fill=color)
    elif shape == "triangle":
        points = [(x, y - half), (x - half, y + half), (x + half, y + half)]
        draw.polygon(points, fill=color)


def get_dominant_relation(pos_a, pos_b):
    """Return whichever spatial relation (left/right/above/below)
    is most dominant between two points (a relative to b) in order
    to deal with ambiguous diagonal relationships between shapes."""
    dx = pos_b[0] - pos_a[0]
    dy = pos_b[1] - pos_a[1]
    # If horizontal separation is larger, decide left/right for a relative to b
    if abs(dx) > abs(dy):
        return "left of" if dx > 0 else "right of"
    # Otherwise decide above/below for a relative to b
    else:
        return "above" if dy > 0 else "below"


def compute_bbox(center, size):
    """Compute bounding box (xmin, ymin, xmax, ymax) for a shape."""
    x, y = center
    half = size // 2
    return (x - half, y - half, x + half, y + half)


def generate_scene(image_id):
    """Generate one synthetic image with multiple shapes and return its caption, main logic of the code."""
    num_shapes = random.choice([3, 4, 5])
    shapes_data = []
    used_positions = []
    used_shape_color_pairs = set()

    # Ensure no two shapes in one image are the same shape and color
    for _ in range(num_shapes):
        while True:
            shape = random.choice(SHAPES)
            color_name, color_val = random.choice(list(COLORS.items()))
            if (shape, color_name) not in used_shape_color_pairs:
                used_shape_color_pairs.add((shape, color_name))
                break

        """
        Ensure shapes don't overlap.
        I just noticed this might create infinite loops if too many shapes are requested.
        Don't request too many shapes.
        The while loop above could also run out of pairs if too many shapes/colors are used.
        3 shapes, 5 colors = 15 unique pairs, so max 15 shapes per image.
        But for now, this is fine.
        """
        while True:
            center = (random.randint(SHAPE_SIZE, IMG_SIZE - SHAPE_SIZE), random.randint(SHAPE_SIZE, IMG_SIZE - SHAPE_SIZE))
            if all(math.dist(center, p) > 60 for p in used_positions):
                used_positions.append(center)
                break

        bbox = compute_bbox(center, SHAPE_SIZE)
        shapes_data.append((shape, color_name, color_val, center, bbox))

    # Choose two shapes at random to create a spatial relation caption
    a, b = random.sample(shapes_data, 2)
    relation = get_dominant_relation(a[3], b[3]) # 3rd value is center
    caption = make_caption(a, b, relation, shapes_data)
    filename = f"scene_{image_id:04d}.png" # I'm expecting 10,000 images max

    # Draw the image
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0)) # Black background
    draw = ImageDraw.Draw(img)
    for shape, _, color_val, center, _ in shapes_data:
        draw_shape(draw, shape, color_val, center, SHAPE_SIZE)
    img.save(os.path.join(OUTPUT_DIR, filename))

    return filename, caption, shapes_data


def make_caption(a, b, relation, shapes_data):
    """Create a caption for each image, for the ground truth csv file. This can be tweaked as needed."""
    shape_a, color_a, _, _, _ = a
    shape_b, color_b, _, _, _ = b

    caption = f"{color_a} {shape_a} {relation} {color_b} {shape_b}"
    distractors = [f"{c} {s}" for s, c, _, _, _ in shapes_data
                   if (s, c) not in [(a[0], a[1]), (b[0], b[1])]]
    if distractors:
        caption += f", with {' and '.join(distractors)} as distractor(s)"
    return caption


# === Generate dataset ===
csv_path = os.path.join(OUTPUT_DIR, "image_captions_ground_truth.csv")
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Each row: filename, caption, and serialized shape data
    writer.writerow(["filename", "caption", "shapes"])
    for i in range(NUM_IMAGES):
        filename, caption, shapes_data = generate_scene(i)
        # Serialize shape info:
        # shape|color|x|y|xmin|ymin|xmax|ymax
        shapes_str = ";".join([
            f"{s}|{c}|{x}|{y}|{xmin}|{ymin}|{xmax}|{ymax}"
            for s, c, _, (x, y), (xmin, ymin, xmax, ymax) in shapes_data
        ])
        writer.writerow([filename, caption, shapes_str])
        # print(f"{filename}: {caption}")
        # print("  shapes:", shapes_str)

print(f"\nGenerated {NUM_IMAGES} images in '{OUTPUT_DIR}' along with 'image_captions_ground_truth.csv'.")
