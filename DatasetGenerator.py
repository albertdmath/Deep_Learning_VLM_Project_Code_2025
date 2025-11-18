import os
import random
import math
import json
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
    if shape == "circle":
        draw.ellipse([x - half, y - half, x + half, y + half], fill=color)
    elif shape == "square":
        draw.rectangle([x - half, y - half, x + half, y + half], fill=color)
    elif shape == "triangle":
        points = [(x, y - half), (x - half, y + half), (x + half, y + half)]
        draw.polygon(points, fill=color)


def get_dominant_relation(pos_a, pos_b):
    """
    Return whichever spatial relation (left/right/above/below)
    is most dominant between two points (a relative to b).
    Returns a natural-language string (e.g. 'left of') used in captions.
    """
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


def make_caption(a, b, relation, shapes_data):
    """
    Create a caption for each image, for the ground truth.
    a, b are shape tuples: (shape, color_name, color_val, center, bbox)
    """
    shape_a, color_a, _, _, _ = a
    shape_b, color_b, _, _, _ = b

    caption = f"{color_a} {shape_a} {relation} {color_b} {shape_b}"
    distractors = [
        f"{c} {s}"
        for s, c, _, _, _ in shapes_data
        if (s, c) not in [(a[0], a[1]), (b[0], b[1])]
    ]
    if distractors:
        caption += f", with {' and '.join(distractors)} as distractor(s)"
    return caption


def generate_scene(image_id):
    """
    Generate one synthetic image with multiple shapes and return:
    - filename
    - caption
    - shapes_data: list of (shape, color_name, color_val, center, bbox)
    - a_idx, b_idx: indices of the two main shapes
    - relation_raw: 'left of', 'right of', 'above', 'below' (A relative to B)
    """
    num_shapes = random.choice([3, 4, 5])  # with distractors
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

        # Ensure shapes don't overlap too much
        while True:
            center = (
                random.randint(SHAPE_SIZE, IMG_SIZE - SHAPE_SIZE),
                random.randint(SHAPE_SIZE, IMG_SIZE - SHAPE_SIZE),
            )
            if all(math.dist(center, p) > 60 for p in used_positions):
                used_positions.append(center)
                break

        bbox = compute_bbox(center, SHAPE_SIZE)
        shapes_data.append((shape, color_name, color_val, center, bbox))

    # Choose two shapes at random to create a spatial relation caption
    a_idx, b_idx = random.sample(range(len(shapes_data)), 2)
    a = shapes_data[a_idx]
    b = shapes_data[b_idx]
    relation_raw = get_dominant_relation(a[3], b[3])  # 3rd value is center
    caption = make_caption(a, b, relation_raw, shapes_data)
    filename = f"scene_{image_id:04d}.png"

    # Draw the image
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))  # Black background
    draw = ImageDraw.Draw(img)
    for shape, _, color_val, center, _ in shapes_data:
        draw_shape(draw, shape, color_val, center, SHAPE_SIZE)
    img.save(os.path.join(OUTPUT_DIR, filename))

    return filename, caption, shapes_data, a_idx, b_idx, relation_raw


# === Generate dataset and JSONL annotations ===
jsonl_path = os.path.join(OUTPUT_DIR, "annotations.jsonl")

with open(jsonl_path, "w") as jsonl_file:
    for i in range(NUM_IMAGES):
        filename, caption, shapes_data, a_idx, b_idx, relation_raw = generate_scene(i)

        # Convert relation phrase to compact code: "left of" -> "A_left_of_B"
        relation_type = "A_" + relation_raw.replace(" ", "_") + "_B"

        # Extract A data
        shape_a, color_a, _, center_a, bbox_a = shapes_data[a_idx]
        # Extract B data
        shape_b, color_b, _, center_b, bbox_b = shapes_data[b_idx]

        num_distractors = len(shapes_data) - 2

        # Natural-language version for the prompt
        relation_nl = {
            "left of": "to the left of",
            "right of": "to the right of",
            "above": "above",
            "below": "below"
        }[relation_raw]

        caption_prompt = (
            f"Is the {color_a} {shape_a} {relation_nl} the {color_b} {shape_b}?"
        )

        # Build JSON exactly as the user specified
        record = {
            "scene_id": filename.replace(".png", ""),
            "A": {
                "shape": shape_a,
                "color": color_a,
                "center": [center_a[0], center_a[1]],
                "bbox": [bbox_a[0], bbox_a[1], bbox_a[2], bbox_a[3]]
            },
            "B": {
                "shape": shape_b,
                "color": color_b,
                "center": [center_b[0], center_b[1]],
                "bbox": [bbox_b[0], bbox_b[1], bbox_b[2], bbox_b[3]]
            },
            "relation_AB": relation_type,
            "num_distractors": num_distractors,
            "caption_prompt": caption_prompt
        }

        jsonl_file.write(json.dumps(record) + "\n")


print(f"\nGenerated {NUM_IMAGES} images in '{OUTPUT_DIR}' and 'annotations.jsonl' with JSON ground truth.")
