import os
import random
import math
import csv
from PIL import Image, ImageDraw

# === Configuration ===
OUTPUT_DIR = "synthetic_dataset"
NUM_IMAGES = 10
IMG_SIZE = 256
SHAPES = ["circle", "square", "triangle"]
COLORS = {
    "red": (220, 20, 60),
    "blue": (30, 144, 255),
    "green": (34, 139, 34),
    "yellow": (255, 215, 0),
    "purple": (147, 112, 219)
}
RELATIONS = ["left of", "right of", "above", "below"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def draw_shape(draw, shape, color, center, size):
    x, y = center
    half = size // 2
    if shape == "circle":
        draw.ellipse([x-half, y-half, x+half, y+half], fill=color)
    elif shape == "square":
        draw.rectangle([x-half, y-half, x+half, y+half], fill=color)
    elif shape == "triangle":
        points = [(x, y-half), (x-half, y+half), (x+half, y+half)]
        draw.polygon(points, fill=color)

def generate_scene(image_id):
    num_shapes = random.choice([3, 4, 5])
    shapes_data = []
    used_positions = []

    for _ in range(num_shapes):
        shape = random.choice(SHAPES)
        color_name, color_val = random.choice(list(COLORS.items()))
        # Ensure shapes don't overlap too much
        while True:
            center = (random.randint(40, 216), random.randint(40, 216))
            if all(math.dist(center, p) > 60 for p in used_positions):
                used_positions.append(center)
                break
        shapes_data.append((shape, color_name, color_val, center))

    # Choose two shapes to describe
    a, b = random.sample(shapes_data, 2)
    relation = random.choice(RELATIONS)
    caption = make_caption(a, b, relation, shapes_data)
    filename = f"scene_{image_id:02d}.png"

    # Draw the image
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for shape, _, color_val, center in shapes_data:
        draw_shape(draw, shape, color_val, center, 40)
    img.save(os.path.join(OUTPUT_DIR, filename))

    return filename, caption

def make_caption(a, b, relation, shapes_data):
    shape_a, color_a, _, pos_a = a
    shape_b, color_b, _, pos_b = b

    # Spatial consistency check
    if relation == "left of" and not (pos_a[0] < pos_b[0]):
        relation = "right of"
    elif relation == "right of" and not (pos_a[0] > pos_b[0]):
        relation = "left of"
    elif relation == "above" and not (pos_a[1] < pos_b[1]):
        relation = "below"
    elif relation == "below" and not (pos_a[1] > pos_b[1]):
        relation = "above"

    caption = f"{color_a} {shape_a} {relation} {color_b} {shape_b}"
    distractors = [f"{c} {s}" for s, c, _, _ in shapes_data if (s, c) not in [(a[0], a[1]), (b[0], b[1])]]
    if distractors:
        caption += f", with {' and '.join(distractors)} as distractor(s)"
    return caption

# === Generate dataset ===
csv_path = os.path.join(OUTPUT_DIR, "captions.csv")
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "caption"])
    for i in range(NUM_IMAGES):
        filename, caption = generate_scene(i)
        writer.writerow([filename, caption])
        print(f"{filename}: {caption}")

print(f"\n✅ Generated {NUM_IMAGES} images in '{OUTPUT_DIR}' with captions saved to 'captions.csv'.")
