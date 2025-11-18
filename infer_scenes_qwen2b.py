from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import json
import jsonlines   # pip install jsonlines

# ------------------------
# SETUP
# ------------------------
model_id = "Qwen/Qwen2-VL-2B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

print("Loading model...")
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
).to(device)

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_id)

folder = Path("./synthetic_dataset")
annotations_path = folder / "annotations.jsonl"
output_file = Path("inference_results.txt")

# Clean output file
output_file.write_text("")   # empty the file at the start

# ------------------------
# LOAD ALL JSON ANNOTATIONS
# ------------------------
print("Reading JSON annotations...")

annotations = {}  # map: scene_id -> record

with jsonlines.open(annotations_path) as reader:
    for record in reader:
        scene_id = record["scene_id"]    # e.g. "scene_0001"
        annotations[scene_id] = record

print(f"Loaded {len(annotations)} annotation entries.")

# ------------------------
# PROCESS IMAGES
# ------------------------
files = sorted(folder.glob("*.png"))
print(f"Found {len(files)} images.")

for img_path in files:
    scene_id = img_path.stem            # "scene_0001"
    if scene_id not in annotations:
        print(f"Warning: No annotation found for {img_path.name}")
        continue

    record = annotations[scene_id]
    prompt_text = record["caption_prompt"]

    print(f"Processing {img_path.name}...")

    # Load image
    raw_image = Image.open(img_path).convert("RGB")

    # Build Qwen2-VL chat prompt
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]

    chat_prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True
    )

    # Prepare model inputs
    inputs = processor(
        images=[raw_image],
        text=[chat_prompt],
        return_tensors="pt",
        padding=True,
    ).to(device, dtype)

    # Generate output
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False
        )

    # Only decode the generated text
    prompt_len = inputs["input_ids"].shape[1]
    text = processor.batch_decode(
        output_ids[:, prompt_len:],
        skip_special_tokens=True
    )[0]

    # Save results
    with output_file.open("a") as f:
        f.write(f"Image: {img_path.name}\n")
        f.write(f"Prompt: {prompt_text}\n")
        f.write(f"Output: {text}\n")
        f.write("-" * 60 + "\n")

print(f"\nAll done. Results saved to: {output_file.resolve()}")
