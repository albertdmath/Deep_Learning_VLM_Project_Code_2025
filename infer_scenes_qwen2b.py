from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import json
import jsonlines   # pip install jsonlines


# SETUP
model_id = "Qwen/Qwen2-VL-2B-Instruct"
BATCH_SIZE = 16   # adjust based on VRAM

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


# Load JSON annotations
print("Reading JSON annotations...")

annotations = {}  # map: scene_id -> record

with jsonlines.open(annotations_path) as reader:
    for record in reader:
        scene_id = record["scene_id"]    # e.g. "scene_0001"
        annotations[scene_id] = record

print(f"Loaded {len(annotations)} annotation entries.")

# Process Images (in batches)
files = sorted(folder.glob("*.png"))
print(f"Found {len(files)} images.")

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

for batch_files in chunked(files, BATCH_SIZE):
    images = []
    prompts = []
    metadata = []

    for img_path in batch_files:
        scene_id = img_path.stem
        if scene_id not in annotations:
            print(f"Warning: No annotation found for {img_path.name}")
            continue

        record = annotations[scene_id]
        prompt_text = record["caption_prompt"]

        raw_image = Image.open(img_path).convert("RGB")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        chat_prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )

        images.append(raw_image)
        prompts.append(chat_prompt)
        metadata.append((img_path.name, prompt_text))

    if len(images) == 0:
        continue

    print(f"Processing batch of {len(images)} images...")

    # Prepare batched inputs
    inputs = processor(
        images=images,
        text=prompts,
        return_tensors="pt",
        padding=True,
    ).to(device, dtype)

    # Generate outputs for batch
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False
        )

    # Decode per sample (strip prompt)
    prompt_len = inputs["input_ids"].shape[1]
    decoded = processor.batch_decode(
        output_ids[:, prompt_len:],
        skip_special_tokens=True
    )

    # Write batch results
    with output_file.open("a") as f:
        for (img_name, prompt), text in zip(metadata, decoded):
            f.write(f"{text}\n")

print(f"\nAll done. Results saved to: {output_file.resolve()}")

