from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import pandas as pd

# ------------------------
# SETUP
# ------------------------
model_id = "Qwen/Qwen2-VL-2B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print("Loading model...")
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
).to(device)

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_id)

# Folder with images
folder = Path("./synthetic_dataset")
output_file = Path("inference_results.txt")

# Store the captions for the text prompts
path = Path("./synthetic_dataset/image_captions_ground_truth.csv")
captions = pd.read_csv(path, usecols=["caption"])["caption"]

# Remove the part about distractors
simple_captions = captions.str.split(",", n=1).str[0]
simple_captions = simple_captions + " - does this caption match the picture? Yes/No"

# Clean/create output file
output_file.write_text("")   # empty the file at the start

# ------------------------
# PROCESS IMAGES
# ------------------------

files = sorted(folder.glob("*.png"))

print(f"Found {len(files)} images.")

for prompt_text, img_path in zip(simple_captions, files):
    print(f"Processing {img_path.name}...")

    # Load image
    raw_image = Image.open(img_path).convert("RGB")

    # Build conversation prompt (Qwen2-VL format)
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

    prompt_len = inputs["input_ids"].shape[1]

    # Decode only the generated answer (without the prompt tokens)
    text = processor.batch_decode(
        output_ids[:, prompt_len:],
        skip_special_tokens=True
    )[0]

    # Save to file
    with output_file.open("a") as f:
        f.write(f"Image: {img_path.name}\n")
        f.write(f"Output: {text}\n")
        f.write("-" * 60 + "\n")

print(f"\nAll done. Results saved to: {output_file.resolve()}")
