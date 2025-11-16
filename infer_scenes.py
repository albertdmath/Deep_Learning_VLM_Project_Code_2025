from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import pandas as pd

# ------------------------
# SETUP
# ------------------------
model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"

print("Loading model...")
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id, 
    dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to("cuda")

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
simple_captions = simple_captions + " - does this caption match the picture? Why?"

# Clean/create output file
output_file.write_text("")   # empty the file at the start

# ------------------------
# PROCESS IMAGES
# ------------------------

files = sorted(folder.glob("*.png"))

print(f"Found {len(files)} images.")

for prompt, img_path in zip(simple_captions, files):
    print(f"Processing {img_path.name}...")

    # Load image
    raw_image = Image.open(img_path).convert("RGB")

    # Build conversation prompt
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Prepare model inputs
    inputs = processor(
        images=raw_image,
        text=prompt,
        return_tensors='pt'
    ).to("cuda", torch.float16)

    # Generate output
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False
        )

    prompt_len = inputs["input_ids"].shape[1]
    text = processor.decode(output_ids[0][prompt_len:], skip_special_tokens=True)

    # Save to file
    with output_file.open("a") as f:
        f.write(f"Image: {img_path.name}\n")
        f.write(f"Output: {text}\n")
        f.write("-" * 60 + "\n")

print(f"\nAll done. Results saved to: {output_file.resolve()}")
