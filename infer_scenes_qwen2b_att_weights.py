from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
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
    dtype=dtype,                 # instead of torch_dtype (HF deprecation warning)
    low_cpu_mem_usage=True,
    attn_implementation="eager", # needed for attentions
).to(device)

model.config.use_cache = False

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_id)

folder = Path("./synthetic_dataset")
annotations_path = folder / "annotations.jsonl"
output_file = Path("inference_results.txt")

attn_folder = Path("attentions_qwen2b")
attn_folder.mkdir(exist_ok=True)

# Clean output file
output_file.write_text("")

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

    # --------------------------------------
    # 1) FORWARD PASS TO GET ATTENTIONS
    # --------------------------------------
    with torch.no_grad():
        base_outputs = model(
            **inputs,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    # Store attentions as a dict: layer_idx -> tensor (num_heads, seq_len, seq_len)
    attentions_cpu = {}
    if base_outputs.attentions is None:
        print("WARNING: model returned no attentions at all.")
    else:
        for layer_idx, layer_attn in enumerate(base_outputs.attentions):
            if layer_attn is None:
                continue
            # layer_attn: (batch, num_heads, seq_len, seq_len) -> remove batch dim
            attentions_cpu[layer_idx] = layer_attn[0].cpu()

    # also save input_ids so you can match tokens later
    input_ids_cpu = inputs["input_ids"][0].cpu()

    # save everything to a .pt file per scene
    torch.save(
        {
            "scene_id": scene_id,
            "input_ids": input_ids_cpu,
            "attentions": attentions_cpu,
        },
        attn_folder / f"{scene_id}_attn.pt",
    )

    # --------------------------------------
    # 2) GENERATION
    # --------------------------------------
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False
        )

    # Only decode the generated text (skip prompt tokens)
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
print(f"Attention tensors saved under: {attn_folder.resolve()}")
