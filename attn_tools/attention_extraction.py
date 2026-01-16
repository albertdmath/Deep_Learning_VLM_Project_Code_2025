import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from functools import lru_cache
from pathlib import Path
from PIL import Image
import jsonlines


MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


@lru_cache(maxsize=1)
def _load_qwen():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)

    model.config.use_cache = False
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    return model, processor, device, dtype


def yield_attention_batches(
    jsonl_path,
    image_root="synthetic_dataset",
    batch_size=4
):
    """
    Streams attention tensors from large JSONL annotation file.

    Each yielded item:
        {
            "scene_id": str,
            "input_ids": Tensor,
            "attentions": Tensor,   # (L, H, S, S)
        }
    """

    model, processor, device, dtype = _load_qwen()
    image_root = Path(image_root)

    def run_batch(batch_records):
        images = []
        prompts = []
        scene_ids = []

        for rec in batch_records:
            sid = rec["scene_id"]
            img_path = image_root / f"{sid}.png"

            images.append(Image.open(img_path).convert("RGB"))
            prompts.append(rec["caption_prompt"])
            scene_ids.append(sid)

        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": text},
                    ],
                }
            ]
            for text in prompts
        ]

        chat_prompts = [
            processor.apply_chat_template(conv, add_generation_prompt=True)
            for conv in conversations
        ]

        inputs = processor(
            images=images,
            text=chat_prompts,
            return_tensors="pt",
            padding=True,
        ).to(device, dtype)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )

        input_ids = inputs["input_ids"].cpu()

        # Stack attentions: (L, B, H, S, S)
        attn = torch.stack([a.cpu() for a in outputs.attentions])

        # Yield per sample: (L, H, S, S)
        for i, sid in enumerate(scene_ids):
            yield {
                "scene_id": sid,
                "input_ids": input_ids[i],
                "attentions": attn[:, i],
            }

    # Stream JSONL
    batch = []

    with jsonlines.open(jsonl_path) as reader:
        for record in reader:
            batch.append(record)
            if len(batch) == batch_size:
                yield from run_batch(batch)
                batch.clear()

        # last partial batch
        if batch:
            yield from run_batch(batch)
