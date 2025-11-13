# https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf

# from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
# import torch
# from PIL import Image
# import requests


# processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")




# model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float32, low_cpu_mem_usage=True) 
# #model.to("cuda:0")

# # prepare image and text prompt, using the appropriate prompt template
# url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
# image = Image.open(requests.get(url, stream=True).raw)

# # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
# # Each value in "content" has to be a list of dicts with types ("text", "image") 
# conversation = [
#     {

#       "role": "user",
#       "content": [
#           {"type": "text", "text": "What is shown in this image?"},
#           {"type": "image"},
#         ],
#     },
# ]
# prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

# # autoregressively complete prompt
# output = model.generate(**inputs, max_new_tokens=100)

# print(processor.decode(output[0], skip_special_tokens=True))


# load_vlm_model_safe.py
# Hugging Face LLaVA v1.6 Mistral inference (CPU/GPU safe)

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests

# -----------------------------
# 1. Auto-detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -----------------------------
# 2. Load processor
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

# -----------------------------
# 3. Load model
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16 if device=="cuda" else torch.float32,
    low_cpu_mem_usage=True
)
model.to(device)

# -----------------------------
# 4. Load a sample image
url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)

# -----------------------------
# 5. Define conversation / prompt
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is shown in this image?"},
            {"type": "image"},
        ],
    },
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# -----------------------------
# 6. Preprocess inputs
inputs = processor(images=image, text=prompt, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}  # move to correct device

# -----------------------------
# 7. Generate output
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=100)

# -----------------------------
# 8. Decode and print
response = processor.decode(output[0], skip_special_tokens=True)
print("Model output:", response)
