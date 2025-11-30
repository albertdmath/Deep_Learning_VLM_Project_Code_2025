import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from transformers import AutoTokenizer

# --- Configuration ---
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
ATTN_FOLDER = Path("attentions_qwen2b")
DATA_FOLDER = Path("synthetic_dataset")
ANNOTATIONS_PATH = DATA_FOLDER / "annotations.jsonl"
OUTPUT_CSV = Path("attention_metrics.csv")

# Map dataset relations to the word we want to track in the prompt
RELATION_MAP = {
    "A_below_B": "below",
    "A_above_B": "above",
    "A_left_of_B": "left",
    "A_right_of_B": "right"
}

def load_jsonl(path):
    with open(path, 'r') as f:
        return {json.loads(line)["scene_id"]: json.loads(line) for line in f}

def get_patch_centers(side=9, w=256, h=256):
    """Returns (9, 9, 2) array of (x,y) centers for visual patches."""
    xs = (np.arange(side) + 0.5) * (w / side)
    ys = (np.arange(side) + 0.5) * (h / side)
    return np.stack(np.meshgrid(xs, ys), axis=-1)

def create_ground_truth_mask(record, centers):
    """Creates a boolean mask where the spatial relation is True."""
    rel = record["relation_AB"]
    bx, by = record["B"]["center"]
    X, Y = centers[..., 0], centers[..., 1]
    
    if rel == "A_below_B": return Y > by
    if rel == "A_above_B": return Y < by
    if rel == "A_left_of_B": return X < bx
    if rel == "A_right_of_B": return X > bx
    return np.zeros_like(X, dtype=bool)

def calculate_iou(attn_grid, gt_mask, top_p=0.5):
    """IoU between top p% attention mass and ground truth mask."""
    flat = attn_grid.flatten()
    # Indices of top mass
    sorted_idx = np.argsort(flat)[::-1]
    cumsum = np.cumsum(flat[sorted_idx])
    cumsum /= (cumsum[-1] + 1e-9)
    
    attn_bool = np.zeros_like(flat, dtype=bool)
    attn_bool[sorted_idx[cumsum <= top_p]] = True
    if not attn_bool.any(): attn_bool[sorted_idx[0]] = True # Keep at least max
    
    attn_bool = attn_bool.reshape(attn_grid.shape)
    
    inter = (attn_bool & gt_mask).sum()
    union = (attn_bool | gt_mask).sum()
    return inter / union if union > 0 else 0.0

def calculate_entropy(vec):
    p = vec / (vec.sum() + 1e-9)
    return -(p * torch.log(p + 1e-9)).sum().item()

def main():
    print("Loading resources...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    annotations = load_jsonl(ANNOTATIONS_PATH)
    files = sorted(ATTN_FOLDER.glob("*_attn.pt"))
    centers = get_patch_centers() # 9x9 grid centers
    
    results = []
    
    print(f"Processing {len(files)} files...")
    for fpath in files:
        try:
            # 1. Load Data
            data = torch.load(fpath, map_location="cpu")
            scene_id = data["scene_id"]
            if scene_id not in annotations: continue
            
            record = annotations[scene_id]
            target_word = RELATION_MAP.get(record["relation_AB"])
            if not target_word: continue

            # 2. Find Tokens
            # Encode word with leading space (common in sentencepiece)
            word_id = tokenizer.encode(" " + target_word, add_special_tokens=False)[0]
            token_matches = (data["input_ids"] == word_id).nonzero(as_tuple=True)[0]
            
            vis_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
            vis_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
            v_start_idx = (data["input_ids"] == vis_start).nonzero(as_tuple=True)[0][0].item()
            v_end_idx = (data["input_ids"] == vis_end).nonzero(as_tuple=True)[0][0].item()
            
            if len(token_matches) == 0: continue
            token_idx = token_matches[0].item() # Use first occurrence
            
            # 3. Process Layers
            gt_mask = create_ground_truth_mask(record, centers)
            
            for layer, attn_tensor in data["attentions"].items():
                # Average heads -> (seq, seq)
                avg_attn = attn_tensor.mean(dim=0)
                # Slice: Target Word -> Visual Tokens (between start/end tags)
                vis_attn = avg_attn[token_idx, v_start_idx+1 : v_end_idx].float()
                
                # Check shape (should be 81 for 9x9)
                if vis_attn.numel() != 81: continue
                
                grid = vis_attn.reshape(9, 9).numpy()
                
                # Metrics
                iou = calculate_iou(grid, gt_mask)
                ent = calculate_entropy(vis_attn)
                
                results.append({
                    "scene_id": scene_id,
                    "relation": record["relation_AB"],
                    "word": target_word,
                    "layer": layer,
                    "iou": iou,
                    "entropy": ent
                })
                
        except Exception as e:
            print(f"Error processing {fpath.name}: {e}")

    # Save
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done. Saved metrics for {len(df)} layer-scene pairs to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()