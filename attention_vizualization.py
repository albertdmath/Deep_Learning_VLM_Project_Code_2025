import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from transformers import AutoTokenizer
import math

# --- Configuration ---
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
ATTN_FOLDER = Path("attentions_qwen2b")
DATA_FOLDER = Path("synthetic_dataset")
ANALYSIS_CSV = Path("attention_metrics.csv")
OUTPUT_FOLDER = Path("visualizations")

# WHICH LAYER DO YOU WANT TO VISUALIZE FOR ALL IMAGES?
TARGET_LAYER = 19 

def normalize(grid):
    """Normalize grid to 0-1 range."""
    g = grid - grid.min()
    return g / (g.max() + 1e-9)

def create_overlay(attn_vector, img_path, output_path, title):
    """Generates and saves the attention overlay image."""
    try:
        # Reshape 81 -> 9x9
        grid = attn_vector.float().reshape(9, 9).numpy()
        grid = normalize(grid)
        
        # Load Original Image
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        
        # Resize heatmap to match image
        heatmap = Image.fromarray((grid * 255).astype(np.uint8))
        heatmap = heatmap.resize((w, h), resample=Image.BILINEAR)
        heatmap_np = np.array(heatmap) / 255.0
        
        # Plot
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img)
        ax.imshow(heatmap_np, alpha=0.5, cmap='jet') # Alpha controls transparency
        ax.set_title(title, fontsize=9)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        
    except Exception as e:
        print(f"Failed to visualize overlay for {img_path}: {e}")

def show_layerwise_grids(layerwise_grids_norm, output_path, scene_id, word):
    """
    Saves a grid plot of normalized attention heatmaps for all layers.
    layerwise_grids_norm: dict {layer_idx: tensor(9,9)} normalized 0-1
    """
    try:
        layers = sorted(layerwise_grids_norm.keys())
        cols = 6
        rows = math.ceil(len(layers) / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(14, 12))
        axes = axes.flatten()

        for i, layer in enumerate(layers):
            g = layerwise_grids_norm[layer].numpy()
            axes[i].imshow(g, cmap="viridis")
            axes[i].set_title(f"L{layer}")
            axes[i].axis("off")

        # Hide unused subplots
        for j in range(i+1, len(axes)):
            axes[j].axis("off")

        plt.suptitle(f"Layer-wise Attention: {scene_id} | '{word}'", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Make room for suptitle
        plt.savefig(output_path)
        plt.close(fig)
    except Exception as e:
        print(f"Failed to save layerwise grid for {scene_id}: {e}")

def main():
    if not ANALYSIS_CSV.exists():
        print("Please run attention_analysis.py first.")
        return
    
    df = pd.read_csv(ANALYSIS_CSV)

    # Load metadata and filter for target layer
    df_layer = df[df["layer"] == TARGET_LAYER]

    # We use the CSV mainly to know which scenes/words are valid targets
    # We will process unique scene-word pairs found in the analysis

    unique_pairs = df_layer[["scene_id", "word", "iou"]].drop_duplicates(subset=["scene_id", "word"])
    
    if unique_pairs.empty:
        print("No valid data found in CSV.")
        return

    # Create output directory
    layer_out_folder = OUTPUT_FOLDER / f"layer_{TARGET_LAYER}"
    grid_out_folder = OUTPUT_FOLDER / "all_layers_grids"
    
    layer_out_folder.mkdir(parents=True, exist_ok=True)
    grid_out_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating visualizations for {len(unique_pairs)} scenes...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Iterate over every unique scene/word pair
    for _, row in unique_pairs.iterrows():
        scene_id = row["scene_id"]
        word = row["word"]
        
        pt_path = ATTN_FOLDER / f"{scene_id}_attn.pt"
        img_path = DATA_FOLDER / f"{scene_id}.png"
        
        if not pt_path.exists() or not img_path.exists(): continue
        
        # Load Attention Tensor
        data = torch.load(pt_path, map_location="cpu")
        
        # Find Token Index
        word_id = tokenizer.encode(" " + word, add_special_tokens=False)[0]
        token_match = (data["input_ids"] == word_id).nonzero(as_tuple=True)[0]
        if len(token_match) == 0: continue
        token_idx = token_match[0].item()
        
        # Find Visual Range
        v_start = (data["input_ids"] == tokenizer.convert_tokens_to_ids("<|vision_start|>")).nonzero()[0].item()
        v_end = (data["input_ids"] == tokenizer.convert_tokens_to_ids("<|vision_end|>")).nonzero()[0].item()
        
        # -- 1. Process for Single Layer Overlay --
        layer_attn = data["attentions"][TARGET_LAYER]
        avg_attn = layer_attn.mean(dim=0)
        vis_attn = avg_attn[token_idx, v_start+1 : v_end]
        
        out_name = layer_out_folder / f"{scene_id}_{word}.png"

        iou = row["iou"]

        title = f"{scene_id} | '{word}' | L{TARGET_LAYER} | IoU: {iou:.2f}"
        create_overlay(vis_attn, img_path, out_name, title)
        
        # -- 2. Process for All Layers Grid --
        layerwise_grids = {}
        for layer, attn_tensor in data["attentions"].items():
            avg = attn_tensor.mean(dim=0)
            vec = avg[token_idx, v_start+1 : v_end].float()
            if vec.numel() == 81:
                grid = vec.reshape(9, 9)
                layerwise_grids[layer] = normalize(grid)
        
        grid_out_name = grid_out_folder / f"{scene_id}_{word}_all_layers.png"
        show_layerwise_grids(layerwise_grids, grid_out_name, scene_id, word)
        
        print(f"Processed {scene_id}")

if __name__ == "__main__":
    main()