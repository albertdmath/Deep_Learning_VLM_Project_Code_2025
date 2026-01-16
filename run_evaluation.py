import torch
import numpy as np
import jsonlines
import h5py
from transformers import AutoTokenizer
from attn_tools.attention_extraction import yield_attention_batches
from attn_tools.attention_metrics import *


# CONFIG
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
ANNOTATION_FILE = "synthetic_dataset/annotations.jsonl"
SIDE = 9 # QWEN uses 81 patches for 256x256 images
BATCH_SIZE = 16
SEED = 8122025

# Map dataset relations to the word we want to track in the prompt
RELATION_MAP = {
    "A_below_B": "below",
    "A_above_B": "above",
    "A_left_of_B": "left",
    "A_right_of_B": "right"
}

def compute_all_stats(grids, gt_rel, gt_union):
    """
    Computes all 5 metrics for (L, H, SIDE, SIDE) attention grid.
    """
    return {
        "iou_relation": calculate_iou(grids, gt_rel),
        "iou_union":    calculate_iou(grids, gt_union),
        "com_relation": calculate_com_distance(grids, gt_rel),
        "com_union":    calculate_com_distance(grids, gt_union),
        "entropy":      calculate_entropy(grids)
    }


# MAIN LOOP        
def main():
    np.random.seed(SEED)
    print("Loading resources...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    annotations = {}
    with jsonlines.open(ANNOTATION_FILE) as reader:
        for rec in reader:
            annotations[rec["scene_id"]] = rec

    centers = get_patch_centers()

    # Open HDF5
    h5_path = "stats.h5"
    h5 = h5py.File(h5_path, "a")

    print("Streaming attention...")
    for item in yield_attention_batches(ANNOTATION_FILE, batch_size=BATCH_SIZE):

        scene_id = item["scene_id"]
        print(f"\nProcessing {scene_id}")

        attn = item["attentions"].float().cpu().numpy()
        assert attn.ndim == 4, "Attention tensor must be (layers, heads, seq, seq)"
        input_ids = item["input_ids"].cpu().numpy()

        record = annotations[scene_id]

        # Resolve tokens
        rel_word = RELATION_MAP[record["relation_AB"]]
        tok_words = {
            "rel": rel_word,
            "A": record["A"]["shape"],
            "B": record["B"]["shape"]
        }

        token_map, visual_range = find_token_and_visual_range(
            input_ids, list(tok_words.values()), tokenizer
        )
        for word in tok_words.values():
            if word not in token_map:
                raise KeyError(f"Token '{word}' not found")

        if len(visual_range) != SIDE * SIDE:
            raise ValueError(f"Expected {SIDE*SIDE} patches, got {len(visual_range)}")


        L, H = attn.shape[:2]

        # Prepare HDF5 group
        scene = h5.require_group(f"scenes/{scene_id}")

        # Relation token stats
        tok_idx = token_map[rel_word][0]
        grids = attn[:, :, tok_idx, visual_range].reshape(L, H, SIDE, SIDE)

        gt_rel = create_gt_relation_mask(record, centers)
        gt_union = create_gt_union_mask(record, centers)
        assert gt_rel.shape == (SIDE, SIDE)
        assert gt_union.shape == (SIDE, SIDE)


        rel_stats = compute_all_stats(grids, gt_rel, gt_union)

        grp = scene.require_group("rel")
        for k, v in rel_stats.items():
            if k in grp: del grp[k]
            grp.create_dataset(k, data=v, compression="gzip")

        # Entity tokens A and B
        for obj in ["A", "B"]:

            word = tok_words[obj]
            tok_idx = token_map[word][0]
            grids = attn[:, :, tok_idx, visual_range].reshape(L, H, SIDE, SIDE)

            gt = create_gt_bbox_mask(record, centers, object=obj)

            obj_stats = {
                "iou_bbox": calculate_iou(grids, gt),
                "com_bbox": calculate_com_distance(grids, gt),
                "entropy":  calculate_entropy(grids)
            }

            grp = scene.require_group(obj)
            for k, v in obj_stats.items():
                if k in grp: del grp[k]
                grp.create_dataset(k, data=v, compression="gzip")

        # BASELINES
        baselines = {
            "uniform":  baseline_uniform(L, H, SIDE),
            "gaussian": baseline_gaussian(L, H, SIDE)
        }
        for name, grid in baselines.items():
            assert grid.shape == (L, H, SIDE, SIDE), f"Baseline '{name}' has wrong shape"
            assert np.allclose(grid.sum(axis=(2,3)), 1.0), f"{name} is not normalized"


        baseline_grp = scene.require_group("baselines")

        for name, grids in baselines.items():

            stats = compute_all_stats(grids, gt_rel, gt_union)

            grp = baseline_grp.require_group(name)

            for k, v in stats.items():
                if k in grp: del grp[k]
                grp.create_dataset(k, data=v, compression="gzip")

        
        # Save metadata
        scene.attrs["relation"] = record["relation_AB"]
        scene.attrs["tokens"] = list(tok_words.values())
        
        # Force write to disk
        h5.flush()

        print(f"{scene_id} saved.")
        

    h5.close()
    print("\nAll statistics written to stats.h5")


if __name__ == "__main__":
    main()