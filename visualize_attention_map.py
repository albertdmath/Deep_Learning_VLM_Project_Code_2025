"""
Utility for generating attention heatmaps on-demand for specific scenes.

This module runs inference for individual scenes and extracts attention patterns
from specific transformer layers and heads. Use this to generate a small number
of figures for your report.

Usage:
    python visualize_attention_map.py scene_0042 15 8
    python visualize_attention_map.py scene_0042 15 8 --token A --output fig1.png
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import jsonlines
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer
from attn_tools.attention_metrics import find_token_and_visual_range

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
ANNOTATION_FILE = "synthetic_dataset/annotations.jsonl"
IMAGE_ROOT = Path("synthetic_dataset")
SIDE = 9  # Qwen uses 81 patches (9x9) for 256x256 images


def visualize_attention_map(
    scene_id: str,
    layer: int,
    head: int,
    token_type: str = "rel",
    output_path: str = None,
    cmap: str = "hot",
    show_grid: bool = True
):
    """
    Generate and save an attention heatmap for a specific scene, layer, and head.

    This function runs inference on-demand for the specified scene. Use this to
    generate a small number of figures for your report. Requires GPU access.

    Args:
        scene_id: Scene identifier (e.g., "scene_0042")
        layer: Transformer layer index (0-based, 0-27 for Qwen-2B)
        head: Attention head index (0-based, 0-15 for Qwen-2B)
        token_type: Which token's attention to visualize. Options:
                    - "rel": relation token (e.g., "left", "right", "above", "below")
                    - "A": object A token (e.g., "circle", "square")
                    - "B": object B token
        output_path: Path to save the PNG file. If None, saves to
                     "visualizations/{scene_id}_L{layer}_H{head}_{token_type}.png"
        cmap: Matplotlib colormap name (default: "hot")
        show_grid: Whether to show grid lines separating patches (default: True)

    Returns:
        np.ndarray: The (9, 9) attention grid that was saved

    Raises:
        ValueError: If scene_id not found, layer/head out of bounds, or token not found
    """

    # Load model and processor
    print(f"Loading model {MODEL_ID}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)

    model.config.use_cache = False
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Load annotation for the scene
    print(f"Loading annotation for {scene_id}...")
    annotation = None
    with jsonlines.open(ANNOTATION_FILE) as reader:
        for record in reader:
            if record["scene_id"] == scene_id:
                annotation = record
                break

    if annotation is None:
        raise ValueError(f"Scene '{scene_id}' not found in {ANNOTATION_FILE}")

    # Load image
    img_path = IMAGE_ROOT / f"{scene_id}.png"
    if not img_path.exists():
        raise ValueError(f"Image not found: {img_path}")

    image = Image.open(img_path).convert("RGB")
    prompt = annotation["caption_prompt"]

    # Prepare input
    print(f"Running inference on {scene_id}...")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    chat_prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True
    )

    inputs = processor(
        images=[image],
        text=[chat_prompt],
        return_tensors="pt",
        padding=True,
    ).to(device, dtype)

    # Run model with attention extraction
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    # Extract attention tensors
    # outputs.attentions is a tuple of (batch, heads, seq, seq) tensors, one per layer
    attentions = torch.stack([a.cpu().float() for a in outputs.attentions])  # (L, B, H, S, S)
    input_ids = inputs["input_ids"].cpu().numpy()[0]  # (S,)

    # Validate layer and head indices
    L, B, H, S, _ = attentions.shape
    if layer < 0 or layer >= L:
        raise ValueError(f"Layer {layer} out of bounds. Model has {L} layers (0-{L-1})")
    if head < 0 or head >= H:
        raise ValueError(f"Head {head} out of bounds. Layer has {H} heads (0-{H-1})")

    # Determine which token to extract attention from
    relation_map = {
        "A_below_B": "below",
        "A_above_B": "above",
        "A_left_of_B": "left",
        "A_right_of_B": "right"
    }

    if token_type == "rel":
        target_word = relation_map[annotation["relation_AB"]]
    elif token_type == "A":
        target_word = annotation["A"]["shape"]
    elif token_type == "B":
        target_word = annotation["B"]["shape"]
    else:
        raise ValueError(f"Invalid token_type '{token_type}'. Must be 'rel', 'A', or 'B'")

    # Find token positions and visual range
    token_map, visual_range = find_token_and_visual_range(
        input_ids,
        [target_word],
        tokenizer
    )

    if target_word not in token_map:
        raise ValueError(f"Token '{target_word}' not found in input sequence")

    if len(visual_range) != SIDE * SIDE:
        raise ValueError(f"Expected {SIDE*SIDE} visual tokens, got {len(visual_range)}")

    # Extract attention from the target token to visual tokens
    token_idx = token_map[target_word][0]
    visual_range_list = list(visual_range)

    # Get attention: (S,) -> slice to visual tokens -> reshape to grid
    attn_vector = attentions[layer, 0, head, token_idx, visual_range_list].numpy()  # (81,)
    attn_grid = attn_vector.reshape(SIDE, SIDE)  # (9, 9)

    # Normalize for visualization (some attention patterns may not sum to 1 over visual tokens only)
    attn_grid_normalized = attn_grid / (attn_grid.sum() + 1e-9)

    # Create output path
    if output_path is None:
        output_dir = Path("visualizations/heatmaps")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{scene_id}_L{layer}_H{head}_{token_type}.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Plot and save
    print(f"Saving attention heatmap to {output_path}...")
    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(attn_grid_normalized, cmap=cmap, interpolation='nearest')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', rotation=270, labelpad=15)

    # Add grid lines
    if show_grid:
        ax.set_xticks(np.arange(SIDE) - 0.5, minor=True)
        ax.set_yticks(np.arange(SIDE) - 0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)

    # Labels
    ax.set_xticks(np.arange(SIDE))
    ax.set_yticks(np.arange(SIDE))
    ax.set_xlabel('Patch Column')
    ax.set_ylabel('Patch Row')
    ax.set_title(f'{scene_id} | Layer {layer} Head {head} | Token: "{target_word}"')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved attention map to {output_path}")
    print(f"  Token: '{target_word}' (position {token_idx})")
    print(f"  Attention sum over visual tokens: {attn_grid.sum():.4f}")
    print(f"  Min: {attn_grid.min():.6f}, Max: {attn_grid.max():.6f}")

    return attn_grid_normalized


def visualize_attention_with_image(
    scene_id: str,
    layer: int,
    head: int,
    token_type: str = "rel",
    output_path: str = None,
    cmap: str = "hot",
    show_grid: bool = True,
    overlay: bool = False,
    alpha: float = 0.6
):
    """
    Generate a side-by-side visualization showing the original image and attention heatmap.

    This creates a figure with two or three subplots:
    - Left: Original scene image
    - Middle (if overlay=True): Attention overlay on image
    - Right: Attention heatmap over visual patches

    Args:
        scene_id: Scene identifier (e.g., "scene_0042")
        layer: Transformer layer index (0-based, 0-27 for Qwen-2B)
        head: Attention head index (0-based, 0-15 for Qwen-2B)
        token_type: Which token's attention to visualize ("rel", "A", or "B")
        output_path: Path to save the PNG file. If None, saves to
                     "visualizations/{scene_id}_L{layer}_H{head}_{token_type}_combined.png"
        cmap: Matplotlib colormap name (default: "hot")
        show_grid: Whether to show grid lines on the heatmap (default: True)
        overlay: Whether to add a middle subplot with attention overlaid on image (default: False)
        alpha: Transparency for overlay (0=transparent, 1=opaque) (default: 0.6)

    Returns:
        tuple: (np.ndarray, PIL.Image) - attention grid and original image
    """

    # Load model and processor
    print(f"Loading model {MODEL_ID}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)

    model.config.use_cache = False
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Load annotation
    print(f"Loading annotation for {scene_id}...")
    annotation = None
    with jsonlines.open(ANNOTATION_FILE) as reader:
        for record in reader:
            if record["scene_id"] == scene_id:
                annotation = record
                break

    if annotation is None:
        raise ValueError(f"Scene '{scene_id}' not found in {ANNOTATION_FILE}")

    # Load image
    img_path = IMAGE_ROOT / f"{scene_id}.png"
    if not img_path.exists():
        raise ValueError(f"Image not found: {img_path}")

    image = Image.open(img_path).convert("RGB")
    prompt = annotation["caption_prompt"]

    # Run inference
    print(f"Running inference on {scene_id}...")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    chat_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=[image], text=[chat_prompt], return_tensors="pt", padding=True).to(device, dtype)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, use_cache=False, return_dict=True)

    # Extract attention
    attentions = torch.stack([a.cpu().float() for a in outputs.attentions])
    input_ids = inputs["input_ids"].cpu().numpy()[0]

    # Validate indices
    L, B, H, S, _ = attentions.shape
    if layer < 0 or layer >= L:
        raise ValueError(f"Layer {layer} out of bounds. Model has {L} layers (0-{L-1})")
    if head < 0 or head >= H:
        raise ValueError(f"Head {head} out of bounds. Layer has {H} heads (0-{H-1})")

    # Get target token
    relation_map = {
        "A_below_B": "below",
        "A_above_B": "above",
        "A_left_of_B": "left",
        "A_right_of_B": "right"
    }

    if token_type == "rel":
        target_word = relation_map[annotation["relation_AB"]]
    elif token_type == "A":
        target_word = annotation["A"]["shape"]
    elif token_type == "B":
        target_word = annotation["B"]["shape"]
    else:
        raise ValueError(f"Invalid token_type '{token_type}'. Must be 'rel', 'A', or 'B'")

    # Find token and extract attention
    token_map, visual_range = find_token_and_visual_range(input_ids, [target_word], tokenizer)
    if target_word not in token_map:
        raise ValueError(f"Token '{target_word}' not found in input sequence")
    if len(visual_range) != SIDE * SIDE:
        raise ValueError(f"Expected {SIDE*SIDE} visual tokens, got {len(visual_range)}")

    token_idx = token_map[target_word][0]
    visual_range_list = list(visual_range)
    attn_vector = attentions[layer, 0, head, token_idx, visual_range_list].numpy()
    attn_grid = attn_vector.reshape(SIDE, SIDE)
    attn_grid_normalized = attn_grid / (attn_grid.sum() + 1e-9)

    # Create output path
    if output_path is None:
        if overlay:
            output_dir = Path("visualizations/overlays")
        else:
            output_dir = Path("visualizations/combined")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{scene_id}_L{layer}_H{head}_{token_type}.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create visualization
    print(f"Saving combined visualization to {output_path}...")

    if overlay:
        # Three subplots: image, overlay, heatmap
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # Left: Original image
        ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title(f'Original Image\n{prompt}', fontsize=10)

        # Middle: Overlay attention on image
        from scipy.ndimage import zoom
        # Upsample attention grid to image size (256x256)
        img_size = image.size[0]  # Should be 256
        zoom_factor = img_size / SIDE
        attn_upsampled = zoom(attn_grid_normalized, zoom_factor, order=1)

        ax2.imshow(image)
        im_overlay = ax2.imshow(attn_upsampled, cmap=cmap, alpha=alpha, interpolation='bilinear')
        ax2.axis('off')
        ax2.set_title(f'Attention Overlay\nAlpha={alpha}', fontsize=10)

        # Right: Attention heatmap
        im = ax3.imshow(attn_grid_normalized, cmap=cmap, interpolation='nearest')
        cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', rotation=270, labelpad=15)

        if show_grid:
            ax3.set_xticks(np.arange(SIDE) - 0.5, minor=True)
            ax3.set_yticks(np.arange(SIDE) - 0.5, minor=True)
            ax3.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)

        ax3.set_xticks(np.arange(SIDE))
        ax3.set_yticks(np.arange(SIDE))
        ax3.set_xlabel('Patch Column')
        ax3.set_ylabel('Patch Row')
        ax3.set_title(f'Attention Heatmap\nLayer {layer}, Head {head}, Token: "{target_word}"')
    else:
        # Two subplots: image, heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Left: Original image
        ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title(f'Original Image\n{prompt}', fontsize=10)

        # Right: Attention heatmap
        im = ax2.imshow(attn_grid_normalized, cmap=cmap, interpolation='nearest')
        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', rotation=270, labelpad=15)

        if show_grid:
            ax2.set_xticks(np.arange(SIDE) - 0.5, minor=True)
            ax2.set_yticks(np.arange(SIDE) - 0.5, minor=True)
            ax2.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)

        ax2.set_xticks(np.arange(SIDE))
        ax2.set_yticks(np.arange(SIDE))
        ax2.set_xlabel('Patch Column')
        ax2.set_ylabel('Patch Row')
        ax2.set_title(f'Attention Heatmap\nLayer {layer}, Head {head}, Token: "{target_word}"')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved combined visualization to {output_path}")
    print(f"  Scene: {scene_id}")
    print(f"  Token: '{target_word}' (position {token_idx})")
    print(f"  Attention sum: {attn_grid.sum():.4f}")

    return attn_grid_normalized, image


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate attention heatmap for a specific scene, layer, and head"
    )
    parser.add_argument("scene_id", type=str, help="Scene ID (e.g., scene_0042)")
    parser.add_argument("layer", type=int, help="Layer index (0-based)")
    parser.add_argument("head", type=int, help="Head index (0-based)")
    parser.add_argument(
        "--token",
        type=str,
        default="rel",
        choices=["rel", "A", "B"],
        help="Which token to visualize attention from (default: rel)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for PNG file (default: visualizations/{scene_id}_L{layer}_H{head}_{token}.png)"
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="hot",
        help="Matplotlib colormap (default: hot)"
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Disable grid lines"
    )
    parser.add_argument(
        "--with-image",
        action="store_true",
        help="Generate side-by-side visualization with original image"
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Add overlay subplot (attention superimposed on image) - requires --with-image"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Transparency for overlay (0=transparent, 1=opaque) (default: 0.6)"
    )

    args = parser.parse_args()

    if args.with_image:
        visualize_attention_with_image(
            scene_id=args.scene_id,
            layer=args.layer,
            head=args.head,
            token_type=args.token,
            output_path=args.output,
            cmap=args.cmap,
            show_grid=not args.no_grid,
            overlay=args.overlay,
            alpha=args.alpha
        )
    else:
        visualize_attention_map(
            scene_id=args.scene_id,
            layer=args.layer,
            head=args.head,
            token_type=args.token,
            output_path=args.output,
            cmap=args.cmap,
            show_grid=not args.no_grid
        )
