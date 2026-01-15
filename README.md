# Attention Alignment in Vision–Language Models

This project analyzes how attention in Vision–Language Models aligns with ground-truth object regions in synthetic scenes with spatial relations (e.g., “left of”, “above”). We extract cross-attention maps for 5k synthetic scenes and evaluate whether correct predictions are supported by meaningful visual grounding or by shortcuts such as language bias or center bias.

## Important Note!

The file with all quantitative metrics is too large to be stored on GH. To generate it locally, run:
```bash
# Creates stats.h5 (GPU required)
python run_evaluation.py
```

<!-- ## What we measure
- IoU with object masks  
- Center-of-mass distance  
- Entropy  
- Mutual information with scene variables  

## What we study
- Entity vs relation token grounding  
- Layer and head specialization  
- Attention differences between correct and incorrect predictions  
- Sensitivity to task difficulty (distractors, layouts, relation type)  

## Output
Tabular metrics per scene, token, layer, and head, with:
- error-conditioned analysis  
- head clustering  
- complexity curves  
- attention–performance correlations -->

## Goal
Quantify when and where VLM attention reflects true visual reasoning vs spurious behavior.

## Generating Figures

Generate attention visualizations on-demand (requires GPU):

```bash
# Heatmap only
python visualize_attention_map.py scene_0042 15 8

# Side-by-side: image + heatmap
python visualize_attention_map.py scene_0100 20 7 --with-image

# With overlay: image + overlay + heatmap
python visualize_attention_map.py scene_0200 25 11 --with-image --overlay

# Visualize object tokens instead of relation token
python visualize_attention_map.py scene_0075 18 5 --with-image --overlay --token A
python visualize_attention_map.py scene_0075 18 5 --with-image --overlay --token B

# Adjust transparency or colormap
python visualize_attention_map.py scene_0300 15 9 --with-image --overlay --alpha 0.4 --cmap viridis
```

**Outputs**: Saved to `visualizations/heatmaps/`, `visualizations/combined/`, or `visualizations/overlays/` depending on options.
