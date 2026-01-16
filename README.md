# Attention Alignment in Vision–Language Models

We examine whether cross-attention weights in vision-language models (VLMs) provide interpretable evidence of relational reasoning. Using a synthetic task and spatial alignment metrics, we analyze attention across layers, heads, and scene complexity. We find that attention often underperforms a uniform baseline, indicating sensitivity to spatial and dataset biases, and that relational alignment emerges transiently in middle layers with limited head specialization. Overall, cross-attention alone provides an incomplete basis for interpreting relational reasoning in VLMs.

## Important Note

All analyses in the Results section are performed in either `attention_analysis.ipynb` or `pca-heads.ipynb`. Running these notebooks requires a file containing all quantitative metrics. Because this file is too large to be stored on GitHub, it must be generated locally using:
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
- attention–performance correlations

## Goal
Quantify when and where VLM attention reflects true visual reasoning vs spurious behavior. -->

## Generating Figures

Generate attention visualizations from the report on-demand (requires GPU):

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
