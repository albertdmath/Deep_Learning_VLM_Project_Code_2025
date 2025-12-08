# Attention Alignment in Vision–Language Models

This project analyzes how attention in Vision–Language Models aligns with ground-truth object regions in synthetic scenes with spatial relations (e.g., “left of”, “between”).

We extract cross-attention maps for ~5k scenes and evaluate whether correct predictions are supported by meaningful visual grounding or by shortcuts such as language bias or center bias.

## What we measure
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
Quantify when and where VLM attention reflects true visual reasoning vs spurious behavior.
