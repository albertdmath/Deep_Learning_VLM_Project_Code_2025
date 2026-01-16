import numpy as np

def find_token_and_visual_range(input_ids, words, tokenizer, verbose=False):
    """
    Args:
        input_ids : np.ndarray (seq_len,)
        words     : list[str]
        tokenizer : HF tokenizer
    Returns:
        token_map : dict[word] -> list[int]   (token positions)
        visual_range : range
    """

    # Decode tokens for debug
    if verbose:
        decoded = tokenizer.convert_ids_to_tokens(input_ids.tolist())
        print("Decoded tokens:")
        for i, tk in enumerate(decoded):
            print(f"{i:3d} | {repr(tk)}")

    # Resolve token indices for words
    token_map = {}

    for word in words:

        word_ids = tokenizer.encode(" " + word, add_special_tokens=False)

        if len(word_ids) != 1:
            raise ValueError(
                f"Word '{word}' maps to multiple tokens: {word_ids}. "
                "This function only supports single-token words."
            )

        target_id = word_ids[0]

        matches = np.where(input_ids == target_id)[0]

        if len(matches) == 0:
            raise ValueError(f"Token '{word}' not found in input_ids")

        token_map[word] = matches.tolist()

    # Find visual token range
    img_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    img_end   = tokenizer.convert_tokens_to_ids("<|vision_end|>")

    start_idx = np.where(input_ids == img_start)[0]
    end_idx   = np.where(input_ids == img_end)[0]

    if len(start_idx) == 0 or len(end_idx) == 0:
        raise ValueError("Vision start/end tokens not found")

    start = start_idx[0] # assumes single vision block
    end   = end_idx[0]

    if start >= end:
        raise ValueError("Invalid vision token ordering")

    visual_range = range(start + 1, end)

    return token_map, visual_range

def get_patch_centers(side=9, w=256, h=256):
    """Returns (9, 9, 2) array of (x,y) centers for visual patches."""
    xs = (np.arange(side) + 0.5) * (w / side)
    ys = (np.arange(side) + 0.5) * (h / side)
    return np.stack(np.meshgrid(xs, ys), axis=-1)

def create_gt_relation_mask(record, centers):
    rel = record["relation_AB"]
    x1, y1, x2, y2 = record["B"]["bbox"]

    X = centers[..., 0]
    Y = centers[..., 1]

    # Remember:
    # y increases downward
    # x increases rightward

    if rel == "A_below_B":
        return Y > y2         # below B's bottom edge
    elif rel == "A_above_B":
        return Y < y1         # above B's top edge
    elif rel == "A_left_of_B":
        return X < x1         # left of B's left edge
    elif rel == "A_right_of_B":
        return X > x2         # right of B's right edge
    else:
        return np.zeros_like(X, dtype=bool)

def create_gt_bbox_mask(record, centers, object="A"):
    """
    record: annotation dict
    patch_centers: (9, 9, 2) array of patch centers in image coordinates (x, y)
    object: either "A" or "B"
    returns: (9, 9) boolean mask where True means patch center is inside object's bbox
    """

    # Extract bbox of A
    x1, y1, x2, y2 = record[object]["bbox"]

    # Split centers
    cx = centers[..., 0]  # (9, 9)
    cy = centers[..., 1]  # (9, 9)

    # Boolean inclusion test
    mask = (
        (cx >= x1) & (cx <= x2) &
        (cy >= y1) & (cy <= y2)
    )

    return mask

def create_gt_union_mask(record, centers):
    A_mask = create_gt_bbox_mask(record, centers, object="A")
    B_mask = create_gt_bbox_mask(record, centers, "B")
    return A_mask | B_mask

def calculate_iou(attn_grids:np.ndarray, gt_mask, top_p=0.5):
    """
    Vectorized IoU for all layers & heads.
    attn_grids: (L, H, 9, 9)
    gt_mask:    (9, 9)
    returns:    (L, H)
    """

    L, H, Ht, Wt = attn_grids.shape
    N = Ht * Wt

    # Flatten spatial grid
    flat = attn_grids.reshape(L, H, N)

    assert np.all(flat >= 0), "Negative attention values"

    # Sort descending
    idx = np.argsort(flat, axis=-1)[..., ::-1]
    sorted_vals = np.take_along_axis(flat, idx, axis=-1)

    # Cumulative mass
    cumsum = np.cumsum(sorted_vals, axis=-1)
    cumsum = cumsum / (cumsum[..., -1:] + 1e-9)

    # Boolean mask for top-p mass
    attn_bool = cumsum <= top_p

    # Ensure at least one True per (layer, head)
    empty = ~attn_bool.any(axis=-1)
    if np.any(empty):
        attn_bool[empty, 0] = True

    # Undo sorting → reconstruct boolean grid in original positions
    bool_flat = np.zeros_like(flat, dtype=bool)
    np.put_along_axis(bool_flat, idx, attn_bool, axis=-1)

    # Reshape back to grids
    attn_mask = bool_flat.reshape(L, H, Ht, Wt)

    # Expand GT mask for broadcasting
    gt = gt_mask[None, None, :, :]

    # IoU
    inter = (attn_mask & gt).sum(axis=(2,3))
    union = (attn_mask | gt).sum(axis=(2,3))

    return np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0)

def calculate_entropy(attn_grids: np.ndarray):
    """
    Vectorized entropy calculator.
    attn_grids: (L, H, 9, 9)
    returns: (L, H)
    """
    L, H, Ht, Wt = attn_grids.shape
    N = Ht * Wt

    # Flatten spatial grid
    flat = attn_grids.reshape(L, H, N)

    # Assertions for validity
    assert np.all(flat >= 0), "Negative values found in attention grids"

    # Entropy (in nats)
    total = np.sum(flat, axis=-1, keepdims=True)
    p = flat / (total + 1e-12)
    entropy = -np.sum(p * np.log(p + 1e-12), axis=-1)

    return entropy

def calculate_com_distance(attn_grids, mask, eps=1e-8):
    """
    attn_grids: (L, H, 9, 9) attention maps
    mask: (9, 9) ground truth mask (binary or soft)
    normalize: if True, normalize distance by max diagonal
    eps: numerical stability for division
    """

    L, H, N, _ = attn_grids.shape  # N = 9

    # Coordinate grid: (9, 9)
    y, x = np.mgrid[0:N, 0:N]  # y=row, x=col

    # Expand for broadcasting
    x = x[None, None, :, :]  # (1, 1, 9, 9)
    y = y[None, None, :, :]  # (1, 1, 9, 9)

    # Attention center of mass (L, H) 
    attn_sum = attn_grids.sum(axis=(-2, -1))+ eps

    ax = (attn_grids * x).sum(axis=(-2, -1), keepdims=False) / attn_sum
    ay = (attn_grids * y).sum(axis=(-2, -1), keepdims=False) / attn_sum

    # Mask center of mass (scalar)
    msum = mask.sum() + eps
    mx = (mask * x[0, 0]).sum() / msum
    my = (mask * y[0, 0]).sum() / msum

    # Distance (L, H) 
    dist = np.sqrt((ax - mx) ** 2 + (ay - my) ** 2)
    return dist

def baseline_uniform(L, H, SIDE=9):
    """Uniform attention over patches."""
    grid = np.ones((L, H, SIDE, SIDE), dtype=np.float32)
    grid /= grid.sum(axis=(2,3), keepdims=True)
    return grid

def baseline_gaussian(L, H, SIDE=9, sigma_range=(0.8, 2.5)):
    y, x = np.mgrid[0:SIDE, 0:SIDE]

    # Broadcast grids: (1,1,S,S)
    x = x[None, None]
    y = y[None, None]

    # Random parameters per (layer, head)
    cx = np.random.uniform(0, SIDE, size=(L, H, 1, 1))
    cy = np.random.uniform(0, SIDE, size=(L, H, 1, 1))
    sigma = np.random.uniform(*sigma_range, size=(L, H, 1, 1))

    # Gaussian kernel
    g = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))

    # Normalize per (layer, head)
    g /= g.sum(axis=(2, 3), keepdims=True)

    return g.astype(np.float32)

