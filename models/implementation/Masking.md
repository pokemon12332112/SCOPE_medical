
# Mask Creation Methodology


## Training: Random Masking

During pre-training with SCPR (Section 3), we use random patch masking to encourage spatially consistent representations.

We divide the input image into:

$$
N_p = 1370
$$

patches. These patches do not form a perfect square grid.

Given an image:

$$
I \in \mathbb{R}^{C \times H \times W},
$$

we estimate the spatial patch layout as:

$$
N_{\text{cols}} = \left\lfloor \sqrt{N_p} \right\rfloor
$$

$$
N_{\text{rows}} = \left\lceil \frac{N_p}{N_{\text{cols}}} \right\rceil
$$

For an image size of:

$$
518 \times 518,
$$

this results in a slightly uneven grid.

Each patch therefore has spatial dimensions:

$$
p_h = \left\lfloor \frac{H}{N_{\text{rows}}} \right\rfloor
$$

$$
p_w = \left\lfloor \frac{W}{N_{\text{cols}}} \right\rfloor
$$

---

### Mask Sampling

For each image in a batch of size $B$, we randomly select:

$$
\lfloor r \cdot N_p \rfloor
$$

patches to mask, where:

$$
r = 0.4
$$

The masked patch indices are defined as:

$$
\mathcal{M}_b = \{\pi_b(i) : i = 1, \ldots, \lfloor r \cdot N_p \rfloor\},
$$

where:

- $\pi_b$ is a random permutation for the $b$-th sample  
- $\mathcal{M}_b$ is the set of masked patch indices  

---

### Spatial Masking Operation

For each selected patch at grid position $(i, j)$, we compute spatial coordinates:

$$
h_0 = i \cdot p_h, \quad
h_1 = \min((i+1) \cdot p_h, H)
$$

$$
w_0 = j \cdot p_w, \quad
w_1 = \min((j+1) \cdot p_w, W)
$$

The corresponding region is zeroed out:

$$
I[:, h_0:h_1, w_0:w_1] = 0
$$

This produces a binary patch mask:

$$
\mathbf{M} \in \{0,1\}^{N_p}
$$

where:

- $\mathbf{M}_{ij} = 1$ indicates a masked patch  

Random masking prevents the model from exploiting spatial biases and ensures robust feature learning across all image regions.

---

## Inference: Anatomical-Specific Masking

For evaluation experiments (Section 4), we employ targeted masking to assess whether models genuinely rely on visual information.

We utilize the **EYEGAZE dataset**, which provides radiologist gaze-based annotations indicating anatomical regions for 1,083 chest X-ray images.

We evaluate under two conditions:

---

### 1. Anatomical Masking

We use the anatomical heatmaps provided by the EYEGAZE dataset.

Procedure:

1. Aggregate and resize all heatmaps for each image.
2. Form a combined heatmap highlighting anatomical regions.
3. Apply an adaptive threshold to select top-activated pixels.
4. Ensure the masked area matches a target ratio:

$$
r \in \{0.1, 0.2, 0.3, 0.4\}
$$

This ensures masking focuses specifically on clinically relevant anatomical regions.

---

### 2. Non-Anatomical Masking

To isolate language prior effects, we mask only visually uninformative regions.

Procedure:

- Random black rectangles are inserted.
- Overlap with anatomical regions is strictly prevented.
- Rectangles of varying sizes are repeatedly sampled.
- Sampling continues until the target masked area is reached (or a sampling limit is met).

This guarantees that only non-anatomical regions are masked.

---

## Purpose of Dual Masking Strategy

This dual masking strategy allows us to:

- Train robust spatial representations (via random masking)
- Systematically evaluate visual grounding capability
- Detect pathological invariance to occlusion
- Distinguish genuine visual reasoning from language-prior exploitation

A model that truly relies on anatomical evidence should show performance degradation under anatomical masking, while remaining relatively stable under non-anatomical masking.