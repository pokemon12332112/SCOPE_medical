## Additional Architectural Details

This section expands on the model architecture specifications briefly mentioned in Section 4 (Implementation Details) of the main paper, providing complete architectural specifications for all components.

### Model Specifications

The visual encoder $$\psi_I$$ uses **DINOv2-ViT-B/14**, outputting:

- Patch tokens  $$N_I$$
- Input resolution: $$518 \times 518$$  
- Feature dimension: $$D_I = 768$$  

The text encoder $$\psi_T$$ is **CXRBert-general** with embedding dimension:

$$
D_T = 768.
$$

Both projection functions $$\phi_I$$ and $$\phi_T$$ are 2-layer MLPs with:

- Hidden dimension: 1536  
- Shared latent dimension:
  $$
  D = 768
  $$
- GELU activation  
- Layer normalization  

---

### SCPR Decoder

The SCPR decoder $$D_\omega$$ is a 4-layer Transformer decoder with:

- 8 attention heads  
- Feedforward dimension:
  $$
  d_{\text{ff}} = 3072
  $$
- Hidden dimension:
  $$
  d_{\text{model}} = 768
  $$
- Masking ratio: 40% during pre-training  

---

### Multimodal Fusion Module

The Multimodal Fusion module is a 2-layer Transformer encoder with:

- 8 attention heads  
- Hidden dimension 768  

---

### Report Generator

The report generator uses **DistilGPT2** with:

- 6 layers  
- 12 attention heads  
- Hidden dimension 768  
- Maximum generation length: 100 tokens  

---

## Training Details

This section provides complete training hyperparameters and schedules that extend the brief description in Section 4 of the main paper.

---

### Optimization and Schedules

We use AdamW with:

$$
\beta_1 = 0.9, \quad \beta_2 = 0.999
$$

- Weight decay: 0.01  
- Gradient clipping maximum norm: 1.0  

Training schedule:

- Pre-training: 20 epochs with 1000-step linear warmup  
- Fine-tuning: 50 epochs with 500-step warmup  

---

### PAPA Hyperparameters

For the optimal transport assignment in Section 3, we use:

- Sinkhorn iterations:
  $$
  L = 3
  $$
- Temperature:
  $$
  \tau = 0.1
  $$
- Entropy regularization:
  $$
  \epsilon = 0.05
  $$

Loss balancing coefficients:

$$
\lambda = 1.0, \quad \lambda_1 = 1.0, \quad \lambda_2 = 0.1.
$$

---

### Data Preprocessing

Images are resized to:

$$
518 \times 518
$$

For multi-view inputs, we use the multiview dataset from MLRG with maximum number of views:

$$
n = 2
$$

Text sequences use:

- Maximum length 256 tokens for reports  
- Maximum length 64 tokens for indications  

---

### Computational Setup

Experiments are conducted using:

- 1 NVIDIA RTX 4090 GPU (24GB)

Training time on MIMIC-CXR:

- Pre-training: ~12 hours  
- Fine-tuning: ~18 hours  