# Implementation Details

Implementation details for Experiment Settings of the main paper.

---

## Prototype Formulation

This details the prototype formulation procedure described in Section 3.3 (PAPA) of the main paper.

We define a set of $K$ pathology prototypes:

$$
P = \{p_1, p_2, \dots, p_K\} \in \mathbb{R}^{K \times D}.
$$

To construct these prototypes with semantically meaningful representations, we follow a systematic procedure based on clinical pathology labels from the CheXpert dataset.

---

## Pathology Concepts

 
Given $K$ pathology concept names $\{c_1, c_2, \dots, c_K\}$ from CheXpert labels, each concept is tokenized into sequences $\{t_1, \dots, t_{L_k}\}$, encoded by the text encoder $\psi_T$ to obtain contextualized embeddings $\{h_1, \dots, h_{L_k}\}$, projected into the shared latent space via $\phi_T$, and finally mean-pooled across the sequence dimension to produce fixed prototype vectors $\{p_1, p_2, \dots, p_K\}$ that serve as semantic anchors for pathology-aware alignment.

Following CheXpert, which defines a standard set of 14 clinical findings for chest X-ray analysis, we use $K = 14$ pathology concepts for prototype construction:

- *enlarged cardiomediastinum*  
- *cardiomegaly*  
- *lung opacity*  
- *lung lesion*  
- *edema*  
- *consolidation*  
- *pneumonia*  
- *atelectasis*  
- *pneumothorax*  
- *pleural effusion*  
- *pleural thickening*  
- *fracture*  
- *support devices*  
- *no finding*  

---

## Construction Procedure

Given the set of pathology concepts:

$$
\mathcal{C} = \{c_1, c_2, \dots, c_K\},
$$

where each $c_k$ corresponds to one of the predefined CheXpert pathology labels, we construct each prototype $p_k$ using the following procedure:

### 1. Tokenization

Each concept name $c_k$ is tokenized using the text encoder's tokenizer  
(WordPiece tokenizer trained on CheXpert data) to produce a sequence of token IDs:

$$
t_k = \{t_1, \dots, t_{L_k}\}.
$$

---

### 2. Text Encoding

The tokenized sequence is passed through the text encoder $\psi_T(\cdot)$ to obtain contextualized token embeddings:

$$
h_k \in \mathbb{R}^{L_k \times D_T},
$$

where $L_k$ is the sequence length for concept $c_k$.

---

### 3. Projection

The encoded representations are projected into the shared latent space using the text projection function $\phi_T(\cdot)$:

$$
\phi_T(h_k) \in \mathbb{R}^{L_k \times D}.
$$

---

### 4. Mean Pooling

To obtain a single prototype vector per concept, we compute the mean across the sequence dimension:

$$
p_k = \frac{1}{L_k} \sum_{i=1}^{L_k} \phi_T(h_k)_i \in \mathbb{R}^{D}.
$$

---

## Final Prototype Set

After processing all pathology concepts, we obtain:

$$
P = \{p_1, p_2, \dots, p_K\}.
$$

Each prototype $p_k$ encodes the semantic meaning of its corresponding pathology concept in the shared vision-language embedding space.

These prototypes remain **fixed during training** and serve as **semantic anchors** for aligning visual and textual features.