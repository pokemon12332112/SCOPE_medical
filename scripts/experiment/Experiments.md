# Experimental Results

This section provides comprehensive experimental results that extend the main paper. We present detailed masking experiments and per-pathology clinical efficacy analyses.

---

## Metrics

We evaluate generated radiology reports using a combination of linguistic and clinical metrics. Linguistic metrics assess surface-level text quality and fluency, while clinical metrics evaluate diagnostic accuracy and correctness of medical content. Given a generated report p and a ground-truth report g, the metrics are defined as follows.

### Linguistic Metrics

**BLEU-n (B-n)** measures n-gram precision with a brevity penalty:

$$
\text{BLEU-n} = \text{BP} \cdot \exp\Bigg( \sum_{k=1}^{n} w_k \log p_k \Bigg)
$$

$$
p_k = \frac{\#\ \text{matched } k\text{-grams}}{\#\ k\text{-grams in } p}
$$

$$
\text{BP} =
\begin{cases}
1, & \text{if } |p| > |g| \\
\exp\Big(1 - |g|/|p|\Big), & \text{otherwise}
\end{cases}
$$

$$
w_k = \frac{1}{n}
$$

**ROUGE-L (R-L)** evaluates the longest common subsequence (LCS) between p and g:

$$
P_L = \frac{LCS(p,g)}{|p|}, \quad
R_L = \frac{LCS(p,g)}{|g|}
$$

$$
F1_L = \frac{(1+\beta^2) P_L R_L}{R_L + \beta^2 P_L}, \quad
\beta = \frac{|g|}{|p|}
$$

**METEOR (MTR)** evaluates a unigram alignment-based score with a fragmentation penalty:

$$
\text{MTR} = F_\text{mean}\cdot(1-\text{Penalty})
$$

$$
F_\text{mean} = \frac{10PR}{R + 9P}
$$

$$
\text{Penalty} = 0.5\left(\frac{\text{chunks}}{\text{matches}}\right)^3
$$

where P and R are unigram precision and recall, **matches** is the number of aligned unigrams, and **chunks** is the number of contiguous matched sequences.

---

### Clinical Metrics

For clinical evaluation, we employ CheXbert to label reports with 14 predefined clinical findings. For each class i, per-class precision, recall, and F1-score are:

$$
P_i = \frac{TP_i}{TP_i + FP_i}, \quad
R_i = \frac{TP_i}{TP_i + FN_i}, \quad
F1_i = \frac{2P_iR_i}{P_i + R_i}
$$

Micro-averaged and macro-averaged scores are defined as:

$$
P_\text{micro} = \frac{\sum_i TP_i}{\sum_i TP_i + \sum_i FP_i}, \quad
P_\text{macro} = \frac{1}{C}\sum_i P_i
$$

$$
R_\text{micro} = \frac{\sum_i TP_i}{\sum_i TP_i + \sum_i FN_i}, \quad
R_\text{macro} = \frac{1}{C}\sum_i R_i
$$

$$
F1_\text{micro} = \frac{2P_\text{micro}R_\text{micro}}{P_\text{micro}+R_\text{micro}}, \quad
F1_\text{macro} = \frac{1}{C}\sum_i F1_i
$$

where $$C$$ is the number of clinical classes.

Finally, **RadGraph F1 (RG)** evaluates correctness of extracted entities and relations:

$$
RG = \frac{2\cdot TP_\text{ent/rel}}{2\cdot TP_\text{ent/rel} + FP_\text{ent/rel} + FN_\text{ent/rel}}
$$

---
## Main Results: Descriptive Accuracy (NLG Metrics)

**Table 1:** Evaluation of descriptive accuracy for our proposed framework **SCOPE** versus existing state-of-the-art methods on **MIMIC-CXR (M-CXR)**, **MIMIC-ABN (M-ABN)**, and **IU X-ray** datasets. Higher is better (↑). **Best** is in **bold**. Runner-up is denoted by *(2nd)*.

> Note: The original LaTeX table uses underline for runner-up; here we mark runner-up values with *(2nd)*.

---

### M-CXR

| Method | Venue | B-1 ↑ | B-2 ↑ | B-3 ↑ | B-4 ↑ | MTR ↑ | R-L ↑ | RG ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SA | EMNLP'23 | - | 0.184 | - | - | - | - | 0.228 |
| MET | CVPR'23 | 0.386 | 0.250 | 0.169 | 0.124 | 0.152 | 0.291 | - |
| KIUT | CVPR'23 | 0.393 | 0.243 | 0.159 | 0.113 | 0.160 | 0.285 | - |
| CoFE | ECCV'24 | - | - | - | 0.125 | 0.176 | 0.304 | - |
| MAN | AAAI'24 | 0.396 | 0.244 | 0.162 | 0.115 | 0.151 | 0.274 | - |
| B-LLM | AAAI'24 | 0.402 | 0.262 | 0.180 | 0.128 | 0.175 | 0.291 | - |
| DCG | ACMMM'24 | 0.397 | 0.258 | 0.166 | 0.126 | 0.162 | 0.295 | - |
| Med-LLM | ACMMM'24 | - | - | - | 0.128 | 0.161 | 0.289 | - |
| SEI | MICCAI'24 | 0.382 | 0.247 | 0.177 | 0.135 | 0.158 | 0.299 | 0.249 |
| FMVP | TMIM'23 | 0.389 | 0.236 | 0.156 | 0.108 | 0.150 | 0.284 | - |
| HERGen | ECCV'24 | 0.395 | 0.248 | 0.169 | 0.122 | 0.156 | 0.285 | - |
| CXRMate | arXiv'23 | 0.361 | 0.223 | 0.150 | 0.108 | 0.159 | 0.263 | 0.238 |
| MLRG | CVPR'25 | 0.411 | 0.277 | 0.204 *(2nd)* | 0.158 *(2nd)* | 0.176 *(2nd)* | 0.320 *(2nd)* | 0.291 *(2nd)* |
| DART | CVPR'25 | 0.437 *(2nd)* | 0.279 *(2nd)* | 0.191 | 0.137 | 0.175 | 0.310 | - |
| **SCOPE (Ours)** | - | **0.452** | **0.334** | **0.269** | **0.227** | **0.204** | **0.373** | **0.347** |
| Δ (abs.) ↑ | - | +0.015 | +0.055 | +0.065 | +0.069 | +0.028 | +0.053 | +0.056 |

---

### M-ABN

| Method | Venue | B-1 ↑ | B-2 ↑ | B-3 ↑ | B-4 ↑ | MTR ↑ | R-L ↑ | RG ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R2Gen | EMNLP'20 | 0.253 | 0.144 | 0.092 | 0.063 | 0.106 | 0.229 | 0.179 |
| CMN | ACL'21 | 0.256 | 0.147 | 0.095 | 0.066 | 0.110 | 0.230 | 0.183 |
| SEI | MICCAI'24 | 0.267 | 0.157 | 0.104 | 0.073 | 0.114 | 0.231 | 0.191 |
| MLRG | CVPR'25 | 0.332 *(2nd)* | 0.199 *(2nd)* | 0.132 *(2nd)* | 0.094 *(2nd)* | 0.136 *(2nd)* | 0.248 *(2nd)* | 0.219 *(2nd)* |
| **SCOPE (Ours)** | - | **0.343** | **0.215** | **0.151** | **0.113** | **0.142** | **0.266** | **0.237** |
| Δ (abs.) ↑ | - | +0.011 | +0.016 | +0.019 | +0.019 | +0.006 | +0.018 | +0.018 |

---

### IU-XRay

| Method | Venue | B-1 ↑ | B-2 ↑ | B-3 ↑ | B-4 ↑ | MTR ↑ | R-L ↑ | RG ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R2Gen | EMNLP'20 | 0.470 | 0.304 | 0.219 | 0.165 | 0.187 | 0.371 | - |
| CMN | ACL'21 | 0.475 | 0.309 | 0.222 | 0.170 | 0.191 | 0.375 | - |
| PPKED | CVPR'21 | 0.483 | 0.315 | 0.224 | 0.168 | 0.190 | 0.376 | - |
| CMCL | ACL'21 | 0.473 | 0.305 | 0.217 | 0.162 | 0.186 | 0.378 | - |
| MSAT | MICCAI'22 | 0.481 | 0.316 | 0.226 | 0.171 | 0.190 | 0.372 | - |
| MET | CVPR'23 | 0.483 | 0.322 | 0.228 | 0.172 | 0.192 | 0.380 | - |
| Med-LLM | ACMMM'24 | - | - | - | 0.168 | 0.209 | 0.381 | - |
| MA | AAAI'24 | **0.501** | 0.328 | 0.230 | 0.170 | 0.213 *(2nd)* | 0.386 | - |
| B-LLM | AAAI'24 | 0.499 *(2nd)* | 0.323 | 0.238 | 0.184 | 0.208 | 0.390 | - |
| DART | CVPR'25 | 0.486 | 0.348 *(2nd)* | 0.265 *(2nd)* | 0.208 *(2nd)* | 0.205 | 0.411 *(2nd)* | - |
| **SCOPE (Ours)** | - | 0.483 | **0.351** | **0.288** | **0.249** | **0.223** | **0.418** | **0.419** |
| Δ (abs.) ↑ | - | -0.018 | +0.003 | +0.023 | +0.041 | +0.010 | +0.007 | - |

---

## Visual Masking Results

### Table 2: Masking Comparison (LLaVARad vs. MLRG vs. SCOPE)

**Table 2.** Comparison of **LLaVARad**, **MLRG**, and **SCOPE** under different masking transformations and ratios. Evaluation metrics include BLEU-4 (B-4), ROUGE-L (R-L), and F1-RadGraph (RG). Values in parentheses indicate percentage drop from baseline (ratio = 0.0). For **Non-Anatomical Masking**, smaller drops indicate robustness to irrelevant occlusions. For **Anatomical Masking**, larger drops demonstrate genuine visual grounding. The last row reports the number of parameters and inference time for each model.



#### Non-Anatomical Masking

| Ratio | LLaVARad B-4 | LLaVARad R-L | LLaVARad RG | MLRG B-4 | MLRG R-L | MLRG RG | SCOPE B-4 | SCOPE R-L | SCOPE RG |
|------:|--------------:|-------------:|------------:|---------:|---------:|--------:|----------:|----------:|---------:|
| 0.4 | 0.152 (-10.1) | 0.301 (-8.8) | 0.311 (-11.1) | 0.217 (-9.6) | 0.380 (-5.2) | 0.361 (-5.2) | 0.237 (-2.9) | 0.402 (-2.4) | 0.402 (-2.9) |
| 0.3 | 0.156 (-7.7)  | 0.312 (-5.5) | 0.335 (-4.3)  | 0.218 (-9.2) | 0.378 (-5.7) | 0.363 (-4.7) | 0.239 (-2.0) | 0.407 (-1.2) | 0.404 (-2.4) |
| 0.2 | 0.160 (-5.3)  | 0.320 (-3.0) | 0.342 (-2.3)  | 0.213 (-11.2) | 0.367 (-8.5) | 0.354 (-7.1) | 0.243 (-0.4) | 0.407 (-1.2) | 0.409 (-1.2) |
| 0.1 | 0.165 (-2.4)  | 0.327 (-0.9) | 0.347 (-0.9)  | 0.231 (-3.7) | 0.389 (-3.0) | 0.374 (-1.8) | 0.243 (-0.4) | 0.409 (-0.7) | 0.411 (-0.7) |
| 0.0 | 0.169         | 0.330        | 0.350         | 0.240    | 0.401    | 0.381    | 0.244     | 0.412     | 0.414    |

#### Anatomical Masking

| Ratio | LLaVARad B-4 | LLaVARad R-L | LLaVARad RG | MLRG B-4 | MLRG R-L | MLRG RG | SCOPE B-4 | SCOPE R-L | SCOPE RG |
|------:|--------------:|-------------:|------------:|---------:|---------:|--------:|----------:|----------:|---------:|
| 0.4 | 0.150 (-11.2) | 0.298 (-9.7) | 0.329 (-6.0) | 0.171 (-28.7) | 0.349 (-13.0) | 0.351 (-7.9) | 0.192 (-21.3) | 0.363 (-11.9) | 0.343 (-17.1) |
| 0.3 | 0.153 (-9.5)  | 0.304 (-7.9) | 0.325 (-7.1) | 0.182 (-24.2) | 0.348 (-13.2) | 0.346 (-9.2) | 0.213 (-12.7) | 0.376 (-8.7) | 0.357 (-13.8) |
| 0.2 | 0.157 (-7.1)  | 0.310 (-6.1) | 0.335 (-4.3) | 0.193 (-19.6) | 0.360 (-10.2) | 0.349 (-8.4) | 0.215 (-11.9) | 0.381 (-7.5) | 0.363 (-12.3) |
| 0.1 | 0.163 (-3.6)  | 0.320 (-3.0) | 0.341 (-2.6) | 0.207 (-13.8) | 0.371 (-7.5) | 0.350 (-8.1) | 0.227 (-7.0) | 0.384 (-6.8) | 0.371 (-10.4) |
| 0.0 | 0.169         | 0.330        | 0.350         | 0.240    | 0.401    | 0.381    | 0.244     | 0.412     | 0.414    |

#### Model Size and Runtime

| Model | Params | Inference Time |
|------|--------:|---------------:|
| LLaVARad | ~7B | 1.71s |
| MLRG | 296M | 0.181s |
| SCOPE (Ours) | 306M | 0.191s |

---

## Visual Masking Experiments

This subsection extends the visual masking experiments presented in Section 4 of the main paper and provides comprehensive quantitative results across all metrics.

Table 2 extends Table 3 from the main paper by including all metrics (BLEU-4, ROUGE-L, and F1-RadGraph). We design controlled masking experiments to assess whether SCOPE depends on visual pathology cues, unlike baseline methods (MLRG, LLaVARad) that may rely on language priors. If SCOPE is visually grounded, performance should noticeably decline when key anatomical regions are occluded, whereas text-driven baselines should remain relatively unaffected.

**Non-Anatomical Masking Robustness.** When non-diagnostic regions are masked, SCOPE achieves the smallest performance drops across all metrics and ratios.

**Anatomical Masking Sensitivity.** When diagnostically critical regions are occluded, SCOPE shows substantial performance drops consistent with genuine visual grounding.

**Optimal Visual Grounding.** SCOPE is robust to irrelevant occlusions yet sensitive to pathological region masking. SCOPE achieves this behavior with competitive efficiency (306M parameters, 0.191s inference) comparable to MLRG (296M, 0.181s), whereas LLaVARad is substantially larger (~7B) and slower (1.71s) while showing weaker visual grounding behavior.

---

## Per-Pathology Clinical Efficacy Results

Table 3 provides detailed per-pathology clinical efficacy results, showing precision, recall, and F1-score breakdowns for each of the 14 clinical findings on MIMIC-CXR.

### Table 3: Per-Pathology Clinical Accuracy on MIMIC-CXR

**Table 3.** Comparison of SEI, MLRG, and our method in terms of clinical accuracy on MIMIC-CXR, where P, R, and F1 denote Precision, Recall, and F1-score, respectively. Win/Loss columns show percentage improvements (+) or degradations (-) of our method vs. MLRG, computed as:

$$
\Delta\% = \frac{v_{\text{ours}} - v_{\text{MLRG}}}{v_{\text{MLRG}}}\times 100\%
$$

| Finding | Freq. (%) | SEI P | SEI R | SEI F1 | MLRG P | MLRG R | MLRG F1 | Ours P | Ours R | Ours F1 | ΔP (%) | ΔR (%) | ΔF1 (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cardiomegaly | 14.8 | 0.599 | 0.633 | 0.616 | 0.629 | 0.570 | 0.598 | 0.684 | 0.651 | 0.668 | +8.7 | +14.2 | +11.7 |
| Lung Opacity | 13.8 | 0.519 | 0.170 | 0.256 | 0.594 | 0.317 | 0.413 | 0.620 | 0.345 | 0.443 | +4.4 | +8.8 | +7.3 |
| Support Devices | 12.8 | 0.763 | 0.708 | 0.734 | 0.768 | 0.788 | 0.778 | 0.816 | 0.809 | 0.813 | +6.3 | +2.7 | +4.5 |
| Pleural Effusion | 12.4 | 0.683 | 0.697 | 0.690 | 0.716 | 0.641 | 0.676 | 0.763 | 0.636 | 0.694 | +6.6 | -0.8 | +2.7 |
| Atelectasis | 10.9 | 0.469 | 0.395 | 0.429 | 0.499 | 0.475 | 0.487 | 0.522 | 0.457 | 0.487 | +4.6 | -3.8 | +0.0 |
| Enlarged Cardiomediastinum | 10.0 | 0.373 | 0.208 | 0.267 | 0.370 | 0.353 | 0.361 | 0.459 | 0.465 | 0.462 | +24.1 | +31.7 | +28.0 |
| Edema | 8.3 | 0.526 | 0.361 | 0.428 | 0.516 | 0.448 | 0.480 | 0.605 | 0.479 | 0.535 | +17.2 | +6.9 | +11.5 |
| Pneumonia | 4.4 | 0.174 | 0.065 | 0.095 | 0.316 | 0.235 | 0.270 | 0.364 | 0.246 | 0.293 | +15.2 | +4.7 | +8.5 |
| Consolidation | 3.3 | 0.218 | 0.194 | 0.205 | 0.259 | 0.150 | 0.190 | 0.317 | 0.159 | 0.212 | +22.4 | +6.0 | +11.6 |
| Lung Lesion | 2.5 | 0.462 | 0.021 | 0.041 | 0.429 | 0.046 | 0.082 | 0.638 | 0.114 | 0.194 | +48.7 | +147.8 | +136.6 |
| No Finding | 2.4 | 0.161 | 0.597 | 0.253 | 0.233 | 0.629 | 0.340 | 0.249 | 0.685 | 0.365 | +6.9 | +8.9 | +7.4 |
| Fracture | 1.8 | 0.000 | 0.000 | 0.000 | 0.174 | 0.021 | 0.037 | 0.361 | 0.067 | 0.113 | +107.5 | +219.0 | +205.4 |
| Pleural Other | 1.6 | 0.167 | 0.022 | 0.039 | 0.231 | 0.054 | 0.087 | 0.338 | 0.119 | 0.176 | +46.3 | +120.4 | +102.3 |
| Pneumothorax | 1.0 | 0.174 | 0.039 | 0.064 | 0.426 | 0.230 | 0.299 | 0.533 | 0.160 | 0.246 | +25.1 | -30.4 | -17.7 |
| **micro avg** | - | 0.523 | 0.410 | 0.460 | 0.549 | 0.468 | 0.505 | 0.597 | 0.500 | 0.545 | +8.7 | +6.8 | +7.9 |
| **macro avg** | - | 0.378 | 0.294 | 0.294 | 0.440 | 0.354 | 0.364 | 0.519 | 0.385 | 0.407 | +18.0 | +8.8 | +11.8 |

---

### Per-Pathology Analysis

**Visual Grounding Benefits Across Pathology Frequencies.**  
Table 3 shows per-pathology clinical efficacy on MIMIC-CXR. SCOPE improves across most findings compared to baselines. The macro-average F1 improvement (+11.8%) exceeds the micro-average gain (+7.9%), indicating stronger performance on less frequent findings.

SCOPE shows particularly strong gains on rare conditions. For example:

- Fracture (1.8%): F1 improves from 0.037 (MLRG) to 0.113 (+205.4%)
- Pleural Other (1.6%): 0.087 → 0.176 (+102.3%)
- Lung Lesion (2.5%): 0.082 → 0.194 (+136.6%)

On more frequent findings such as Cardiomegaly (14.8%) and Support Devices (12.8%), improvements are more modest (+11.7% and +4.5%, respectively).

This trend supports the visual grounding hypothesis: for rare pathologies with limited textual co-occurrence patterns, language-prior-driven baselines cannot rely on memorized associations, and SCOPE’s grounding mechanisms (PAPA + SCPR) provide greater benefits.

---