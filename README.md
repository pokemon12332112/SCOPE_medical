<div align="center">

# SCOPE


<div align="center">

</div>

</div>

---


## Requirements

```bash
# create virtual environment
conda create -n scope python=3.11

# install dependencies
pip install -r requirements.txt
````



## 📂 Datasets

### Medical Images

* **MIMIC-CXR / MIMIC-ABN** — PhysioNet, with data systematically organized under root directories labeled `p10` through `p19`, maintaining consistency with MIMIC-CXR's default configuration.
* **IU X-ray** — NIH, its root directory is the `NLMCXR_png`.
* **Two-View CXR** — aggregated studies with two views from MIMIC-CXR + IU X-ray

```
files/
├── p10
    └── p10000032
            └── s50414267
               ├── 02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg
               └── 174413ec-4ec4c1f7-34ea26b7-c5f994f8-79ef1962.jpg
├── p11
├── p12
├── p13
├── p14
├── p15
├── p16
├── p17
├── p18
├── p19
└── NLMCXR_png
   ├── CXR1_1_IM-0001-3001.png
   ├── CXR1_1_IM-0001-4001.png
   └── CXR2_IM-0652-1001.png
```

  
### Reorganization of Raw Radiology Reports
- MIMIC-CXR: five_work_mimic_cxr_annotation_v1.1.json
- MIMIC-ABN: mlrg_mimic_abn_annotation_v1.1.json
- Two-view CXR: mlrg_multiview_cxr_annotation_v1.1.json
- IU-Xray: iu_xray.json
- View Position for all datasets: five_work_mimic_cxr_view_position_v1.1.json


## Training

**1. Download checkpoints for architecture and metrics.**
- For CE metrics calculation: `chexbert.pth`, `radgraph`, and `bert-base-uncased`.
- For model initialization: `microsoft/rad-dino` (image encoder), `microsoft/BiomedVLP-CXR-BERT-specialized` (text encoder), `distilbert/distilgpt2` (define text generator), and `cvt2distilgpt2` (initialize text generator).
- Checkpoint directory: Place all checkpoints in a local directory (e.g., "checkpoints"), and configure the `--ckpt_zoo_dir` argument in the corresponding `script/**/**.sh` file.

| **Checkpoint**                             | **Variable name**     |
| ------------------------------------------ | --------------------- |
| `chexbert.pth`                             | `chexbert_path`       |
| `bert-base-uncased`                        | `bert_path`           |
| `radgraph`                                 | `radgraph_path`       |
| `microsoft/rad-dino`                       | `rad_dino_path`       |
| `microsoft/BiomedVLP-CXR-BERT-specialized` | `cxr_bert_path`       |
| `distilbert/distilgpt2`                    | `distilgpt2_path`     |
| `cvt2distilgpt2`                           | `cvt2distilgpt2_path` |


**2. Pretrain and Finetune**

Two-stage training:
- Pretraining with Self-Consistent Patch Reconstruction and Pathology-Aware Prototype Alignment:
```bash
cd scripts/cxr
bash pretrain.sh
```
- Auto-regressive Finetuning:
```
cd scripts/CXR
bash finetune.sh
```
## Testing
---
- To test the model 
```
cd scripts/CXR
bash test.sh
`` -->
