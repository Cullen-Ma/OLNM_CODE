# SAM-Net: Sparse-Aware Multimodal Learning for Occult Lymph Node Metastasis Prediction

This repository contains the implementation of **Sparse-Aware Multimodal Fusion Network (SAM-Net)** and its integration with a **Clinical Decision Agent (CDA)** for preoperative prediction of **occult lymph node metastasis (OLNM)** in oral squamous cell carcinoma (OSCC).

## Overview

Accurate preoperative prediction of OLNM remains challenging because PET and CT provide fundamentally different types of information. PET contains sparse metabolic signals that may indicate metastatic lesions, whereas CT provides dense anatomical context. Conventional multimodal fusion methods may dilute these sparse but clinically important functional cues.

To address this challenge, we propose **SAM-Net**, a sparse-aware multimodal architecture that explicitly models the heterogeneous characteristics of PET/CT data.

### Key Components

* **Sparse-Aware Multimodal Fusion:** Jointly models complementary metabolic information from PET and anatomical information from CT.
* **Differentiable Gumbel-Softmax Top-k Routing:** Dynamically selects salient metabolic regions from PET while retaining global anatomical context.
* **Local Feature Enhancement:** Focuses representation learning on clinically informative regions.
* **Clinical Decision Agent (CDA):** Integrates anatomy parsing, uncertainty-guided evidence aggregation, and pseudo-report generation to support clinical decision-making, including scenarios with missing clinical text.
* **Interpretability:** Provides risk-stratified predictions and visualization of relevant image regions using methods such as Grad-CAM.


## Repository Structure

```text
├── ct_clip.py
├── ctvit_add_pet.py
├── data_inference.py
├── zero_shot_our_crossval.py
├── ct_vocabfine_train_and_inference_cross_validation_*.py
├── cv_oral_*_fusion_*.slurm
└── README.md
```

## Training

An example training configuration is provided in:

```text
cv_oral_delatesjt_all_gumble32_fusion_mlp.slurm
```

The training pipeline supports multimodal CT/PET input, Gumbel-based local feature selection, multimodal feature fusion, and stratified cross-validation.

## Citation

If you find this repository useful for your research, please cite our paper:

> *Multimodal Learning with Clinical Decision Support for Preoperative Prediction of Occult Lymph Node Metastasis in Oral Squamous Cell Carcinoma*
