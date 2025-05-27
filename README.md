# Parameter-Efficient Fine-Tuning of 3D DDPM for MRI Image Generation Using Tensor Networks

## > This repository contains the official implementation of **Parameter-Efficient Fine-Tuning of 3D DDPM for MRI Image Generation Using Tensor Networks**.

## > Congratulations 🎉🎉 Our work is early accepted by **MICCAI 2025** 🎉🎉

## 🔍 Introduction

This project provides the official PyTorch implementation of our MICCAI 2025 paper, where we introduce a novel method for Parameter-Efficient Fine-Tuning (PEFT) of 3D convolutional networks using **tensor networks**. Our approach enables efficient adaptation of large 3D models while maintaining strong performance across medical imaging tasks.

**Key highlights:**

- 🧠 Introduced **tensor decomposition-based PEFT** for 3D CNNs  
- 📦 Compatible with standard 3D backbones (e.g., 3D UNet, VNet)  
- 🚀 Significant reduction in trainable parameters without compromising accuracy  

## 📄 Paper

- **Title:** XXX  
- **Authors:** [Author List]  
- **Conference:** MICCAI 2025 (Early Accept)  
- **[Link to Paper (arXiv or MICCAI site)](URL)**

## 📁 Repository Structure

```bash
.
├── models/              # 3D CNN backbones and tensor network modules
├── configs/             # YAML configuration files for training/evaluation
├── data/                # Dataset handling and preprocessing
├── scripts/             # Training and evaluation scripts
├── utils/               # Utility functions
├── requirements.txt     # Python dependencies
└── README.md            # Project description
```

## 🚀 Getting Started

### 1. Clone the repository

git clone https://github.com/your_username/XXX.git
cd XXX


### 2. Set up environment

conda create -n xxx_env python=3.10
conda activate xxx_env
pip install -r requirements.txt


### 3. Prepare data

Instructions for preparing your dataset go here.

### 4. Train the model

python scripts/train.py --config configs/your_config.yaml



## 📊 Results

| Dataset | Method | Params (M) | Dice Score (%) |
|---------|--------|------------|----------------|
| Task A  | Ours   | 2.3        | 89.7           |
| Task B  | Ours   | 3.1        | 91.2           |

## 🤝 Citation

If you find this work useful, please consider citing our paper:

@inproceedings{yourbibtex2025,
title={XXX},
author={Author1, A. and Author2, B. and Author3, C.},
booktitle={Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
year={2025}
}


## 📬 Contact

For questions or collaborations, please contact [your.email@example.com].

## 🙏 Acknowledgements

We would like to thank [XXX](https://github.com/xxx/xxx) for their open-sourced code, which served as a valuable reference for this work.

