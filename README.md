# Parameter-Efficient Fine-Tuning of 3D DDPM for MRI Image Generation Using Tensor Networks

<p align="center">
  <a href="https://arxiv.org/abs/xxxx.xxxxx">
    <img src="https://img.shields.io/badge/Paper-PDF-red?logo=adobeacrobatreader&style=for-the-badge" alt="paper"/>
  </a>
  <a href="[https://github.com/your_username/your_repo](https://github.com/TUAT-Novice/tenvoo/)">
    <img src="https://img.shields.io/badge/Code-GitHub-blue?logo=github&style=for-the-badge" alt="code"/>
  </a>
</p>

### This repository contains the official implementation of **"Parameter-Efficient Fine-Tuning of 3D DDPM for MRI Image Generation Using Tensor Networks"**.

### Congratulations !! Our work is early accepted by **MICCAI 2025** 🎉🎉

## 🔍 Introduction

This repository provides the official implementation of **Tensor Volumetric Operator (TenVOO)**, a novel parameter-efficient fine-tuning (PEFT) method specifically designed for 3D convolutional networks. TenVOO enables efficient adaptation of large-scale 3D models while preserving spatial interactions, and demonstrates strong performance across various Magnetic Resonance Imaging (MRI) generation tasks.

**Key highlights:**

- 🧠 Introduced tensor decomposition-based PEFT, TenVOO, for 3D CNNs  
- 📦 Compatible with standard 3D backbones (e.g., 3D UNet)  
- 📈 Significant reduction in trainable parameters without compromising spatial understanding

## 📁 Repository Structure

```bash
.
├── prepare/              # Details for environment setup and data preparation
├── peft/                 # PEFT codes for fine-tuning
├── scripts/              # Training and evaluation scripts
├── dataset.py            # Define Dataset and DataLoader class
├── ddpm_unet.py          # Define the MONAI DDPM model
├── pretrain_ddpm.py      # Code for pretraining DDPM
├── ft_ddpm.py            # Code for fine-tuning DDPM
├── eval.py               # Code for model evaluation
├── utils.py              # Utilities
├── med3d.py              # Define the Med3D, only for eval.py
└── README.md             # Project description
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/TUAT-Novice/tenvoo.git
cd tenvoo/
```

### 2. Set up environment & Prepare data

Please refer to [`./prepare`](./prepare)

### 3. Train the model

To train you MONAI DDPM, you can modify the config in  [`./scripts/run_pretrain_ddpm.sh`](./scripts/run_pretrain_ddpm.sh), then run:

```bash
python scripts/run_pretrain_ddpm.sh
```

To fine-tune based on a pre-trained DDPM, please modify the config in  [`./scripts/run_ft_ddpm.sh`](./scripts/run_ft_ddpm.sh), then run:

```bash
python scripts/run_ft_ddpm.sh
```

To evaluate the DDPM, please modify the config in  [`./scripts/run_eval.sh`](./scripts/run_eval.sh), then run:

```bash
python scripts/run_eval.sh
```

## 📊 Results

| Dataset | Method | Params (M) | Dice Score (%) |
|---------|--------|------------|----------------|
| Task A  | Ours   | 2.3        | 89.7           |
| Task B  | Ours   | 3.1        | 91.2           |

## 🤝 Citation

If you find this work useful, please consider citing our paper:

```bibtex
@inproceedings{yourbibtex2025,
  title={Your Paper Title},
  author={Author1, A. and Author2, B. and Author3, C.},
  booktitle={Proceedings of the MICCAI},
  year={2025}
}
```

## 📬 Contact

For questions or collaborations, please contact s237857s@st.go.tuat.ac.jp or nkvhua@outlook.com.

## 🙏 Acknowledgements

We would like to thank [QuanTA](https://github.com/quanta-fine-tuning/quanta) for their open-sourced code, which served as a valuable reference for this work.

