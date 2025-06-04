## 1. Environment Setup

### Our Environment

- Python 3.11  
- PyTorch 2.2.0  
- CUDA 12.1  

### Required Dependencies

To install the required dependencies for TenVOO, please run:

```bash
pip install -r requirements.txt
```
or

```bash
pip install -r prepare/requirements.txt
```

### Evaluation Tools

If you want to evaluate your diffusion models as we do, you also need to download the resnet50 checkpoint from [Med3D](https://github.com/Tencent/MedicalNet), and install the [torch-two-sample](https://github.com/josipd/torch-two-sample/) repository by yourself.

## 2. Data Preparation

### 📥 Download

We use the following datasets for training and evaluation:

- [UK Biobank](https://www.ukbiobank.ac.uk/) – pretraining
- [BraTS 2021](https://www.med.upenn.edu/cbica/brats2021/) – fine-tuning and evaluation
- [ADNI](https://adni.loni.usc.edu/) – fine-tuning and evaluation
- [PPMI](https://www.ppmi-info.org/) – fine-tuning and evaluation

Please apply for access and download the datasets from their official websites if needed.

---

### 🧹 Preprocessing

- **UK Biobank**: We use the officially preprocessed T1-weighted MRI images, already registered to the MNI template.

- **BraTS 2021**: We use the raw T1-weighted images. A filtered list of valid samples is provided in [`./prepare/clean_brats.txt`](./clean_brats.txt). If needed, you can put this txt file into the root path of BraTS2021.

- **ADNI & PPMI**: We pre-process the T1-weighted images from ADNI and PPMI following the steps below using [CAT12.9](https://neuro-jena.github.io/cat12-help/cat.html).

| Step       |
|----------------|
| SANLM denoising (first)   |
| Affine Preprocessing   |
| Correct center-of-mass   |
| Affine registration   |
| TPM-based Segmentation & Registration  |
| Global intensity correction |
| SANLM denoising (second)  |
| Fast Optimized Shooting registration  |
| LAS |
| ROI segmentation |
| Enhanced Blood Vessel Correction |
| AMAP |
| Final cleanup |

Our pre-processing may be overly aggressive, as we aimed to ensure spatial consistency across ADNI/PPMI and the UK Biobank. The goal is to demonstrate that baseline methods such as LoRA fail even when spatial consistency is maintained. You may consider using an alternative, smoother pre-processing pipeline.

---

### 🗂️ Data Directory Structure

The final output should match the directory organized in [`./dataset.py`](../dataset.py).

If your data structure is different, you can modify the `GLOB_PATHS` variable in [`./dataset.py`](../dataset.py) to match your local layout.
