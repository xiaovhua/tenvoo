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

- **BraTS 2021**: We use the raw T1-weighted images. A filtered list of valid samples is provided in [`./prepare/clean_brats.txt`](./clean_brats.txt). If needed, you can put the txt file into the root path of BraTS2021.

- **ADNI & PPMI**: We pre-process the T1-weighted images from ADNI and PPMI following the steps below.

| Tool       | Step       |
|----------------|----------------|
| Row 1, Col 1   | Row 1, Col 2   |
| Row 2, Col 1   | Row 2, Col 2   |
| Row 3, Col 1   | Row 3, Col 2   |
| Row 4, Col 1   | Row 4, Col 2   |
| Row 5, Col 1   | Row 5, Col 2   |
| Row 6, Col 1   | Row 6, Col 2   |
| Row 7, Col 1   | Row 7, Col 2   |

---

### 🗂️ Data Directory Structure

The final output should match the directory organized as follows:

```bash
/your_dataset_root/
├── train/
│   ├── subject_01/
│   │   └── T1_brain_to_MNI.nii.gz
│   └── ...
├── val/
│   ├── subject_XX/
│   │   └── T1_brain_to_MNI.nii.gz
│   └── ...
```

If your data structure is different, you can modify the `GLOB_PATHS` variable in [`./dataset.py`](../dataset.py) to match your local layout.



## 3. Tutorial and Quick Start
### Training
To start training, run:
python pretrain_ddpm.py --config configs/your_config.yaml
### Inference / Generation
To generate MRI samples:
python generate.py --config configs/your_config.yaml --ckpt path/to/your_model.ckpt

git clone https://github.com/josipd/torch-two-sample.git
cd torch-two-sample
pip install .
### Run Evaluation
python eval.py --config configs/your_config.yaml
