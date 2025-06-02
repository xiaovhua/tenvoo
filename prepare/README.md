## 1. Environment Setup

**Our Environment:**

- Python 3.11  
- PyTorch 2.2.0  
- CUDA 12.1  

**Required Dependencies:**

To install the required dependencies for TenVOO, please run:

```bash
pip install -r requirements.txt
```

## 2. Data Preparation

We use [UK Biobank](https://www.ukbiobank.ac.uk/) for pretraining, and [BraTS2021](https://www.med.upenn.edu/cbica/brats2021/), [ADNI](https://adni.loni.usc.edu/), [PPMI](https://www.ppmi-info.org/) for fine-tuning and evaluation.


/your_dataset_root/
├── train/
│   ├── subject_01/
│   │   └── T1_brain_to_MNI.nii.gz
│   └── ...
├── val/
│   ├── subject_XX/
│   │   └── T1_brain_to_MNI.nii.gz
│   └── ...




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
