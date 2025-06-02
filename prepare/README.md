## 1. Environment Setup

**Required Environment:**

- Python 3.11  
- PyTorch 2.2.0  
- CUDA 12.1  

### Step 1: Install PyTorch

Please install PyTorch manually based on your system and GPU configuration using the official guide:  
👉 https://pytorch.org/get-started/locally/

### Step 2: Install Python dependencies

Once PyTorch is installed, run:

```bash
pip install -r requirements.txt


## 2. Data Preparation
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
