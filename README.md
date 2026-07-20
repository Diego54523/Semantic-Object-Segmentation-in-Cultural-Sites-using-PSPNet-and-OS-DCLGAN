# Semantic Object Segmentation in Cultural Sites using PSPNet and OS-DCLGAN

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)

This repository contains the official implementation of the project **"Semantic Object Segmentation in Cultural Sites using PSPNet and OS-DCLGAN"**. 

The goal of this project is to perform Semantic Segmentation of Cultural Heritage objects (e.g., paintings, sculptures, display cases) in egocentric vision using the **EGO-CH-OBJ-SEG** dataset. To overcome the high cost of manual pixel-perfect annotation, we leverage synthetic data and bridge the *Domain Gap* between synthetic and real domains using a Joint-Training approach combining **PSPNet** and **OS-DCLGAN** (One-Sided Dual Contrastive Learning GAN).

## 🚀 Key Features
* **Unsupervised & Semi-Supervised Domain Adaptation (UDA/SSDA):** Achieves acceptable segmentation results with 0% real annotated data, peaking in efficiency with just 50% real data.
* **One-Sided GAN Architecture:** Replaces the heavy CycleGAN with a Contrastive Learning approach (PatchNCE Loss + CBAM), cutting VRAM usage in half and avoiding steganography artifacts.
* **Exact RGB Matching:** Custom vectorized data engineering to seamlessly map noisy RGB masks into discrete class indices.
* **HPC & SLURM Ready:** Includes a checkpointing logic to bypass 12-hour timeout limits on computing clusters.

---

## 📁 Project Structure

The repository is organized as follows:

```text
SEMANTIC-OBJECT-SEGMENTATION...
│
├── dataset/
│   └── dataset.py               # Custom PyTorch Dataset (Data loading, RGB Exact Matching, Augmentation)
│
├── docs/
│   └── Report_...pdf            # Full project report/thesis detailing methodology and results
│
├── models/
│   ├── networks.py              # OS-DCLGAN architecture (Generator and Discriminator)
│   ├── PSPNet.py                # Pyramid Scene Parsing Network implementation (ResNet-50 backbone)
│   └── train.py                 # Core Joint-Training loop (GAN + PSPNet) with checkpointing
│
├── notebook/
│   └── preliminary_results.ipynb # Jupyter notebook for dataset exploration and visualization
│
├── utility/
│   ├── cbam.py                  # Convolutional Block Attention Module
│   ├── evaluate.py              # Inference script computing mIoU, PA, mPA, and FWAVACC
│   ├── osudl_loss.py            # Custom loss functions for Domain Adaptation
│   ├── patchnce.py              # Noise Contrastive Estimation (PatchNCE) loss logic
│   └── Resnet.py                # ResNet blocks and utilities
│
├── .env                         # Environment variables for dynamic dataset paths
├── .gitignore                   # Ignored files (__pycache__, datasets, weights)
├── debug_dataset.py             # Script to verify DataLoader, RGB matching, and tensor shapes
├── evaluation.sh                # SLURM script to run evaluate.py on the cluster
├── run_train.sh                 # SLURM script to launch the Joint-Training via Job Arrays
└── requirements.txt             # Python dependencies
```

## ⚙️ Installation & Setup

1. **Clone the repository:**
```bash
git clone [https://github.com/Diego54523/Semantic-Object-Segmentation-in-Cultural-Sites-using-PSPNet-and-OS-DCLGAN.git](https://github.com/Diego54523/Semantic-Object-Segmentation-in-Cultural-Sites-using-PSPNet-and-OS-DCLGAN.git)
cd Semantic-Object-Segmentation-in-Cultural-Sites-using-PSPNet-and-OS-DCLGAN
```

## 🔧 Installazione

**Install the dependences:**

It is recommended to utilize a *virtual environment* or a Apptainer/Docker container.

```bash
pip install -r requirements.txt
```

## ⚙️ Environment setup

Create a file `.env` into the main directory stating the dataset path:

```env
DATA_PATH = path/to/dataset
```

## 🧠 Training
The training script execute an simultaneous update *end-to-end* of both OS-DCLGAN and PSPNet. Includes a phase of asymmetric *warm-up* to prevent the collaps of of the segmenter during the early (noisy) epochs of the generator.

**To start the local training:**

```bash
python models/train.py --real_percentage 0.50
```

**To launch the training on a SLURM cluster:**

```bash
sbatch run_train.sh
```

Markdown
> **Note:** The script includes an automatic resuming logic. If the job reaches the cluster timeout (e.g., 12 hours), relaunching the script will load the latest available `.pth` checkpoint and resume training from there.

---

## 📊 Evaluation & Results

To evaluate the model on the real test set, run the evaluation script. This will output the Pixel Accuracy (PA), Mean Pixel Accuracy (mPA), Mean IoU, and FWAVACC.

```bash
python utility/evaluate.py --weights_path path/to/weights.pth
# or via SLURM: sbatch evaluation.sh
```

## 📚 References & Acknowledgments

* **Baseline idea and EGO-CH dataset from:** *Semantic Segmentation of Cultural Heritage Objects in Egocentric Videos* (ICPR 2020) - [ICPR2020_Segmentation_CR.pdf](https://iplab.dmi.unict.it/legacy/EGO-CH-OBJ-SEG/downloads/ICPR2020_Segmentation_CR.pdf)
* **Segmentation architecture based on:** [PSPNet (Pyramid Scene Parsing Network)](https://github.com/hszhao/PSPNet) - CVPR 2017.
* **Unsupervised translation inspired by:** [OSUDL (One-Sided Unsupervised Domain Adaptation via Dual Contrastive Learning)](https://github.com/RedPotatoChip/OSUDL).

---

## 📄 License

This project is released under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Author 

* **Diego Martinez** 
