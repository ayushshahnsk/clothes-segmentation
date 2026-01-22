# 🧥 AI Clothes Segmentation & Virtual Try-On System

An end-to-end AI-powered system for **cloth segmentation**, **image analytics**, and **virtual try-on**, built using deep learning and computer vision techniques.  
This project integrates a custom-trained UNet segmentation model with the state-of-the-art **VITON-HD Virtual Try-On** framework, providing a research-grade fashion AI system.

---

## 🚀 Key Features

### 👕 Clothes Segmentation (UNet)
- Pixel-level cloth segmentation
- Binary mask generation
- Cloth extraction and overlay
- Batch and single image testing
- Model training & evaluation support

### 📊 Image Analytics
- Pixel statistics (cloth vs background)
- Cloth area percentage calculation
- Graph analytics and visual summaries

### 🖥 Interactive Streamlit UI
- Upload cloth images for segmentation
- View mask, extracted cloth, and comparison
- Download results
- Analytics dashboard
- Integrated Virtual Try-On (dataset-based)

### 👗 Virtual Try-On (VITON-HD)
- AI-powered human-cloth synthesis
- Dataset-based try-on using pretrained VITON-HD models
- Research-grade virtual dressing system

---

## 📁 Project Folder Structure

CLOTHES-SEGMENTATION/
│
├── VITON-HD/
│ ├── assets/
│ ├── checkpoints/ # Pretrained VITON-HD models
│ ├── datasets/ # Dataset (test, cloth, masks, pose, parse)
│ ├── networks.py
│ ├── datasets.py
│ ├── test.py # VITON-HD inference script
│ ├── utils.py
│ ├── LICENSE
│ └── README.md # Original VITON-HD documentation
│
├── app.py # Streamlit UI Application
├── batch_test.py # Batch cloth segmentation
├── cloth_seg_model.pth # Trained UNet segmentation model
├── dataset.py # Custom PyTorch Dataset
├── evaluate_masks.py # Mask evaluation (IoU, Dice)
├── model.py # UNet model architecture
├── overlay_mask.py # Cloth overlay and extraction
├── random_test.py # Random image testing
├── side_by_side.py # Comparison generator
├── test_model.py # Single image test
├── train.py # UNet training script
└── readme.md # Project notes

---

## ⚙ Installation

### 1️⃣ Clone Repository
git clone https://github.com/shadow2496/VITON-HD

### 2️⃣ Install Dependencies
- `pip install torch torchvision opencv-python streamlit numpy matplotlib`
- `pip install pillow scipy tqdm`

---

## 🧠 UNet Cloth Segmentation
### Train Model
- `python train.py`
### Test Model (Single Image)
- `python test_model.py`
### Random Image Test
- `python random_test.py`
### Batch Test
- `python batch_test.py`
### Evaluate Masks
- `python evaluate_masks.py`

---

## 👗 Virtual Try-On (VITON-HD)
### Navigate to VITON-HD Folder
- `cd VITON-HD`
### Run Try-On Model
- `python test.py --name viton_test --dataset_dir datasets --dataset_mode test --dataset_list test_pairs.txt --checkpoint_dir checkpoints --save_dir results`
### Output images will be saved in:
- `VITON-HD/results/viton_test/`

---

## 🖥 Streamlit Web Application
### Run the full UI:
- `streamlit run app.py`

---

## 🧪 Sample Commands Summary
| Task           | Command                                |
| -------------- | -------------------------------------- |
| Train UNet     | `python train.py`                      |
| Single Test    | `python test_model.py`                 |
| Random Test    | `python random_test.py`                |
| Batch Test     | `python batch_test.py`                 |
| Evaluate Masks | `python evaluate_masks.py`             |
| Run VITON-HD   | `python test.py --name viton_test ...` |
| Run UI         | `streamlit run app.py`                 |

---

## 📚 Dataset & Models Source
### Virtual Try-On Framework:
`[All pretrained models and dataset used for try-on are credited to the original authors.]`

`(https://github.com/shadow2496/VITON-HD)`

---

## 👨‍💻 Author
### Ayush Shah (www.github.com/ayushshahnsk)