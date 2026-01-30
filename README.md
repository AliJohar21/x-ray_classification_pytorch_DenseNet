# Chest X-Ray Image Classification using PyTorch & DenseNet

## 📌 Project Overview
This project implements a deep learning solution for automated disease detection from chest X-ray images. Using the **PyTorch** framework, it leverages **Transfer Learning** with a pre-trained **DenseNet** architecture to classify X-rays into two categories: **Normal** and **Pneumonia**.

The project focuses on reproducibility, robust data preprocessing, and comprehensive evaluation metrics to ensure reliable performance in medical image analysis.

## 🚀 Key Features
* **High Performance:** Achieved a **Test Accuracy of 94.43%**.
* **Reproducibility:** Strict seeding of random number generators (Seed: \`42\`) for consistent results across runs.
* **Data Exploration (EDA):** Visual sanity checks to verify data integrity and label correctness.
* **Transfer Learning:** Utilizes a pre-trained **DenseNet** model, fine-tuned for binary classification.
* **Robust Pipeline:** Includes custom data loading, transformations, and augmentation.
* **Evaluation:** Detailed performance analysis using Test Accuracy, Confusion Matrix, and Classification Reports.

## 🛠️ Technologies Used
* **Python 3.13**
* **PyTorch** (Deep Learning framework)
* **Torchvision** (Image transformations and pre-trained models)
* **NumPy** (Numerical computations)
* **Matplotlib & Seaborn** (Visualization)
* **Scikit-Learn** (Evaluation metrics)
* **PIL** (Image processing)

## 📂 Dataset Structure
The project expects the dataset to be organized in the standard folder structure:
```
root_dir/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## ⚙️ Setup & Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AliJohar21/x-ray_classification_pytorch_DenseNet.git
    cd x-ray_classification_pytorch_DenseNet
    ```

2.  **Install dependencies:**
    Ensure you have the required libraries installed. You can install them using pip:
    ```bash
    pip install torch torchvision numpy matplotlib seaborn scikit-learn pillow
    ```

## 🏃‍♂️ Usage
1.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook x-ray_classification_pytorch_DenseNet.ipynb
    ```
2.  Run the cells sequentially.
3.  **Step 1:** System sets up seeds and checks for GPU availability.
4.  **Step 2:** Data exploration visualizes sample X-rays.
5.  **Training:** The model trains on the training set and validates on the validation set.
6.  **Evaluation:** The final model is tested on the unseen test set, generating a confusion matrix and accuracy report.

## 📊 Results
The notebook generates the following evaluation artifacts:
* **Test Accuracy:** Percentage of correctly classified images in the test set.
* **Confusion Matrix:** Heatmap visualizing True Positives, False Positives, etc.
* **Classification Report:** Precision, Recall, and F1-Score for both classes.
