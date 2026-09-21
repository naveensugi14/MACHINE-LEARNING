# Experiment 9: Perceptron vs Multilayer Perceptron (A/B Experiment)

## Overview
This repository contains the complete implementation, dataset, hyperparameter tuning experiments, and formal report for **Experiment 9: Perceptron vs Multilayer Perceptron (A/B Experiment) with Hyperparameter Tuning**.

The objective is to compare a **Single-Layer Perceptron Learning Algorithm (PLA)** implemented from scratch against a **Multilayer Perceptron (MLP)** with non-linear activation functions on the 62-class English Handwritten Characters dataset.

## Directory Structure
```
exp9/
├── dataset/
│   ├── english_characters_data.csv    # Feature matrix (3,410 samples × 64 features)
│   └── english_characters_target.csv  # 62 class character labels (0-9, A-Z, a-z)
├── experiment_9.py                    # Standalone Python execution script
├── Experiment_9.ipynb                 # Fully executed Jupyter Notebook
├── Experiment_9_Lab_Record.pdf        # Compiled 9-page PDF Report
├── Experiment_9_Lab_Record.tex        # LaTeX source code for report
├── requirements.txt                   # Required Python dependencies
└── README.md                          # Experiment documentation
```

## Dataset Specifications
- **Dataset Name**: English Handwritten Characters Dataset
- **Number of Samples**: 3,410 instances (55 samples per class)
- **Number of Features**: 64 numerical feature attributes (8×8 pixel grid)
- **Target Classes**: 62 classes (Digits `0–9`, Uppercase `A–Z`, Lowercase `a–z`)
- **Preprocessing**: Standardized using `StandardScaler` ($\mu=0, \sigma=1$)

## Model Architecture & Hyperparameter Tuning
1. **Model A: Single-Layer Perceptron (PLA)**: Custom One-vs-Rest (OvR) binary perceptrons with step activation.
2. **Model B: Multilayer Perceptron (MLP)**: Hyperparameter tuning grid over:
   - **Activations**: ReLU (optimal: 88.71%), Tanh (85.34%), Logistic (79.18%)
   - **Optimizers**: Adam (lr=0.001, 38 epochs), SGD (lr=0.01, 150+ epochs), L-BFGS
   - **Architectures**: Single layer `(64,)`, Two layers `(128, 64)`, Three layers `(128, 64, 32)`
   - **Batch Sizes**: 32, 64, 128, 256

## Setup and Execution Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Python Script
```bash
python experiment_9.py
```

### 3. Open Jupyter Notebook
```bash
jupyter notebook Experiment_9.ipynb
```

## Key Results Summary
| Model Architecture | Test Accuracy | Precision | Recall | Macro F1-Score | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Single-Layer Perceptron (PLA) | 38.27% | 0.3912 | 0.3827 | 0.3745 | 0.125 s |
| Baseline MLP (64,) | 88.71% | 0.8920 | 0.8871 | 0.8812 | 1.450 s |
| **Tuned MLP (128, 64)** | **92.38%** | **0.9285** | **0.9238** | **0.9215** | **2.180 s** |

## Author
- **Student Name**: Naveenkumar S
- **Register Number**: 3122247001038
- **GitHub Repository**: [https://github.com/naveensugi14/MACHINE-LEARNING](https://github.com/naveensugi14/MACHINE-LEARNING)
