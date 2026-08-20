# Experiment 5: Decision Tree and Random Forest - A Comparative Classification Study

**Course**: ICS1512 – Machine Learning Laboratory  
**Department**: Computer Science and Engineering, Sri Sivasubramaniya Nadar College of Engineering  
**Student Name**: NAVEENKUMAR S  
**Register Number**: 3122247001038  
**Faculty**: Dr. Poreddy Ajay Kumar Reddy  
**Submission Date**: 16/08/2026  
**GitHub Repository**: [MACHINE-LEARNING](https://github.com/naveensugi14/MACHINE-LEARNING)

---

## 📌 Project Overview

This repository implements, evaluates, and compares a single non-parametric classification algorithm (**Decision Tree**) and an ensemble tree-based algorithm (**Random Forest**) applied to the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset for binary tumor diagnosis.

The study explores feature correlation analysis, 5-fold stratified cross-validation hyperparameter optimization using `GridSearchCV`, decision rules, impurity measures (Gini index and Entropy), and generalization capabilities on unseen test data.

---

## 🎯 Objectives

- Implement single non-parametric **Decision Tree** and ensemble **Random Forest** classifiers.
- Conduct Exploratory Data Analysis (EDA) including class distribution analysis and full 30-feature correlation heatmaps.
- Systematically search hyperparameter spaces using 5-fold cross-validation (`GridSearchCV`).
- Compare performance metrics: **Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC**.
- Analyze decision boundaries, impurity criteria, tree depth constraints, and the variance-reduction benefits of bootstrap aggregation (bagging) in Random Forests.

---

## 📊 Dataset Summary

- **Dataset**: Wisconsin Diagnostic Breast Cancer (WDBC) Dataset
- **Source**: UCI Machine Learning Repository
- **Total Samples**: 569 instances
- **Features**: 30 continuous real-valued numerical attributes computed from digitized FNA images (10 Mean metrics, 10 Standard Error metrics, 10 Worst metrics).
- **Target Classes**:
  - `Benign (B)`: 357 samples (62.7%) — Label 0
  - `Malignant (M)`: 212 samples (37.3%) — Label 1
- **Train-Test Split**: 80% Training (455 samples), 20% Testing (114 samples) with `random_state = 42`.
- **Feature Scaling**: Tree-based models are scale-invariant; raw feature scaling is preserved.

---

## ⚙️ Model Architectures & Hyperparameter Optimization

### 1. Decision Tree Classifier
- **Working Principle**: Recursively partitions feature space into axis-aligned hyperplanes to minimize node impurity.
- **Tuned Hyperparameters (`GridSearchCV`, 5-Fold CV)**:
  - `criterion`: `['gini', 'entropy']`
  - `max_depth`: `[3, 5, 7, 10, None]`
  - `min_samples_split`: `[2, 5, 10]`
  - `min_samples_leaf`: `[1, 2, 4]`
- **Best Parameters**: `criterion='gini'`, `max_depth=7`, `min_samples_leaf=1`
- **Best 5-Fold CV Accuracy**: **94.07%** (F1: 0.9218)

### 2. Random Forest Classifier
- **Working Principle**: Builds an ensemble of $B$ decorrelated decision trees using Bootstrap Aggregation (Bagging) and Random Subspace Feature Selection ($m = \sqrt{p}$ or $\log_2 p$).
- **Tuned Hyperparameters (`GridSearchCV`, 5-Fold CV)**:
  - `n_estimators`: `[50, 100, 200]`
  - `max_depth`: `[5, 10, None]`
  - `max_features`: `['sqrt', 'log2']`
  - `bootstrap`: `[True, False]`
- **Best Parameters**: `bootstrap=False`, `max_depth=5`, `max_features='log2'`, `n_estimators=100`
- **Best 5-Fold CV Accuracy**: **96.48%** (F1: 0.9528)

---

## 📈 Performance Results

### 1. 5-Fold Cross-Validation Accuracy Comparison ($K=5$)

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Average Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Decision Tree** | 0.9451 | 0.9231 | 0.9451 | 0.9341 | 0.9560 | **0.9407** |
| **Random Forest** | **0.9560** | **0.9780** | **0.9780** | **0.9451** | **0.9670** | **0.9648** |

### 2. Final Test Set Evaluation ($N_{\text{test}} = 114$)

| Metric | Decision Tree (Tuned) | Random Forest (Tuned) | Performance Gain |
|:---|:---:|:---:|:---:|
| **Accuracy** | 93.86% (0.9386) | **96.49% (0.9649)** | +2.63% |
| **Precision** | 94.87% (0.9487) | **100.00% (1.0000)** | +5.13% |
| **Recall** | 88.10% (0.8810) | **90.48% (0.9048)** | +2.38% |
| **F1-Score** | 91.36% (0.9136) | **95.00% (0.9500)** | +3.64% |
| **ROC-AUC** | 92.25% (0.9225) | **99.34% (0.9934)** | +7.09% |

### Key Inferences:
1. **Random Forest Superiority**: Random Forest outperformed the single Decision Tree across all metrics, achieving **100.00% Precision** (zero false positive benign predictions) and an exceptional **0.9934 ROC-AUC**.
2. **Variance Reduction**: Bootstrap aggregation in Random Forests decorrelates individual tree errors, successfully stabilizing model predictions and mitigating overfitting.

---

## 🚀 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/naveensugi14/MACHINE-LEARNING.git
   cd MACHINE-LEARNING
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook Ex-5.ipynb
   ```

---

## 📁 Repository Structure

```text
├── Ex-5.ipynb                            # Complete Jupyter Notebook for Experiment 5
├── Experiment-5_Documentation.pdf        # Academic Laboratory Record PDF
├── requirements.txt                      # Environment Dependencies
└── README.md                             # Repository Documentation for Experiment 5
```

---

## 📜 References

1. W. H. Wolberg, W. N. Street, and O. L. Mangasarian, "Breast Cancer Wisconsin (Diagnostic) Data Set," *UCI Machine Learning Repository*, 1995.
2. L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.
3. L. Breiman et al., *Classification and Regression Trees*, Wadsworth, Belmont, CA, 1984.
4. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR*, vol. 12, pp. 2825–2830, 2011.
