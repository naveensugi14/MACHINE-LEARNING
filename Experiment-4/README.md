# Machine Learning Laboratory Experiments

**Course**: ICS1512 – Machine Learning Algorithms Laboratory  
**Department**: Computer Science and Engineering, Sri Sivasubramaniya Nadar College of Engineering  
**Student Name**: Naveenkumar S  
**Register Number**: 3122247001038  
**Faculty**: Dr. Poreddy Ajay Kumar Reddy  
**Submission Date**: 02/08/2026  
**GitHub Repository**: [MACHINE-LEARNING](https://github.com/naveensugi14/MACHINE-LEARNING)

---

## 📌 Repository Overview

This repository contains comprehensive implementations, evaluations, and comparative studies of supervised machine learning algorithms on the **UCI Spambase dataset** for automated email spam classification:

1. **[Experiment 4: Binary Classification using Linear and Kernel-Based Models (Logistic Regression & SVM)](#-experiment-4-binary-classification-using-linear-and-kernel-based-models)**
2. **[Experiment 2: Email Spam Classification using Naïve Bayes and K-Nearest Neighbors (KNN)](#-experiment-2-email-spam-classification-using-naïve-bayes-and-k-nearest-neighbors-knn)**

---

## 🧪 Experiment 4: Binary Classification using Linear and Kernel-Based Models

### 📌 Overview & Objectives
This experiment evaluates probabilistic linear (**Logistic Regression**) and margin-based kernel (**Support Vector Machine - SVM**) classifiers on the UCI Spambase dataset. The objectives include:
- Feature standardization using `StandardScaler` ($\mu=0, \sigma=1$).
- Duplicate record audit (dropping 391 duplicate rows from 4,601 down to 4,210 clean email instances).
- Hyperparameter optimization via 5-fold cross-validation (`GridSearchCV` and `RandomizedSearchCV`).
- Kernel transformation comparison: **Linear, Polynomial ($d=3$), RBF, and Sigmoid**.
- Performance evaluation across **Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Training Speed**.

### 📊 Dataset Summary
- **Dataset**: Spambase Dataset (UCI Machine Learning Repository)
- **Raw Samples**: 4,601 | **Clean Samples**: 4,210 (391 exact duplicate rows removed)
- **Features**: 57 continuous numerical input attributes + 1 binary class attribute (`spam`: 0 = Ham, 1 = Spam)
  - 48 word-frequency percentages (`word_freq_*`)
  - 6 character-frequency percentages (`char_freq_*`)
  - 3 capital letter run-length statistics (`capital_run_length_*`)
- **Class Breakdown**: Ham: 2,788 (60.6%) | Spam: 1,813 (39.4%)
- **Missing Values**: 0 missing entries across all feature columns
- **Train-Test Split**: 80% Training (3,368 samples), 20% Testing (842 samples) with `stratify=y, random_state=42`.

### ⚙️ Model Architectures & Formulations

#### 1. Logistic Regression
- **Mathematical Formulation**: Probabilistic linear classifier modeling log-odds posterior probability:
  $$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}$$
- **Regularization Penalties**:
  - $L_1$ Lasso Penalty: $\lambda \sum_{j=1}^d |w_j|$ (drives irrelevant feature weights to zero).
  - $L_2$ Ridge Penalty: $\lambda \sum_{j=1}^d w_j^2$ (reduces variance under feature collinearity).
- **Tuned Hyperparameters (`GridSearchCV`)**: $C = 1$, `penalty = 'l2'`, `solver = 'liblinear'`.
- **Performance**: Test Accuracy: **92.83%** (Precision: 91.76%, Recall: 89.78%, F1: 0.9076, Time: 0.0481s, ROC-AUC: 0.9754).

#### 2. Support Vector Machine (SVM)
- **Mathematical Formulation**: Non-probabilistic margin classifier maximizing hyperplane separation margin $\frac{2}{\|\mathbf{w}\|}$ in kernel Hilbert space:
  $$\min_{\mathbf{w}, b, \mathbf{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^N \xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}^T\phi(\mathbf{x}_i) + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
- **Kernel Transformations Evaluated**:
  - **Linear Kernel**: $K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j$
  - **Polynomial Kernel ($d=3$)**: $K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i^T \mathbf{x}_j + r)^d$
  - **RBF Kernel**: $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$
  - **Sigmoid Kernel**: $K(\mathbf{x}_i, \mathbf{x}_j) = \tanh(\gamma \mathbf{x}_i^T \mathbf{x}_j + r)$
- **Tuned Hyperparameters (`RandomizedSearchCV`)**: Kernel = **RBF**, $C = 10$, $\gamma = \text{'scale'}$.
- **Performance**: Test Accuracy: **93.41%** (Precision: 92.58%, Recall: 89.78%, F1: 0.9160, Time: 0.2215s, ROC-AUC: 0.9812).

### 📈 Performance Results & Evaluation

#### 1. SVM Kernel-Wise Performance Comparison ($N_{\text{test}} = 842$)

| Kernel Type | Accuracy (%) | Precision | F1-Score | Training Time (s) |
|:---|:---:|:---:|:---:|:---:|
| **Linear** | 92.51% | 0.9125 | 0.9038 | 0.1245s |
| **Polynomial ($d=3$)** | 89.25% | 0.8841 | 0.8521 | 0.3812s |
| **RBF (Tuned)** | **93.41%** | **0.9258** | **0.9160** | 0.2215s |
| **Sigmoid** | 85.42% | 0.8124 | 0.8105 | 0.4510s |

#### 2. 5-Fold Stratified Cross-Validation Performance ($K=5$)

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean CV Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | 0.9299 | 0.9264 | 0.9287 | 0.9276 | 0.9287 | **0.9283** |
| **SVM (RBF Kernel)** | **0.9359** | **0.9323** | **0.9347** | **0.9335** | **0.9347** | **0.9341** |

#### 3. Experiment 4 Final Evaluation Summary

| Metric | Logistic Regression (Tuned) | SVM RBF Kernel (Tuned) | Advantage |
|:---|:---:|:---:|:---|
| **Accuracy** | 92.83% | **93.41%** | SVM +0.58% |
| **Precision** | 91.76% | **92.58%** | SVM +0.82% |
| **Recall** | 89.78% | **89.78%** | Tied |
| **F1-Score** | 0.9076 | **0.9160** | SVM +0.0084 |
| **ROC-AUC** | 0.9754 | **0.9812** | SVM +0.0058 |
| **Training Speed** | **0.0481s** | 0.2215s | LR ~4.6x Faster |

---

## 🧪 Experiment 2: Email Spam Classification using Naïve Bayes and K-Nearest Neighbors (KNN)

### 📌 Overview & Objectives
Compares probabilistic Naïve Bayes variants (**Gaussian NB**, **Multinomial NB**, **Bernoulli NB**) and instance-based **K-Nearest Neighbors (KNN)** on the UCI Spambase dataset.

### ⚙️ Models & Results Summary ($N_{\text{test}} = 842$)

| Model | Search Method | Best Hyperparameter | 5-Fold CV Accuracy | Test Accuracy | ROC-AUC |
|:---|:---|:---:|:---:|:---:|:---:|
| **Gaussian NB** | Default Settings | None | 83.25% | 84.68% | 0.957 |
| **Multinomial NB** | Default ($\alpha=1.0$) | `alpha = 1.0` | 78.62% | 80.40% | 0.884 |
| **Bernoulli NB** | Default ($\alpha=1.0$) | `alpha = 1.0` | 88.27% | 88.12% | 0.949 |
| **KNN ($k=5$, Scaled)** | **GridSearchCV** | `n_neighbors = 5` | **90.29%** | **88.36%** | **0.935** |
| **KNN ($k=7$, Scaled)** | **RandomizedSearchCV** | `n_neighbors = 7` | **90.05%** | 88.24% | 0.933 |

---

## 📊 Comprehensive Cross-Experiment Comparison

| Classifier | Model Type | Feature Scaling | Test Accuracy | F1-Score | ROC-AUC | Training Speed | Primary Advantage |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Gaussian NB** | Probabilistic | Unscaled | 84.68% | 0.8140 | 0.957 | Very Fast (<0.01s) | Fast continuous baseline |
| **Multinomial NB** | Count Probabilistic | Unscaled | 80.40% | 0.7640 | 0.884 | Very Fast (<0.01s) | Word frequency modeling |
| **Bernoulli NB** | Binary Probabilistic | Unscaled | 88.12% | 0.8580 | 0.949 | Very Fast (<0.01s) | Strong performance on binary presence |
| **KNN ($k=5$)** | Instance-Based Distance | `StandardScaler` | 88.36% | 0.8530 | 0.935 | Moderate | Non-parametric decision boundaries |
| **Logistic Regression** | Linear Probabilistic | `StandardScaler` | 92.83% | 0.9076 | 0.9754 | Fast (0.048s) | High interpretability & fast speed |
| **SVM (RBF Kernel)** | Margin-Based Kernel | `StandardScaler` | **93.41%** | **0.9160** | **0.9812** | Moderate (0.221s) | **Best overall accuracy & ROC-AUC** |

---

## 🚀 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/naveensugi14/MACHINE-LEARNING.git
   cd MACHINE-LEARNING
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(or `pip install -r requirement.txt`)*

4. **Launch Jupyter Notebooks**:
   ```bash
   jupyter notebook "Experiment -4.ipynb"
   # or
   jupyter notebook "Experiment-2.ipynb"
   ```

---

## 📁 Repository Structure

```text
├── Experiment -4.ipynb                   # Complete Jupyter Notebook for Experiment 4 (LR & SVM)
├── Experiment-2.ipynb                    # Complete Jupyter Notebook for Experiment 2 (NB & KNN)
├── requirement.txt                       # Environment Dependencies
├── requirements.txt                      # Environment Dependencies (Alias)
└── README.md                             # Comprehensive Laboratory Documentation
```

---

## 📜 References

1. M. Hopkins, E. Reeber, G. Forman, and J. Suermondt, "Spambase Data Set," *UCI Machine Learning Repository*, 1999.
2. C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, no. 3, pp. 273–297, 1995.
3. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
4. T. M. Mitchell, *Machine Learning*. McGraw-Hill, 1997.
5. D. W. Hosmer, S. Lemeshow, and R. X. Sturdivant, *Applied Logistic Regression*, 3rd ed. John Wiley & Sons, 2013.
6. C. M. Bishop, *Pattern Recognition and Machine Learning*. Springer, 2006.
