# Experiment 2: Email Spam Classification using Naïve Bayes and K-Nearest Neighbors (KNN)

**Course**: ICS1512 – Machine Learning Algorithms Laboratory  
**Department**: Computer Science and Engineering, Sri Sivasubramaniya Nadar College of Engineering  
**Student Name**: Naveenkumar S  
**Register Number**: 3122247001038  
**Faculty**: Dr. Poreddy Ajay Kumar Reddy  
**Submission Date**: 02/08/2026  
**GitHub Repository**: [MACHINE-LEARNING](https://github.com/naveensugi14/MACHINE-LEARNING)

---

## 📌 Project Overview

This repository implements, evaluates, and compares probabilistic Naïve Bayes variants (**Gaussian NB**, **Multinomial NB**, **Bernoulli NB**) and instance-based **K-Nearest Neighbors (KNN)** algorithms on the **UCI Spambase dataset** for automated email spam classification.

The study investigates data preprocessing (duplicate record removal and `StandardScaler` feature normalization), hyperparameter optimization via 5-fold cross-validation (`GridSearchCV` and `RandomizedSearchCV`), and comprehensive model evaluation across key metrics including Accuracy, Precision, Recall, F1-Score, ROC-AUC, Precision-Recall (PR) curves, and execution efficiency.

---

## 🎯 Objectives

- Implement four fundamental machine learning classifiers: **Gaussian Naïve Bayes**, **Multinomial Naïve Bayes**, **Bernoulli Naïve Bayes**, and **K-Nearest Neighbors (KNN)**.
- Perform exploratory data analysis (EDA) including feature distributions, correlation heatmaps, missing value validation, and boxplot outlier detection.
- Clean and preprocess the dataset by identifying and dropping **391 exact duplicate records** (reducing sample size from 4,601 to 4,210 clean email rows).
- Apply feature standardization (`StandardScaler`) to prevent high-magnitude features from dominating distance-based KNN distance calculations.
- Execute hyperparameter tuning over $k \in \{1, 3, 4, 5, 7, 9, 11\}$ using 5-fold Stratified Cross-Validation (`GridSearchCV` and `RandomizedSearchCV`).
- Evaluate test set performance using an **80-20 train-test split** ($N_{\text{test}} = 842$) across Accuracy, Precision, Recall, F1-Score, ROC Curves, and PR Curves.

---

## 📊 Dataset Summary

- **Dataset**: Spambase Dataset
- **Source**: UCI Machine Learning Repository
- **Raw Samples**: 4,601 email records
- **Clean Samples**: 4,210 email records (after removing 391 exact duplicate rows)
- **Features**: 57 continuous numerical input attributes + 1 binary target class attribute (`class` / `spam`)
  - 48 word-frequency percentages (`word_freq_*`)
  - 6 character-frequency percentages (`char_freq_*`)
  - 3 capital letter run-length statistics (`capital_run_length_*`)
- **Missing Values**: 0 missing entries across all feature columns
- **Class Distribution**:
  - `Ham (Non-Spam = 0)`: 2,788 samples (60.6%)
  - `Spam (Spam = 1)`: 1,813 samples (39.4%)
- **Train-Test Split**: 80% Training (3,368 samples), 20% Testing (842 samples) with `random_state=42`.
- **Feature Scaling**: `StandardScaler` ($\mu=0, \sigma=1$) applied for distance-based KNN modeling; unscaled for Naïve Bayes variants.

---

## ⚙️ Model Architectures & Hyperparameter Optimization

### 1. Gaussian Naïve Bayes (Gaussian NB)
- **Mathematical Formulation**: Assumes continuous features given a class follow a Gaussian (normal) distribution:
  $$P(x_i \mid C_k) = \frac{1}{\sqrt{2\pi\sigma_k^2}} \exp\left(-\frac{(x_i - \mu_k)^2}{2\sigma_k^2}\right)$$
- **Hyperparameters**: Default settings (None).
- **Validation**: 5-Fold Stratified Cross-Validation.
- **Performance**: 5-Fold CV Accuracy: **83.25%**, Test Set Accuracy: **84.68%** (ROC-AUC: 0.957).

### 2. Multinomial Naïve Bayes (Multinomial NB)
- **Mathematical Formulation**: Operates on feature frequency counts using additive Laplace smoothing ($\alpha = 1.0$):
  $$P(x_i \mid C_k) = \frac{N_{y,i} + \alpha}{N_y + \alpha n}$$
- **Hyperparameters**: Default additive smoothing ($\alpha = 1.0$).
- **Validation**: 5-Fold Stratified Cross-Validation.
- **Performance**: 5-Fold CV Accuracy: **78.62%**, Test Set Accuracy: **80.40%** (ROC-AUC: 0.884).

### 3. Bernoulli Naïve Bayes (Bernoulli NB)
- **Mathematical Formulation**: Evaluates binary feature presence or absence ($x_i > 0$):
  $$P(\mathbf{x} \mid C_k) = \prod_{i=1}^n P(i \mid C_k)^{x_i} (1 - P(i \mid C_k))^{(1 - x_i)}$$
- **Hyperparameters**: Default binarization threshold ($x_i > 0$, $\alpha = 1.0$).
- **Validation**: 5-Fold Stratified Cross-Validation.
- **Performance**: 5-Fold CV Accuracy: **88.27%**, Test Set Accuracy: **88.12%** (ROC-AUC: 0.949).

### 4. K-Nearest Neighbors (KNN)
- **Mathematical Formulation**: Non-parametric algorithm classifying samples based on Euclidean distance to training points and majority voting:
  $$d(\mathbf{x}, \mathbf{x}') = \sqrt{\sum_{i=1}^n (x_i - x'_i)^2}, \quad \hat{y} = \arg\max_{c} \sum_{i \in N_k(\mathbf{x})} \mathbb{I}(y_i = c)$$
- **Hyperparameter Optimization**:
  - `GridSearchCV` (5-Fold CV over $k \in \{1, 3, 4, 5, 7, 9, 11\}$): Optimal $n\_neighbors = 5$ (Mean CV Accuracy: **90.29%**).
  - `RandomizedSearchCV` (5-Fold CV): Optimal $n\_neighbors = 7$ (Mean CV Accuracy: **90.05%**).
- **Performance**: Test Set Accuracy ($k=5$, Scaled): **88.36%** (ROC-AUC: 0.935).

---

## 📈 Performance Results

### 1. 5-Fold Cross-Validation Summary ($K=5$)

| Model | Search Method | Best Hyperparameter | Mean 5-Fold CV Accuracy |
|:---|:---|:---:|:---:|
| **Gaussian NB** | Default Settings | None | 83.25% |
| **Multinomial NB** | Default ($\alpha=1.0$) | `alpha = 1.0` | 78.62% |
| **Bernoulli NB** | Default ($\alpha=1.0$) | `alpha = 1.0` | 88.27% |
| **KNN ($k=5$)** | **GridSearchCV** | `n_neighbors = 5` | **90.29%** |
| **KNN ($k=7$)** | **RandomizedSearchCV** | `n_neighbors = 7` | **90.05%** |

### 2. Test Set Evaluation Summary ($N_{\text{test}} = 842$, 80-20 Split)

| Model | Test Accuracy | Precision (Ham = 0) | Precision (Spam = 1) | Recall (Ham = 0) | Recall (Spam = 1) | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Gaussian NB** | 84.68% | 0.97 | 0.75 | 0.76 | 0.97 | 0.957 |
| **Multinomial NB** | 80.40% | 0.82 | 0.79 | 0.85 | 0.74 | 0.884 |
| **Bernoulli NB** | 88.12% | 0.87 | 0.90 | 0.93 | 0.82 | 0.949 |
| **KNN ($k=5$, Scaled)** | **88.36%** | **0.88** | **0.89** | **0.93** | **0.82** | **0.935** |

### 💡 Key Inferences & Findings

1. **Top Performing Classifier**: **KNN ($k=5$, Scaled)** achieved the highest test accuracy (**88.36%**) and highest 5-fold cross-validation accuracy (**90.29%**).
2. **Top Naïve Bayes Model**: **Bernoulli NB** achieved strong test accuracy (**88.12%**) and CV accuracy (**88.27%**), outperforming Gaussian and Multinomial NB by effectively capturing binary feature presence/absence ($x_i > 0$) in spam detection.
3. **Importance of Feature Scaling**: Applying `StandardScaler` was critical for KNN to ensure continuous attributes with larger ranges (e.g., `capital_run_length_total`) did not skew Euclidean distance calculations.
4. **Computational Efficiency Trade-Off**: Naïve Bayes models trained rapidly (5–7 milliseconds) with minimal computational overhead, whereas KNN inference slowed down during testing due to instance-based distance comparisons against all training samples ($O(n \cdot d)$ complexity).

---

## 🚀 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/naveensugi14/MACHINE-LEARNING.git
   cd MACHINE-LEARNING
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(or `pip install -r requirement.txt`)*

4. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook "Experiment-2.ipynb"
   ```

---

## 📁 Repository Structure

```text
├── Experiment-2.ipynb                   # Complete Jupyter Notebook for Experiment 2
├── requirement.txt                      # Environment Dependencies
├── requirements.txt                     # Environment Dependencies (Alias)
└── README.md                            # Project Documentation for Experiment 2
```

---

## 📜 References

1. M. Hopkins, E. Reeber, G. Forman, and J. Suermondt, "Spambase Data Set," *UCI Machine Learning Repository*, 1999.
2. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
3. T. M. Mitchell, *Machine Learning*. McGraw-Hill, 1997.
4. T. Cover and P. Hart, "Nearest neighbor pattern classification," *IEEE Transactions on Information Theory*, vol. 13, no. 1, pp. 21–27, 1967.
