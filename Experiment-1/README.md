# Machine Learning Laboratory Experiments

**Course**: ICS1512 – Machine Learning Algorithms Laboratory  
**Department**: Computer Science and Engineering, Sri Sivasubramaniya Nadar College of Engineering  
**Student Name**: Naveenkumar S  
**Register Number**: 3122247001038  
**Faculty**: Dr. Poreddy Ajay Kumar Reddy  
**Submission Date**: 28/07/2026 – 02/08/2026  
**GitHub Repository**: [MACHINE-LEARNING](https://github.com/naveensugi14/MACHINE-LEARNING)

---

## 📌 Repository Overview

This repository contains comprehensive implementations, automated EDA utilities, evaluations, and comparative studies of supervised machine learning algorithms:

1. **[Experiment 1: Exploratory Data Analysis & Data Preprocessing Utility (`perform_eda.ipynb`)](#-experiment-1-exploratory-data-analysis--data-preprocessing)**
2. **[Experiment 3: Regression Analysis using Linear and Regularized Models](#-experiment-3-regression-analysis-using-linear-and-regularized-models)**
3. **[Experiment 4: Binary Classification using Linear and Kernel-Based Models](#-experiment-4-binary-classification-using-linear-and-kernel-based-models)**
4. **[Experiment 2: Email Spam Classification using Naïve Bayes and K-Nearest Neighbors (KNN)](#-experiment-2-email-spam-classification-using-naïve-bayes-and-k-nearest-neighbors-knn)**

---

## 🧪 Experiment 1: Exploratory Data Analysis & Data Preprocessing

### 📌 Overview & Aim
Develop and execute a universal Exploratory Data Analysis function (`perform_eda(file_path)` in `perform_eda.ipynb`) across **5 multi-domain benchmark datasets** using NumPy, Pandas, SciPy, Matplotlib, and Seaborn:
- Automated dataset structure inspection (`df.shape`, `df.info()`, `df.describe()`).
- Data hygiene audits: missing value counting (`df.isnull().sum()`) and duplicate row detection (`df.duplicated().sum()`).
- Domain-specific visual diagnostic plots: histograms with KDE, boxplots, violin plots, scatter plots, correlation heatmaps, pairplots, QQ plots for normality audit, and 2D spatial image grid renderings ($28 \times 28$ grayscale matrices).

### 📊 Benchmark Datasets Evaluated

| Dataset Name | Domain / ML Task | Samples | Features | Key Visual Diagrams | Primary EDA Finding / Takeaway |
|:---|:---|:---:|:---:|:---|:---|
| **Iris Dataset** | Multi-Class Classification | 150 | 4 continuous + 1 target | Histograms, Boxplots, Scatter Plots, Heatmap, Pairplot | *Iris-setosa* is linearly separable along petal dimensions ($r=0.96$). |
| **Loan Amount Prediction** | Regression / Classification | 4,269 | 12 continuous + 1 target | Histograms, Boxplots, Scatter Plot, Heatmap, Pairplot | High correlation ($r=0.93$) between income & loan amount; asset scale variance requires `StandardScaler`. |
| **Predicting Diabetes** | Binary Clinical Classification | 768 | 8 continuous + 1 target | Histograms, Boxplots, Scatter Plot, Heatmap, Pairplot | Identified physiological zero anomalies (zero Insulin, Glucose, BMI) representing missing clinical data. |
| **Email Spam Classification** | Binary Text Classification | 5,572 | Text Payload Attributes | Class Bar Chart, Histograms, Boxplots, QQ Plots | Severe class imbalance (87% ham vs 13% spam); spam emails exhibit longer message lengths. |
| **Handwritten Recognition (MNIST)** | Multi-Class Image Classification | 10,000 | 784 pixels + 1 target | Class Distribution Bar, $28 \times 28$ Grayscale Render, Pixel Boxplots | Reshaping flat 784-vectors into $28 \times 28$ matrices restores spatial digit stroke geometry. |

---

## 🧪 Experiment 3: Regression Analysis using Linear and Regularized Models

### 📌 Overview & Objectives
Predict continuous loan request amounts on the Kaggle "Predict Loan Amount Data" dataset (20,000 records, target range \$6,185.48 to \$576,335.68) using unregularized (**Linear Regression**) and regularized (**Ridge, Lasso, Elastic Net**) regression models.

### ⚙️ Models & Results Summary ($N_{\text{test}} = 4,000$, 80-20 Split)

| Model | Search Method | Best Hyperparameters | Mean 5-Fold CV $R^2$ | Test MAE (\$) | Test RMSE (\$) | Test $R^2$ Score | Train Time |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Linear Regression** | Baseline | None | 0.930188 | \$10,891.31 | \$15,263.10 | 0.935589 | 0.0178s |
| **Ridge Regression** | `GridSearchCV` | `alpha = 10` | 0.930192 | \$10,891.34 | \$15,263.13 | 0.935589 | **0.0168s** |
| **Lasso Regression** | `GridSearchCV` | `alpha = 100` | **0.930243** | **\$10,891.18** | **\$15,262.93** | **0.935591** | 1.8107s |
| **Elastic Net (Tuned)** | `GridSearchCV` | `alpha = 0.01, l1_ratio = 0.8` | 0.930190 | \$10,891.25 | \$15,263.02 | 0.935590 | 0.1160s |

#### 💡 Key Inferences
- `Property Price` is the strongest predictor of requested loan amount ($r=0.96$).
- **Lasso ($\alpha=100$)** offers the sparsest interpretable model by driving non-informative feature weights (`Age`, `Dependents`) to zero without sacrificing accuracy.

---

## 🧪 Experiment 4: Binary Classification using Linear and Kernel-Based Models

### 📌 Overview & Objectives
Evaluates probabilistic linear (**Logistic Regression**) and margin-based kernel (**Support Vector Machine - SVM**) classifiers on the UCI Spambase dataset (4,210 clean records).

### ⚙️ Models & Results Summary ($N_{\text{test}} = 842$)

| Model | Hyperparameters | 5-Fold CV Accuracy | Test Accuracy | Precision | F1-Score | ROC-AUC | Train Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression (Tuned)** | $C=1, \text{penalty}=\text{'l2'}$ | 92.83% | 92.83% | 0.9176 | 0.9076 | 0.9754 | **0.0481s** |
| **SVM (Linear Kernel)** | Default | - | 92.51% | 0.9125 | 0.9038 | - | 0.1245s |
| **SVM (Poly Kernel, $d=3$)** | Default | - | 89.25% | 0.8841 | 0.8521 | - | 0.3812s |
| **SVM (RBF Kernel, Tuned)** | $C=10, \gamma=\text{'scale'}$ | **93.41%** | **93.41%** | **0.9258** | **0.9160** | **0.9812** | 0.2215s |
| **SVM (Sigmoid Kernel)** | Default | - | 85.42% | 0.8124 | 0.8105 | - | 0.4510s |

---

## 🧪 Experiment 2: Email Spam Classification using Naïve Bayes and K-Nearest Neighbors (KNN)

### 📌 Overview & Objectives
Compares Naïve Bayes variants (**Gaussian NB**, **Multinomial NB**, **Bernoulli NB**) and instance-based **K-Nearest Neighbors (KNN)** on the UCI Spambase dataset.

### ⚙️ Models & Results Summary ($N_{\text{test}} = 842$)

| Model | Hyperparameters | 5-Fold CV Accuracy | Test Accuracy | Precision (Ham) | Precision (Spam) | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Gaussian NB** | Default | 83.25% | 84.68% | 0.97 | 0.75 | 0.957 |
| **Multinomial NB** | $\alpha=1.0$ | 78.62% | 80.40% | 0.82 | 0.79 | 0.884 |
| **Bernoulli NB** | $\alpha=1.0$ | 88.27% | 88.12% | 0.87 | 0.90 | 0.949 |
| **KNN ($k=5$, Scaled)** | $n\_neighbors = 5$ | **90.29%** | **88.36%** | **0.88** | **0.89** | **0.935** |

---

## 📊 Master Overview Table of All Experiments

| Exp # | Task Focus | Datasets / Models Evaluated | Key Outcome / Best Performer | Top Metric |
|:---:|:---|:---|:---|:---:|
| **Exp 1** | Universal EDA Utility & Preprocessing | Iris, Loan, Diabetes, Spam, MNIST | Universal function `perform_eda(file_path)` | 5 Multi-Domain Benchmark Audits |
| **Exp 3** | Loan Amount Prediction (Regression) | Linear, Ridge, Lasso, Elastic Net | **Lasso ($\alpha=100$)** | $R^2 = \mathbf{0.9356}$, $\text{MAE} = \mathbf{\$10,891}$ |
| **Exp 4** | Email Spam Classification (Kernel/Linear) | Logistic Regression, SVM (Linear, Poly, RBF, Sigmoid) | **SVM (RBF Kernel)** | Accuracy = $\mathbf{93.41\%}$, $\text{AUC} = \mathbf{0.9812}$ |
| **Exp 2** | Email Spam Classification (NB & KNN) | Gaussian, Multinomial, Bernoulli NB, KNN ($k=5, 7$) | **KNN ($k=5$, Scaled)** | Accuracy = $\mathbf{88.36\%}$, CV Acc = $\mathbf{90.29\%}$ |

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
   jupyter notebook perform_eda.ipynb
   # or
   jupyter notebook "Experiment -3.ipynb"
   # or
   jupyter notebook "Experiment -4.ipynb"
   # or
   jupyter notebook "Experiment-2.ipynb"
   ```

---

## 📁 Repository Structure

```text
├── perform_eda.ipynb                     # Universal EDA Script for Experiment 1
├── Experiment -3.ipynb                   # Complete Jupyter Notebook for Experiment 3 (Regression)
├── Experiment -4.ipynb                   # Complete Jupyter Notebook for Experiment 4 (LR & SVM)
├── Experiment-2.ipynb                    # Complete Jupyter Notebook for Experiment 2 (NB & KNN)
├── requirement.txt                       # Environment Dependencies
├── requirements.txt                      # Environment Dependencies (Alias)
└── README.md                             # Comprehensive Laboratory Documentation
```

---

## 📜 References

1. Kaggle, "Predict Loan Amount Data," 2021.
2. M. Hopkins, E. Reeber, G. Forman, and J. Suermondt, "Spambase Data Set," *UCI Machine Learning Repository*, 1999.
3. R. Fisher, "The use of multiple measurements in taxonomic problems," *Annals of Eugenics*, 1936.
4. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
5. C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, no. 3, pp. 273–297, 1995.
6. T. M. Mitchell, *Machine Learning*. McGraw-Hill, 1997.
7. D. W. Hosmer, S. Lemeshow, and R. X. Sturdivant, *Applied Logistic Regression*, 3rd ed. John Wiley & Sons, 2013.
