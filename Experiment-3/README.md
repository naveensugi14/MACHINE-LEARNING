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

This repository contains comprehensive implementations, evaluations, and comparative studies of supervised machine learning algorithms:

1. **[Experiment 3: Regression Analysis using Linear and Regularized Models](#-experiment-3-regression-analysis-using-linear-and-regularized-models)**
2. **[Experiment 4: Binary Classification using Linear and Kernel-Based Models](#-experiment-4-binary-classification-using-linear-and-kernel-based-models)**
3. **[Experiment 2: Email Spam Classification using Naïve Bayes and K-Nearest Neighbors (KNN)](#-experiment-2-email-spam-classification-using-naïve-bayes-and-k-nearest-neighbors-knn)**

---

## 🧪 Experiment 3: Regression Analysis using Linear and Regularized Models

### 📌 Overview & Objectives
This experiment predicts continuous loan request amounts using unregularized (**Linear Regression**) and regularized (**Ridge, Lasso, Elastic Net**) regression models on the Kaggle "Predict Loan Amount Data" dataset. Key objectives:
- Perform data preprocessing: missing value imputation (median for continuous, mode for categorical), label encoding, and `StandardScaler` feature scaling.
- Implement baseline **Linear Regression (OLS)** and regularized variants (**Ridge $L_2$**, **Lasso $L_1$**, **Elastic Net $L_1+L_2$**).
- Execute hyperparameter tuning via 5-fold cross-validation (`GridSearchCV`).
- Evaluate predictive performance using **MAE, MSE, RMSE, and $R^2$ Score**.
- Analyze feature coefficient shrinkage, model sparsity, collinearity handling, and bias-variance characteristics.

### 📊 Dataset Summary
- **Dataset Name**: Predict Loan Amount Data (Kaggle)
- **Total Records**: 20,000 applicants
- **Features**: 22 input features + 1 continuous target variable (`Loan Amount Request (USD)` ranging from \$6,185.48 to \$576,335.68)
- **Missing Value Imputation**: Median imputation for numerical features; mode imputation for categorical attributes.
- **Train-Test Split**: 80% Training (16,000 samples), 20% Testing (4,000 samples) with `random_state=42`.
- **Feature Scaling**: `StandardScaler` fit on training split and applied to test split.

### ⚙️ Model Architectures & Formulations

#### 1. Ordinary Least Squares (OLS) Linear Regression
- **Mathematical Formulation**: Fits a hyperplane minimizing sum of squared residuals:
  $$\hat{y} = \beta_0 + \sum_{i=1}^n \beta_i x_i, \quad \min_{\mathbf{\beta}} \sum_{j=1}^m (y_j - \hat{y}_j)^2$$
- **Performance**: Test $R^2$: **0.935589**, MAE: **\$10,891.31**, RMSE: **\$15,263.10**, Training Time: **0.0178s**.

#### 2. Ridge Regression ($L_2$ Regularization)
- **Mathematical Formulation**: Adds quadratic $L_2$ penalty to shrink weights and handle multicollinearity:
  $$\min_{\mathbf{\beta}} \sum_{j=1}^m (y_j - \hat{y}_j)^2 + \alpha \sum_{i=1}^n \beta_i^2$$
- **Tuned Hyperparameters (`GridSearchCV`)**: Best $\alpha = 10$ (Mean CV $R^2$: **0.930192**).
- **Performance**: Test $R^2$: **0.935589**, MAE: \$10,891.34, RMSE: \$15,263.13, Training Time: **0.0168s**.

#### 3. Lasso Regression ($L_1$ Regularization)
- **Mathematical Formulation**: Adds absolute $L_1$ penalty to drive uninformative feature coefficients strictly to zero:
  $$\min_{\mathbf{\beta}} \sum_{j=1}^m (y_j - \hat{y}_j)^2 + \alpha \sum_{i=1}^n |\beta_i|$$
- **Tuned Hyperparameters (`GridSearchCV`)**: Best $\alpha = 100$ (Mean CV $R^2$: **0.930243**).
- **Performance**: Test $R^2$: **0.935591**, MAE: **\$10,891.18**, RMSE: **\$15,262.93**, Training Time: 1.8107s.

#### 4. Elastic Net Regression ($L_1 + L_2$ Regularization)
- **Mathematical Formulation**: Combines $L_1$ and $L_2$ penalties with mixing parameter $r \in [0, 1]$ (`l1_ratio`):
  $$\min_{\mathbf{\beta}} \sum_{j=1}^m (y_j - \hat{y}_j)^2 + \alpha \left( r \sum_{i=1}^n |\beta_i| + \frac{1-r}{2} \sum_{i=1}^n \beta_i^2 \right)$$
- **Tuned Hyperparameters (`GridSearchCV`)**: Best $\alpha = 0.01$, `l1_ratio = 0.8` (Mean CV $R^2$: **0.930190**).
- **Performance**: Test $R^2$: **0.935590**, MAE: \$10,891.25, RMSE: \$15,263.02, Training Time: 0.1160s.

### 📈 Performance Results & Evaluation

#### 1. Hyperparameter Tuning & 5-Fold CV Summary ($K=5$)

| Model | Search Method | Best Hyperparameters | Mean 5-Fold CV $R^2$ |
|:---|:---|:---:|:---:|
| **Ridge Regression** | `GridSearchCV` | `alpha = 10` | 0.930192 |
| **Lasso Regression** | `GridSearchCV` | `alpha = 100` | **0.930243** |
| **Elastic Net Regression** | `GridSearchCV` | `alpha = 0.01, l1_ratio = 0.8` | 0.930190 |

#### 2. Test Set Evaluation Summary ($N_{\text{test}} = 4,000$, 80-20 Split)

| Model | MAE (\$) | MSE ($\times 10^8$) | RMSE (\$) | Test $R^2$ Score | Train Time (s) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Linear Regression** | \$10,891.31 | 2.3296 | \$15,263.10 | 0.935589 | 0.0178s |
| **Ridge Regression ($\alpha=10$)** | \$10,891.34 | 2.3296 | \$15,263.13 | 0.935589 | **0.0168s** |
| **Lasso Regression ($\alpha=100$)** | **\$10,891.18** | **2.3296** | **\$15,262.93** | **0.935591** | 1.8107s |
| **Elastic Net (Tuned)** | \$10,891.25 | 2.3296 | \$15,263.02 | 0.935590 | 0.1160s |
| **Default Elastic Net (Untuned)** | \$17,137.95 | 5.1655 | \$22,727.76 | 0.857181 | 0.1160s |

#### 💡 Key Inferences & Insights
1. **Dominant Predictor**: `Property Price` shows a strong linear correlation ($r=0.96$) with loan request amount and dominates feature weight magnitudes across all models.
2. **Best Model Selection**: **Lasso Regression ($\alpha=100$)** achieved the best overall MAE (\$10,891.18) and highest $R^2$ (0.935591) while zeroing out uninformative features (`Age`, `Dependents`) to create a sparse, interpretable model.
3. **Hyperparameter Tuning Impact**: Grid search tuning for Elastic Net boosted test $R^2$ from 0.8572 to **0.9356** and reduced MAE by over \$6,246.
4. **No Overfitting**: All models demonstrated minimal gap between train $R^2$ (0.9356) and 5-fold CV $R^2$ (0.9302), confirming excellent generalization.

---

## 🧪 Experiment 4: Binary Classification using Linear and Kernel-Based Models

### 📌 Overview & Objectives
Evaluates probabilistic linear (**Logistic Regression**) and margin-based kernel (**Support Vector Machine - SVM**) classifiers on the UCI Spambase dataset.

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

## 📊 Summary of All Implemented Experiments

| Exp # | Focus / Task | Models Evaluated | Best Model | Top Metric Achieved |
|:---:|:---|:---|:---|:---:|
| **Exp 3** | Loan Amount Request Prediction (Regression) | Linear, Ridge, Lasso, Elastic Net | **Lasso ($\alpha=100$)** | $R^2 = \mathbf{0.9356}$, $\text{MAE} = \mathbf{\$10,891}$ |
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
   jupyter notebook "Experiment -3.ipynb"
   # or
   jupyter notebook "Experiment -4.ipynb"
   # or
   jupyter notebook "Experiment-2.ipynb"
   ```

---

## 📁 Repository Structure

```text
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
3. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
4. C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, no. 3, pp. 273–297, 1995.
5. T. M. Mitchell, *Machine Learning*. McGraw-Hill, 1997.
6. D. W. Hosmer, S. Lemeshow, and R. X. Sturdivant, *Applied Logistic Regression*, 3rd ed. John Wiley & Sons, 2013.
