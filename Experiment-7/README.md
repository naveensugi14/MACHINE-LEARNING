# Experiment 7: Dimensionality Reduction and Model Evaluation (With and Without PCA)

**Course:** ICS1512 Machine Learning Algorithms Laboratory  
**Author:** [Naveenkumar S](https://github.com/naveensugi14/MACHINE-LEARNING) (Register No: `3122247001038`)  
**Faculty:** Dr. Poreddy Ajay Kumar Reddy  
**Repository:** [https://github.com/naveensugi14/MACHINE-LEARNING](https://github.com/naveensugi14/MACHINE-LEARNING)

---

## 📌 Project Overview

This repository contains the complete experimental workflow, empirical benchmarks, visualization scripts, Jupyter notebooks, and LaTeX laboratory manual for **Experiment 7: Dimensionality Reduction and Model Evaluation (With and Without PCA)** conducted on the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset.

The core objective is to analyze the mathematical and empirical impacts of dimensionality reduction using **Principal Component Analysis (PCA)** across ten diverse machine learning algorithms under **5-Fold Stratified Cross-Validation** and comprehensive hyperparameter optimization.

---

## 📊 Dataset Description

- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC) Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **Total Samples:** 569 instances
- **Total Features:** 30 continuous numerical attributes (10 mean metrics, 10 standard error metrics, and 10 worst metrics)
- **Target Variable:** Binary Diagnosis — `Benign (B)`: 357 samples (62.74%) | `Malignant (M)`: 212 samples (37.26%)
- **Missing Values:** 0
- **Train/Test Split:** 80% Training ($N_{\text{train}} = 455$) / 20% Testing ($N_{\text{test}} = 114$), Stratified (`random_state=42`)
- **Scaling:** Standardized using `StandardScaler` ($\mu = 0, \sigma = 1$)

---

## 🔬 PCA Dimensionality Reduction Summary

- **Total Original Features:** 30
- **Target Variance Threshold:** $\ge 95\%$
- **Optimal Principal Components ($k$):** **10 Components**
- **Cumulative Explained Variance Retained:** **95.21%**
- **Dimensionality Reduction Ratio:** **66.7% compression** (from 30 down to 10 features)
- **Primary Benefit:** Effectively eliminates severe pairwise multi-collinearity ($\rho > 0.90$) among nuclear morphometric descriptors (radius, perimeter, area) while accelerating model convergence.

---

## 🤖 Models Evaluated

1. **Support Vector Machine (SVM)** (`linear`, `rbf` kernels; $C \in [0.1, 1, 10, 100]$)
2. **Gaussian Naïve Bayes** ($\text{var\_smoothing} \in [10^{-11}, \dots, 10^{-7}]$)
3. **k-Nearest Neighbors (KNN)** ($k \in [3, 5, 7, 9, 11]$; `uniform`, `distance`; `euclidean`, `manhattan`)
4. **Logistic Regression** ($C \in [0.01, 0.1, 1, 10, 100]$; `lbfgs` solver)
5. **Decision Tree (CART)** ($\text{max\_depth} \in [\text{None}, 3, 5, 10, 15]$; $\text{min\_samples\_split} \in [2, 5, 10]$)
6. **Random Forest** ($n_{\text{estimators}} \in [100, 200]$; $\text{max\_depth} \in [\text{None}, 5, 10, 15]$)
7. **AdaBoost** ($n_{\text{estimators}} \in [50, 100, 200]$; $\text{learning\_rate} \in [0.01, 0.1, 1.0]$)
8. **Gradient Boosting** ($n_{\text{estimators}} \in [50, 100, 200]$; $\text{learning\_rate} \in [0.01, 0.1, 0.2]$; $\text{max\_depth} \in [2, 3, 5]$)
9. **XGBoost** ($n_{\text{estimators}} \in [50, 100, 200]$; $\text{learning\_rate} \in [0.01, 0.1, 0.2]$; $\text{max\_depth} \in [3, 5, 7]$)
10. **Stacked Ensemble** (Base: SVM + KNN + Random Forest; Meta-Learner: Logistic Regression with $C \in [0.01, 0.1, 1, 10, 100]$)

---

## 📈 Performance Summary: No-PCA vs. With-PCA

| Classifier | No-PCA 5-Fold CV | With-PCA 5-Fold CV | No-PCA Test Acc | With-PCA Test Acc | With-PCA Test F1 | With-PCA Test ROC-AUC | Optimal Hyperparameters (With-PCA) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **SVM** | $97.58\%$ | $97.58\%$ | $97.37\%$ | $96.49\%$ | $0.9487$ | **$0.9954$** | $C=1, \text{kernel}=\text{linear}$ |
| **Naïve Bayes** | $93.85\%$ | $92.09\%$ | $92.11\%$ | $89.47\%$ | $0.8571$ | $0.9613$ | $\text{var\_smoothing}=10^{-11}$ |
| **KNN** | $96.92\%$ | $96.70\%$ | $93.86\%$ | **$95.61\%$** $\uparrow$ | **$0.9351$** | **$0.9838$** | $k=5, \text{metric}=\text{euclidean}$ |
| **Logistic Regression** | $97.14\%$ | **$97.58\%$** $\uparrow$ | $96.49\%$ | **$97.37\%$** $\uparrow$ | **$0.9620$** | **$0.9970$** (Best) | $C=1.0$ |
| **Decision Tree** | $93.85\%$ | $93.41\%$ | $93.86\%$ | **$95.61\%$** $\uparrow$ | **$0.9367$** | **$0.9603$** | $\text{depth}=\text{None}, \text{split}=10, \text{leaf}=4$ |
| **Random Forest** | $96.26\%$ | $95.82\%$ | $97.37\%$ | $93.86\%$ | $0.9136$ | $0.9926$ | $n_{\text{est}}=100, \text{depth}=\text{None}$ |
| **AdaBoost** | $96.48\%$ | $96.48\%$ | $97.37\%$ | **$97.37\%$** | **$0.9620$** | **$0.9917$** | $n_{\text{est}}=50, \text{learning\_rate}=1.0$ |
| **Gradient Boosting** | $96.04\%$ | $95.60\%$ | $96.49\%$ | $94.74\%$ | $0.9231$ | $0.9931$ | $n_{\text{est}}=50, \eta=0.1, \text{depth}=2$ |
| **XGBoost** | $97.14\%$ | $96.92\%$ | $97.37\%$ | **$97.37\%$** | **$0.9620$** | **$0.9921$** | $n_{\text{est}}=200, \eta=0.2, \text{depth}=3$ |
| **Stacked Ensemble** | $97.36\%$ | $96.92\%$ | $95.61\%$ | **$96.49\%$** $\uparrow$ | **$0.9487$** | **$0.9954$** | $\text{meta\_C}=1.0$ |

---

## 🔍 Key Findings & Discussion

1. **Distance-Based & Metric Learners (KNN):**
   - Test accuracy increased by **$+1.75\%$** ($93.86\% \rightarrow 95.61\%$). In 30 dimensions, Euclidean distance is corrupted by noisy, correlated attributes (curse of dimensionality). In the 10-dimensional orthogonal subspace, pairwise distances accurately reflect true cluster geometry.
2. **Linear Hyperplane Models (Logistic Regression):**
   - Test accuracy increased to **$97.37\%$** and achieved the highest overall ROC-AUC of **$0.9970$**. Removing extreme geometric multi-collinearity ($\rho \approx 0.99$) regularized the Hessian matrix and prevented weight explosion.
3. **Variance Reduction Across Folds:**
   - Unpruned Decision Trees experienced a reduction in cross-validation standard deviation from $\sigma = 3.61\%$ down to $\sigma = 2.94\%$, confirming that PCA projection reduces fold-level overfitting.
4. **Ensemble & Boosted Learners (AdaBoost & XGBoost):**
   - Maintained joint-top accuracy of **$97.37\%$** and near-perfect sensitivity on only 10 principal components while training up to $3\times$ faster.

---

## 📁 Repository Structure

```
├── Experiment-7.ipynb          # Comprehensive Jupyter Notebook with all experiments
├── run_experiment7.py          # Standalone Python runner script for all 10 models
├── generate_all_plots.py       # High-resolution figure generator
├── generate_latex.py           # Programmatic generator for Manual.tex
├── Manual.tex                  # Complete LaTeX source code for the lab report
├── Manual.pdf                  # Compiled 15-page publication-quality PDF report
├── requirements.txt            # Python environment dependencies
├── results.json                # Raw cross-validation and test set evaluation metrics
├── wdbc.data                   # WDBC raw dataset
├── wdbc.names                  # WDBC attribute description file
└── figures/                    # High-resolution generated plots (300 DPI)
    ├── fig1_class_distribution.png
    ├── fig2_correlation_heatmap.png
    ├── fig3_scree_plot.png
    ├── fig4_cumulative_variance.png
    ├── fig5_pca_2d_projection.png
    ├── fig6_cv_accuracy_comparison.png
    ├── fig7_test_auc_comparison.png
    ├── fig8_confusion_matrix_svm.png
    ├── fig9_confusion_matrix_logreg.png
    ├── fig10_confusion_matrix_xgboost.png
    ├── fig11_confusion_matrix_stacking.png
    ├── fig12_roc_curves.png
    └── fig13_precision_recall_curves.png
```

---

## 🚀 How to Run & Reproduce

### 1. Set Up Python Environment
```bash
# Clone repository
git clone https://github.com/naveensugi14/MACHINE-LEARNING.git
cd MACHINE-LEARNING

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Experiments & Generate Figures
```bash
# Run all model training and evaluation
python run_experiment7.py

# Generate all 13 publication-quality figures
python generate_all_plots.py
```

### 3. Compile LaTeX PDF Report
```bash
# Generate Manual.tex
python generate_latex.py

# Compile to PDF using pdflatex
pdflatex -interaction=nonstopmode Manual.tex
pdflatex -interaction=nonstopmode Manual.tex
```

---

## 📚 References

1. W. H. Wolberg, W. N. Street, and O. L. Mangasarian, "Breast Cancer Wisconsin (Diagnostic) Data Set," *UCI Machine Learning Repository*, 1995.
2. F. Pedregosa *et al.*, "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
3. K. Pearson, "On lines and planes of closest fit to systems of points in space," *Philosophical Magazine*, vol. 2, no. 11, pp. 559–572, 1901.
4. T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. ACM SIGKDD*, 2016.
5. D. H. Wolpert, "Stacked generalization," *Neural Networks*, vol. 5, no. 2, pp. 241–259, 1992.
