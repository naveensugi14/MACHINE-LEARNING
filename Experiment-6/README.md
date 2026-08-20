# Experiment 6: Bagging, Boosting, and Stacked Ensemble Models

**Course**: ICS1512 – Machine Learning Laboratory  
**Department**: Computer Science and Engineering, Sri Sivasubramaniya Nadar College of Engineering  
**Student Name**: NAVEENKUMAR S  
**Register Number**: 3122247001038  
**Faculty**: Dr. Poreddy Ajay Kumar Reddy  
**Submission Date**: 19/08/2026  

---

## 📌 Project Overview

This repository implements, evaluates, and compares advanced **Ensemble Learning Strategies**—specifically **Bagging (Bootstrap Aggregation)**, **Boosting (AdaBoost and Gradient Boosting)**, and **Stacked Ensembles (Stacking)**—applied to the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset for binary classification of tumor malignancy.

Ensemble methods combine multiple base models to construct a predictive framework superior to any individual estimator in terms of accuracy, stability, variance reduction, and bias mitigation.

---

## 🎯 Objectives

- Understand key ensemble learning paradigms: Bagging, Boosting, and Stacking.
- Implement a **Bagging Classifier** using Decision Trees as base estimators.
- Implement **Boosting Classifiers** using **AdaBoost** and **Gradient Boosting**.
- Construct a **Stacked Ensemble Classifier** combining heterogeneous base learners (SVM, Naïve Bayes, Decision Tree) with a **Logistic Regression** meta-learner.
- Explore hyperparameter search spaces using 5-fold cross-validation.
- Evaluate models using standard classification metrics: **Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC**.
- Analyze the bias-variance tradeoff across ensemble architectures.

---

## 📊 Dataset Summary

- **Dataset**: Wisconsin Diagnostic Breast Cancer (WDBC)
- **Source**: UCI Machine Learning Repository
- **Total Samples**: 569 instances
- **Features**: 30 continuous real-valued numerical attributes computed from digitized images of fine needle aspirates (FNA) of breast masses.
- **Target Classes**:
  - `M` (Malignant): 212 samples (37.26%) — Positive Class (1)
  - `B` (Benign): 357 samples (62.74%) — Negative Class (0)
- **Train-Test Split**: 80% Training (455 samples), 20% Testing (114 samples) with Stratification.

---

## ⚙️ Ensemble Model Architectures

1. **Bagging Classifier**:
   - **Base Estimator**: `DecisionTreeClassifier(max_depth=5)`
   - **Tuned Hyperparameters**: `n_estimators=50`, `max_samples=1.0`, `max_features=1.0`
   - **Mechanism**: Reduces model variance through parallel bootstrap sampling with replacement.

2. **AdaBoost Classifier**:
   - **Base Estimator**: Decision Tree Stumps (`max_depth=1`)
   - **Tuned Hyperparameters**: `n_estimators=100`, `learning_rate=0.1`
   - **Mechanism**: Sequential adaptive boosting that iteratively reweights misclassified training instances.

3. **Gradient Boosting Classifier**:
   - **Base Estimator**: Shallow Decision Trees
   - **Tuned Hyperparameters**: `n_estimators=100`, `learning_rate=0.05`, `max_depth=2`
   - **Mechanism**: Sequential gradient-based optimization minimizing loss function along negative gradient.

4. **Stacked Ensemble (Stacking)**:
   - **Heterogeneous Base Learners**: Support Vector Machine (`SVC`), Naïve Bayes (`GaussianNB`), Decision Tree (`DecisionTreeClassifier`)
   - **Meta-Learner**: `LogisticRegression(max_iter=1000)`
   - **Mechanism**: Out-of-fold cross-validation predictions from base estimators are concatenated as meta-features to train the meta-classifier.

---

## 📈 Performance Results

All models were evaluated on the independent test set (114 samples):

| Model | Accuracy (%) | Precision | Recall | F1-Score | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Bagging** | 95.61% | 0.9524 | 0.9302 | 0.9412 | 0.9866 |
| **AdaBoost** | **96.49%** | **0.9756** | 0.9302 | 0.9524 | 0.9921 |
| **Gradient Boosting** | 95.61% | 0.9524 | 0.9302 | 0.9412 | 0.9925 |
| **Stacked Ensemble** | **96.49%** | 0.9535 | **0.9535** | **0.9535** | **0.9971** |

### Key Inferences:
- **Best Generalization & Sensitivity**: The **Stacked Ensemble** achieved the highest ROC-AUC (**0.9971**) and highest Recall (**0.9535**), correctly identifying malignant cases with minimal false negatives.
- **Highest Precision**: **AdaBoost** achieved the highest Precision (**0.9756**), ensuring minimal false positive alarms.

---

## 🚀 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/username/repository.git
   cd repository
   ```

2. **Set up a Python virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Jupyter Notebook**:
   ```bash
   jupyter notebook Experiment-6.ipynb
   ```

---

## 📁 File Structure

```text
├── Experiment-6.ipynb                    # Complete Jupyter Notebook with code & execution output
├── Experiment_6_Lab_Record.tex           # Formal Academic Laboratory LaTeX Report
├── Experiment_6_Lab_Record.pdf           # Compiled PDF Lab Report
├── wdbc.data                             # Wisconsin Diagnostic Breast Cancer Dataset
├── requirements.txt                      # Python Dependencies
├── generate_exp6_assets.py               # Script generating high-res figures & metrics
├── exp6_assets/                          # Plots & Confusion Matrices
│   ├── class_distribution.png
│   ├── correlation_matrix.png
│   ├── cm_bagging.png
│   ├── cm_adaboost.png
│   ├── cm_gradient_boosting.png
│   ├── cm_stacking.png
│   └── roc_curves.png
└── README.md                             # Repository Documentation
```

---

## 📜 References

1. Breiman, L. (1996). "Bagging predictors." *Machine Learning*, 24(2), 123-140.
2. Freund, Y., & Schapire, R. E. (1997). "A decision-theoretic generalization of on-line learning and an application to boosting." *Journal of Computer and System Sciences*, 55(1), 119-139.
3. Wolpert, D. H. (1992). "Stacked generalization." *Neural Networks*, 5(2), 241-259.
4. Scikit-learn Documentation: [Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html).
