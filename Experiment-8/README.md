# Experiment 8: Clustering Human Activity Recognition Data

## Overview
This repository contains the complete implementation, dataset, analysis, and formal report for **Experiment 8: Clustering Human Activity Recognition Data using K-Means, DBSCAN, and Hierarchical Clustering**.

The objective is to group 561-dimensional sensor telemetry vectors extracted from 3-axis accelerometer and gyroscope signals into 6 human physical activity classes (*Walking*, *Walking Upstairs*, *Walking Downstairs*, *Sitting*, *Standing*, *Laying*) without ground truth supervision during model fitting.

## Directory Structure
```
exp8/
├── dataset/
│   ├── har_data.csv             # Feature matrix (10,299 samples × 561 features)
│   └── har_target.csv           # Ground truth activity class labels
├── experiment_8.py              # Standalone Python execution script
├── Experiment_8.ipynb           # Fully executed Jupyter Notebook
├── Experiment_8_Lab_Record.pdf  # Compiled 11-page PDF Report
├── Experiment_8_Lab_Record.tex  # LaTeX source code for report
├── requirements.txt             # Required Python dependencies
└── README.md                    # Experiment documentation
```

## Dataset Specifications
- **Dataset Name**: Human Activity Recognition Using Smartphones Dataset
- **Number of Samples**: 10,299 instances
- **Number of Features**: 561 continuous sensor features
- **Target Classes**: 6 physical activity states
- **Preprocessing**: Standardized using `StandardScaler` ($\mu=0, \sigma=1$)

## Algorithms Evaluated
1. **Model A: K-Means Clustering**: Partitioning algorithm using Euclidean distance and inertia minimization ($k=2 \dots 8$).
2. **Model B: DBSCAN**: Density-based clustering with noise identification ($\epsilon=15.0, 	ext{min\_samples}=10$).
3. **Model C: Hierarchical Agglomerative Clustering (HAC)**: Bottom-up tree hierarchy with Ward linkage ($k=6$).

## Setup and Execution Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Python Script
```bash
python experiment_8.py
```

### 3. Open Jupyter Notebook
```bash
jupyter notebook Experiment_8.ipynb
```

## Key Results Summary
| Algorithm | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index | ARI | NMI |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **K-Means ($k=6$)** | **0.1517** | **1.8342** | **1245.8** | **0.4312** | **0.5784** |
| DBSCAN ($\epsilon=15, 	ext{min}=10$) | 0.1205 | 2.4510 | 412.3 | 0.1845 | 0.2310 |
| **Hierarchical (Ward, $k=6$)** | 0.1428 | 1.9120 | 1182.4 | 0.4180 | 0.5621 |

## Author
- **Student Name**: Naveenkumar S
- **Register Number**: 3122247001038
- **GitHub Repository**: [https://github.com/naveensugi14/MACHINE-LEARNING](https://github.com/naveensugi14/MACHINE-LEARNING)
