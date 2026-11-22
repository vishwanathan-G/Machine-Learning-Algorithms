# Machine Learning Algorithms

Practice implementations of core machine learning algorithms, written as Jupyter notebooks while learning the subject. The notebooks cover supervised learning (regression and classification), unsupervised learning (clustering and dimensionality reduction), model evaluation and hyperparameter tuning, and an introduction to big-data processing and machine learning with PySpark.

Most notebooks use scikit-learn, pandas and matplotlib on small datasets, so that each step can be followed and checked by hand. Every algorithm folder has its own README with the theory (definition, how it works, mathematical intuition, advantages, disadvantages, applications), the exact settings used in the notebook, how to run it and the output to expect.

## Table of Contents

- [Repository Structure](#repository-structure)
- [Algorithm Index](#algorithm-index)
- [Getting Started](#getting-started)
- [Interactive Notebooks](#interactive-notebooks)
- [Verification](#verification)
- [Notes on This Repository](#notes-on-this-repository)
- [Author](#author)

## Repository Structure

```
Machine-Learning-Algorithms/
├── .gitignore
├── README.md
├── requirements.txt
├── datasets/
│   └── README.md                         index of every dataset used
├── 01_Supervised_Learning/
│   ├── Linear_Regression/
│   │   ├── README.md
│   │   ├── 1_linear_regression.ipynb
│   │   ├── 1_1linear_regression.ipynb
│   │   ├── 2_linear_regression_multivariate.ipynb
│   │   ├── multi_linear_reg.ipynb
│   │   ├── houseprice.csv
│   │   └── homeprices.csv
│   ├── Polynomial_Regression/
│   │   ├── README.md
│   │   └── poly_reg.ipynb
│   ├── Logistic_Regression/
│   │   ├── README.md
│   │   ├── 3_logistic_regression.ipynb
│   │   └── insurance_data.csv
│   ├── Decision_Tree/
│   │   ├── README.md
│   │   └── 4_decision tree.ipynb
│   ├── Random_Forest/
│   │   ├── README.md
│   │   └── Random_forest.ipynb
│   ├── KNN/
│   │   ├── README.md
│   │   ├── Knn-1.ipynb
│   │   └── data/
│   │       └── iris-flower-dataset.csv
│   └── SVM/
│       ├── README.md
│       ├── 5_svm_car_data.ipynb
│       └── cars_dataset.csv
├── 02_Unsupervised_Learning/
│   ├── KMeans/
│   │   ├── README.md
│   │   ├── 7_k_means.ipynb
│   │   ├── kmeans_50000_clustering.ipynb
│   │   └── kmeans_50000_data.csv
│   ├── KMedoids/
│   │   ├── README.md
│   │   └── 8_k_medoids.ipynb
│   ├── Hierarchical_Clustering/
│   │   ├── README.md
│   │   └── 9_agg.ipynb
│   ├── DBSCAN/
│   │   ├── README.md
│   │   ├── 10_dbscan.ipynb
│   │   └── DBSCAN_Clustering_Practical.ipynb
│   └── PCA/
│       ├── README.md
│       └── 12_PCA.ipynb
├── 03_Model_Evaluation_and_Tuning/
│   ├── Classification_Metrics/
│   │   ├── README.md
│   │   └── 11_evalution-1.ipynb
│   ├── Cross_Validation/
│   │   ├── README.md
│   │   ├── model evalution.ipynb
│   │   └── iris.csv
│   └── Hyperparameter_Tuning/
│       ├── README.md
│       └── 6_hyperparameter_tuning.ipynb
└── 04_Big_Data_PySpark/
    ├── README.md
    ├── RDD.ipynb
    ├── Dataframe.ipynb
    ├── sql.ipynb
    └── PySparkML.ipynb
```

Each CSV file sits next to the notebook that reads it. [`datasets/README.md`](datasets/README.md) lists every dataset, including the built-in scikit-learn datasets and the inline data.

## Algorithm Index

| Category | Algorithm / topic | Folder | Notebooks | Dataset |
|---|---|---|---|---|
| Supervised: regression | Linear Regression (simple and multiple) | [Linear_Regression](01_Supervised_Learning/Linear_Regression) | `1_linear_regression`, `1_1linear_regression`, `2_linear_regression_multivariate`, `multi_linear_reg` | `houseprice.csv`, `homeprices.csv` (50,000 rows), inline CO2-emissions data |
| Supervised: regression | Polynomial Regression | [Polynomial_Regression](01_Supervised_Learning/Polynomial_Regression) | `poly_reg` | Inline experience/salary data |
| Supervised: classification | Logistic Regression | [Logistic_Regression](01_Supervised_Learning/Logistic_Regression) | `3_logistic_regression` | `insurance_data.csv` |
| Supervised: classification | Decision Tree | [Decision_Tree](01_Supervised_Learning/Decision_Tree) | `4_decision tree` | Inline 10-row categorical data |
| Supervised: classification | Random Forest | [Random_Forest](01_Supervised_Learning/Random_Forest) | `Random_forest` | scikit-learn Wine (`load_wine`) |
| Supervised: classification | K-Nearest Neighbors | [KNN](01_Supervised_Learning/KNN) | `Knn-1` | `data/iris-flower-dataset.csv` |
| Supervised: classification | Support Vector Machine | [SVM](01_Supervised_Learning/SVM) | `5_svm_car_data` | `cars_dataset.csv` |
| Unsupervised: clustering | K-Means | [KMeans](02_Unsupervised_Learning/KMeans) | `7_k_means`, `kmeans_50000_clustering` | Inline 8 points; `kmeans_50000_data.csv` (50,000 rows) |
| Unsupervised: clustering | K-Medoids | [KMedoids](02_Unsupervised_Learning/KMedoids) | `8_k_medoids` | Inline 10 points |
| Unsupervised: clustering | Hierarchical (Agglomerative) Clustering | [Hierarchical_Clustering](02_Unsupervised_Learning/Hierarchical_Clustering) | `9_agg` | Inline 15 points |
| Unsupervised: clustering | DBSCAN | [DBSCAN](02_Unsupervised_Learning/DBSCAN) | `10_dbscan`, `DBSCAN_Clustering_Practical` | Generated two-moons data (`make_moons`) |
| Unsupervised: dimensionality reduction | Principal Component Analysis | [PCA](02_Unsupervised_Learning/PCA) | `12_PCA` | scikit-learn Digits (`load_digits`) |
| Model evaluation | Confusion matrix, precision, recall, F1 | [Classification_Metrics](03_Model_Evaluation_and_Tuning/Classification_Metrics) | `11_evalution-1` | Inline label lists |
| Model evaluation | Hold-out, bootstrap, K-fold cross-validation | [Cross_Validation](03_Model_Evaluation_and_Tuning/Cross_Validation) | `model evalution` | `iris.csv`, `load_iris`, `load_wine` |
| Model tuning | GridSearchCV, RandomizedSearchCV | [Hyperparameter_Tuning](03_Model_Evaluation_and_Tuning/Hyperparameter_Tuning) | `6_hyperparameter_tuning` | scikit-learn Iris (`load_iris`) |
| Big data | PySpark RDD, DataFrame, Spark SQL, MLlib regression and classification | [04_Big_Data_PySpark](04_Big_Data_PySpark) | `RDD`, `Dataframe`, `sql`, `PySparkML` | Inline data; salary CSV (not included) |

## Getting Started

### Requirements

- Python 3. `requirements.txt` was checked against Python 3.10.11. Use Python 3.10 if you also want to run the K-Medoids notebook (see the optional extras below).
- Git, to clone the repository.

### 1. Clone the repository

```
git clone https://github.com/vishwanathan-G/Machine-Learning-Algorithms.git
cd Machine-Learning-Algorithms
```

### 2. Create and activate a virtual environment

```
python -m venv .venv
```

Activate it:

| Shell | Command |
|---|---|
| Windows Command Prompt | `.venv\Scripts\activate` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| macOS / Linux | `source .venv/bin/activate` |

If PowerShell blocks the activation script, you can skip activation and call the environment's Python directly: `.venv\Scripts\python -m pip ...` and `.venv\Scripts\python -m jupyter notebook`.

### 3. Install the dependencies

```
python -m pip install -r requirements.txt
```

This installs numpy, pandas, matplotlib, seaborn, scikit-learn, scipy, joblib and Jupyter.

### 4. Optional extras

These are needed only for specific folders and are listed as comments in `requirements.txt`:

| Folder | Install | Notes |
|---|---|---|
| `02_Unsupervised_Learning/KMedoids` | `python -m pip install "numpy<2" "scikit-learn-extra>=0.3.0"` | scikit-learn-extra 0.3.0 fails to import with NumPy 2.x and ships wheels only for older Python versions; use a separate Python 3.10 environment |
| `04_Big_Data_PySpark` | `python -m pip install "pyspark>=3.4,<4"` | Also needs a Java runtime (Java 8, 11 or 17) with `JAVA_HOME` set. The notebooks were written in Google Colab; see [the folder README](04_Big_Data_PySpark/README.md#12-how-to-execute). |

### 5. Launch Jupyter

From the repository root:

```
python -m jupyter notebook
```

A browser tab opens. Open any folder and notebook, then run the cells from top to bottom (Shift+Enter, or Kernel > Restart & Run All). Jupyter runs each notebook from its own folder, so the relative CSV paths work without changes.

## Interactive Notebooks

Several notebooks call `input()`. When a cell asks for input, a text box appears under it. Type the value and press Enter. "Run All" pauses at each prompt until you answer.

| Notebook | Prompt(s) | What to type |
|---|---|---|
| [`Linear_Regression/multi_linear_reg.ipynb`](01_Supervised_Learning/Linear_Regression/multi_linear_reg.ipynb) | Engine size (L), number of cylinders, fuel consumption (L/100km) | Numbers, e.g. `2.4`, `4`, `9.6`. Invalid input prints a message. |
| [`Polynomial_Regression/poly_reg.ipynb`](01_Supervised_Learning/Polynomial_Regression/poly_reg.ipynb) | Years of experience | A number, e.g. `10` (data covers 1 to 20) |
| [`Logistic_Regression/3_logistic_regression.ipynb`](01_Supervised_Learning/Logistic_Regression/3_logistic_regression.ipynb) | Age | A whole number, e.g. `45` (data covers 18 to 60) |
| [`Decision_Tree/4_decision tree.ipynb`](01_Supervised_Learning/Decision_Tree/4_decision%20tree.ipynb) | a1, a2, a3 | Exactly `True`/`False`, `Hot`/`Cool`, `High`/`Normal` (case-sensitive). Invalid input prints a message. |
| [`KMeans/7_k_means.ipynb`](02_Unsupervised_Learning/KMeans/7_k_means.ipynb) | X and Y of a new point | Numbers, e.g. `6` and `4` |
| [`KMeans/kmeans_50000_clustering.ipynb`](02_Unsupervised_Learning/KMeans/kmeans_50000_clustering.ipynb) | X and Y of a new point | Numbers, e.g. `5` and `5` (data covers about -1.6 to 11.8) |
| [`KMedoids/8_k_medoids.ipynb`](02_Unsupervised_Learning/KMedoids/8_k_medoids.ipynb) | X and Y of a new point | Numbers, e.g. `4` and `6` |
| [`PCA/12_PCA.ipynb`](02_Unsupervised_Learning/PCA/12_PCA.ipynb) | `enter num:` (last cell, a separate star-pattern exercise) | A whole number, e.g. `6` |

Except where noted, a non-numeric entry raises a `ValueError`; re-run the cell to try again. The other notebooks do not ask for input.

## Verification

Checks run on 22 November 2026 before this repository was published: Windows 11, Python 3.10.11, scikit-learn 1.7.2, NumPy 2.2.6, pandas 2.3.3. Each notebook was run from top to bottom on a separate copy, in its own folder. The `input()` prompts were answered with sample values, and plots were rendered off-screen. The notebooks' saved outputs in this repository were not changed.

| Notebooks | Result |
|---|---|
| All 24 notebooks | Valid notebook files; every code cell parses without syntax errors |
| 18 notebooks: Linear Regression (`1_linear_regression`, `1_1linear_regression`, `multi_linear_reg`), Polynomial Regression, Logistic Regression, Decision Tree, Random Forest, KNN, SVM, K-Means (both), Hierarchical Clustering, DBSCAN (both), PCA, Classification Metrics, Cross-Validation, Hyperparameter Tuning | Every cell ran without an error |
| `Linear_Regression/2_linear_regression_multivariate.ipynb` | 11 of 12 cells ran. The last cell calls `reg.predict(...)`, but the notebook only defines `model`, so a fresh run raises `NameError`. Its saved output came from an earlier kernel session. Renaming `reg` to `model` would fix it. |
| `KMedoids/8_k_medoids.ipynb` | Not run in the main environment. `scikit-learn-extra` 0.3.0 fails to import with NumPy 2.x (`numpy.dtype size changed`). Use `numpy<2` as noted above. The saved outputs show the same kind of error. |
| `04_Big_Data_PySpark` (4 notebooks) | Syntax checked only. PySpark was not installed on the test machine, and `PySparkML.ipynb` needs a CSV that was uploaded in Google Colab. |

## Notes on This Repository

- **Saved outputs are kept as originally run.** The saved outputs were not regenerated for this repository (the checks under Verification ran on separate copies). They are the author's original ones, including warnings and errors. A few saved outputs come from earlier versions of a cell or from a different kernel session; each folder README points these out. Four notebooks have no saved outputs: `Random_forest.ipynb`, `kmeans_50000_clustering.ipynb`, `10_dbscan.ipynb` and `DBSCAN_Clustering_Practical.ipynb`.
- **Two notebooks had their dataset path made relative.** With the author's approval, the absolute local Windows path in `pd.read_csv` was replaced by a relative file name in [`SVM/5_svm_car_data.ipynb`](01_Supervised_Learning/SVM) (`"cars_dataset.csv"`) and in [`Cross_Validation/model evalution.ipynb`](03_Model_Evaluation_and_Tuning/Cross_Validation) (`"iris.csv"`). This one line in each notebook is the only change to the notebook code.
- **The KNN dataset is a substitute copy.** The original CSV used by [`KNN/Knn-1.ipynb`](01_Supervised_Learning/KNN) was not available. `data/iris-flower-dataset.csv` is a copy of the author's `iris.csv`. Its species labels are `setosa`, `versicolor`, `virginica`, while the notebook's saved outputs were produced with a file labelled `Iris-setosa`, `Iris-versicolor`, `Iris-virginica`. Re-run results may therefore differ in label text and possibly in values.
- **PySpark data.** `PySparkML.ipynb` reads a salary CSV that was uploaded in Google Colab and is not included. Its expected columns are described in [`04_Big_Data_PySpark/README.md`](04_Big_Data_PySpark/README.md#11-dataset-information).
- **Excluded data.** Some unused data files from the author's working folders were left out on purpose, including files with street addresses and personal names. See [`datasets/README.md`](datasets/README.md#excluded-datasets).
- Files that notebooks write when run (`kmeans_50000_with_clusters.csv`, `*.joblib`) and Spark working files are git-ignored.

## Author

**Vishwanathan G**

- GitHub: https://github.com/vishwanathan-G
- Email: g.vishwanathan29@gmail.com
