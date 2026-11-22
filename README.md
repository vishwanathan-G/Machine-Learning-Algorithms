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
- [Sources and Attribution](#sources-and-attribution)
- [Changes on 22 November 2026](#changes-on-22-november-2026)
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
│   │   ├── Linear Regression_Delivery Time Prediction.ipynb
│   │   ├── Multiple Linear Regression__House Price Prediction.ipynb
│   │   ├── 1_1linear_regression.ipynb
│   │   ├── multi_linear_reg.ipynb
│   │   ├── 01_linear_delivery_time.csv
│   │   ├── 02_multiple_linear_house_price.csv
│   │   └── homeprices.csv
│   ├── Polynomial_Regression/
│   │   ├── README.md
│   │   ├── poly_reg.ipynb
│   │   ├── polynomial_Regression_machine_temperature.ipynb
│   │   └── 03_polynomial_machine_temperature.csv
│   ├── Logistic_Regression/
│   │   ├── README.md
│   │   ├── logistic_Regression_customer_churn.ipynb
│   │   └── 04_logistic_customer_churn.csv
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
│   │   ├── 8_k_medoids.ipynb
│   │   └── requirements-kmedoids.txt     separate environment for this notebook
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
    ├── PySparkML.ipynb
    └── pyspark_salary_dataset_1000.csv
```

Each CSV file sits next to the notebook that reads it. [`datasets/README.md`](datasets/README.md) lists every dataset, including the built-in scikit-learn datasets and the inline data.

## Algorithm Index

| Category | Algorithm / topic | Folder | Notebooks | Dataset |
|---|---|---|---|---|
| Supervised: regression | Linear Regression (simple and multiple) | [Linear_Regression](01_Supervised_Learning/Linear_Regression) | `Linear Regression_Delivery Time Prediction`, `Multiple Linear Regression__House Price Prediction`, `1_1linear_regression`, `multi_linear_reg` | `01_linear_delivery_time.csv` (250 rows), `02_multiple_linear_house_price.csv` (250 rows), `homeprices.csv` (50,000 rows), inline CO2-emissions data |
| Supervised: regression | Polynomial Regression | [Polynomial_Regression](01_Supervised_Learning/Polynomial_Regression) | `poly_reg`, `polynomial_Regression_machine_temperature` | Inline experience/salary data; `03_polynomial_machine_temperature.csv` (250 rows) |
| Supervised: classification | Logistic Regression | [Logistic_Regression](01_Supervised_Learning/Logistic_Regression) | `logistic_Regression_customer_churn` | `04_logistic_customer_churn.csv` (250 rows) |
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
| Big data | PySpark RDD, DataFrame, Spark SQL, MLlib regression and classification | [04_Big_Data_PySpark](04_Big_Data_PySpark) | `RDD`, `Dataframe`, `sql`, `PySparkML` | Inline data; `pyspark_salary_dataset_1000.csv` (1,000 rows) |

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
| `02_Unsupervised_Learning/KMedoids` | In a separate Python 3.10 environment: `python -m pip install -r 02_Unsupervised_Learning/KMedoids/requirements-kmedoids.txt` | scikit-learn-extra 0.3.0 is compiled against NumPy 1.x and fails to import with NumPy 2.x, so this notebook needs its own environment. `requirements-kmedoids.txt` pins the tested versions (NumPy 1.26.4, scikit-learn 1.7.2, scikit-learn-extra 0.3.0). See [the folder README](02_Unsupervised_Learning/KMedoids/README.md#12-how-to-execute). |
| `04_Big_Data_PySpark` | `python -m pip install "pyspark>=3.4"` | Also needs a Java runtime with `JAVA_HOME` set (Java 8, 11 or 17 for PySpark 3.x; Java 17 or newer for PySpark 4.x). The notebooks were written in Google Colab. The salary CSV used by `PySparkML.ipynb` is included in the folder; see [the folder README](04_Big_Data_PySpark/README.md#12-how-to-execute) for the path change needed to run it locally. |

### 5. Launch Jupyter

From the repository root:

```
python -m jupyter notebook
```

A browser tab opens. Open any folder and notebook, then run the cells from top to bottom (Shift+Enter, or Kernel > Restart & Run All). Jupyter runs each notebook from its own folder, so the relative CSV paths work without changes. The one exception is the Google Colab path `/content/...` in `PySparkML.ipynb` (see the PySpark row above).

## Interactive Notebooks

Several notebooks call `input()`. When a cell asks for input, a text box appears under it. Type the value and press Enter. "Run All" pauses at each prompt until you answer.

| Notebook | Prompt(s) | What to type |
|---|---|---|
| [`Linear_Regression/Linear Regression_Delivery Time Prediction.ipynb`](01_Supervised_Learning/Linear_Regression/Linear%20Regression_Delivery%20Time%20Prediction.ipynb) | Distance in km, asked twice (with the trained model, then with the model re-loaded from `delivery_model.pkl`) | A number, e.g. `12.5` (data covers 1.15 to 29.62) |
| [`Linear_Regression/Multiple Linear Regression__House Price Prediction.ipynb`](01_Supervised_Learning/Linear_Regression/Multiple%20Linear%20Regression__House%20Price%20Prediction.ipynb) | Area (sq ft), number of bedrooms, house age, distance from the city (km) | Numbers, e.g. `1500`, `3`, `10`, `8.5`. Bedrooms and age must be whole numbers. |
| [`Linear_Regression/multi_linear_reg.ipynb`](01_Supervised_Learning/Linear_Regression/multi_linear_reg.ipynb) | Engine size (L), number of cylinders, fuel consumption (L/100km) | Numbers, e.g. `2.4`, `4`, `9.6`. Invalid input prints a message. |
| [`Polynomial_Regression/poly_reg.ipynb`](01_Supervised_Learning/Polynomial_Regression/poly_reg.ipynb) | Years of experience | A number, e.g. `10` (data covers 1 to 20) |
| [`Polynomial_Regression/polynomial_Regression_machine_temperature.ipynb`](01_Supervised_Learning/Polynomial_Regression/polynomial_Regression_machine_temperature.ipynb) | Operating hours, asked twice (degree 2 model, then degree 3 model) | A number, e.g. `6` (data covers 1.00 to 11.92) |
| [`Decision_Tree/4_decision tree.ipynb`](01_Supervised_Learning/Decision_Tree/4_decision%20tree.ipynb) | a1, a2, a3 | Exactly `True`/`False`, `Hot`/`Cool`, `High`/`Normal` (case-sensitive). Invalid input prints a message. |
| [`KMeans/7_k_means.ipynb`](02_Unsupervised_Learning/KMeans/7_k_means.ipynb) | X and Y of a new point | Numbers, e.g. `6` and `4` |
| [`KMeans/kmeans_50000_clustering.ipynb`](02_Unsupervised_Learning/KMeans/kmeans_50000_clustering.ipynb) | X and Y of a new point | Numbers, e.g. `5` and `5` (data covers about -1.6 to 11.8) |
| [`KMedoids/8_k_medoids.ipynb`](02_Unsupervised_Learning/KMedoids/8_k_medoids.ipynb) | X and Y of a new point | Numbers, e.g. `4` and `6` |
| [`PCA/12_PCA.ipynb`](02_Unsupervised_Learning/PCA/12_PCA.ipynb) | `enter num:` (last cell, a separate star-pattern exercise) | A whole number, e.g. `6` |

Except where noted, a non-numeric entry raises a `ValueError`; re-run the cell to try again. The other notebooks do not ask for input.

## Verification

Checks run on 22 November 2026: Windows 11, Python 3.10.11, scikit-learn 1.7.2, NumPy 2.2.6, pandas 2.3.3, unless stated otherwise in the table.

- **First check.** Each notebook was run from top to bottom on a separate copy, in its own folder. The `input()` prompts were answered with sample values, and plots were rendered off-screen. This check did not change the saved outputs.
- **After the code review.** Two notebooks were re-run in place, so their saved outputs were regenerated: `7_k_means` after a code fix, and `8_k_medoids` in its own environment. `Knn-1` was re-run with its recovered original dataset, and `PySparkML` was run in part on a copy. See [Changes on 22 November 2026](#changes-on-22-november-2026).
- **After the notebook replacement.** The four notebooks added on 22 November 2026 were each run from top to bottom on a copy with Python 3.10.11 and scikit-learn 1.7.2, with sample values for the `input()` prompts. Only `logistic_Regression_customer_churn` was re-run in place (after its `y_proba` fix), so only its saved outputs were regenerated. The other three keep the author's original saved outputs.

| Notebooks | Result |
|---|---|
| The 21 notebooks of the first check that are still in the repository | Valid notebook files; every code cell parses without syntax errors |
| 16 notebooks: Linear Regression (`1_1linear_regression`, `multi_linear_reg`), Polynomial Regression (`poly_reg`), Decision Tree, Random Forest, KNN, SVM, K-Means (both), Hierarchical Clustering, DBSCAN (both), PCA, Classification Metrics, Cross-Validation, Hyperparameter Tuning | Every cell ran without an error in the first check |
| `Linear_Regression/Linear Regression_Delivery Time Prediction.ipynb` | 12 of 12 code cells ran (inputs 34 and 34). Saved outputs are the author's original run: R2 94.48% (printed as `Accuracy (R² Score)`), 122.16 minutes for 34 km. The split has no `random_state`, so a re-run gives a different split and R2. |
| `Linear_Regression/Multiple Linear Regression__House Price Prediction.ipynb` | 9 of 9 code cells ran (inputs 2670, 1, 6, 6.89). Saved outputs are the author's original run: R2 96.26%. The split has no `random_state`. |
| `Polynomial_Regression/polynomial_Regression_machine_temperature.ipynb` | 10 of 10 code cells ran (inputs 24 and 24). Saved outputs are the author's original run: R2 97.94% (degree 2) and 97.93% (degree 3). |
| `Logistic_Regression/logistic_Regression_customer_churn.ipynb` | After the `y_proba` fix: 9 of 9 code cells ran; saved outputs regenerated (accuracy 72.00%; the hard-coded sample customer is predicted as Churn with probability 82.34%). They match the author's earlier saved run in 8 of 9 code cells; the 9th, the data preview, differs only in pandas' plain-text wrapping. |
| `KMeans/7_k_means.ipynb` | After the plot-loop fix: 6 of 6 code cells ran (inputs X = 6, Y = 4); saved outputs regenerated; the plot shows all four clusters |
| `KMedoids/8_k_medoids.ipynb` | 6 of 6 code cells ran without errors in the separate `requirements-kmedoids.txt` environment (NumPy 1.26.4, scikit-learn-extra 0.3.0; inputs X = 4, Y = 6); saved outputs regenerated. It does not run in the main environment, because scikit-learn-extra 0.3.0 fails to import with NumPy 2.x. |
| `KNN/Knn-1.ipynb` | Re-run with the recovered original dataset: all 13 saved code-cell outputs were reproduced exactly (accuracy 1.0 and 0.9666666666666667, both confusion matrices). The notebook file was not changed. |
| `04_Big_Data_PySpark/PySparkML.ipynb` | Partly run on a copy with Python 3.13.7, PySpark 4.1.1 and Java 17 (Colab upload cell skipped, `/content/` path pointed at the local CSV). The CSV-based part (data preview, `VectorAssembler`, `randomSplit` with 838 training and 162 test rows, `LinearRegression` predictions, R2 = 0.9958205217455208) reproduced the saved Colab outputs exactly. Cells 6 to 8 and 21, which create DataFrames from Python lists, could not run locally (see the next row). |
| `04_Big_Data_PySpark`: `RDD`, `Dataframe`, `sql` | Not run locally. On the test laptop Spark's Python worker crashes even for a three-line `parallelize().collect()` test, so cells that create RDDs or DataFrames from Python lists cannot run there; this is a local environment problem. Their saved outputs come from Google Colab, where they ran. |

## Notes on This Repository

- **Saved outputs.** Most saved outputs are the author's original ones, including warnings and errors. The exceptions are `7_k_means.ipynb` and `8_k_medoids.ipynb`, which were re-run on 22 November 2026 after the code review, and `logistic_Regression_customer_churn.ipynb`, which was re-run the same day after its `y_proba` fix; their saved outputs were regenerated. The saved outputs of the four PySpark notebooks come from Google Colab. A few saved outputs in other notebooks come from earlier versions of a cell or from a different kernel session; each folder README points these out. Four notebooks have no saved outputs: `Random_forest.ipynb`, `kmeans_50000_clustering.ipynb`, `10_dbscan.ipynb` and `DBSCAN_Clustering_Practical.ipynb`.
- **Two notebooks had their dataset path made relative.** With the author's approval, the absolute local Windows path in `pd.read_csv` was replaced by a relative file name in [`SVM/5_svm_car_data.ipynb`](01_Supervised_Learning/SVM) (`"cars_dataset.csv"`) and in [`Cross_Validation/model evalution.ipynb`](03_Model_Evaluation_and_Tuning/Cross_Validation) (`"iris.csv"`). The fixes made after the code review are listed under [Changes on 22 November 2026](#changes-on-22-november-2026).
- **Recovered datasets.** Two datasets that were missing earlier were found in the author's own files and are now included: the original Iris CSV used by [`KNN/Knn-1.ipynb`](01_Supervised_Learning/KNN) (`data/iris-flower-dataset.csv`, labels `Iris-setosa`, `Iris-versicolor`, `Iris-virginica`), and `pyspark_salary_dataset_1000.csv` (1,000 rows) used by [`PySparkML.ipynb`](04_Big_Data_PySpark). Both match the notebooks' saved outputs.
- **Local paths in saved outputs.** In 6 notebooks, the Windows username in file paths inside saved warning outputs was replaced with `<user>` (for example `C:\Users\<user>\AppData\...`). The warning text is otherwise unchanged. A scan of all files found no usernames, local drive paths, phone numbers or credentials, and no email addresses other than the author contact under [Author](#author).
- **Excluded data.** Some unused data files from the author's working folders were left out on purpose, including files with street addresses and personal names. They are listed in [`datasets/README.md`](datasets/README.md#excluded-datasets), together with the two tutorial datasets removed with the codebasics-based notebooks.
- Files that notebooks write when run (`kmeans_50000_with_clusters.csv`, `*.joblib`, and `*.pkl` such as `delivery_model.pkl`) and Spark working files are git-ignored.

## Sources and Attribution

Some notebooks closely follow public tutorials or notebooks. The code similarity scores and identical-line shares below come from a comparison with those sources.

| Notebook | Source | Similarity |
|---|---|---|
| [`PCA/12_PCA.ipynb`](02_Unsupervised_Learning/PCA) | [codebasics/py](https://github.com/codebasics/py), folder `ML/18_PCA` (`PCA_tutorial_digits`) | 0.94; 84% of code lines identical |
| [`Hyperparameter_Tuning/6_hyperparameter_tuning.ipynb`](03_Model_Evaluation_and_Tuning/Hyperparameter_Tuning) | [codebasics/py](https://github.com/codebasics/py), folder `ML/15_gridsearch` | 0.97; 96% of code lines identical |
| [`SVM/5_svm_car_data.ipynb`](01_Supervised_Learning/SVM) | `svm_car_data.ipynb` in [karthikeyanthanigai/SVM-for-car-dataset](https://github.com/karthikeyanthanigai/SVM-for-car-dataset) (created 2020) | 88% of code lines identical; same 12-cell structure and image |

- Three earlier notebooks that followed the codebasics tutorials, `1_linear_regression`, `2_linear_regression_multivariate` and `3_logistic_regression`, were replaced by the author's own implementations on 22 November 2026 (see [Changes on 22 November 2026](#changes-on-22-november-2026)).
- `cars_dataset.csv` is byte-identical to the file in karthikeyanthanigai/SVM-for-car-dataset. The data is the UCI Car Evaluation dataset (Bohanec and Rajkovic, 1988). Changes in this copy of the SVM notebook: intermediate arrays are printed, explanatory comments were added, the dataset is read from a relative path, and the source's "Open in Colab" badge (cell 0) was replaced by an attribution note.
- **Licences.** The codebasics/py repository has no licence file, and karthikeyanthanigai/SVM-for-car-dataset has no licence. Without a licence, the source authors keep all rights to their code. Before making this repository public, either replace the closely matching notebooks (`5_svm_car_data`, `12_PCA` and `6_hyperparameter_tuning`) with own implementations, keep them out of the public release, or get permission from the source authors.
- The other notebooks showed low similarity (below 0.4) to these sources, and `1_1linear_regression.ipynb` and `multi_linear_reg.ipynb` did not match them; they appear to be the author's own work. The replacement notebooks `Linear Regression_Delivery Time Prediction`, `Multiple Linear Regression__House Price Prediction` and `logistic_Regression_customer_churn` have code similarity scores of 0.30 to 0.37 to the tutorial sources. They are the author's own implementations, taken from the author's practice repository, where they were written as solutions to a set of regression practice questions. `polynomial_Regression_machine_temperature` is a further own notebook for the same practice questions.

## Changes on 22 November 2026

Changes made after a code-review round, and the replacement of three tutorial-based notebooks by the author's own implementations. Each folder README lists the changes for its folder.

- **Notebook replacement (Linear, Logistic and Polynomial Regression).** `1_linear_regression.ipynb`, `2_linear_regression_multivariate.ipynb` and `3_logistic_regression.ipynb`, which closely followed the codebasics tutorials, were removed together with `houseprice.csv` and `insurance_data.csv`. (`2_linear_regression_multivariate.ipynb` had received a `reg` to `model` fix and a re-run earlier in this round.) They were replaced by the author's own implementations from the author's practice repository, written as solutions to a set of regression practice questions, each with its practice CSV: `Linear Regression_Delivery Time Prediction.ipynb`, `Multiple Linear Regression__House Price Prediction.ipynb` and `logistic_Regression_customer_churn.ipynb` (code similarity to the tutorial sources 0.30 to 0.37). A fourth own notebook, `polynomial_Regression_machine_temperature.ipynb`, was added to Polynomial Regression. In the churn notebook, the ROC cell used `y_proba`, which was never defined, so a fresh run raised a `NameError`; one added line in the metrics cell now defines it, and the notebook was re-run and its saved outputs regenerated. The other three new notebooks keep the author's original saved outputs. All four ran from top to bottom on a copy (see [Verification](#verification)).
- **K-Means.** The plot loop in `7_k_means.ipynb` drew only 3 of the 4 clusters; it now loops over `kmeans.n_clusters` with four colours. Re-run with X = 6, Y = 4; the plot shows all four clusters.
- **K-Medoids.** No code change. New `requirements-kmedoids.txt` pins the versions under which `8_k_medoids.ipynb` runs (NumPy 1.26.4 with scikit-learn-extra 0.3.0). Re-run in that environment; the old `ValueError` and `NameError` outputs were replaced.
- **PySpark.** `PySparkML.ipynb` cell 9 (`model.fit(x,y)` in scikit-learn style) is now a markdown note explaining why it does not work in Spark. `pyspark_salary_dataset_1000.csv` was recovered and is included. A local test reproduced the CSV-based outputs exactly.
- **SVM.** Attribution note added to the notebook (cell 0) and the README, with the licence caveat.
- **KNN.** The substitute Iris copy was replaced by the recovered original dataset, with which the notebook reproduces its saved outputs exactly. The notebook is unchanged.
- **Attribution.** Source notes were added to the PCA and Hyperparameter Tuning READMEs and to this README (see [Sources and Attribution](#sources-and-attribution)). The notes added to the Linear Regression and Logistic Regression READMEs were removed again with the notebooks they described; those READMEs now state that their notebooks are the author's own. The code of `12_PCA` and `6_hyperparameter_tuning` was not changed.
- **Privacy.** The Windows username in file paths inside saved outputs was replaced with `<user>` in 7 notebooks, 3 of which were later removed in the notebook replacement, and in the saved warnings of the new Delivery Time and Machine Temperature notebooks. Six notebooks in the repository now contain such paths.

## Author

**Vishwanathan G**

- GitHub: https://github.com/vishwanathan-G
- Email: g.vishwanathan29@gmail.com
