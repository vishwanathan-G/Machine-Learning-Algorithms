# Datasets Index

Each CSV file lives next to the notebook that reads it, so the notebooks can use simple relative paths. This folder holds no data files. It is an index of every dataset used in the repository.

## Data files in the repository

| File | Folder | Used by | Rows x columns | What it contains | Origin |
|---|---|---|---|---|---|
| `houseprice.csv` | [`01_Supervised_Learning/Linear_Regression`](../01_Supervised_Learning/Linear_Regression) | `1_linear_regression.ipynb`, `2_linear_regression_multivariate.ipynb` | 6 x 4 | `area`, `bedrooms` (one value missing), `age`, `price` | Small practice table; origin not documented |
| `homeprices.csv` | [`01_Supervised_Learning/Linear_Regression`](../01_Supervised_Learning/Linear_Regression) | `1_1linear_regression.ipynb` | 50,000 x 4 | `area` (800 to 5,000), `bedrooms` (1 to 7), `age` (0 to 50), `price` (217,000 to 967,000); no missing values | Appears to be synthetic (generated) data; origin not documented |
| `insurance_data.csv` | [`01_Supervised_Learning/Logistic_Regression`](../01_Supervised_Learning/Logistic_Regression) | `3_logistic_regression.ipynb` | 400 x 5 | `User ID`, `Gender`, `Age` (18 to 60), `EstimatedSalary`, `Purchased` (0/1: 257/143). The notebook uses `Age` and `Purchased`. | Appears to match a widely used "Social Network Ads" practice dataset (same columns and row count) |
| `iris-flower-dataset.csv` | [`01_Supervised_Learning/KNN/data`](../01_Supervised_Learning/KNN/data) | `Knn-1.ipynb` | 150 x 5 | `sepal_length`, `sepal_width`, `petal_length`, `petal_width` (cm), `species` (`setosa`, `versicolor`, `virginica`, 50 each) | Classic Iris flower dataset (also in the UCI Machine Learning Repository). This is a copy of the author's `iris.csv`, supplied because the original file was not available. See the note below. |
| `cars_dataset.csv` | [`01_Supervised_Learning/SVM`](../01_Supervised_Learning/SVM) | `5_svm_car_data.ipynb` | 1,728 x 7 | Six categorical attributes (`buying`, `maint`, `doors`, `persons`, `lug_boot`, `safety`) and the target `car` (`unacc` 1,210, `acc` 384, `good` 69, `vgood` 65) | Appears to be the UCI Car Evaluation dataset, with door and person counts written as words (`two`, `three`, `four`, `5more`, `more`) |
| `kmeans_50000_data.csv` | [`02_Unsupervised_Learning/KMeans`](../02_Unsupervised_Learning/KMeans) | `kmeans_50000_clustering.ipynb` | 50,000 x 3 | `X`, `Y` coordinates and `Original_Group` (`Group_1` to `Group_5`, 10,000 each). The model uses only `X` and `Y`. | Synthetic; the notebook describes it as generated practice data with five groups |
| `iris.csv` | [`03_Model_Evaluation_and_Tuning/Cross_Validation`](../03_Model_Evaluation_and_Tuning/Cross_Validation) | `model evalution.ipynb` (hold-out section) | 150 x 5 | Same contents as `iris-flower-dataset.csv` above | Classic Iris flower dataset |

**Note on the KNN copy of Iris.** `Knn-1.ipynb`'s saved outputs were produced with a file whose species labels were `Iris-setosa`, `Iris-versicolor` and `Iris-virginica`. The copy in this repository uses `setosa`, `versicolor` and `virginica`. Re-running the notebook prints labels without the `Iris-` prefix, and results may not match the saved outputs exactly.

## Built-in scikit-learn datasets

These are loaded by code and need no file.

| Loader | Used by | Size | Contents |
|---|---|---|---|
| `load_wine()` | `01_Supervised_Learning/Random_Forest/Random_forest.ipynb`, `03_Model_Evaluation_and_Tuning/Cross_Validation/model evalution.ipynb` | 178 x 13, 3 classes | Chemical analysis of wines from three cultivars |
| `load_iris()` | `03_Model_Evaluation_and_Tuning/Cross_Validation/model evalution.ipynb`, `03_Model_Evaluation_and_Tuning/Hyperparameter_Tuning/6_hyperparameter_tuning.ipynb` | 150 x 4, 3 classes | Iris flower measurements |
| `load_digits()` | `02_Unsupervised_Learning/PCA/12_PCA.ipynb` | 1,797 x 64, 10 classes | 8 x 8 images of handwritten digits, pixel values 0 to 16 |

## Generated data

| Generator | Used by | Size | Contents |
|---|---|---|---|
| `make_moons(n_samples=300, noise=0.05)` | `02_Unsupervised_Learning/DBSCAN/10_dbscan.ipynb` (no seed), `02_Unsupervised_Learning/DBSCAN/DBSCAN_Clustering_Practical.ipynb` (`random_state=42`) | 300 x 2 | Two interleaving half-circles |

## Inline data (typed into the notebook)

| Notebook | Size | Contents |
|---|---|---|
| `01_Supervised_Learning/Linear_Regression/multi_linear_reg.ipynb` | 9 x 4 | `ENGINESIZE`, `CYLINDERS`, `FUELCONSUMPTION_COMB`, `CO2EMISSIONS` (typed in from an image, according to a code comment) |
| `01_Supervised_Learning/Polynomial_Regression/poly_reg.ipynb` | 20 x 2 | `Experience` (1 to 20), `Salary` (50,000 to 400,000) |
| `01_Supervised_Learning/Decision_Tree/4_decision tree.ipynb` | 10 x 4 | Categorical `a1` (True/False), `a2` (Hot/Cool), `a3` (High/Normal), `Classification` (Yes/No) |
| `02_Unsupervised_Learning/KMeans/7_k_means.ipynb` | 8 x 2 | `X`, `Y` points |
| `02_Unsupervised_Learning/KMedoids/8_k_medoids.ipynb` | 10 x 2 | `x`, `y` points |
| `02_Unsupervised_Learning/Hierarchical_Clustering/9_agg.ipynb` | 15 x 2 | `X`, `Y` points in three groups |
| `03_Model_Evaluation_and_Tuning/Classification_Metrics/11_evalution-1.ipynb` | 8 label pairs | `y_true`, `y_pred` binary labels |
| `04_Big_Data_PySpark/RDD.ipynb` | small lists | Integer lists |
| `04_Big_Data_PySpark/Dataframe.ipynb`, `04_Big_Data_PySpark/sql.ipynb` | 10 rows | Employee table (`id`, `name`, `salary`; plus `Dept` in `sql.ipynb`) |
| `04_Big_Data_PySpark/PySparkML.ipynb` | 3 rows each | Toy `Gender`/`Age`/`Salary`/`label` rows |

## Referenced but not included

| File | Referenced by | Notes |
|---|---|---|
| `pyspark_salary_dataset_1000.csv` | `04_Big_Data_PySpark/PySparkML.ipynb` | Uploaded in Google Colab and read from `/content/`. Expected columns: `ID` (integer), `Experience` (years, decimal), `Salary` (integer). The saved outputs show 1,000 rows. See [`04_Big_Data_PySpark/README.md`](../04_Big_Data_PySpark/README.md#11-dataset-information). |
| `canada_per_capita_income.csv` | Exercise text in `01_Supervised_Learning/Linear_Regression/1_linear_regression.ipynb` | Mentioned in a markdown exercise only; no code reads it |
| `hiring.csv` | Exercise text in `01_Supervised_Learning/Linear_Regression/2_linear_regression_multivariate.ipynb` | Mentioned in a markdown exercise only; no code reads it |

## Files written when notebooks run

These files are git-ignored (see `.gitignore`).

| File | Written by |
|---|---|
| `kmeans_50000_with_clusters.csv` | `02_Unsupervised_Learning/KMeans/kmeans_50000_clustering.ipynb` |
| `kmeans_model.joblib` | `02_Unsupervised_Learning/KMeans/kmeans_50000_clustering.ipynb` |

## Excluded datasets

These files were in the author's working folders but are intentionally not part of this repository:

| File | Reason |
|---|---|
| `MELBOURNE_HOUSE_PRICES_LESS.csv`, `Melbourne_housing_FULL.csv` | Not used by any notebook; about 12.3 MB; contain street addresses and real-estate agent names |
| `diabetes_dataset.csv` | Not used by any notebook |
