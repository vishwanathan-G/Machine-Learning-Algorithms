# Datasets Index

Each CSV file lives next to the notebook that reads it, so the notebooks can use simple relative paths. This folder holds no data files. It is an index of every dataset used in the repository.

## Data files in the repository

| File | Folder | Used by | Rows x columns | What it contains | Origin |
|---|---|---|---|---|---|
| `houseprice.csv` | [`01_Supervised_Learning/Linear_Regression`](../01_Supervised_Learning/Linear_Regression) | `1_linear_regression.ipynb`, `2_linear_regression_multivariate.ipynb` | 6 x 4 | `area`, `bedrooms` (one value missing), `age`, `price` | The 6-row house-price data of the codebasics "py" tutorials ([github.com/codebasics/py](https://github.com/codebasics/py)); the two notebooks follow its linear regression tutorials `ML/1_linear_reg` and `ML/2_linear_reg_multivariate`. That repository has no licence file. |
| `homeprices.csv` | [`01_Supervised_Learning/Linear_Regression`](../01_Supervised_Learning/Linear_Regression) | `1_1linear_regression.ipynb` | 50,000 x 4 | `area` (800 to 5,000), `bedrooms` (1 to 7), `age` (0 to 50), `price` (217,000 to 967,000); no missing values | Appears to be synthetic (generated) data; origin not documented |
| `insurance_data.csv` | [`01_Supervised_Learning/Logistic_Regression`](../01_Supervised_Learning/Logistic_Regression) | `3_logistic_regression.ipynb` | 400 x 5 | `User ID`, `Gender`, `Age` (18 to 60), `EstimatedSalary`, `Purchased` (0/1: 257/143). The notebook uses `Age` and `Purchased`. | Appears to match a widely used "Social Network Ads" practice dataset (same columns and row count) |
| `iris-flower-dataset.csv` | [`01_Supervised_Learning/KNN/data`](../01_Supervised_Learning/KNN/data) | `Knn-1.ipynb` | 150 x 5 | `sepal_length`, `sepal_width`, `petal_length`, `petal_width` (cm), `species` (`Iris-setosa`, `Iris-versicolor`, `Iris-virginica`, 50 each) | Classic Iris flower dataset (also in the UCI Machine Learning Repository). This is the original file used by the notebook, recovered from the author's own files (an `IRIS.csv`). Re-running `Knn-1.ipynb` with it reproduces all of the notebook's saved outputs exactly. |
| `cars_dataset.csv` | [`01_Supervised_Learning/SVM`](../01_Supervised_Learning/SVM) | `5_svm_car_data.ipynb` | 1,728 x 7 | Six categorical attributes (`buying`, `maint`, `doors`, `persons`, `lug_boot`, `safety`) and the target `car` (`unacc` 1,210, `acc` 384, `good` 69, `vgood` 65) | The UCI Car Evaluation dataset (Bohanec and Rajkovic, 1988), with door and person counts written as words (`two`, `three`, `four`, `5more`, `more`). The file is byte-identical to `cars_dataset.csv` in [karthikeyanthanigai/SVM-for-car-dataset](https://github.com/karthikeyanthanigai/SVM-for-car-dataset), the repository the notebook is adapted from. That repository has no licence; see the [SVM README](../01_Supervised_Learning/SVM/README.md#source-and-attribution). |
| `kmeans_50000_data.csv` | [`02_Unsupervised_Learning/KMeans`](../02_Unsupervised_Learning/KMeans) | `kmeans_50000_clustering.ipynb` | 50,000 x 3 | `X`, `Y` coordinates and `Original_Group` (`Group_1` to `Group_5`, 10,000 each). The model uses only `X` and `Y`. | Synthetic; the notebook describes it as generated practice data with five groups |
| `iris.csv` | [`03_Model_Evaluation_and_Tuning/Cross_Validation`](../03_Model_Evaluation_and_Tuning/Cross_Validation) | `model evalution.ipynb` (hold-out section) | 150 x 5 | The same 150 measurements as `iris-flower-dataset.csv` above, with species labels without the `Iris-` prefix (`setosa`, `versicolor`, `virginica`, 50 each) | Classic Iris flower dataset |
| `pyspark_salary_dataset_1000.csv` | [`04_Big_Data_PySpark`](../04_Big_Data_PySpark) | `PySparkML.ipynb` (linear regression) | 1,000 x 3 | `ID` (1 to 1,000), `Experience` (years, decimal, 1 to 20), `Salary` (integer, 22,250 to 122,182); no missing values. Only `Experience` and `Salary` are used. | Recovered from the author's own files; its rows match the notebook's saved output. In Google Colab it was uploaded and read from `/content/`; to run locally, change that path (see [`04_Big_Data_PySpark/README.md`](../04_Big_Data_PySpark/README.md#12-how-to-execute)). |

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
| `canada_per_capita_income.csv` | Exercise text in `01_Supervised_Learning/Linear_Regression/1_linear_regression.ipynb` | Mentioned in a markdown exercise only; no code reads it. The exercise refers to the exercise folder of the codebasics tutorial. |
| `hiring.csv` | Exercise text in `01_Supervised_Learning/Linear_Regression/2_linear_regression_multivariate.ipynb` | Mentioned in a markdown exercise only; no code reads it. The exercise refers to the exercise folder of the codebasics tutorial. |

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
