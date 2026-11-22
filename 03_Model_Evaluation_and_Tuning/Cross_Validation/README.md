# Model Evaluation: Hold-out, Bootstrap and K-Fold Cross-Validation

## Files in this folder

| File | Description |
|---|---|
| `model evalution.ipynb` | Compares ways of estimating model performance: a hold-out split, bootstrap resampling, K-fold cross-validation and 100 repeated random splits. Also compares a lazy learner (KNN) with an eager learner (decision tree), and KNN with a random forest on the Wine dataset. |
| `iris.csv` | Iris data, 150 rows x 5 columns, used by the hold-out section |

## 1. Algorithm Name

Model evaluation and validation techniques: hold-out validation, bootstrap resampling, K-fold cross-validation and repeated random sub-sampling. Also a lazy vs eager learner comparison (KNN vs decision tree).

## 2. Definition

Model evaluation estimates how well a trained model will perform on data it has not seen. Each technique differs in how it divides the available data between training and testing, and in how many times it repeats that division.

## 3. How It Works

| Technique | Procedure |
|---|---|
| Hold-out | Split once into a training set and a test set (for example 80/20). Train on one and score on the other. |
| Repeated hold-out | Repeat the random hold-out split many times and average the scores. |
| K-fold cross-validation | Split the data into K equal folds. Train K times, each time holding out a different fold for testing, and average the K scores. |
| Bootstrap | Draw many samples of size n with replacement, train a model on each, and look at the distribution of the scores. |
| Lazy vs eager learners | A lazy learner (KNN) stores the training data and does the work at prediction time. An eager learner (decision tree) builds a model during training. |

## 4. Why It Is Used

- A single split can be lucky or unlucky. Repeating or rotating the split gives a more reliable estimate.
- Cross-validation uses every row for both training and testing, which matters for small datasets.
- The bootstrap also shows the variability (uncertainty) of a model's performance.
- A reliable estimate is needed to compare models fairly and to tune hyperparameters (see `../Hyperparameter_Tuning`).

## 5. Mathematical Intuition

- K-fold estimate: `score_cv = (1/K) * sum(score_k)`. Its spread (standard deviation) shows how stable the model is.
- Each bootstrap sample of size `n` leaves out about `(1 - 1/n)^n ≈ 36.8%` of the rows. These out-of-bag rows can serve as an untouched test set.
- Mean squared error for regression: `MSE = (1/n) * sum((y - y_hat)^2)`. Accuracy for classification: `correct / total`.
- R2 (the default `score` of regression models in scikit-learn): `R2 = 1 - SS_res / SS_tot`. It is not an error, so higher is better.

## 6. Important Parameters

**General**

| Function / class | Parameter | Default | Meaning |
|---|---|---|---|
| `train_test_split` | `test_size` / `train_size` | 0.25 test | Split proportions |
| `train_test_split` | `random_state` | `None` | Seed; `None` gives a different split each run |
| `train_test_split` | `stratify` | `None` | Keep class proportions in both parts |
| `KFold` | `n_splits` | `5` | Number of folds |
| `KFold` | `shuffle`, `random_state` | `False`, `None` | Shuffle before splitting, and its seed |
| `cross_val_score` | `cv` | 5-fold | Folds or a splitter object |
| `cross_val_score` | `scoring` | `None` | `None` uses the estimator's `score`: R2 for regressors, accuracy for classifiers |
| `sklearn.utils.resample` | `replace`, `n_samples` | `True`, all rows | Bootstrap sampling |

**Used in this notebook**

| Section | Settings |
|---|---|
| Hold-out (iris.csv) | `train_test_split(X, y, train_size=0.8)` with no `random_state`; `KNeighborsClassifier(n_neighbors=3)`; all 4 features |
| Bootstrap | `n_iterations = 10000`; `resample(X, y)`; `LinearRegression()`; feature petal length, target petal width |
| K-fold | `KFold(n_splits=5, shuffle=True, random_state=42)`; `cross_val_score(LinearRegression(), X, y, cv=kf)` with the default scorer |
| Lazy learner | `KNeighborsClassifier(n_neighbors=3)`; petal length only; `test_size=0.3, random_state=42` |
| Eager learner | `DecisionTreeClassifier(random_state=42)`; same data and split as the lazy learner |
| Wine comparison | 100 seeds; `test_size=0.3, random_state=seed`; `KNeighborsClassifier(n_neighbors=3)` vs `RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed)` |
| Repeated hold-out | 100 runs of `shuffle` and then `train_test_split(test_size=0.2)`; `LinearRegression()` on petal length to petal width |

## 7. Advantages

- Hold-out: simple and fast; suitable for large datasets.
- K-fold: lower-variance estimate; every row is used for testing exactly once.
- Bootstrap: shows the uncertainty of the estimate and works with small samples.
- Repeated hold-out: easy to implement, and averages out the luck of a single split.

## 8. Disadvantages

- Hold-out: the estimate depends heavily on one random split.
- K-fold: K times the training cost; folds must be shuffled or stratified when the data is ordered by class.
- Bootstrap: many model fits; scoring on rows that were also used for training gives an optimistic estimate.
- Repeated hold-out: some rows may never be tested and others tested many times.

## 9. Real-World Applications

- Choosing between candidate models before deployment.
- Hyperparameter tuning (grid or random search with cross-validation).
- Reporting a model's expected accuracy with an uncertainty range.
- Checking model stability on small datasets, as in medical or scientific studies.

## 10. Python Libraries Used

| Library | Used for |
|---|---|
| `pandas` | Loading `iris.csv`, building the Wine DataFrame |
| `numpy` | Averages |
| `sklearn.datasets` (`load_iris`, `load_wine`) | Built-in datasets |
| `sklearn.model_selection` (`train_test_split`, `KFold`, `cross_val_score`) | Splitting and cross-validation |
| `sklearn.utils` (`resample`, `shuffle`) | Bootstrap sampling and shuffling |
| `sklearn.linear_model.LinearRegression` | Regression model |
| `sklearn.neighbors.KNeighborsClassifier` | Lazy learner |
| `sklearn.tree.DecisionTreeClassifier` | Eager learner |
| `sklearn.ensemble.RandomForestClassifier` | Wine comparison |
| `sklearn.metrics` (`accuracy_score`, `mean_squared_error`) | Scores |

## 11. Dataset Information

| Data | Used in | Size | Notes |
|---|---|---|---|
| `iris.csv` (this folder) | Hold-out section | 150 rows x 5 columns | `sepal_length`, `sepal_width`, `petal_length`, `petal_width`, `species` (`setosa`, `versicolor`, `virginica`, 50 each). The same 150 measurements as `01_Supervised_Learning/KNN/data/iris-flower-dataset.csv`, whose species labels carry an `Iris-` prefix. |
| `load_iris()` (built-in) | Bootstrap, K-fold, lazy/eager learners, repeated hold-out | 150 rows x 4 features | Regression sections use petal length (column 2) to predict petal width (column 3). Classification sections use petal length to predict the species (0, 1, 2). |
| `load_wine()` (built-in) | Wine comparison and display | 178 rows x 13 features, 3 classes | Chemical analysis of wines |

**Change made for this repository.** With the author's approval, one line was changed. In the `pd.read_csv` call, the absolute local Windows path was replaced by the relative path `"iris.csv"`, so the notebook runs from this folder on any machine. No other code was changed, and the saved outputs are the author's original ones.

## 12. How to Execute

1. Set up the environment as described in the [root README](../../README.md#getting-started).
2. Run `python -m jupyter notebook` from the repository root and open `03_Model_Evaluation_and_Tuning/Cross_Validation/model evalution.ipynb`.
3. Run the cells from top to bottom. There are no input prompts.

The bootstrap cell fits 10,000 models, and the Wine comparison fits 100 random forests, so these two cells take longer than the others.

## 13. Expected Output

Saved output from the author's run:

| Section | Printed by the code | Saved output |
|---|---|---|
| Hold-out (KNN, iris.csv) | `Accuracy Score:` | 1.0 |
| Bootstrap (10,000 fits) | first 10 MSE values, then `Bootstrap Avg. MSE:` | 0.04263 |
| K-fold (5 folds) | the 5 fold scores, then `K-Fold Cross-Validation MSE:` and their mean | `[0.9283 0.9444 0.9293 0.9136 0.8964]`, mean 0.9224 |
| Lazy learner (KNN, k = 3) | `Lazy Learner Accuracy (KNN):` | 0.9778 |
| Eager learner (decision tree) | `Eager Learner Accuracy (Decision Tree):` | 0.9556 |
| Wine: KNN vs random forest | `Avg KNN Accuracy on Wine Dataset:` and `Avg Random Forest Accuracy on Wine Dataset:` | No saved output |
| Wine DataFrame | the full table | `[178 rows x 13 columns]` |
| Repeated hold-out (100 runs) | `Average MSE over 100 runs:` | 0.04362 |

Notes on how the markdown cells relate to the code and saved outputs:

- The first section is titled "Holdout Method (Regression)" and a following markdown cell says "MSE: 0.047". The code trains a KNN classifier on all four features and prints accuracy. The hold-out split has no `random_state`, so its accuracy can change from run to run.
- A code cell after the hold-out section is empty but still has a saved output: an array of 30 predicted species names (`setosa`, `versicolor`, `virginica`). The labels match the format used in `iris.csv`.
- The bootstrap markdown says "Trains 100 models"; the code uses 10,000 iterations. Each bootstrap model is scored on the full dataset, including rows it was trained on.
- In the K-fold section, `cross_val_score` uses `LinearRegression`'s default scorer, which is R2. The printed values (around 0.92) are R2 scores, although the label says MSE. To get MSE, pass `scoring='neg_mean_squared_error'` and negate the result.
- The summary table in the notebook lists approximate values (for example bootstrap 0.053 to 0.057, KNN 0.972, tree 0.951) that differ from the saved outputs above. It also notes that KNN performs lower than the decision tree, while in the saved run KNN scored higher (0.9778 vs 0.9556).
