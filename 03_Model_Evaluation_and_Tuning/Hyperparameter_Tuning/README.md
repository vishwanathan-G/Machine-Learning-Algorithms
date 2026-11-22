# Hyperparameter Tuning

## Files in this folder

| File | Description |
|---|---|
| `6_hyperparameter_tuning.ipynb` | Finds the best model and hyperparameters for Iris classification. It moves from manual trial and error, to `cross_val_score` in a loop, to `GridSearchCV` and `RandomizedSearchCV`, and finally a grid search over three model types (SVM, Random Forest, Logistic Regression). |

## Source and attribution

`6_hyperparameter_tuning.ipynb` follows the codebasics "py" grid search tutorial ([github.com/codebasics/py](https://github.com/codebasics/py), folder `ML/15_gridsearch`). A comparison gave a code similarity score of 0.97, and 96% of the code lines are identical. The codebasics/py repository has no licence file.

## 1. Algorithm Name

Hyperparameter tuning with cross-validation: Grid Search (`GridSearchCV`) and Randomized Search (`RandomizedSearchCV`).

## 2. Definition

Hyperparameters are settings chosen before training, such as an SVM's `C` and `kernel` or the number of trees in a forest. The model does not learn them from the data. Hyperparameter tuning searches for the combination that gives the best validation performance, usually measured with K-fold cross-validation.

## 3. How It Works

1. Define the model and a search space, for example `C` in `[1, 10, 20]` and `kernel` in `['rbf', 'linear']`.
2. **Grid search** tries every combination (here 3 x 2 = 6). **Randomized search** tries only `n_iter` randomly chosen combinations.
3. Each combination is scored with K-fold cross-validation, and the mean score is recorded.
4. The best combination is reported (`best_params_`, `best_score_`). With `refit=True`, the default, the model is retrained on all the data with those settings (`best_estimator_`).

## 4. Why It Is Used

- Default hyperparameters are rarely the best for a given dataset.
- Manual trial and error is slow and easy to get wrong. Grid and random search are systematic and reproducible.
- Cross-validation inside the search reduces the risk of picking settings that only suit one lucky split.

## 5. Mathematical Intuition

- Grid search evaluates the Cartesian product of the parameter lists: with `p` parameters of `m` values each, there are `m^p` combinations, each fitted `K` times.
- Randomized search fixes the budget at `n_iter * K` fits, regardless of the size of the space. When only a few parameters really matter, random sampling covers their range more efficiently than a grid.
- Selection rule: `best = argmax over theta of (1/K) * sum(score_k(theta))`.
- SVM parameters searched here: `C` trades margin width against training errors, and `kernel` chooses a linear or RBF boundary (see `../../01_Supervised_Learning/SVM/README.md`).

## 6. Important Parameters

**General**

| Class | Parameter | Default | Meaning |
|---|---|---|---|
| `GridSearchCV` | `param_grid` | (required) | Dictionary of parameter lists to try |
| `GridSearchCV`, `RandomizedSearchCV` | `cv` | 5-fold | Cross-validation folds |
| `GridSearchCV`, `RandomizedSearchCV` | `scoring` | `None` | `None` uses the estimator's `score` (accuracy for classifiers) |
| `GridSearchCV`, `RandomizedSearchCV` | `refit` | `True` | Retrain the best model on all the data |
| `GridSearchCV`, `RandomizedSearchCV` | `n_jobs` | `None` | Parallel fits |
| `GridSearchCV`, `RandomizedSearchCV` | `return_train_score` | `False` | Also record training scores |
| `RandomizedSearchCV` | `param_distributions` | (required) | Lists or distributions to sample from |
| `RandomizedSearchCV` | `n_iter` | `10` | Number of combinations sampled |
| `RandomizedSearchCV` | `random_state` | `None` | Seed for the sampling |

**Used in this notebook**

| Step | Model and settings |
|---|---|
| Approach 1: manual | `train_test_split(iris.data, iris.target, test_size=0.3)` (no `random_state`); `svm.SVC(kernel='rbf', C=30, gamma='auto')` |
| Approach 2: `cross_val_score` | `cv=5`; `SVC(kernel='linear', C=10)`, `SVC(kernel='rbf', C=10)`, `SVC(kernel='rbf', C=20)`, all with `gamma='auto'` |
| Loop | kernels `['rbf', 'linear']` x `C` `[1, 10, 20]`, `cv=5`, mean score per combination |
| `GridSearchCV` | `svm.SVC(gamma='auto')`, grid `C: [1, 10, 20]`, `kernel: ['rbf', 'linear']`, `cv=5`, `return_train_score=False` |
| `RandomizedSearchCV` | same model and grid, `cv=5`, `n_iter=2`, no `random_state` |
| Model comparison | `GridSearchCV(cv=5)` on `svm.SVC(gamma='auto')` with `C: [1, 10, 20]` and `kernel: ['rbf', 'linear']`; `RandomForestClassifier()` with `n_estimators: [1, 5, 10]`; `LogisticRegression(solver='liblinear', multi_class='auto')` with `C: [1, 5, 10]` |

## 7. Advantages

- Grid search is exhaustive within the grid and easy to understand.
- Randomized search scales to large search spaces with a fixed budget.
- Both use cross-validation, so the chosen settings are less likely to overfit one split.
- `cv_results_` records every combination for later analysis.

## 8. Disadvantages

- Grid search cost grows exponentially with the number of parameters.
- Randomized search may miss the best combination.
- Both can be slow for expensive models. Bayesian optimisation or successive halving are alternatives.
- Choosing the best score over many combinations on the same folds can still be slightly optimistic. A separate test set, or nested cross-validation, gives an unbiased final estimate.

## 9. Real-World Applications

- Tuning gradient-boosting, random forest or SVM models for production.
- Choosing the regularisation strength for linear models.
- Comparing several model families on the same data before deployment.
- Tuning neural network learning rates and layer sizes (often with randomized or Bayesian search).

## 10. Python Libraries Used

| Library | Used for |
|---|---|
| `sklearn.datasets` | `load_iris` |
| `sklearn.svm` | `SVC` |
| `pandas` | DataFrames for the data and for `cv_results_` |
| `numpy` | `np.average` of fold scores |
| `sklearn.model_selection` (`train_test_split`, `cross_val_score`, `GridSearchCV`, `RandomizedSearchCV`) | Splitting, cross-validation and search |
| `sklearn.ensemble.RandomForestClassifier` | Model comparison |
| `sklearn.linear_model.LogisticRegression` | Model comparison |

## 11. Dataset Information

- Built-in scikit-learn Iris dataset, loaded with `datasets.load_iris()`. No file is needed.
- 150 rows x 4 features: sepal length, sepal width, petal length and petal width (cm).
- Target: 3 species (`setosa`, `versicolor`, `virginica`), 50 rows each. The notebook adds the species names as a `flower` column for display.
- The notebook's third markdown cell shows an image from `iris_petal_sepal.png`. That image is not included in this repository, so it does not display.

## 12. How to Execute

1. Set up the environment as described in the [root README](../../README.md#getting-started).
2. Run `python -m jupyter notebook` from the repository root and open `03_Model_Evaluation_and_Tuning/Hyperparameter_Tuning/6_hyperparameter_tuning.ipynb`.
3. Run the cells from top to bottom. There are no input prompts.

**scikit-learn version note.** The model-comparison cell creates `LogisticRegression(solver='liblinear', multi_class='auto')`. The `multi_class` parameter was deprecated in scikit-learn 1.5 and removed in 1.8. With scikit-learn 1.5 to 1.7 the cell gives a `FutureWarning`. With 1.8 or later, creating the model is expected to fail with a `TypeError` about the unexpected keyword. `requirements.txt` records scikit-learn 1.7.2 as the version it was checked against.

## 13. Expected Output

Saved output from the author's run:

| Step | Saved output |
|---|---|
| Approach 1: `SVC(rbf, C=30)` test accuracy | 0.9778 (the split has no `random_state`, so this changes between runs) |
| `cross_val_score`, linear, C = 10 | `[1.0, 1.0, 0.9, 0.9667, 1.0]` |
| `cross_val_score`, rbf, C = 10 | `[0.9667, 1.0, 0.9667, 0.9667, 1.0]` |
| `cross_val_score`, rbf, C = 20 | `[0.9667, 1.0, 0.9, 0.9667, 1.0]` |
| Loop: mean scores | rbf_1 0.98, rbf_10 0.98, rbf_20 0.9667, linear_1 0.98, linear_10 0.9733, linear_20 0.9667 |
| `GridSearchCV` `best_params_` | `{'C': 1, 'kernel': 'rbf'}` |
| `GridSearchCV` `best_score_` | 0.98 |
| `RandomizedSearchCV` (`n_iter=2`) | C = 10, rbf: 0.98; C = 1, linear: 0.98 (the two combinations are sampled at random each run) |

Saved output of the model comparison:

| Model | Best score | Best parameters |
|---|---|---|
| SVM | 0.9800 | `{'C': 1, 'kernel': 'rbf'}` |
| Random Forest | 0.9533 | `{'n_estimators': 5}` |
| Logistic Regression | 0.9667 | `{'C': 5}` |

The notebook concludes that an SVM with `C=1` and `kernel='rbf'` is the best of the models tried for this problem. The Random Forest has no `random_state`, so its score and best `n_estimators` can change between runs.

The notebook also shows the full `cv_results_` dictionary and DataFrame (fit and score times, per-fold scores, ranks) and the result of `dir(clf)`. The saved `dir(clf)` list includes attributes such as `iid` that current scikit-learn versions no longer have, so that output appears to come from an older scikit-learn installation.

## Changes (22 Nov 2026)

- `6_hyperparameter_tuning.ipynb`: unchanged.
- This README: added the Source and attribution note.
