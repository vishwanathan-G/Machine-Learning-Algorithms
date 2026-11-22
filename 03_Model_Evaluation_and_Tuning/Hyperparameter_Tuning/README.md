# Hyperparameter Tuning

## Files in this folder

| File | Description |
|---|---|
| `hyperparameter_tuning_churn.ipynb` | Tunes three classifiers on the customer-churn practice dataset: `GridSearchCV` for logistic regression, `RandomizedSearchCV` for a random forest and an SVM, all scored with ROC AUC, then compares the tuned models once on a held-out test set |
| `04_logistic_customer_churn.csv` | Copy of the owner's churn practice dataset from [`01_Supervised_Learning/Logistic_Regression`](../../01_Supervised_Learning/Logistic_Regression), so this folder runs on its own (250 rows x 6 columns) |

## Authorship

**`hyperparameter_tuning_churn.ipynb` was written by Claude (an AI assistant) on 22 November 2026, at the repository owner's request, as a reference implementation. It is not the owner's own work, and the owner has not yet reworked it.** The notebook's first cell says the same. The dataset is the owner's own practice dataset.

It replaces `6_hyperparameter_tuning.ipynb`, which closely followed a public tutorial without a licence (see the root README's [Sources and Attribution](../../README.md#sources-and-attribution)). That notebook was removed from this repository; it remains in the owner's local course folder.

## 1. Algorithm Name

Hyperparameter tuning with cross-validation: Grid Search (`GridSearchCV`) and Randomized Search (`RandomizedSearchCV`).

## 2. Definition

Hyperparameters are settings chosen before training, such as an SVM's `C` and `gamma` or the depth of the trees in a forest. The model does not learn them from the data. Hyperparameter tuning searches for the combination that gives the best validation performance, usually measured with K-fold cross-validation.

## 3. How It Works

1. Define the model and a search space, for example `C` in `[0.01, 0.1, 1, 10, 100]` and `class_weight` in `[None, 'balanced']`.
2. **Grid search** tries every combination (here 5 x 2 = 10). **Randomized search** tries only `n_iter` combinations sampled from lists or probability distributions.
3. Each combination is scored with K-fold cross-validation, and the mean score is recorded.
4. The best combination is reported (`best_params_`, `best_score_`). With `refit=True`, the default, the model is retrained on all the training data with those settings (`best_estimator_`).
5. The tuned model is evaluated once on data that played no part in the search.

## 4. Why It Is Used

- Default hyperparameters are rarely the best for a given dataset.
- Manual trial and error is slow and easy to get wrong. Grid and random search are systematic and reproducible.
- Cross-validation inside the search reduces the risk of picking settings that only suit one lucky split.

## 5. Mathematical Intuition

- Grid search evaluates the Cartesian product of the parameter lists: with `p` parameters of `m` values each, there are `m^p` combinations, each fitted `K` times.
- Randomized search fixes the budget at `n_iter * K` fits, regardless of the size of the space. When only a few parameters really matter, random sampling covers their range more efficiently than a grid.
- Parameters such as `C` and `gamma` matter on a multiplicative scale, so they are sampled from a log-uniform distribution: every factor of 10 is equally likely.
- Selection rule: `best = argmax over theta of (1/K) * sum(score_k(theta))`. Picking the best of many combinations on the same folds is slightly optimistic, which is why a separate test set is kept.
- ROC AUC is the probability that a randomly chosen churner gets a higher model score than a randomly chosen non-churner. It does not depend on the 0.5 decision threshold.

## 6. Important Parameters

**General**

| Class | Parameter | Default | Meaning |
|---|---|---|---|
| `GridSearchCV` | `param_grid` | (required) | Dictionary of parameter lists to try |
| `RandomizedSearchCV` | `param_distributions` | (required) | Lists or distributions to sample from |
| `RandomizedSearchCV` | `n_iter` | `10` | Number of combinations sampled |
| both | `cv` | 5-fold | Cross-validation folds |
| both | `scoring` | `None` | `None` uses the estimator's `score` (accuracy for classifiers) |
| both | `refit` | `True` | Retrain the best model on all the training data |
| both | `n_jobs` | `None` | Parallel fits |
| `RandomizedSearchCV` | `random_state` | `None` | Seed for the sampling |

**Used in this notebook**

| Step | Model and settings |
|---|---|
| Split | `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`: 200 training and 50 test rows |
| Folds | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, scoring `roc_auc` everywhere |
| Baseline | `make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))` with `cross_val_score` |
| Logistic regression | `GridSearchCV`, same pipeline, `C: [0.01, 0.1, 1, 10, 100]`, `class_weight: [None, 'balanced']` |
| Random forest | `RandomizedSearchCV(RandomForestClassifier(random_state=42), n_iter=25, random_state=42, n_jobs=-1)` over `n_estimators` 100-499, `max_depth` `[None, 3, 5, 8]`, `min_samples_leaf` 1-9, `max_features` `['sqrt', 'log2', None]` |
| SVM | `RandomizedSearchCV(make_pipeline(StandardScaler(), SVC()), n_iter=25, random_state=42)` over `C` log-uniform 0.01-100, `gamma` log-uniform 0.001-1, `class_weight` `[None, 'balanced']` |

## 7. Advantages

- Grid search is exhaustive within the grid and easy to understand.
- Randomized search scales to large search spaces with a fixed budget.
- Both use cross-validation, so the chosen settings are less likely to overfit one split.
- `cv_results_` records every combination for later analysis.

## 8. Disadvantages

- Grid search cost grows exponentially with the number of parameters.
- Randomized search may miss the best combination.
- Both can be slow for expensive models. Bayesian optimisation or successive halving are alternatives.
- With a small dataset, cross-validation scores are noisy, and the best score is an optimistic estimate. This notebook shows that effect (see Expected Output).

## 9. Real-World Applications

- Tuning gradient-boosting, random forest or SVM models for production.
- Choosing the regularisation strength for linear models.
- Comparing several model families on the same data before deployment, as here for customer churn.
- Tuning neural network learning rates and layer sizes (often with randomized or Bayesian search).

## 10. Python Libraries Used

| Library | Used for |
|---|---|
| `pandas`, `numpy` | Loading the CSV and result tables |
| `scipy.stats` (`randint`, `loguniform`) | Distributions for randomized search |
| `sklearn.model_selection` (`train_test_split`, `StratifiedKFold`, `GridSearchCV`, `RandomizedSearchCV`, `cross_val_score`) | Splitting, cross-validation and search |
| `sklearn.preprocessing.StandardScaler`, `sklearn.pipeline.make_pipeline` | Scaling inside a pipeline |
| `sklearn.linear_model.LogisticRegression`, `sklearn.ensemble.RandomForestClassifier`, `sklearn.svm.SVC` | The three models |
| `sklearn.metrics` (`roc_auc_score`, `f1_score`, `accuracy_score`) | Test-set evaluation |

## 11. Dataset Information

- File: `04_logistic_customer_churn.csv`, 250 rows x 6 columns, a copy of the owner's practice dataset (supplied with the owner's regression and classification practice questions).
- Features: `Monthly_Bill`, `Tenure_Months`, `Support_Calls`, `Late_Payments`, `Monthly_Usage_Hours`.
- Target: `Churn` (1 = the customer left). 25.6% of the customers churn and 74.4% do not, so the classes are imbalanced.

## 12. How to Execute

1. Set up the environment as described in the [root README](../../README.md#getting-started).
2. Run `python -m jupyter notebook` from the repository root and open `03_Model_Evaluation_and_Tuning/Hyperparameter_Tuning/hyperparameter_tuning_churn.ipynb`.
3. Run the cells from top to bottom. There are no input prompts. The searches take about 20 seconds on a laptop CPU.

## 13. Expected Output

Results from running the notebook on 22 Nov 2026 (Python 3.10.11, scikit-learn 1.7.2), saved in the notebook. The split, folds and random searches use `random_state=42`, so a re-run with the same library versions gives the same numbers.

- Baseline logistic regression: mean CV ROC AUC 0.861 (std 0.047).
- Logistic regression grid search: best `C = 1`, `class_weight = None`, mean CV ROC AUC 0.861. The next-best settings (`C = 10` or `100`, with or without `'balanced'`) score 0.860, so the choice makes almost no difference.
- Random forest randomized search: best `max_depth = 3`, `max_features = 'sqrt'`, `min_samples_leaf = 9`, `n_estimators = 148`, mean CV ROC AUC 0.810.
- SVM randomized search: best `C` about 13.15, `gamma` about 0.0029, `class_weight = None`, mean CV ROC AUC 0.858.

**Tuned models on the 50-row test set:**

| Model | Best CV ROC AUC | Test ROC AUC | Test F1 (churn) | Test accuracy |
|---|---|---|---|---|
| Logistic regression (grid) | 0.861 | 0.634 | 0.000 | 0.68 |
| Random forest (randomized) | 0.810 | 0.628 | 0.000 | 0.74 |
| SVM (randomized) | 0.858 | 0.663 | 0.143 | 0.76 |

What these results show:

- **Test scores are much lower than the cross-validation scores** (ROC AUC about 0.63-0.66 against 0.81-0.86). With only 250 customers, and only about a quarter of the 50 test customers churning, a single test set gives a noisy estimate, and the best cross-validation score is an optimistic one.
- **At the default 0.5 threshold the models rarely predict churn.** Logistic regression and the random forest catch no churners on the test set (F1 0.000). The random forest's accuracy of 0.74 equals the share of customers who do not churn, and logistic regression is below it. Accuracy is therefore a misleading score here.
- Possible next steps (not done in the notebook): tune `class_weight` or the decision threshold for churn recall, use repeated cross-validation for steadier estimates, and collect more data.

## Changes (22 Nov 2026)

- Added `hyperparameter_tuning_churn.ipynb`, a reference implementation written by Claude at the owner's request (see Authorship), and a copy of the churn practice dataset.
- Removed `6_hyperparameter_tuning.ipynb`, which closely followed an unlicensed public tutorial. It remains in the owner's local course folder.
- This README was rewritten for the new notebook. The scikit-learn version note about `multi_class` no longer applies, because that notebook is gone.
