# Logistic Regression

## Files in this folder

| File | Description |
|---|---|
| `logistic_Regression_customer_churn.ipynb` | Binary classification: predicts whether a customer churns from five account features, with feature scaling, accuracy, a classification report, a confusion matrix and a ROC curve; then predicts a hard-coded sample customer |
| `04_logistic_customer_churn.csv` | 250 rows x 6 columns (`Monthly_Bill`, `Tenure_Months`, `Support_Calls`, `Late_Payments`, `Monthly_Usage_Hours`, `Churn`) |

## Source

`logistic_Regression_customer_churn.ipynb` is the author's own work. It comes from the author's own practice repository (`vishwanathan-G/ML_project`), where it was written as a solution to a set of regression practice questions; `04_logistic_customer_churn.csv` is the practice dataset supplied with those questions. It replaced a notebook that closely followed the codebasics logistic regression tutorial (see [Changes](#changes-22-nov-2026)). A comparison with the tutorial sources gave code similarity scores of 0.30 to 0.37 for the replacement notebooks.

## 1. Algorithm Name

Logistic Regression, supervised learning, classification.

## 2. Definition

Logistic regression is a classification model that estimates the probability that an input belongs to the positive class. It computes a linear combination of the features and passes it through the sigmoid function, which maps any number into the range 0 to 1.

## 3. How It Works

1. Compute a linear score `z = b0 + b1*x1 + ... + bn*xn`.
2. Convert the score to a probability with the sigmoid `p = 1 / (1 + e^(-z))`.
3. Learn the coefficients by minimising log-loss (cross-entropy) with an iterative solver. scikit-learn uses `lbfgs` by default and adds L2 regularisation.
4. Predict class 1 when `p >= 0.5`, otherwise class 0.

## 4. Why It Is Used

- It is the standard baseline for yes/no problems.
- It gives probabilities, not just labels, so the decision threshold can be moved.
- The coefficients are interpretable as changes in log-odds.

## 5. Mathematical Intuition

- Sigmoid: `sigmoid(z) = 1 / (1 + e^(-z))`. `sigmoid(0) = 0.5`.
- Log-odds are linear in the features: `log(p / (1 - p)) = b0 + b1*x1 + ...`.
- Log-loss: `L = -(1/n) * sum(y*log(p) + (1 - y)*log(1 - p))`.
- The decision boundary is where `z = 0` (`p = 0.5`). With one feature this is the value `x = -b0 / b1`; with several features it is a flat hyperplane.
- Standardisation (used in the notebook): `x_scaled = (x - mean) / std`, with the mean and standard deviation of each feature taken from the training set.
- ROC curve: the true positive rate `TP / (TP + FN)` plotted against the false positive rate `FP / (FP + TN)` as the threshold moves from 1 to 0. AUC is the area under that curve (0.5 for random ranking, 1.0 for a perfect one).

## 6. Important Parameters

**General (`sklearn.linear_model.LogisticRegression`)**

| Parameter | Default | Meaning |
|---|---|---|
| `C` | `1.0` | Inverse regularisation strength; smaller means stronger regularisation |
| `penalty` | `'l2'` | Type of regularisation (deprecated in scikit-learn 1.8, where `l1_ratio` controls the L1/L2 mix) |
| `solver` | `'lbfgs'` | Optimisation algorithm |
| `max_iter` | `100` | Maximum solver iterations |
| `class_weight` | `None` | Re-weight classes, e.g. `'balanced'` for imbalanced data |

**Used in this notebook**

- Features: `Monthly_Bill`, `Tenure_Months`, `Support_Calls`, `Late_Payments`, `Monthly_Usage_Hours` (every column except the target). Target: `Churn` (1 = churn, 0 = no churn).
- `train_test_split(x, y, train_size=0.8, random_state=42)`, which gives 200 training and 50 test rows. The fixed `random_state` gives the same split on every run.
- `StandardScaler()`: `fit_transform` on the training features, `transform` on the test features and on the sample customer.
- `LogisticRegression(max_iter=1000)`; all other parameters keep their defaults (`C=1.0`, `solver='lbfgs'`, L2 regularisation, no class weights).
- `model.predict` uses the default 0.5 threshold. `model.predict_proba(...)[:, 1]` gives the churn probability, used for the ROC curve, the AUC and the sample customer.

## 7. Advantages

- Fast to train and to predict.
- Outputs probabilities that are easy to threshold.
- Coefficients are interpretable.
- Regularisation is built in.

## 8. Disadvantages

- The decision boundary is linear in the features unless features are engineered.
- Sensitive to strongly correlated features and to outliers.
- Can underfit complex relationships.
- Needs feature scaling for good solver behaviour when features have very different ranges.

## 9. Real-World Applications

- Predicting whether a customer will buy a product or respond to an offer.
- Credit default and fraud risk scoring.
- Medical risk prediction (disease present / absent).
- Email spam detection and churn prediction.

## 10. Python Libraries Used

| Library | Used for |
|---|---|
| `pandas` | Loading the CSV; building the one-row DataFrame for the sample customer |
| `sklearn.model_selection.train_test_split` | Train/test split |
| `sklearn.preprocessing.StandardScaler` | Feature scaling |
| `sklearn.linear_model.LogisticRegression` | The model |
| `sklearn.metrics` (`accuracy_score`, `classification_report`, `confusion_matrix`, `roc_curve`, `roc_auc_score`) | Evaluation |
| `matplotlib.pyplot` | Figure with two panels; the ROC curve |
| `seaborn` | Heatmap of the confusion matrix |

## 11. Dataset Information

- File: `04_logistic_customer_churn.csv`, 250 rows x 6 columns, no missing values. Practice dataset supplied with the author's regression practice questions.
- Columns: `Monthly_Bill` (20.34 to 179.95), `Tenure_Months` (2 to 72), `Support_Calls` (0 to 9), `Late_Payments` (0 to 5), `Monthly_Usage_Hours` (5.01 to 79.74), `Churn` (0 = 186 rows, 1 = 64 rows).
- The classes are imbalanced: about a quarter of the customers churn. The 50-row test set has 36 non-churn and 14 churn rows (the `support` column of the classification report).

## 12. How to Execute

1. Set up the environment as described in the [root README](../../README.md#getting-started). The notebook also needs `seaborn`, which `requirements.txt` installs.
2. Run `python -m jupyter notebook` from the repository root and open `01_Supervised_Learning/Logistic_Regression/logistic_Regression_customer_churn.ipynb`.
3. Run the cells from top to bottom.
4. The notebook does not ask for input. To score a different customer, change the values in the `sample_customer` cell (the second-to-last code cell) and run it again.

## 13. Expected Output

- The first five rows of the data (in plain text pandas shortens the column list with `...`).
- `LogisticRegression(max_iter=1000)` after fitting.
- `Accuracy Score: <value>%` and a classification report for the 50 test rows.
- A figure with two panels: the confusion matrix as a heatmap (`No Churn` / `Churn` labels), and the ROC curve with the AUC in its legend and a dashed diagonal for random ranking.
- For the sample customer: `Prediction Result: Churn` or `No Churn`, and `Churn Probability: <value>%`.

Saved output (regenerated on 22 Nov 2026, see [Changes](#changes-22-nov-2026)):

| Cell | Saved output |
|---|---|
| Accuracy (test set, 50 rows) | `Accuracy Score: 72.00%` |
| Class 0 (no churn): precision / recall / F1 | 0.76 / 0.89 / 0.82 (support 36) |
| Class 1 (churn): precision / recall / F1 | 0.50 / 0.29 / 0.36 (support 14) |
| Confusion matrix (rows = true, columns = predicted) | No Churn: 32 correct, 4 predicted as Churn. Churn: 10 predicted as No Churn, 4 correct. |
| ROC curve legend | `ROC Curve (AUC = 0.76)` |
| Sample customer (`Monthly_Bill` 120.50, `Tenure_Months` 3, `Support_Calls` 5, `Late_Payments` 2, `Monthly_Usage_Hours` 15.0) | `Prediction Result: Churn`, `Churn Probability: 82.34%` |

The model finds 4 of the 14 churners in the test set (recall 0.29 for class 1); most of its errors are churners predicted as staying. Because the split uses `random_state=42`, a new run in the same environment gives the same values.

## Changes (22 Nov 2026)

- **Notebook replacement.** `3_logistic_regression.ipynb` and `insurance_data.csv` were removed from the repository. The notebook closely followed the codebasics "py" logistic regression tutorial (code similarity 0.85). `logistic_Regression_customer_churn.ipynb` and `04_logistic_customer_churn.csv` were added: the author's own implementation (see [Source](#source)).
- **Fix.** Cell 6 (the plot cell) drew the ROC curve from `y_proba`, which the notebook never defined, so a fresh run raised a `NameError`. One line was added to cell 5 (the metrics cell): `y_proba = model.predict_proba(X_test_scaled)[:, 1]`. The notebook was then re-run and its saved outputs regenerated. They match the author's earlier saved run in 8 of 9 code cells; the 9th, the data preview, differs only in pandas' plain-text wrapping.
- **Test** (Python 3.10.11, scikit-learn 1.7.2, run from top to bottom on a copy): after the fix, 9 of 9 code cells ran.
- This README: the Source and attribution note about the removed notebook was replaced by the [Source](#source) note, and the files table and sections 5, 6 and 10 to 13 were rewritten for the new notebook.
