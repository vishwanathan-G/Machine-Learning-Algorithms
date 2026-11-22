# Logistic Regression

## Files in this folder

| File | Description |
|---|---|
| `3_logistic_regression.ipynb` | Binary classification: predicts whether a person buys insurance from their age, then works through the sigmoid calculation by hand; asks for an age |
| `insurance_data.csv` | 400 rows x 5 columns (User ID, Gender, Age, EstimatedSalary, Purchased) |

## Source and attribution

`3_logistic_regression.ipynb` follows the codebasics "py" logistic regression tutorial ([github.com/codebasics/py](https://github.com/codebasics/py), folder `ML/7_logistic_reg`). A comparison gave a code similarity score of 0.85, and 70% of the code lines are identical. The codebasics/py repository has no licence file.

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
- With one feature, the decision boundary is the value of `x` where `z = 0`, i.e. `x = -b0 / b1`.
- The notebook's markdown describes `coef_` and `intercept_` as `m` and `b` in `y = m*x + b`. In logistic regression that line gives `z`, the input to the sigmoid, not the prediction itself. The notebook's own `prediction_function` applies the sigmoid to it.

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

- Feature: `Age` only. Target: `Purchased`.
- `train_test_split(df[['Age']], df['Purchased'], train_size=0.8)`, which gives 320 training and 80 test rows. No `random_state` is set, so the split changes on every run.
- `LogisticRegression()` with all defaults.
- A hand-written `sigmoid` and `prediction_function(age)` using the rounded coefficients `0.042` and `-1.53`.

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
| `pandas` | Loading the CSV |
| `matplotlib.pyplot` | Scatter plot of age vs purchase |
| `sklearn.model_selection.train_test_split` | Train/test split |
| `sklearn.linear_model.LogisticRegression` | The model |
| `math` (standard library) | `math.exp` in the hand-written sigmoid |

## 11. Dataset Information

- File: `insurance_data.csv`, 400 rows x 5 columns, no missing values.
- Columns: `User ID`, `Gender` (204 Female, 196 Male), `Age` (18 to 60), `EstimatedSalary` (15,000 to 150,000), `Purchased` (0 = 257 rows, 1 = 143 rows).
- Only `Age` and `Purchased` are used by the notebook.
- The columns and row count appear to match a widely used "Social Network Ads" practice dataset. Its exact origin is not documented in the notebook.

## 12. How to Execute

1. Set up the environment as described in the [root README](../../README.md#getting-started).
2. Run `python -m jupyter notebook` from the repository root and open `01_Supervised_Learning/Logistic_Regression/3_logistic_regression.ipynb`.
3. Run the cells from top to bottom.
4. The prediction cell asks `Enter the age:`. Type a whole number such as `30` or `50` and press Enter (the data covers ages 18 to 60). A decimal or non-numeric entry raises a `ValueError` because the code uses `int(...)`.

## 13. Expected Output

- The 400-row DataFrame and a scatter plot of `Purchased` (0 or 1) against `Age`, drawn with red `+` markers.
- `y_train`, `X_test`, the predicted labels for the 80 test rows and `predict_proba` (two probabilities per row).
- `model.score(X_test, y_test)`: the accuracy on the test set.
- After the age input, one of: `The person will purchase the insurance` or `The person will not purchase the insurance`.
- `coef_`, `intercept_`, and the hand-calculated probabilities for ages 35 and 43.

Saved output from the author's run:

| Cell | Saved output |
|---|---|
| `model.score(X_test, y_test)` | 0.8375 |
| `model.coef_` | 0.04150133 |
| `model.intercept_` | -1.52726963 |
| `prediction_function(35)` | 0.485 (below 0.5, so "will not buy") |
| `prediction_function(43)` | 0.569 (above 0.5, so "will buy") |
| Age input cell | `The person will purchase the insurance` (the age typed was not recorded) |

Because the split has no `random_state`, a new run gives a different split, score and coefficients. The saved outputs also come from more than one run: the two cells that display `y_predicted` show different arrays.

The last markdown cell reads "0.485 is more than 0.5" for age 43. The value computed for 43 is 0.569.

## Changes (22 Nov 2026)

- `3_logistic_regression.ipynb`: the Windows username in a file path inside a saved warning output was replaced with `<user>` (for example `C:\Users\<user>\AppData\...`). The warning text is otherwise unchanged. The code and the other saved outputs are unchanged.
- This README: added the Source and attribution note.
