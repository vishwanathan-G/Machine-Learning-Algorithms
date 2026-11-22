# Polynomial Regression

## Files in this folder

| File | Description |
|---|---|
| `poly_reg.ipynb` | Compares plain linear regression with polynomial regression of degree 3 and 4 on inline experience/salary data, plots all fits and predicts a salary for a user-entered number of years |
| `polynomial_Regression_machine_temperature.ipynb` | Polynomial regression of degree 2 and degree 3 for machine temperature from operating hours: R2 of both models on a hold-out split, a prediction from each for user-entered operating hours, and a plot of both fits |
| `03_polynomial_machine_temperature.csv` | 250 rows x 2 columns (`Operating_Hours`, `Machine_Temperature_C`) |

## Source

`polynomial_Regression_machine_temperature.ipynb` is the author's own work, written as a solution to the same set of regression practice questions as the new notebooks in `../Linear_Regression` and `../Logistic_Regression`. `03_polynomial_machine_temperature.csv` is the practice dataset supplied with those questions.

## 1. Algorithm Name

Polynomial Regression, supervised learning, regression.

## 2. Definition

Polynomial regression fits a curve instead of a straight line. It expands the input `x` into powers (`x, x^2, ..., x^d`) and then fits an ordinary linear regression on those expanded features. The model is still linear in its coefficients, which is why it can be solved with `LinearRegression`.

## 3. How It Works

1. Choose a degree `d`.
2. Transform each input value into the feature vector `[1, x, x^2, ..., x^d]` (`PolynomialFeatures`).
3. Fit linear regression on the transformed features by least squares.
4. To predict, transform the new input with the same transformer and call `predict`.

## 4. Why It Is Used

- Many relationships curve: growth that speeds up, diminishing returns, saturation.
- It is an easy step up from linear regression and reuses the same fitting method.
- Comparing degrees on one plot shows clearly how model flexibility changes the fit.

## 5. Mathematical Intuition

- Degree-`d` model: `y_hat = b0 + b1*x + b2*x^2 + ... + bd*x^d`.
- The cost is the same as linear regression: `MSE = (1/n) * sum((y - y_hat)^2)`.
- A higher degree lowers the training error but increases variance. Beyond the training range the highest power dominates, so the curve can rise or fall steeply. This is the bias-variance trade-off.

## 6. Important Parameters

**General (`sklearn.preprocessing.PolynomialFeatures`)**

| Parameter | Default | Meaning |
|---|---|---|
| `degree` | `2` | Highest power to generate |
| `include_bias` | `True` | Add a column of ones (the `x^0` term) |
| `interaction_only` | `False` | With several features, keep only interaction terms such as `x1*x2` |

`LinearRegression` parameters are listed in [`../Linear_Regression/README.md`](../Linear_Regression/README.md#6-important-parameters).

**Used in `poly_reg.ipynb`**

- `LinearRegression()` on the raw `Experience` column (baseline).
- `PolynomialFeatures(degree=3)` followed by `LinearRegression()`.
- `PolynomialFeatures(degree=4)` followed by `LinearRegression()`.
- All models are trained on all 20 rows (no train/test split).
- Curves are drawn on a grid `np.arange(X.min(), X.max() + 0.1, 0.1)` so they look smooth.

**Used in `polynomial_Regression_machine_temperature.ipynb`**

- Feature `Operating_Hours`, target `Machine_Temperature_C`.
- `PolynomialFeatures(degree=2)` and `PolynomialFeatures(degree=3)`, other parameters at their defaults (so `include_bias=True`). Each is applied with `fit_transform` to the whole `Operating_Hours` column before the split.
- `train_test_split(..., train_size=0.8, random_state=42)` on each expanded matrix, which gives 200 training and 50 test rows. With the same `random_state`, both models use the same rows.
- `LinearRegression()` with all defaults for each degree, evaluated with `r2_score` on the test set and printed as a percentage.
- Curves are drawn on `np.linspace(x.min(), x.max(), 300)` (300 points across the data range).

## 7. Advantages

- Captures curved relationships with a simple, well-understood method.
- Reuses linear regression tooling and its closed-form solution.
- The degree is a single, easy-to-understand control over flexibility.

## 8. Disadvantages

- High degrees overfit and oscillate, especially with few data points.
- Extrapolation is unreliable: predictions outside the training range can grow very quickly.
- The number of features grows quickly when there are many input columns.
- Powers of large values differ greatly in scale; scaling often helps numerically.

## 9. Real-World Applications

- Salary or price curves that rise faster at higher levels.
- Growth curves in biology and population studies.
- Calibration curves for sensors and instruments.
- Modelling non-linear trends in physics and engineering measurements, such as machine temperature over operating time.

## 10. Python Libraries Used

| Library | Used for | Notebooks |
|---|---|---|
| `numpy` | Building the smooth plotting grid (`np.arange` in `poly_reg`, `np.linspace` in the machine temperature notebook) | both |
| `pandas` | Building the DataFrame (`poly_reg`); loading the CSV (machine temperature) | both |
| `matplotlib.pyplot` | 2 x 2 grid of plots (`poly_reg`); 1 x 2 grid of plots (machine temperature) | both |
| `sklearn.linear_model.LinearRegression` | Linear and polynomial fits | both |
| `sklearn.preprocessing.PolynomialFeatures` | Creating polynomial features | both |
| `sklearn.model_selection.train_test_split` | Hold-out split | machine temperature |
| `sklearn.metrics.r2_score` | Evaluation | machine temperature |

## 11. Dataset Information

**`poly_reg.ipynb`**

- Inline data defined in the notebook: 20 rows x 2 columns.
- `Experience`: 1 to 20 (years).
- `Salary`: 50,000 to 400,000. It rises slowly for the first few years and then by about 20,000 per year.
- No external file is needed.

**`polynomial_Regression_machine_temperature.ipynb`**

- File: `03_polynomial_machine_temperature.csv`, 250 rows x 2 columns, no missing values. Practice dataset supplied with the author's regression practice questions.
- `Operating_Hours`: 1.00 to 11.92.
- `Machine_Temperature_C`: 39.94 to 166.15. The temperature rises faster as the hours increase, so the relationship curves upward.

## 12. How to Execute

1. Set up the environment as described in the [root README](../../README.md#getting-started).
2. Run `python -m jupyter notebook` from the repository root and open `01_Supervised_Learning/Polynomial_Regression/` and then a notebook. Jupyter runs the notebook from its own folder, so the relative CSV path resolves.
3. Run the cells from top to bottom.
4. Answer the prompts in the text box under the cell and press Enter:

| Notebook | Prompt | What to type |
|---|---|---|
| `poly_reg.ipynb` (last cell) | `Enter years of experience:` | A number such as `10` or `12.5`. The training data covers 1 to 20 years, so values in that range show how the models interpolate. Values outside it show how they extrapolate. |
| `polynomial_Regression_machine_temperature.ipynb` (two cells: degree 2, then degree 3) | `Enter Operating Hours:` (asked twice) | A number such as `6` or `9.5`. The data covers 1.00 to 11.92 hours; values outside that range are extrapolations. |

In both notebooks the input is read with `float(...)`, so a non-numeric entry raises a `ValueError`.

## 13. Expected Output

### `poly_reg.ipynb`

- The 20-row DataFrame.
- A 2 x 2 figure of plots, each with the actual salaries as blue points:
  1. Linear regression (red line)
  2. Polynomial regression, degree 3 (green curve)
  3. Linear regression again (the third subplot repeats the first)
  4. Polynomial regression, degree 4 (purple curve)
- After the input, three printed predictions: linear, degree 3 and degree 4.
- Saved output from the author's run (input `25` years, outside the training range):

| Model | Saved prediction for 25 years |
|---|---|
| Linear | $468,784.21 |
| Polynomial, degree 3 | $479,299.21 |
| Polynomial, degree 4 | $630,984.31 |

The spread between the three values for an input beyond the data shows how differently the models extrapolate.

### `polynomial_Regression_machine_temperature.ipynb`

- The first five rows of the data.
- The R2 score of each model on the 50 test rows, printed as `Degree 2 R² Score: <value>%` and `Degree 3 R² Score: <value>%`.
- After each input: `Predicted Temperature (Degree 2): <value> °C` or `Predicted Temperature (Degree 3): <value> °C`, followed by a scikit-learn `UserWarning` that `X does not have valid feature names, but PolynomialFeatures was fitted with feature names` (a plain list is passed to `transform`). The plot cell shows the same warning twice.
- A 1 x 2 figure: the actual data as blue points with the degree 2 fit (red dashed curve, left) and the degree 3 fit (red solid curve, right).
- Saved output from the author's run:

| Cell | Saved output |
|---|---|
| R2, degree 2 (test set) | `Degree 2 R² Score: 97.94%` |
| R2, degree 3 (test set) | `Degree 3 R² Score: 97.93%` |
| Prediction, degree 2 | `Predicted Temperature (Degree 2): 470.70 °C` |
| Prediction, degree 3 | `Predicted Temperature (Degree 3): 359.73 °C` |

The operating hours typed were not recorded in the saved output. Both predictions are far above the highest temperature in the data (166.15 °C), so they are extrapolations beyond the range the models were trained on. Inside that range the two fits are almost the same, as their R2 scores show; outside it they can differ widely. Because the split uses `random_state=42`, every run uses the same training and test rows.

## Changes (22 Nov 2026)

- **Added** `polynomial_Regression_machine_temperature.ipynb` and `03_polynomial_machine_temperature.csv`, the author's own notebook and its practice dataset (see [Source](#source)). The saved outputs are the author's original run and were not regenerated. The Windows username in the file paths of the saved warning outputs was replaced with `<user>`.
- **Test** (Python 3.10.11, scikit-learn 1.7.2, run from top to bottom on a copy): 10 of 10 code cells ran (inputs 24 and 24).
- `poly_reg.ipynb` is unchanged.
- This README: the files table and sections 6 and 9 to 13 were updated for both notebooks, and the Source note was added.
