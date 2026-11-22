# Linear Regression

## Files in this folder

| File | Description |
|---|---|
| `1_linear_regression.ipynb` | Simple linear regression: house price from area (6-row `houseprice.csv`) |
| `1_1linear_regression.ipynb` | Multiple linear regression on 50,000 rows with an 80/20 train/test split and MAE, MSE, RMSE, R2 |
| `2_linear_regression_multivariate.ipynb` | Multiple linear regression on `houseprice.csv` after filling a missing value with the median |
| `multi_linear_reg.ipynb` | Multiple linear regression for CO2 emissions from inline vehicle data; asks for input |
| `houseprice.csv` | 6 rows x 4 columns (area, bedrooms, age, price) |
| `homeprices.csv` | 50,000 rows x 4 columns (area, bedrooms, age, price) |

## 1. Algorithm Name

Linear Regression (simple and multiple), supervised learning, regression.

## 2. Definition

Linear regression models a continuous target as a weighted sum of input features plus a constant. Simple linear regression uses one feature. Multiple (multivariate) linear regression uses two or more.

## 3. How It Works

1. Assume the target is roughly a straight-line (or flat-plane) function of the features.
2. Choose the weights (coefficients) and the intercept that make the predicted values as close as possible to the actual values, by minimising the sum of squared errors (ordinary least squares).
3. Use the fitted equation to predict the target for new inputs.

## 4. Why It Is Used

- It is the standard first model for a numeric target and a baseline for more complex models.
- The coefficients are directly interpretable: each one is the change in the prediction for a one-unit change in that feature, holding the others fixed.
- It trains almost instantly, even on large tables.

## 5. Mathematical Intuition

- Simple: `y_hat = m * x + b`, where `m` is the slope (`coef_`) and `b` is the intercept (`intercept_`).
- Multiple: `y_hat = b0 + b1*x1 + b2*x2 + ... + bn*xn`.
- Cost function (mean squared error): `MSE = (1/n) * sum((y - y_hat)^2)`.
- Closed-form least-squares solution: `w = (X^T X)^-1 X^T y`. Gradient descent reaches the same minimum iteratively.
- Common metrics:
  - `MAE = mean(|y - y_hat|)`
  - `RMSE = sqrt(MSE)`, in the same units as the target
  - `R2 = 1 - SS_res / SS_tot`, the share of the target's variance the model explains (1.0 is a perfect fit)

## 6. Important Parameters

**General (`sklearn.linear_model.LinearRegression`)**

| Parameter | Default | Meaning |
|---|---|---|
| `fit_intercept` | `True` | Whether to learn the constant term `b` |
| `positive` | `False` | Force all coefficients to be non-negative |
| `copy_X` | `True` | Copy the input instead of overwriting it |
| `n_jobs` | `None` | Number of CPU cores for large multi-target problems |

**Used in these notebooks**

- `LinearRegression()` with all defaults in every notebook.
- `1_1linear_regression.ipynb`: `train_test_split(X, y, test_size=0.2, random_state=42)`.
- `2_linear_regression_multivariate.ipynb`: missing `bedrooms` value filled with the column median (`fillna(df.bedrooms.median())`).
- The other notebooks train on all rows (no train/test split).

## 7. Advantages

- Simple, fast and easy to explain.
- Coefficients show the direction and size of each feature's effect.
- Has an exact solution; no tuning is needed for the basic model.
- Works well when the relationship really is close to linear.

## 8. Disadvantages

- Cannot capture curved relationships unless features are transformed (see `../Polynomial_Regression`).
- Sensitive to outliers, because errors are squared.
- Correlated features (multicollinearity) make individual coefficients unstable.
- Predictions outside the range of the training data (extrapolation) can be unreliable.
- Assumes errors with roughly constant variance and no strong dependence between observations.

## 9. Real-World Applications

- Estimating house or property prices from size, age and rooms.
- Forecasting sales or demand from spend and seasonality features.
- Estimating vehicle fuel use or emissions from engine characteristics.
- Trend analysis in finance, economics and operations.

## 10. Python Libraries Used

| Library | Used for | Notebooks |
|---|---|---|
| `pandas` | Loading CSVs, building DataFrames | all |
| `numpy` | `np.sqrt` for RMSE | `1_1linear_regression`, also imported in `1_linear_regression` and `2_linear_regression_multivariate` |
| `matplotlib.pyplot` | Line plot, scatter plot, histogram | `1_linear_regression`, `1_1linear_regression` |
| `sklearn.linear_model` (`LinearRegression`, `linear_model`) | The model | all |
| `sklearn.model_selection.train_test_split` | Hold-out split | `1_1linear_regression` |
| `sklearn.metrics` (`mean_absolute_error`, `mean_squared_error`, `r2_score`) | Evaluation | `1_1linear_regression` |

## 11. Dataset Information

| Notebook | Data | Size | Notes |
|---|---|---|---|
| `1_linear_regression.ipynb` | `houseprice.csv` | 6 rows x 4 columns | Only `area` (feature) and `price` (target) are used. Practice table; origin not documented. |
| `2_linear_regression_multivariate.ipynb` | `houseprice.csv` | 6 rows x 4 columns | Features `area`, `bedrooms`, `age`; target `price`. One `bedrooms` value is missing (`NaN`) and is filled with the median (4.0). |
| `1_1linear_regression.ipynb` | `homeprices.csv` | 50,000 rows x 4 columns | Features `area` (800 to 5,000), `bedrooms` (1 to 7), `age` (0 to 50); target `price` (217,000 to 967,000). No missing values. Appears to be synthetic data; origin not documented. |
| `multi_linear_reg.ipynb` | Inline Python dictionary | 9 rows x 4 columns | `ENGINESIZE`, `CYLINDERS`, `FUELCONSUMPTION_COMB` (features) and `CO2EMISSIONS` (target). A code comment says the values were typed in from an image, leaving out a row with a missing CO2 value. |

The markdown "Exercise" cells at the end of `1_linear_regression.ipynb` and `2_linear_regression_multivariate.ipynb` mention `canada_per_capita_income.csv` and `hiring.csv`. No code uses those files, and they are not in this repository.

## 12. How to Execute

1. Set up the environment as described in the [root README](../../README.md#getting-started).
2. From the repository root run `python -m jupyter notebook`.
3. Open `01_Supervised_Learning/Linear_Regression/` and then a notebook. Jupyter runs the notebook from its own folder, so the relative CSV paths resolve.
4. Run the cells from top to bottom (Shift+Enter, or Kernel > Restart & Run All).

`multi_linear_reg.ipynb` asks for three values in a text box under the cell. Press Enter after each one:

| Prompt | What to type | Range in the inline data |
|---|---|---|
| `Enter Engine Size (L):` | a decimal number, e.g. `2.4` | 1.5 to 3.7 |
| `Enter Number of Cylinders:` | a whole number, e.g. `4` | 4 or 6 |
| `Enter Fuel Consumption (L/100km):` | a decimal number, e.g. `9.6` | 5.9 to 11.6 |

If you type something that is not a number, the cell prints `Invalid input. Please enter numeric values.`

## 13. Expected Output

### `1_linear_regression.ipynb`

- The DataFrame from `houseprice.csv` and a red line plot of `price` against `area` (pentagon markers).
- `model.predict` for two areas, plus `coef_` and `intercept_`, and a hand calculation of `m * x + b` for an area of 3,200.
- Saved output from the author's run:

| Cell | Saved output |
|---|---|
| `model.predict([[5000]])` | 913,240.12 |
| `model.predict([[4000]])` | 745,930.57 |
| `model.coef_` | 167.30954677 |
| `model.intercept_` | 76,692.38 |
| `3200*167.30954677 + 76692.3818707813` | 612,082.93 |

- The markdown headings above the prediction cells mention 3,300 and 5,000 sq ft. The code predicts for 5,000 and 4,000 sq ft. A scikit-learn `UserWarning` about missing feature names appears because plain lists are passed to `predict`.

### `1_1linear_regression.ipynb`

- Prints the shape `(50000, 4)`, a missing-value count per column (all 0) and `Train size: 40000 | Test size: 10000`.
- Prints MAE, MSE, RMSE and R2 on the test set.
- Shows two plots side by side: actual vs predicted price (with a red dashed diagonal), and a 50-bin histogram of residuals.
- Predicts a price for a new house with area 3000, 2 bedrooms, age 40.
- Saved output from the author's run:

| Metric (test set) | Saved output |
|---|---|
| MAE | 20,074.53 |
| MSE | 633,903,560.84 |
| RMSE | 25,177.44 |
| R2 | 0.9750 |
| Predicted price (3000, 2, 40) | 550,059.8 |

### `2_linear_regression_multivariate.ipynb`

- Shows the median of `bedrooms` (4.0), the DataFrame after filling, the feature matrix and the target.
- Saved output from the author's run: coefficients `[112.06244194, 23388.88007794, -3231.71790863]` for `area`, `bedrooms`, `age`, and intercept `221323.0018654043`. `model.predict([[5000, 3, 12]])` gives 813,021.24, and the hand calculation in the next cell gives the same value.
- Notes on the saved outputs:
  - The cell whose code is `model.intercept_` shows the three coefficients, and the cell whose code is `model.coef_` shows the intercept. The saved outputs appear to come from before the code of those two cells was swapped.
  - The last prediction cell calls `reg.predict([[1000, 3, 20]])`, but the model in this notebook is named `model`. Its saved output (338,917.73) comes from an earlier session. In a fresh kernel that cell raises a `NameError` because `reg` is never defined.
  - The markdown headings mention different inputs (3,000 sq ft / 3 bedrooms / 40 years, and 2,500 / 4 / 5) from the ones the code uses.

### `multi_linear_reg.ipynb`

- Shows the 9-row inline DataFrame, fits the model, then asks for the three inputs and prints them with `Predicted CO₂ Emission: <value> g/km`.
- Saved output from the author's run: `Engine Size: 2.4` and `Predicted CO₂ Emission: 198.37 g/km`. The other two input values were not recorded. The current code also prints the cylinder and fuel-consumption values, which the saved output does not include, so it comes from an earlier version of the cell.
