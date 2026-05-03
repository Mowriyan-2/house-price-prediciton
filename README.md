# 🏠 House Price Prediction

An end-to-end machine learning pipeline to predict California median house prices using the [California Housing Prices dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices) from the 1990 U.S. Census.

---

## 📌 Project Overview

This project tackles a real-world regression problem: given a census block in California, predict its median house value. It walks through the full ML lifecycle — from raw data exploration to a tuned, deployable model — using clean, production-style code with sklearn pipelines.

The pipeline follows this structured sequence:

**EDA → Preprocessing → Baseline Model → Model Selection (CV) → Hyperparameter Tuning → Final Evaluation → Inference**

Each stage builds on the previous one. Design decisions are justified by data insights rather than guesswork — for example, the choice of imputation strategy comes from EDA, the choice of model comes from cross-validation, and the final hyperparameters come from a systematic grid search.

---

## 📂 Dataset

| Detail | Info |
|---|---|
| Source | [Kaggle – California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices) |
| File | `housing.csv` |
| Rows | 20,640 |
| Target | `median_house_value` (USD) |
| Era | 1990 U.S. Census |

> **Important:** Each row represents a **census block** — a small geographic unit — not an individual house. Features like `total_rooms` and `population` refer to the entire block, not a single property. This distinction matters when engineering features later.

### Features

| Feature | Type | Description |
|---|---|---|
| `longitude` | Numerical | How far west the block is; higher value = further west |
| `latitude` | Numerical | How far north the block is; higher value = further north |
| `housing_median_age` | Numerical | Median age of houses in the block |
| `total_rooms` | Numerical | Total number of rooms across all houses in the block |
| `total_bedrooms` | Numerical | Total bedrooms in the block — **has missing values** |
| `population` | Numerical | Total number of people residing in the block |
| `households` | Numerical | Total number of households in the block |
| `median_income` | Numerical | Median household income (in tens of thousands of USD) |
| `ocean_proximity` | Categorical | Distance to ocean: `NEAR BAY`, `INLAND`, `<1H OCEAN`, `NEAR OCEAN`, `ISLAND` |

---

## 🔍 Exploratory Data Analysis (EDA)

EDA is the foundation of the entire project. Before writing a single line of model code, the data is thoroughly explored to understand its structure, distributions, quirks, and relationships. Every preprocessing and modelling decision downstream is informed by these findings.

### Missing Values
Only `total_bedrooms` has missing values — approximately 207 rows (~1% of the dataset). Since the missingness is small and likely random (not systematically related to house price), **median imputation** is the right choice. Mean imputation would be pulled by the skewed outliers in this column; dropping the rows would discard valid training data.

### Target Distribution
`median_house_value` is **right-skewed** with a hard artificial cap at **$500,000**. This cap is a data collection artefact — any house worth more than $500K was simply recorded as $500,000. This means hundreds of expensive blocks (especially in coastal California) are all clumped at exactly $500K even though their true values differ. The model cannot distinguish between them, which creates a systematic blind spot at the upper end of the price range.

### Feature Distributions
Several features (`total_rooms`, `total_bedrooms`, `population`, `households`) are heavily right-skewed with large outliers. For example, some blocks report over 30,000 total rooms — likely large apartment complexes. Tree-based models handle these outliers natively without transformation, but linear models are significantly affected. This is one reason tree-based models outperform linear ones on this dataset.

### Correlation Analysis
The heatmap reveals two important structural patterns in the data:

- **`median_income` is the strongest predictor** of house value, with a correlation of ~0.69. This makes strong intuitive sense — wealthier neighbourhoods have higher property values. No other single feature comes close.
- **High multicollinearity** exists among block-level count features. `total_rooms`, `total_bedrooms`, `households`, and `population` are all highly correlated with each other because they all scale with the size of the block. This redundancy hurts linear models (inflated coefficients, unstable fits) but does not affect tree-based models, since trees select the most informative feature independently at each split.

### Ocean Proximity
`INLAND` blocks are the most common category and have the lowest median prices. `ISLAND` is rare and extremely expensive. This categorical feature adds meaningful geographic pricing context beyond what raw coordinates alone can capture.

---

## ⚙️ Preprocessing Pipeline

All preprocessing is wrapped inside a `sklearn.pipeline.Pipeline` combined with a `ColumnTransformer`. This architecture is intentional and critical — it ensures that **no information from the test set leaks into the training process**, which is a common and serious mistake when preprocessing is done naively before splitting.

For example, if you compute the median of `total_bedrooms` on the full dataset and then impute it, the test set's values have influenced the imputation. With a pipeline, the median is computed only on training data and then applied to the test set — this is the statistically correct approach.

### Numerical Features
```
SimpleImputer(strategy="median") → StandardScaler()
```
- **Median imputation** fills the missing `total_bedrooms` values robustly without being influenced by outliers
- **StandardScaler** normalises all numerical features to zero mean and unit variance. This is required for linear models (Ridge, Lasso) to converge correctly and for their coefficients to be fairly comparable. Tree-based models do not need scaling, but including it in the pipeline causes no harm.

### Categorical Features
```
SimpleImputer(strategy="most_frequent") → OneHotEncoder(handle_unknown="ignore")
```
- `ocean_proximity` is converted into 5 binary columns — one per category. This lets the model treat each category independently without imposing any artificial ordering.
- `handle_unknown="ignore"` ensures the pipeline does not crash if an unseen category appears at inference time, making the system more robust in production.

### Train / Test Split
- **80% training, 20% test** with `random_state=42` for reproducibility
- The test set is kept completely sealed until final evaluation. It is never used during model selection or hyperparameter tuning. Using the test set to make modelling decisions is a form of overfitting — the reported performance would be optimistically biased.

---

## 📊 Baseline Model

Before trying any complex model, a **Linear Regression** baseline is established — trained on the full training set without cross-validation or tuning.

This serves as a minimum performance benchmark. Linear Regression assumes a linear relationship between features and the target. Given the non-linearity of housing prices (income does not scale linearly with price, location effects are complex), this model is expected to underfit significantly. But that is the point — it sets a floor. If a fancier model cannot convincingly beat this, something is wrong with the modelling approach.

Establishing a baseline first also prevents a common mistake: spending hours tuning a complex model only to find it barely improves over something trivially simple.

---

## 🤖 Model Selection via Cross-Validation

Rather than evaluating models on the test set to pick the best one (which would overfit the model choice to the test set), **5-fold cross-validation** is used entirely within the training data.

The training set is split into 5 equal folds. Each model is trained on 4 folds and evaluated on the remaining fold — repeated 5 times so every fold serves as the validation set exactly once. The average RMSE across 5 folds is the CV score. This gives a reliable, unbiased estimate of how each model will generalise to unseen data.

### Models Compared

| Model | Why it was included |
|---|---|
| `LinearRegression` | Baseline — establishes the minimum acceptable performance |
| `Ridge` | Linear model with L2 regularisation — reduces overfitting caused by multicollinear features |
| `Lasso` | Linear model with L1 regularisation — performs automatic feature selection by zeroing out weak predictors |
| `RandomForestRegressor` | Ensemble of independent decision trees — robust to outliers, captures non-linearity, no scaling needed |
| `HistGradientBoostingRegressor` | Sequential gradient boosted trees — fast, highly accurate, handles missing values natively |

### Why Gradient Boosting Wins

**Winner: `HistGradientBoostingRegressor`** — lowest CV RMSE and highest R² across all 5 folds.

Gradient boosting builds trees **sequentially**, where each new tree is specifically trained to correct the prediction errors made by all previous trees. Unlike Random Forest which builds trees independently in parallel, gradient boosting focuses its learning effort on the hardest-to-predict samples. This makes it particularly effective at capturing the complex, non-linear, interaction-heavy relationships in housing data — like how the combination of high income + coastal location + newer housing drives prices far more than any single factor alone.

| Model | CV RMSE | CV MAE | CV R² |
|---|---|---|---|
| HistGradientBoostingRegressor | ✅ Best | ✅ Best | ✅ Best |
| RandomForestRegressor | 2nd | 2nd | 2nd |
| Ridge | — | — | — |
| Lasso | — | — | — |
| LinearRegression | — | — | — |

*(Run the notebook to see actual metric values)*

---

## 🎛️ Hyperparameter Tuning

Once `HistGradientBoostingRegressor` is identified as the best architecture, its hyperparameters are tuned using **GridSearchCV** — an exhaustive search over all combinations of a defined parameter grid, with each combination evaluated using 5-fold CV.

### Parameter Grid

```python
param_grid = {
    "model__learning_rate":     [0.03, 0.05, 0.1],      # step size per boosting round
    "model__max_depth":         [None, 3, 6],            # max tree depth
    "model__max_leaf_nodes":    [15, 31, 63],            # max leaves per tree
    "model__min_samples_leaf":  [20, 50, 100],           # min samples at each leaf
    "model__l2_regularization": [0.0, 0.1, 1.0]         # L2 penalty on leaf values
}
```

This covers **243 combinations** (3⁵), each evaluated with 5-fold CV — a total of **1,215 model fits**. GridSearchCV selects the combination with the lowest average CV RMSE.

### What Each Parameter Controls

| Parameter | Effect |
|---|---|
| `learning_rate` | Lower = more careful, slower learning. Often better generalisation but needs more boosting rounds |
| `max_depth` / `max_leaf_nodes` | Controls tree complexity. Deeper trees can overfit; shallow trees underfit |
| `min_samples_leaf` | Prevents learning from very small sample groups — acts as a regulariser against noise |
| `l2_regularization` | Penalises large leaf prediction values, similar to Ridge regression but inside the boosting framework |

### Best Parameters Found

```python
learning_rate=0.1,
max_depth=None,
max_leaf_nodes=63,
min_samples_leaf=20,
l2_regularization=0.1
```

The model is then retrained on the **entire training set** using these best parameters before final evaluation. Retraining on the full training set (rather than just the 4/5 used during CV) gives the model the most data to learn from.

---

## 📉 Final Evaluation

The tuned model is evaluated on the held-out test set — data it has never seen at any point during training or tuning.

### Metrics

| Metric | What it measures |
|---|---|
| **RMSE** | Average prediction error in dollars. Penalises large errors more heavily due to squaring |
| **MAE** | Average absolute prediction error in dollars. More interpretable and less sensitive to outliers |
| **R²** | Proportion of variance in house prices explained by the model. 1.0 = perfect, 0.0 = no better than predicting the mean |

| Split | RMSE | MAE | R² |
|---|---|---|---|
| Train | — | — | — |
| Test | — | — | — |

*(Run the notebook to populate with actual values)*

### Residual Analysis

Two residual plots are generated to check for systematic model errors:

- **Residuals vs Predictions**: Residuals (actual − predicted) plotted against predicted values. A well-behaved model shows residuals randomly scattered around zero. A fan shape indicates the model is less accurate for expensive houses.
- **Residual Distribution Histogram**: Checks whether errors are approximately normally distributed. Strong skew indicates the model systematically under- or over-predicts — often caused by the $500K cap on the target.

---

## 🔮 Inference

A clean, reusable `predict_house_price()` function wraps the entire trained pipeline for single-sample predictions. It accepts all 9 raw input features and returns the predicted median house value in dollars — no manual preprocessing required.

```python
predict_house_price(
    model=hgb_best,
    longitude=-122.230,
    latitude=37.880,
    housing_median_age=41,
    total_rooms=880,
    total_bedrooms=129,    # can be np.nan — pipeline imputes automatically
    population=322,
    households=126,
    median_income=8.3252,
    ocean_proximity="NEAR BAY"
)
# → Predicted median house value: ~$358,000
```

Missing values in `total_bedrooms` are handled automatically by the pipeline's imputer — no preprocessing needed at inference time.

---

## 🛠️ Tech Stack

| Category | Libraries |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Modelling & evaluation | `scikit-learn` |
| Models used | `LinearRegression`, `Ridge`, `Lasso`, `RandomForestRegressor`, `HistGradientBoostingRegressor` |
| Tuning | `GridSearchCV`, `KFold`, `cross_validate` |

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/Mowriyan-2/<repo-name>.git
cd <repo-name>

# 2. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn

# 3. Add the dataset
# Download housing.csv from Kaggle and place it in the project root
# https://www.kaggle.com/datasets/camnugent/california-housing-prices

# 4. Run the notebook
jupyter notebook 15_4_house_price_prediction.ipynb
```

---

## 📈 Areas to Improve

### 1. Feature Engineering — Create Smarter Features

**The Problem:**

The current dataset uses raw block-level counts like `total_rooms` and `total_bedrooms`. These numbers are misleading in isolation. A block with 5,000 total rooms sounds enormous — but if it also has 2,000 households, each household only has 2.5 rooms on average, which is actually quite cramped. The model is trying to learn house prices from counts that don't reflect what individual households actually experience.

**The Fix:**

```python
df["rooms_per_household"]      = df["total_rooms"]    / df["households"]
df["bedrooms_per_room"]        = df["total_bedrooms"] / df["total_rooms"]
df["population_per_household"] = df["population"]     / df["households"]
```

**Why each feature matters:**
- `rooms_per_household` captures spaciousness directly. More rooms per household = larger homes = higher prices.
- `bedrooms_per_room` — a low ratio suggests more living/common space, associated with higher-end properties.
- `population_per_household` measures crowding. Dense, overcrowded blocks tend to have lower property values.

These three derived features are among the strongest predictors on this dataset and typically reduce RMSE by 5–10% without any model changes at all.

---

### 2. Target Transformation — Fix the Skewed and Capped Target

**The Problem:**

The target variable `median_house_value` has two serious issues:

1. **Right skew** — most houses cluster in the $100K–$300K range, but a small number of very expensive houses stretch the distribution to the right. Most regression models assume roughly symmetric errors. When the target is skewed, models make disproportionately large errors on expensive houses.

2. **Artificial cap at $500,000** — any house worth more than $500K was recorded as exactly $500,000. This creates a giant spike at the top of the distribution. The model sees hundreds of houses all priced identically at $500K even though their true values differ — it literally cannot learn the real relationship at the upper end.

**The Fix — Log Transformation:**

`np.log1p` (log of value + 1) compresses the scale, dramatically reduces skew, and brings the distribution much closer to a normal bell curve:

```python
import numpy as np

y_train_log = np.log1p(y_train)
model.fit(X_train, y_train_log)

y_pred_log = model.predict(X_test)
y_pred     = np.expm1(y_pred_log)   # inverse of log1p

rmse = root_mean_squared_error(y_test, y_pred)
```

---

### 3. Spatial Features — Give the Model a Sense of Geography

**The Problem:**

`latitude` and `longitude` are fed to the model as raw floating point numbers. The model has no understanding that they represent geography. A tree-based model can technically learn geographic patterns, but it has to discover them laboriously — one coordinate split at a time — with no concept that two houses 0.01 degrees apart are neighbours.

**Fix 1 — Geographic Clustering with KMeans:**

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10, random_state=42)
df["location_cluster"] = kmeans.fit_predict(df[["latitude", "longitude"]]).astype(str)
```

**Fix 2 — Distance to Key Economic Hubs:**

```python
SF_LAT, SF_LON = 37.7749, -122.4194
LA_LAT, LA_LON = 34.0522, -118.2437

df["dist_to_sf"] = np.sqrt((df["latitude"] - SF_LAT)**2 + (df["longitude"] - SF_LON)**2)
df["dist_to_la"] = np.sqrt((df["latitude"] - LA_LAT)**2 + (df["longitude"] - LA_LON)**2)
```

---

### 4. Try More Powerful Boosting Models

**The Problem:**

`HistGradientBoostingRegressor` is scikit-learn's built-in boosting tool. It is solid, but dedicated libraries like XGBoost and LightGBM are specifically engineered for tabular prediction and consistently outperform it.

**The Fix:**

```bash
pip install xgboost lightgbm
```

```python
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

xgb  = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                    subsample=0.8, colsample_bytree=0.8, random_state=42)

lgbm = LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63,
                     subsample=0.8, colsample_bytree=0.8, random_state=42)
```

**Why they are better:** LightGBM grows trees leaf-wise (more efficient), supports native categoricals without one-hot encoding, and supports early stopping — automatically stopping when validation RMSE plateaus.

---

### 5. Smarter Hyperparameter Tuning

**The Problem:**

`GridSearchCV` with 243 combinations tests every single configuration exhaustively. This is slow. Research shows most of the gain comes from a small number of important parameters — exhaustive search wastes compute on unimportant regions of the search space.

**The Fix — RandomizedSearchCV:**

Sample random configurations from continuous distributions. Near-optimal results in 30–50 trials instead of 243:

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_dist = {
    "model__learning_rate":     uniform(0.01, 0.2),
    "model__max_leaf_nodes":    randint(15, 127),
    "model__min_samples_leaf":  randint(10, 100),
    "model__l2_regularization": uniform(0.0, 2.0),
    "model__max_iter":          randint(100, 600),
    "model__max_bins":          [63, 127, 255],
}

random_search = RandomizedSearchCV(
    hgb_pipe, param_dist, n_iter=50, cv=5,
    scoring="neg_root_mean_squared_error",
    random_state=42, n_jobs=-1
)
random_search.fit(X_train, y_train)
```

Also tune `max_iter` (default 100 — likely too few) and `max_bins` (default 255 — try lower for faster training).

---

### 6. Error Analysis — Understand Where the Model Fails

**The Problem:**

A single RMSE on the full test set hides where the model makes systematic mistakes. A model with good overall RMSE can still be wildly inaccurate for specific segments like coastal luxury properties or capped $500K houses.

**The Fix:**

```python
results = X_test.copy()
results["actual"]    = y_test.values
results["predicted"] = test_final_pred
results["abs_error"] = (results["actual"] - results["predicted"]).abs()

# Error by ocean proximity
print(results.groupby("ocean_proximity")["abs_error"].mean().sort_values(ascending=False))

# Capped houses — true value is unknown but >= $500K
capped = results[results["actual"] == 500_000]
print(f"Capped houses MAE: ${capped['abs_error'].mean():,.0f}")

# Error by income bracket
results["income_bracket"] = pd.cut(results["median_income"], bins=5,
    labels=["Very Low", "Low", "Medium", "High", "Very High"])
print(results.groupby("income_bracket")["abs_error"].mean())
```

---

### 7. Spatial Cross-Validation — Get an Honest Performance Estimate

**The Problem:**

`KFold(shuffle=True)` splits data randomly. Houses that are geographically close have very similar prices (spatial autocorrelation). Random splits allow neighbouring houses to appear in both train and test folds, making CV scores appear better than they really are — a form of data leakage specific to geographic datasets.

**The Fix — GroupKFold on Geographic Clusters:**

```python
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans

kmeans    = KMeans(n_clusters=20, random_state=42)
geo_groups = kmeans.fit_predict(X_train[["latitude", "longitude"]])

group_kfold = GroupKFold(n_splits=5)
scores = cross_validate(
    pipe, X_train, y_train,
    cv=group_kfold.split(X_train, y_train, groups=geo_groups),
    scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"}
)
print("Spatial CV RMSE:", -scores["test_rmse"].mean())
```

Spatial CV gives an honest estimate of generalisation to new geographic areas — the real-world deployment scenario. Your RMSE will be higher than random KFold but far more truthful.

---

## 👤 Author

**Mowriyan B**
B.Tech, IIT Kharagpur (2024–2028)
📧 mowriyan52@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/mowriyan-b-560133379/)
🐙 [GitHub](https://github.com/Mowriyan-2)
