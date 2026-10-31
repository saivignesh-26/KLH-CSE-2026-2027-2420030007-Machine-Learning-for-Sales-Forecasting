import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = "data/processed/rossmann_processed.csv"

OUTPUT_DIR = "outputs/residual_hybrid_forecast"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. LOAD PROCESSED DATA
# ============================================================

print("\nLoading processed dataset...")

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print(
    "Dataset shape:",
    df.shape
)


# Convert Date
df["Date"] = pd.to_datetime(
    df["Date"]
)

# Sort by date
df = df.sort_values(
    "Date"
).reset_index(drop=True)


# ============================================================
# 3. SELECT STORE
# ============================================================

store_id = 1

store_data = df[
    df["Store"] == store_id
].copy()

store_data = store_data.sort_values(
    "Date"
).reset_index(drop=True)

print(
    "\nSelected Store:",
    store_id
)

print(
    "Store records:",
    len(store_data)
)


# ============================================================
# 4. GET LAST AVAILABLE DATE
# ============================================================

last_date = store_data["Date"].max()

print(
    "Last available date:",
    last_date.date()
)


# ============================================================
# 5. CREATE FUTURE DATES
# ============================================================

forecast_days = 30

future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=forecast_days,
    freq="D"
)


# ============================================================
# 6. GET LATEST STORE INFORMATION
# ============================================================

latest_row = store_data.iloc[-1]


# ============================================================
# 7. CREATE FUTURE DATAFRAME
# ============================================================

future_df = pd.DataFrame({

    "Date": future_dates,

    "Store": store_id,

    "Open": latest_row["Open"],

    "Promo": latest_row["Promo"],

    "SchoolHoliday": latest_row["SchoolHoliday"],

    "StateHoliday": latest_row["StateHoliday"],

    "StoreType": latest_row["StoreType"],

    "Assortment": latest_row["Assortment"],

    "CompetitionDistance":
        latest_row["CompetitionDistance"],

    "CompetitionOpenSinceMonth":
        latest_row["CompetitionOpenSinceMonth"],

    "CompetitionOpenSinceYear":
        latest_row["CompetitionOpenSinceYear"],

    "Promo2":
        latest_row["Promo2"],

    "Promo2SinceWeek":
        latest_row["Promo2SinceWeek"],

    "Promo2SinceYear":
        latest_row["Promo2SinceYear"],

    "PromoInterval":
        latest_row["PromoInterval"]
})


# ============================================================
# 8. CREATE TIME FEATURES
# ============================================================

future_df["Year"] = (
    future_df["Date"].dt.year
)

future_df["Month"] = (
    future_df["Date"].dt.month
)

future_df["Day"] = (
    future_df["Date"].dt.day
)

future_df["WeekOfYear"] = (
    future_df["Date"].dt.isocalendar().week.astype(int)
)

future_df["DayOfWeek"] = (
    future_df["Date"].dt.dayofweek + 1
)

future_df["IsWeekend"] = (
    future_df["DayOfWeek"] >= 6
).astype(int)


# ============================================================
# 9. PREPARE FEATURES
# ============================================================

feature_columns = [

    "Store",
    "DayOfWeek",
    "Open",
    "Promo",
    "SchoolHoliday",
    "StateHoliday",
    "StoreType",
    "Assortment",
    "CompetitionDistance",
    "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear",
    "Promo2",
    "Promo2SinceWeek",
    "Promo2SinceYear",
    "PromoInterval",
    "Year",
    "Month",
    "Day",
    "WeekOfYear",
    "IsWeekend"
]


X_future = future_df[
    feature_columns
].copy()


# ============================================================
# 10. CONVERT CATEGORICAL FEATURES
# ============================================================

categorical_features = [
    "StateHoliday",
    "StoreType",
    "Assortment",
    "PromoInterval"
]

for col in categorical_features:

    X_future[col] = (
        X_future[col]
        .astype(str)
    )


# ============================================================
# 11. HANDLE NUMERICAL FEATURES
# ============================================================

numerical_features = [
    col
    for col in feature_columns
    if col not in categorical_features
]

for col in numerical_features:

    X_future[col] = pd.to_numeric(
        X_future[col],
        errors="coerce"
    )

    X_future[col] = (
        X_future[col]
        .fillna(0)
    )


# ============================================================
# 12. TRAIN BASE XGBOOST FOR FORECASTING
# ============================================================
#
# We recreate the same XGBoost configuration used in
# the residual-hybrid experiment.
#
# Training data:
# First 150,000 rows of the complete historical dataset.
# ============================================================

print(
    "\nPreparing base XGBoost model..."
)


# Prepare complete historical features
X_history = df.drop(
    columns=[
        "Sales",
        "Customers",
        "SalesPerCustomer",
        "Date"
    ],
    errors="ignore"
).copy()

y_history = df["Sales"].copy()


# Convert categorical columns
history_categorical = X_history.select_dtypes(
    include=["object", "string"]
).columns.tolist()

for col in history_categorical:

    X_history[col] = (
        X_history[col]
        .astype(str)
    )


# Numerical columns
history_numerical = X_history.select_dtypes(
    include=[
        "int64",
        "int32",
        "float64",
        "float32"
    ]
).columns.tolist()


# Use the first 150,000 rows
base_train_size = min(
    150000,
    len(X_history)
)

X_base_train = X_history.iloc[
    :base_train_size
].copy()

y_base_train = y_history.iloc[
    :base_train_size
].copy()


print(
    "XGBoost training rows:",
    len(X_base_train)
)


# ============================================================
# 13. XGBOOST PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            history_categorical
        ),

        (
            "numerical",

            "passthrough",

            history_numerical
        )
    ]
)


# ============================================================
# 14. XGBOOST MODEL
# ============================================================

xgb_model = XGBRegressor(

    n_estimators=200,

    max_depth=8,

    learning_rate=0.1,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=42,

    n_jobs=-1
)


base_xgb_pipeline = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            xgb_model
        )
    ]
)


# ============================================================
# 15. TRAIN XGBOOST
# ============================================================

print(
    "\nTraining XGBoost..."
)

base_xgb_pipeline.fit(
    X_base_train,
    y_base_train
)

print(
    "XGBoost training completed."
)


# ============================================================
# 16. PREPARE RESIDUAL TRAINING DATA
# ============================================================
#
# The residual model must learn the remaining errors of XGBoost.
#
# We use historical data after the first 150,000 rows.
# ============================================================

residual_start = 150000

X_residual_train = X_history.iloc[
    residual_start:
].copy()

y_residual_train = y_history.iloc[
    residual_start:
].copy()


# To keep training practical, use at most 200,000 rows
residual_max_rows = min(
    200000,
    len(X_residual_train)
)

X_residual_train = X_residual_train.iloc[
    :residual_max_rows
].copy()

y_residual_train = y_residual_train.iloc[
    :residual_max_rows
].copy()


print(
    "Residual training rows:",
    len(X_residual_train)
)


# ============================================================
# 17. GENERATE XGBOOST PREDICTIONS
# ============================================================

print(
    "\nGenerating XGBoost residual-training predictions..."
)

xgb_residual_predictions = (
    base_xgb_pipeline.predict(
        X_residual_train
    )
)


# ============================================================
# 18. CALCULATE RESIDUALS
# ============================================================

residual_values = (
    y_residual_train.values
    -
    xgb_residual_predictions
)


print(
    "Residual calculation completed."
)

print(
    "Residual mean:",
    np.mean(residual_values)
)

print(
    "Residual standard deviation:",
    np.std(residual_values)
)


# ============================================================
# 19. TRANSFORM RESIDUAL FEATURES
# ============================================================

fitted_preprocessor = (
    base_xgb_pipeline
    .named_steps["preprocessor"]
)

X_residual_transformed = (
    fitted_preprocessor.transform(
        X_residual_train
    )
)


# ============================================================
# 20. TRAIN RESIDUAL RANDOM FOREST
# ============================================================

print(
    "\nTraining residual Random Forest..."
)

residual_model = RandomForestRegressor(

    n_estimators=100,

    max_depth=12,

    min_samples_leaf=5,

    random_state=42,

    n_jobs=-1
)


residual_model.fit(
    X_residual_transformed,
    residual_values
)

print(
    "Residual Random Forest training completed."
)


# ============================================================
# 21. GENERATE FUTURE XGBOOST PREDICTIONS
# ============================================================

print(
    "\nGenerating 30-day XGBoost forecast..."
)

xgb_future_predictions = (
    base_xgb_pipeline.predict(
        X_future
    )
)


# ============================================================
# 22. TRANSFORM FUTURE FEATURES
# ============================================================

X_future_transformed = (
    fitted_preprocessor.transform(
        X_future
    )
)


# ============================================================
# 23. GENERATE RESIDUAL CORRECTIONS
# ============================================================

print(
    "Generating residual corrections..."
)

future_residual_corrections = (
    residual_model.predict(
        X_future_transformed
    )
)


# ============================================================
# 24. FINAL HYBRID FORECAST
# ============================================================

final_predictions = (
    xgb_future_predictions
    +
    future_residual_corrections
)


# Sales cannot be negative
final_predictions = np.maximum(
    final_predictions,
    0
)


# ============================================================
# 25. CREATE FORECAST OUTPUT
# ============================================================

forecast_output = pd.DataFrame({

    "Date":
        future_dates,

    "Store":
        store_id,

    "XGBoost_Prediction":
        xgb_future_predictions,

    "Residual_Correction":
        future_residual_corrections,

    "Predicted_Sales":
        final_predictions
})


# ============================================================
# 26. SAVE FORECAST
# ============================================================

forecast_path = os.path.join(
    OUTPUT_DIR,
    "residual_hybrid_sales_forecast.csv"
)

forecast_output.to_csv(
    forecast_path,
    index=False
)


# ============================================================
# 27. SAVE SUMMARY
# ============================================================

summary = {

    "Store":
        int(store_id),

    "Last_Historical_Date":
        str(last_date.date()),

    "Forecast_Days":
        forecast_days,

    "Total_Forecast_Sales":
        float(
            final_predictions.sum()
        ),

    "Average_Daily_Forecast":
        float(
            final_predictions.mean()
        ),

    "Minimum_Daily_Forecast":
        float(
            final_predictions.min()
        ),

    "Maximum_Daily_Forecast":
        float(
            final_predictions.max()
        )
}


summary_path = os.path.join(
    OUTPUT_DIR,
    "forecast_summary.json"
)

with open(
    summary_path,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# 28. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 75)
print("30-DAY RESIDUAL HYBRID FORECAST")
print("=" * 75)

print(
    forecast_output.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

print("\n" + "=" * 75)

print(
    "Total Forecast Sales:",
    f"{final_predictions.sum():,.2f}"
)

print(
    "Average Daily Forecast:",
    f"{final_predictions.mean():,.2f}"
)

print(
    "Minimum Daily Forecast:",
    f"{final_predictions.min():,.2f}"
)

print(
    "Maximum Daily Forecast:",
    f"{final_predictions.max():,.2f}"
)

print("=" * 75)

print(
    "\nForecast saved to:"
)

print(
    forecast_path
)

print(
    "\nSummary saved to:"
)

print(
    summary_path
)

print(
    "\nResidual hybrid forecasting completed successfully."
)