import pandas as pd
import numpy as np
import joblib
import os

print("==============================================")
print("PHASE 6: E-COMMERCE SALES FORECASTING")
print("==============================================")

# -------------------------------------------------
# PATHS
# -------------------------------------------------

DATA_PATH = "data/processed/rossmann_processed.csv"
MODEL_PATH = "models/xgboost.pkl"
OUTPUT_DIR = "outputs/forecast"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

print("\nLoading processed dataset...")

df = pd.read_csv(
    DATA_PATH,
    low_memory=False,
    dtype={
        "StateHoliday": str,
        "StoreType": str,
        "Assortment": str,
        "PromoInterval": str
    }
)

df["Date"] = pd.to_datetime(df["Date"])

print("Dataset loaded successfully.")

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

print("\nLoading XGBoost model...")

model = joblib.load(MODEL_PATH)

print("XGBoost model loaded successfully.")

# -------------------------------------------------
# SELECT STORE
# -------------------------------------------------

store_id = 1

store_data = df[df["Store"] == store_id].copy()

if store_data.empty:
    raise ValueError(f"Store {store_id} was not found in the dataset.")

store_data = store_data.sort_values("Date")

last_row = store_data.iloc[-1].copy()

last_date = store_data["Date"].max()

print("\nSelected Store:", store_id)
print("Last available date:", last_date.date())

# -------------------------------------------------
# CREATE FUTURE DATES
# -------------------------------------------------

future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=30,
    freq="D"
)

future_df = pd.DataFrame({
    "Date": future_dates
})

# -------------------------------------------------
# COPY LATEST STORE FEATURES
# -------------------------------------------------

# Features that describe the store and its environment
store_features = [
    "Store",
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
    "PromoInterval"
]

for feature in store_features:
    future_df[feature] = last_row[feature]

# -------------------------------------------------
# CREATE TIME FEATURES
# -------------------------------------------------

future_df["Year"] = future_df["Date"].dt.year
future_df["Month"] = future_df["Date"].dt.month
future_df["Day"] = future_df["Date"].dt.day
future_df["WeekOfYear"] = (
    future_df["Date"]
    .dt.isocalendar()
    .week
    .astype(int)
)

future_df["DayOfWeek"] = future_df["Date"].dt.dayofweek + 1

future_df["IsWeekend"] = (
    future_df["DayOfWeek"]
    .isin([6, 7])
    .astype(int)
)

# -------------------------------------------------
# ENSURE CORRECT DATA TYPES
# -------------------------------------------------

categorical_features = [
    "StateHoliday",
    "StoreType",
    "Assortment",
    "PromoInterval"
]

for column in categorical_features:
    future_df[column] = (
        future_df[column]
        .fillna("None")
        .astype(str)
    )

# Numerical features
numerical_features = [
    "Store",
    "DayOfWeek",
    "Open",
    "Promo",
    "SchoolHoliday",
    "CompetitionDistance",
    "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear",
    "Promo2",
    "Promo2SinceWeek",
    "Promo2SinceYear",
    "Year",
    "Month",
    "Day",
    "WeekOfYear",
    "IsWeekend"
]

for column in numerical_features:
    future_df[column] = pd.to_numeric(
        future_df[column],
        errors="coerce"
    )

# Replace any remaining numerical missing values
future_df[numerical_features] = (
    future_df[numerical_features]
    .fillna(0)
)

# -------------------------------------------------
# PREPARE MODEL INPUT
# -------------------------------------------------

X_future = future_df.drop(
    columns=["Date"]
)

# These columns were removed during model training
# and therefore must NOT be supplied to the model.
columns_to_remove = [
    "Sales",
    "Customers",
    "SalesPerCustomer"
]

for column in columns_to_remove:
    if column in X_future.columns:
        X_future = X_future.drop(
            columns=[column]
        )

# -------------------------------------------------
# FORCE COLUMN ORDER USED DURING TRAINING
# -------------------------------------------------

expected_columns = [
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

X_future = X_future[
    expected_columns
]

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

print("\nForecasting next 30 days...")

predictions = model.predict(X_future)

predictions = np.maximum(
    predictions,
    0
)

# -------------------------------------------------
# SAVE FORECAST
# -------------------------------------------------

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Store": store_id,
    "Predicted_Sales": predictions
})

output_path = os.path.join(
    OUTPUT_DIR,
    "sales_forecast.csv"
)

forecast_df.to_csv(
    output_path,
    index=False
)

# -------------------------------------------------
# DISPLAY RESULTS
# -------------------------------------------------

print("\n==============================================")
print("30-DAY SALES FORECAST")
print("==============================================")

print(
    forecast_df.to_string(index=False)
)

print("\n==============================================")
print("FORECAST COMPLETED SUCCESSFULLY")
print("==============================================")

print("\nForecast saved at:")
print(output_path)

print("\nTotal predicted sales:")
print(
    round(
        forecast_df["Predicted_Sales"].sum(),
        2
    )
)