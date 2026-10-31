import pandas as pd
import numpy as np
import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor


print("==============================================")
print("PHASE 4: MACHINE LEARNING MODEL TRAINING")
print("==============================================")


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

DATA_PATH = "data/processed/rossmann_processed.csv"
MODEL_PATH = "models"

os.makedirs(MODEL_PATH, exist_ok=True)


# --------------------------------------------------
# 2. LOAD PROCESSED DATASET
# --------------------------------------------------

print("\nLoading processed dataset...")

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 3. CONVERT DATE
# --------------------------------------------------

df["Date"] = pd.to_datetime(df["Date"])


# --------------------------------------------------
# 4. SORT DATA CHRONOLOGICALLY
# --------------------------------------------------

print("\nSorting data by date...")

df = df.sort_values("Date").reset_index(drop=True)


# --------------------------------------------------
# 5. REMOVE UNNECESSARY / LEAKAGE COLUMNS
# --------------------------------------------------

print("\nRemoving unnecessary columns...")

columns_to_drop = [
    "Sales",
    "Customers",
    "SalesPerCustomer",
    "Date"
]

X = df.drop(columns=columns_to_drop)
y = df["Sales"]

# --------------------------------------------------
# 6. IDENTIFY CATEGORICAL AND NUMERICAL FEATURES
# --------------------------------------------------

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["int64", "float64", "int32"]
).columns.tolist()


# Convert all categorical columns to strings
# This prevents mixed integer/string encoding errors
for column in categorical_features:
    X[column] = X[column].astype(str)


print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)

# --------------------------------------------------
# 7. TIME-BASED TRAIN / TEST SPLIT
# --------------------------------------------------

print("\nCreating time-based train/test split...")

split_index = int(len(X) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# --------------------------------------------------
# 8. PREPROCESSING
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# --------------------------------------------------
# 9. LINEAR REGRESSION
# --------------------------------------------------

print("\n==============================================")
print("Training Linear Regression...")
print("==============================================")

linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

linear_model.fit(X_train, y_train)

print("Linear Regression training completed.")


joblib.dump(
    linear_model,
    f"{MODEL_PATH}/linear_regression.pkl"
)

print("Linear Regression model saved.")


# --------------------------------------------------
# 10. RANDOM FOREST
# --------------------------------------------------

print("\n==============================================")
print("Training Random Forest...")
print("==============================================")


# Random Forest can be computationally expensive
# on the complete Rossmann dataset.
# We use a representative training sample.

sample_size = min(150000, len(X_train))

X_train_rf = X_train.iloc[:sample_size]
y_train_rf = y_train.iloc[:sample_size]


print("Random Forest training rows:", sample_size)


rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


rf_model.fit(
    X_train_rf,
    y_train_rf
)

print("Random Forest training completed.")


joblib.dump(
    rf_model,
    f"{MODEL_PATH}/random_forest.pkl"
)

print("Random Forest model saved.")


# --------------------------------------------------
# 11. XGBOOST
# --------------------------------------------------

print("\n==============================================")
print("Training XGBoost...")
print("==============================================")


sample_size_xgb = min(150000, len(X_train))

X_train_xgb = X_train.iloc[:sample_size_xgb]
y_train_xgb = y_train.iloc[:sample_size_xgb]


print("XGBoost training rows:", sample_size_xgb)


xgb_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


xgb_model.fit(
    X_train_xgb,
    y_train_xgb
)

print("XGBoost training completed.")


joblib.dump(
    xgb_model,
    f"{MODEL_PATH}/xgboost.pkl"
)

print("XGBoost model saved.")


# --------------------------------------------------
# 12. SAVE TEST DATA
# --------------------------------------------------

print("\nSaving test data for Phase 5...")


test_data = X_test.copy()

test_data["Actual_Sales"] = y_test.values

test_data.to_csv(
    "data/processed/test_data.csv",
    index=False
)


# --------------------------------------------------
# 13. FINAL MESSAGE
# --------------------------------------------------

print("\n==============================================")
print("PHASE 4 COMPLETED SUCCESSFULLY")
print("==============================================")

print("\nModels created:")

print("1. Linear Regression")
print("2. Random Forest")
print("3. XGBoost")

print("\nSaved models:")

print("models/linear_regression.pkl")
print("models/random_forest.pkl")
print("models/xgboost.pkl")

print("\nTest data saved at:")
print("data/processed/test_data.csv")

print("\nReady for Phase 5:")
print("Model Evaluation and Best Model Selection")