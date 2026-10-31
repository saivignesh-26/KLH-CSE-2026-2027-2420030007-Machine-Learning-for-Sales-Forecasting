import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = "data/processed/rossmann_processed.csv"

XGB_MODEL_PATH = "models/xgboost.pkl"

OUTPUT_DIR = "outputs/residual_hybrid"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading processed dataset...")

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print("Dataset shape:", df.shape)

df["Date"] = pd.to_datetime(df["Date"])

# Chronological ordering
df = df.sort_values("Date").reset_index(drop=True)


# ============================================================
# 3. PREPARE FEATURES AND TARGET
# ============================================================

TARGET = "Sales"

X = df.drop(
    columns=[
        "Sales",
        "Customers",
        "SalesPerCustomer",
        "Date"
    ],
    errors="ignore"
)

y = df[TARGET].copy()


# Convert categorical columns to string
categorical_columns = X.select_dtypes(
    include=["object", "string"]
).columns

for col in categorical_columns:
    X[col] = X[col].astype(str)


# ============================================================
# 4. CHRONOLOGICAL SPLIT
# ============================================================
#
# Total data:
#     1,017,209
#
# Development data:
#     80%
#
# Final test:
#     20%
#
# Development data is further divided into:
#
#     Base-model training : 60%
#     Residual training   : 20%
#     Final test          : 20%
#
# The final test is NEVER used for training.
# ============================================================

total_size = len(X)

test_start = int(total_size * 0.80)

development_X = X.iloc[:test_start].copy()
development_y = y.iloc[:test_start].copy()

X_test = X.iloc[test_start:].copy()
y_test = y.iloc[test_start:].copy()

# Split development data into base-training and
# residual-training portions
base_train_end = int(len(development_X) * 0.75)

X_base_train = development_X.iloc[:base_train_end].copy()
y_base_train = development_y.iloc[:base_train_end].copy()

X_residual_train = development_X.iloc[base_train_end:].copy()
y_residual_train = development_y.iloc[base_train_end:].copy()


print("\nChronological Split")
print("--------------------------------")
print("Base model training :", len(X_base_train))
print("Residual training   :", len(X_residual_train))
print("Final test          :", len(X_test))


# ============================================================
# 5. LOAD EXISTING XGBOOST MODEL
# ============================================================
#
# IMPORTANT:
# The existing XGBoost model was trained on the original
# first 150,000 rows.
#
# To build a clean residual-learning experiment, we train
# a NEW base XGBoost model using ONLY X_base_train.
#
# This prevents the residual model from learning errors
# generated from observations that trained the base model.
# ============================================================

print("\nTraining leakage-safe base XGBoost model...")


# We use the same XGBoost configuration as your existing model.
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


# ============================================================
# 6. IDENTIFY FEATURES
# ============================================================

categorical_features = X_base_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numerical_features = X_base_train.select_dtypes(
    include=["int64", "int32", "float64", "float32"]
).columns.tolist()


# ============================================================
# 7. BASE MODEL PREPROCESSOR
# ============================================================

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


# ============================================================
# 8. BASE XGBOOST MODEL
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
# 9. LIMIT BASE TRAINING SIZE
# ============================================================
#
# Your original model used at most 150,000 rows for RF/XGB.
#
# We keep the same limit so the comparison remains practical.
# ============================================================

MAX_BASE_TRAIN_SIZE = 150000

if len(X_base_train) > MAX_BASE_TRAIN_SIZE:

    X_base_train_model = X_base_train.iloc[
        :MAX_BASE_TRAIN_SIZE
    ].copy()

    y_base_train_model = y_base_train.iloc[
        :MAX_BASE_TRAIN_SIZE
    ].copy()

else:

    X_base_train_model = X_base_train.copy()

    y_base_train_model = y_base_train.copy()


print(
    "Rows used for base XGBoost:",
    len(X_base_train_model)
)


# ============================================================
# 10. TRAIN BASE XGBOOST
# ============================================================

base_xgb_pipeline.fit(
    X_base_train_model,
    y_base_train_model
)

print("Base XGBoost training completed.")


# ============================================================
# 11. GENERATE RESIDUAL-TRAINING PREDICTIONS
# ============================================================

print("\nGenerating base XGBoost predictions for residual learning...")

xgb_residual_pred = base_xgb_pipeline.predict(
    X_residual_train
)


# ============================================================
# 12. CALCULATE RESIDUALS
# ============================================================
#
# Residual:
#
#     residual = actual - predicted
#
# The residual model learns these remaining errors.
# ============================================================

residuals = (
    y_residual_train.values
    -
    xgb_residual_pred
)


print("\nResidual statistics")
print("--------------------------------")
print(
    "Mean residual:",
    np.mean(residuals)
)

print(
    "Std residual :",
    np.std(residuals)
)


# ============================================================
# 13. TRAIN RESIDUAL RANDOM FOREST
# ============================================================
#
# The residual model receives the ORIGINAL features and
# learns the systematic errors left by XGBoost.
# ============================================================

print("\nTraining residual correction model...")


residual_rf = RandomForestRegressor(

    n_estimators=100,

    max_depth=12,

    min_samples_leaf=5,

    random_state=42,

    n_jobs=-1
)


# We need a transformed numerical matrix because the
# residual model cannot directly process categorical strings.
#
# Use the already-fitted XGBoost preprocessor.
# ============================================================

fitted_preprocessor = base_xgb_pipeline.named_steps[
    "preprocessor"
]

X_residual_transformed = fitted_preprocessor.transform(
    X_residual_train
)


print(
    "Residual feature matrix shape:",
    X_residual_transformed.shape
)


residual_rf.fit(
    X_residual_transformed,
    residuals
)

print("Residual correction model trained.")


# ============================================================
# 14. CREATE FUNCTION FOR METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    non_zero = y_true != 0

    if non_zero.sum() > 0:

        mape = np.mean(
            np.abs(
                (
                    y_true[non_zero]
                    -
                    y_pred[non_zero]
                )
                /
                y_true[non_zero]
            )
        ) * 100

    else:

        mape = np.nan

    r2 = r2_score(
        y_true,
        y_pred
    )

    return mae, rmse, mape, r2


# ============================================================
# 15. EVALUATE BASE XGBOOST ON FINAL TEST
# ============================================================

print("\nGenerating final test predictions...")

xgb_test_pred = base_xgb_pipeline.predict(
    X_test
)


# ============================================================
# 16. TRANSFORM FINAL TEST FEATURES
# ============================================================

X_test_transformed = fitted_preprocessor.transform(
    X_test
)


# ============================================================
# 17. PREDICT RESIDUAL CORRECTION
# ============================================================

residual_test_pred = residual_rf.predict(
    X_test_transformed
)


# ============================================================
# 18. FINAL RESIDUAL-CORRECTED PREDICTION
# ============================================================

hybrid_test_pred = (
    xgb_test_pred
    +
    residual_test_pred
)


# Prevent negative sales predictions
hybrid_test_pred = np.maximum(
    hybrid_test_pred,
    0
)


# ============================================================
# 19. CALCULATE PERFORMANCE
# ============================================================

xgb_mae, xgb_rmse, xgb_mape, xgb_r2 = calculate_metrics(
    y_test,
    xgb_test_pred
)

hybrid_mae, hybrid_rmse, hybrid_mape, hybrid_r2 = calculate_metrics(
    y_test,
    hybrid_test_pred
)


# ============================================================
# 20. CREATE RESULT TABLE
# ============================================================

results = pd.DataFrame({

    "Model": [

        "XGBoost Baseline",

        "XGBoost + Residual Correction"
    ],

    "MAE": [

        xgb_mae,

        hybrid_mae
    ],

    "RMSE": [

        xgb_rmse,

        hybrid_rmse
    ],

    "MAPE": [

        xgb_mape,

        hybrid_mape
    ],

    "R2": [

        xgb_r2,

        hybrid_r2
    ]
})


# ============================================================
# 21. SAVE RESULTS
# ============================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "residual_hybrid_model_comparison.csv"
)

results.to_csv(
    results_path,
    index=False
)


# ============================================================
# 22. SAVE PREDICTIONS
# ============================================================

predictions = pd.DataFrame({

    "Actual_Sales": y_test.values,

    "XGBoost_Prediction": xgb_test_pred,

    "Residual_Correction": residual_test_pred,

    "Hybrid_Prediction": hybrid_test_pred
})


predictions_path = os.path.join(
    OUTPUT_DIR,
    "residual_hybrid_predictions.csv"
)

predictions.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# 23. SAVE MODEL INFORMATION
# ============================================================

model_information = {

    "Base_Model":
        "XGBoost",

    "Residual_Model":
        "Random Forest",

    "Base_Training_Rows":
        int(len(X_base_train_model)),

    "Residual_Training_Rows":
        int(len(X_residual_train)),

    "Final_Test_Rows":
        int(len(X_test)),

    "Formula":
        "Final Prediction = XGBoost Prediction + Residual Correction",

    "XGBoost_MAE":
        float(xgb_mae),

    "XGBoost_RMSE":
        float(xgb_rmse),

    "XGBoost_MAPE":
        float(xgb_mape),

    "XGBoost_R2":
        float(xgb_r2),

    "Hybrid_MAE":
        float(hybrid_mae),

    "Hybrid_RMSE":
        float(hybrid_rmse),

    "Hybrid_MAPE":
        float(hybrid_mape),

    "Hybrid_R2":
        float(hybrid_r2)
}


with open(
    os.path.join(
        OUTPUT_DIR,
        "residual_hybrid_information.json"
    ),
    "w"
) as f:

    json.dump(
        model_information,
        f,
        indent=4
    )


# ============================================================
# 24. DETERMINE BEST MODEL
# ============================================================

if hybrid_r2 > xgb_r2:

    best_model = "XGBoost + Residual Correction"

else:

    best_model = "XGBoost Baseline"


with open(
    os.path.join(
        OUTPUT_DIR,
        "best_model.txt"
    ),
    "w"
) as f:

    f.write(best_model)


# ============================================================
# 25. PRINT FINAL RESULTS
# ============================================================

print("\n")
print("=" * 75)
print("FINAL RESIDUAL HYBRID RESULTS")
print("=" * 75)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\n" + "=" * 75)

print(
    "Base XGBoost R²:",
    f"{xgb_r2:.6f}"
)

print(
    "Residual Hybrid R²:",
    f"{hybrid_r2:.6f}"
)

print("\n" + "=" * 75)

if hybrid_r2 > xgb_r2:

    improvement = (
        hybrid_r2
        -
        xgb_r2
    )

    print(
        "RESULT: Residual hybrid improved over XGBoost."
    )

    print(
        f"R² improvement: {improvement:.6f}"
    )

else:

    difference = (
        xgb_r2
        -
        hybrid_r2
    )

    print(
        "RESULT: XGBoost remains better."
    )

    print(
        f"R² difference: {difference:.6f}"
    )

print("=" * 75)


# ============================================================
# 26. OUTPUT FILES
# ============================================================

print("\nFiles saved:")

print(
    results_path
)

print(
    predictions_path
)

print(
    os.path.join(
        OUTPUT_DIR,
        "residual_hybrid_information.json"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "best_model.txt"
    )
)

print("\nResidual hybrid experiment completed successfully.")