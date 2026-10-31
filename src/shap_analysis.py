import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt


print("==============================================")
print("PHASE 6: SHAP EXPLAINABILITY ANALYSIS")
print("==============================================")


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

DATA_PATH = "data/processed/rossmann_processed.csv"
MODEL_PATH = "models/xgboost.pkl"
OUTPUT_PATH = "outputs/shap"

os.makedirs(OUTPUT_PATH, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("\nLoading processed dataset...")

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 3. LOAD XGBOOST PIPELINE
# --------------------------------------------------

print("\nLoading XGBoost base model...")

model_pipeline = joblib.load(MODEL_PATH)

print("XGBoost base model loaded successfully.")


# --------------------------------------------------
# 4. PREPARE FEATURES
# --------------------------------------------------

print("\nPreparing features for SHAP...")

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)

columns_to_drop = [
    "Sales",
    "Customers",
    "SalesPerCustomer",
    "Date"
]

X = df.drop(
    columns=columns_to_drop
)


# --------------------------------------------------
# 5. CONVERT CATEGORICAL COLUMNS
# --------------------------------------------------

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

for column in categorical_features:
    X[column] = X[column].astype(str)


# --------------------------------------------------
# 6. SELECT SHAP SAMPLE
# --------------------------------------------------

print("\nPreparing SHAP sample...")

sample_size = min(2000, len(X))

X_sample = X.sample(
    n=sample_size,
    random_state=42
)

print("SHAP sample size:", len(X_sample))


# --------------------------------------------------
# 7. GET PREPROCESSOR AND XGBOOST MODEL
# --------------------------------------------------

print("\nExtracting preprocessing and XGBoost components...")

preprocessor = model_pipeline.named_steps[
    "preprocessor"
]

xgb_model = model_pipeline.named_steps[
    "model"
]

print("Preprocessor extracted.")
print("XGBoost model extracted.")


# --------------------------------------------------
# 8. TRANSFORM DATA
# --------------------------------------------------

print("\nTransforming data for SHAP...")

X_transformed = preprocessor.transform(
    X_sample
)

print(
    "Transformed feature shape:",
    X_transformed.shape
)


# --------------------------------------------------
# 9. GET FEATURE NAMES
# --------------------------------------------------

feature_names = preprocessor.get_feature_names_out()

print(
    "Number of transformed features:",
    len(feature_names)
)


# --------------------------------------------------
# 10. CREATE SHAP EXPLAINER
# --------------------------------------------------

print("\nCreating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(
    xgb_model
)

print("SHAP TreeExplainer created successfully.")


# --------------------------------------------------
# 11. CALCULATE SHAP VALUES
# --------------------------------------------------

print("\nCalculating SHAP values...")

shap_values = explainer.shap_values(
    X_transformed
)

print("SHAP calculation completed.")


# --------------------------------------------------
# 12. SHAP FEATURE IMPORTANCE
# --------------------------------------------------

print("\nCalculating mean absolute SHAP importance...")

mean_abs_shap = np.abs(
    shap_values
).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Absolute_SHAP": mean_abs_shap
})

importance_df = importance_df.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
).reset_index(drop=True)


# --------------------------------------------------
# 13. SAVE FEATURE IMPORTANCE CSV
# --------------------------------------------------

importance_df.to_csv(
    f"{OUTPUT_PATH}/shap_feature_importance.csv",
    index=False
)

print(
    "\nFeature importance saved to:"
    f" {OUTPUT_PATH}/shap_feature_importance.csv"
)


# --------------------------------------------------
# 14. SHAP FEATURE IMPORTANCE BAR PLOT
# --------------------------------------------------

print("\nCreating SHAP feature importance plot...")

plt.figure()

shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=feature_names,
    plot_type="bar",
    show=False,
    max_display=20
)

plt.title(
    "Top Features Affecting XGBoost Sales Predictions"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_PATH}/shap_feature_importance.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 15. SHAP SUMMARY DOT PLOT
# --------------------------------------------------

print("Creating SHAP summary plot...")

plt.figure()

shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=feature_names,
    show=False,
    max_display=20
)

plt.title(
    "SHAP Summary of XGBoost Sales Predictions"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_PATH}/shap_summary.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 16. SAVE SHAP VALUES
# --------------------------------------------------

print("\nSaving individual SHAP values...")

shap_sample_df = pd.DataFrame(
    shap_values,
    columns=feature_names
)

shap_sample_df.to_csv(
    f"{OUTPUT_PATH}/shap_values.csv",
    index=False
)

print(
    f"SHAP values saved to:"
    f" {OUTPUT_PATH}/shap_values.csv"
)


# --------------------------------------------------
# 17. PRINT TOP FEATURES
# --------------------------------------------------

print("\n==============================================")
print("TOP FEATURES ACCORDING TO SHAP")
print("==============================================")

print(
    importance_df.head(15).to_string(
        index=False
    )
)


# --------------------------------------------------
# 18. FINAL INFORMATION
# --------------------------------------------------

print("\n==============================================")
print("SHAP ANALYSIS COMPLETED SUCCESSFULLY")
print("==============================================")

print("\nGenerated files:")

print("- shap_feature_importance.png")
print("- shap_summary.png")
print("- shap_feature_importance.csv")
print("- shap_values.csv")

print("\nInterpretation:")
print(
    "SHAP explains the contribution of input features "
    "to the XGBoost base-model predictions."
)

print(
    "The residual Random Forest correction layer "
    "is not directly explained by this SHAP analysis."
)