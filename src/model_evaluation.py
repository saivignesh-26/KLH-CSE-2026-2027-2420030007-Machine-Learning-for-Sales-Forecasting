import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt


print("==============================================")
print("PHASE 5: MODEL EVALUATION AND COMPARISON")
print("==============================================")


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

TEST_PATH = "data/processed/test_data.csv"
MODEL_PATH = "models"
OUTPUT_PATH = "outputs/evaluation"

os.makedirs(OUTPUT_PATH, exist_ok=True)


# --------------------------------------------------
# 2. LOAD TEST DATA
# --------------------------------------------------

print("\nLoading test dataset...")

test_df = pd.read_csv(
    TEST_PATH,
    low_memory=False
)

print("Test dataset shape:", test_df.shape)


# --------------------------------------------------
# 3. SEPARATE FEATURES AND TARGET
# --------------------------------------------------

X_test = test_df.drop(columns=["Actual_Sales"])
y_test = test_df["Actual_Sales"]


# --------------------------------------------------
# 4. LOAD TRAINED MODELS
# --------------------------------------------------

print("\nLoading trained models...")

linear_model = joblib.load(
    f"{MODEL_PATH}/linear_regression.pkl"
)

rf_model = joblib.load(
    f"{MODEL_PATH}/random_forest.pkl"
)

xgb_model = joblib.load(
    f"{MODEL_PATH}/xgboost.pkl"
)

print("All models loaded successfully.")


# --------------------------------------------------
# 5. REMOVE DATE COLUMN
# --------------------------------------------------

# Date was used only for chronological splitting.
# The trained models do not use it.

if "Date" in X_test.columns:
    X_test["Date"] = pd.to_datetime(X_test["Date"])


# --------------------------------------------------
# 6. PREDICTIONS
# --------------------------------------------------

print("\nGenerating predictions...")


print("Predicting with Linear Regression...")
linear_predictions = linear_model.predict(X_test)


print("Predicting with Random Forest...")
rf_predictions = rf_model.predict(X_test)


print("Predicting with XGBoost...")
xgb_predictions = xgb_model.predict(X_test)


print("Predictions generated successfully.")


# --------------------------------------------------
# 7. EVALUATION FUNCTION
# --------------------------------------------------

def calculate_metrics(actual, predicted):

    mae = np.mean(
        np.abs(actual - predicted)
    )

    rmse = np.sqrt(
        np.mean((actual - predicted) ** 2)
    )

    # Avoid division by zero in MAPE
    non_zero = actual != 0

    mape = np.mean(
        np.abs(
            (actual[non_zero] - predicted[non_zero])
            / actual[non_zero]
        )
    ) * 100

    ss_total = np.sum(
        (actual - np.mean(actual)) ** 2
    )

    ss_residual = np.sum(
        (actual - predicted) ** 2
    )

    r2 = 1 - (
        ss_residual / ss_total
    )

    return mae, rmse, mape, r2


# --------------------------------------------------
# 8. CALCULATE METRICS
# --------------------------------------------------

print("\nCalculating evaluation metrics...")


linear_mae, linear_rmse, linear_mape, linear_r2 = calculate_metrics(
    y_test.values,
    linear_predictions
)


rf_mae, rf_rmse, rf_mape, rf_r2 = calculate_metrics(
    y_test.values,
    rf_predictions
)


xgb_mae, xgb_rmse, xgb_mape, xgb_r2 = calculate_metrics(
    y_test.values,
    xgb_predictions
)


# --------------------------------------------------
# 9. CREATE COMPARISON TABLE
# --------------------------------------------------

results = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Random Forest",
        "XGBoost"
    ],

    "MAE": [
        linear_mae,
        rf_mae,
        xgb_mae
    ],

    "RMSE": [
        linear_rmse,
        rf_rmse,
        xgb_rmse
    ],

    "MAPE (%)": [
        linear_mape,
        rf_mape,
        xgb_mape
    ],

    "R2": [
        linear_r2,
        rf_r2,
        xgb_r2
    ]

})


# --------------------------------------------------
# 10. DISPLAY RESULTS
# --------------------------------------------------

print("\n==============================================")
print("MODEL PERFORMANCE COMPARISON")
print("==============================================")

print(
    results.to_string(
        index=False
    )
)


# --------------------------------------------------
# 11. SAVE RESULTS
# --------------------------------------------------

results.to_csv(
    f"{OUTPUT_PATH}/model_comparison.csv",
    index=False
)

print("\nModel comparison saved to:")

print(
    f"{OUTPUT_PATH}/model_comparison.csv"
)


# --------------------------------------------------
# 12. IDENTIFY BEST MODEL
# --------------------------------------------------

# Higher R2 is better

best_model_index = results["R2"].idxmax()

best_model = results.loc[
    best_model_index,
    "Model"
]

best_r2 = results.loc[
    best_model_index,
    "R2"
]


print("\n==============================================")
print("BEST MODEL")
print("==============================================")

print("Best model:", best_model)
print("Best R2:", round(best_r2, 4))


# --------------------------------------------------
# 13. SAVE BEST MODEL INFORMATION
# --------------------------------------------------

with open(
    f"{OUTPUT_PATH}/best_model.txt",
    "w"
) as file:

    file.write(
        f"Best Model: {best_model}\n"
    )

    file.write(
        f"R2 Score: {best_r2:.4f}\n"
    )


# --------------------------------------------------
# 14. ACTUAL VS PREDICTED DATA
# --------------------------------------------------

prediction_results = pd.DataFrame({

    "Actual_Sales": y_test.values,

    "Linear_Regression": linear_predictions,

    "Random_Forest": rf_predictions,

    "XGBoost": xgb_predictions

})


prediction_results.to_csv(
    f"{OUTPUT_PATH}/predictions.csv",
    index=False
)


print("\nPrediction results saved.")


# --------------------------------------------------
# 15. ACTUAL VS PREDICTED GRAPH
# --------------------------------------------------

print("\nCreating actual vs predicted graph...")


# Display first 200 test observations
plot_data = prediction_results.iloc[:200]


plt.figure(figsize=(14, 7))

plt.plot(
    plot_data["Actual_Sales"].values,
    label="Actual Sales"
)

plt.plot(
    plot_data["Linear_Regression"].values,
    label="Linear Regression"
)

plt.plot(
    plot_data["Random_Forest"].values,
    label="Random Forest"
)

plt.plot(
    plot_data["XGBoost"].values,
    label="XGBoost"
)

plt.title(
    "Actual vs Predicted Sales"
)

plt.xlabel(
    "Test Observations"
)

plt.ylabel(
    "Sales"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_PATH}/actual_vs_predicted.png"
)

plt.close()


print("Actual vs predicted graph saved.")


# --------------------------------------------------
# 16. MODEL COMPARISON GRAPH
# --------------------------------------------------

print("\nCreating model comparison graph...")


plt.figure(figsize=(9, 6))

plt.bar(
    results["Model"],
    results["R2"]
)

plt.title(
    "R² Score Comparison"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "R² Score"
)

plt.xticks(
    rotation=15
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_PATH}/model_r2_comparison.png"
)

plt.close()


print("Model comparison graph saved.")


# --------------------------------------------------
# 17. FINAL MESSAGE
# --------------------------------------------------

print("\n==============================================")
print("PHASE 5 COMPLETED SUCCESSFULLY")
print("==============================================")

print("\nGenerated files:")

print("- model_comparison.csv")
print("- best_model.txt")
print("- predictions.csv")
print("- actual_vs_predicted.png")
print("- model_r2_comparison.png")

print("\nNext:")
print("PHASE 6 - SHAP EXPLAINABILITY AND FORECASTING")