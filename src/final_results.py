import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


print("==============================================")
print("STEP 3: FINAL RESULTS AND PERFORMANCE GRAPHS")
print("==============================================")


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

OUTPUT_PATH = "outputs/final_results"

os.makedirs(OUTPUT_PATH, exist_ok=True)


# --------------------------------------------------
# 2. FINAL MODEL PERFORMANCE DATA
# --------------------------------------------------

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "XGBoost",
        "XGBoost + Residual Correction"
    ],

    "MAE": [
        1704.514802,
        1117.365609,
        929.158936,
        884.966806
    ],

    "RMSE": [
        2468.010027,
        1713.089747,
        1370.574287,
        1316.909380
    ],

    "MAPE": [
        31.199927,
        20.044207,
        16.239846,
        15.200174
    ],

    "R2": [
        0.592194,
        0.803519,
        0.874233,
        0.883889
    ]
})


# --------------------------------------------------
# 3. SAVE FINAL PERFORMANCE TABLE
# --------------------------------------------------

results.to_csv(
    f"{OUTPUT_PATH}/final_performance_metrics.csv",
    index=False
)

print("\nFinal performance table:")
print(results.to_string(index=False))


# --------------------------------------------------
# 4. MODEL PERFORMANCE COMPARISON
# --------------------------------------------------

print("\nCreating model performance comparison...")

fig, ax = plt.subplots(figsize=(11, 6))

x = np.arange(len(results))
width = 0.2

ax.bar(
    x - 1.5 * width,
    results["MAE"],
    width,
    label="MAE"
)

ax.bar(
    x - 0.5 * width,
    results["RMSE"],
    width,
    label="RMSE"
)

ax.bar(
    x + 0.5 * width,
    results["MAPE"] * 50,
    width,
    label="MAPE × 50"
)

ax.bar(
    x + 1.5 * width,
    results["R2"] * 2000,
    width,
    label="R² × 2000"
)

ax.set_xticks(x)
ax.set_xticklabels(
    [
        "Linear Regression",
        "Random Forest",
        "XGBoost",
        "XGBoost +\nResidual Correction"
    ]
)

ax.set_ylabel("Scaled Metric Value")
ax.set_title(
    "Performance Comparison of Sales Forecasting Models"
)

ax.legend()

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_PATH}/model_performance_comparison.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 5. ACTUAL VS PREDICTED
# --------------------------------------------------

print("Creating actual vs predicted graph...")


prediction_path = (
    "outputs/residual_hybrid/"
    "residual_hybrid_predictions.csv"
)

if os.path.exists(prediction_path):

    predictions = pd.read_csv(prediction_path)

    print(
        "Loaded hybrid prediction file:",
        predictions.shape
    )

    # Try to identify actual/prediction columns
    print(
        "Prediction columns:",
        predictions.columns.tolist()
    )

    # Expected columns from the hybrid experiment
    actual_col = None
    xgb_col = None
    hybrid_col = None

    for col in predictions.columns:

        lower_col = col.lower()

        if (
            actual_col is None
            and (
                lower_col == "actual_sales"
                or lower_col == "actual"
                or lower_col == "sales"
            )
        ):
            actual_col = col

        if (
            xgb_col is None
            and (
                "xgb" in lower_col
                and "prediction" in lower_col
            )
        ):
            xgb_col = col

        if (
            hybrid_col is None
            and (
                "hybrid" in lower_col
                or "residual" in lower_col
            )
        ):
            hybrid_col = col

    # Fallback based on common column names
    if actual_col is None and "Actual_Sales" in predictions.columns:
        actual_col = "Actual_Sales"

    if xgb_col is None and "XGB_Prediction" in predictions.columns:
        xgb_col = "XGB_Prediction"

    if hybrid_col is None and "Hybrid_Prediction" in predictions.columns:
        hybrid_col = "Hybrid_Prediction"

    if (
        actual_col is not None
        and xgb_col is not None
        and hybrid_col is not None
    ):

        plot_data = predictions.head(300).copy()

        plt.figure(figsize=(12, 6))

        plt.plot(
            plot_data[actual_col].values,
            label="Actual Sales"
        )

        plt.plot(
            plot_data[xgb_col].values,
            label="XGBoost Prediction"
        )

        plt.plot(
            plot_data[hybrid_col].values,
            label="Residual Hybrid Prediction"
        )

        plt.xlabel("Test Sample")
        plt.ylabel("Sales")

        plt.title(
            "Actual vs Predicted Sales"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            f"{OUTPUT_PATH}/actual_vs_predicted.png",
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "Actual vs predicted graph created."
        )

    else:

        print(
            "Could not automatically identify "
            "prediction columns."
        )

else:

    print(
        "Hybrid prediction file not found:"
    )

    print(prediction_path)


# --------------------------------------------------
# 6. ERROR COMPARISON
# --------------------------------------------------

print("\nCreating prediction error comparison...")


if os.path.exists(prediction_path):

    predictions = pd.read_csv(prediction_path)

    actual_col = None
    xgb_col = None
    hybrid_col = None

    for col in predictions.columns:

        lower_col = col.lower()

        if (
            actual_col is None
            and (
                lower_col == "actual_sales"
                or lower_col == "actual"
            )
        ):
            actual_col = col

        if (
            xgb_col is None
            and "xgb" in lower_col
            and "prediction" in lower_col
        ):
            xgb_col = col

        if (
            hybrid_col is None
            and "hybrid" in lower_col
        ):
            hybrid_col = col

    if (
        actual_col is not None
        and xgb_col is not None
        and hybrid_col is not None
    ):

        sample = predictions.head(300).copy()

        xgb_error = (
            sample[actual_col]
            - sample[xgb_col]
        )

        hybrid_error = (
            sample[actual_col]
            - sample[hybrid_col]
        )

        plt.figure(figsize=(12, 6))

        plt.plot(
            xgb_error.values,
            label="XGBoost Error"
        )

        plt.plot(
            hybrid_error.values,
            label="Residual Hybrid Error"
        )

        plt.axhline(
            y=0,
            linestyle="--"
        )

        plt.xlabel("Test Sample")
        plt.ylabel("Prediction Error")

        plt.title(
            "Prediction Error Comparison"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            f"{OUTPUT_PATH}/error_comparison.png",
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "Error comparison graph created."
        )


# --------------------------------------------------
# 7. 30-DAY FORECAST
# --------------------------------------------------

print("\nCreating 30-day forecast graph...")


forecast_path = (
    "outputs/residual_hybrid_forecast/"
    "residual_hybrid_sales_forecast.csv"
)

if os.path.exists(forecast_path):

    forecast = pd.read_csv(
        forecast_path
    )

    forecast["Date"] = pd.to_datetime(
        forecast["Date"]
    )

    print(
        "Forecast records:",
        len(forecast)
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        forecast["Date"],
        forecast["Predicted_Sales"],
        marker="o",
        label="Residual Hybrid Forecast"
    )

    plt.xlabel("Date")
    plt.ylabel("Predicted Sales")

    plt.title(
        "30-Day Sales Forecast Using XGBoost + Residual Correction"
    )

    plt.xticks(rotation=45)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_PATH}/30_day_forecast.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "30-day forecast graph created."
    )

else:

    print(
        "Forecast file not found:"
    )

    print(forecast_path)


# --------------------------------------------------
# 8. SUMMARY
# --------------------------------------------------

print("\n==============================================")
print("STEP 3 COMPLETED")
print("==============================================")

print("\nGenerated files:")

print("- final_performance_metrics.csv")
print("- model_performance_comparison.png")
print("- actual_vs_predicted.png")
print("- error_comparison.png")
print("- 30_day_forecast.png")

print(
    "\nProposed model:"
)

print(
    "XGBoost + Residual Correction"
)

print(
    "\nFinal R2:",
    results.loc[
        results["Model"]
        == "XGBoost + Residual Correction",
        "R2"
    ].iloc[0]
)

print(
    "Final MAE:",
    results.loc[
        results["Model"]
        == "XGBoost + Residual Correction",
        "MAE"
    ].iloc[0]
)

print(
    "Final RMSE:",
    results.loc[
        results["Model"]
        == "XGBoost + Residual Correction",
        "RMSE"
    ].iloc[0]
)

print(
    "Final MAPE:",
    results.loc[
        results["Model"]
        == "XGBoost + Residual Correction",
        "MAPE"
    ].iloc[0]
)

print("\nAll Step 3 results generated.")