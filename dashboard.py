import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc


# =========================================================
# PATHS
# =========================================================

MODEL_RESULTS = (
    "outputs/final_results/"
    "final_performance_metrics.csv"
)

FORECAST_FILE = (
    "outputs/residual_hybrid_forecast/"
    "residual_hybrid_sales_forecast.csv"
)

SHAP_FILE = (
    "outputs/shap/"
    "shap_feature_importance.csv"
)

PREDICTIONS_FILE = (
    "outputs/residual_hybrid/"
    "residual_hybrid_predictions.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

print("==============================================")
print("LOADING FINAL DASHBOARD DATA")
print("==============================================")

model_df = pd.read_csv(MODEL_RESULTS)

forecast_df = pd.read_csv(
    FORECAST_FILE
)

shap_df = pd.read_csv(
    SHAP_FILE
)

predictions_df = pd.read_csv(
    PREDICTIONS_FILE
)

forecast_df["Date"] = pd.to_datetime(
    forecast_df["Date"]
)

print("Model results:", model_df.shape)
print("Forecast data:", forecast_df.shape)
print("SHAP data:", shap_df.shape)
print("Prediction data:", predictions_df.shape)

print("Dashboard data loaded successfully.")


# =========================================================
# PROPOSED MODEL
# =========================================================

proposed_model_row = model_df.loc[
    model_df["Model"]
    == "XGBoost + Residual Correction"
].iloc[0]

proposed_model = proposed_model_row["Model"]

proposed_mae = proposed_model_row["MAE"]

proposed_rmse = proposed_model_row["RMSE"]

proposed_mape = proposed_model_row["MAPE"]

proposed_r2 = proposed_model_row["R2"]


# =========================================================
# FORECAST SUMMARY
# =========================================================

total_forecast = forecast_df[
    "Predicted_Sales"
].sum()

average_forecast = forecast_df[
    "Predicted_Sales"
].mean()

minimum_forecast = forecast_df[
    "Predicted_Sales"
].min()

maximum_forecast = forecast_df[
    "Predicted_Sales"
].max()


# =========================================================
# MODEL PERFORMANCE GRAPH
# =========================================================

model_fig = px.bar(
    model_df,
    x="Model",
    y="R2",
    title="Model Performance Comparison (R²)",
    text_auto=".3f"
)

model_fig.update_layout(
    yaxis_title="R² Score",
    xaxis_title="Machine Learning Model",
    xaxis_tickangle=-15
)


# =========================================================
# ACTUAL VS PREDICTED GRAPH
# =========================================================

prediction_sample = predictions_df.head(300).copy()

actual_vs_pred_fig = px.line(
    prediction_sample,
    y=[
        "Actual_Sales",
        "XGBoost_Prediction",
        "Hybrid_Prediction"
    ],
    title="Actual vs Predicted Sales"
)

prediction_sample["Sample"] = range(
    1,
    len(prediction_sample) + 1
)

actual_vs_pred_fig = px.line(
    prediction_sample,
    x="Sample",
    y=[
        "Actual_Sales",
        "XGBoost_Prediction",
        "Hybrid_Prediction"
    ],
    title="Actual vs Predicted Sales"
)

actual_vs_pred_fig.update_layout(
    xaxis_title="Test Sample",
    yaxis_title="Sales",
    legend_title="Prediction Type"
)


# =========================================================
# ERROR COMPARISON GRAPH
# =========================================================

error_df = prediction_sample.copy()

error_df["XGBoost_Error"] = (
    error_df["Actual_Sales"]
    - error_df["XGBoost_Prediction"]
)

error_df["Hybrid_Error"] = (
    error_df["Actual_Sales"]
    - error_df["Hybrid_Prediction"]
)

error_long = error_df[
    [
        "XGBoost_Error",
        "Hybrid_Error"
    ]
].copy()

error_long["Sample"] = range(
    1,
    len(error_long) + 1
)

error_long = error_long.melt(
    id_vars="Sample",
    var_name="Model",
    value_name="Prediction_Error"
)

error_fig = px.line(
    error_long,
    x="Sample",
    y="Prediction_Error",
    color="Model",
    title="Prediction Error Comparison"
)

error_fig.update_layout(
    xaxis_title="Test Sample",
    yaxis_title="Prediction Error"
)


# =========================================================
# 30-DAY FORECAST GRAPH
# =========================================================

forecast_fig = px.line(
    forecast_df,
    x="Date",
    y="Predicted_Sales",
    markers=True,
    title=(
        "30-Day Sales Forecast Using "
        "XGBoost + Residual Correction"
    )
)

forecast_fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Predicted Sales"
)


# =========================================================
# SHAP FEATURE IMPORTANCE
# =========================================================

shap_top = shap_df.head(15).copy()

shap_top = shap_top.sort_values(
    "Mean_Absolute_SHAP",
    ascending=True
)

shap_fig = px.bar(
    shap_top,
    x="Mean_Absolute_SHAP",
    y="Feature",
    orientation="h",
    title=(
        "Top Features Affecting "
        "XGBoost Sales Predictions"
    )
)

shap_fig.update_layout(
    xaxis_title="Mean Absolute SHAP Value",
    yaxis_title="Feature"
)


# =========================================================
# DASH APP
# =========================================================

app = Dash(__name__)

app.title = (
    "Explainable Sales Forecasting Dashboard"
)


# =========================================================
# DASHBOARD LAYOUT
# =========================================================

app.layout = html.Div(

    style={
        "fontFamily": "Arial",
        "padding": "25px",
        "backgroundColor": "#f5f5f5"
    },

    children=[

        # =================================================
        # TITLE
        # =================================================

        html.H1(
            "Explainable Sales Forecasting Dashboard",
            style={
                "textAlign": "center",
                "marginBottom": "10px"
            }
        ),

        html.P(
            "An Explainable Machine Learning Approach "
            "for Sales Forecasting and Demand Prediction",
            style={
                "textAlign": "center",
                "fontSize": "18px"
            }
        ),


        # =================================================
        # PROPOSED MODEL
        # =================================================

        html.Div(
            [

                html.H2(
                    "Proposed Hybrid Model",
                    style={
                        "textAlign": "center"
                    }
                ),

                html.H1(
                    proposed_model,
                    style={
                        "textAlign": "center"
                    }
                ),

                html.P(
                    "The proposed system combines an "
                    "XGBoost base prediction with a "
                    "Random Forest residual-correction stage.",
                    style={
                        "textAlign": "center",
                        "fontSize": "16px"
                    }
                )

            ],

            style={
                "backgroundColor": "white",
                "padding": "20px",
                "marginTop": "25px",
                "marginBottom": "25px",
                "borderRadius": "10px",
                "boxShadow": (
                    "0 2px 5px rgba(0,0,0,0.1)"
                )
            }
        ),


        # =================================================
        # PERFORMANCE KPI CARDS
        # =================================================

        html.Div(

            style={
                "display": "flex",
                "gap": "20px",
                "marginBottom": "30px"
            },

            children=[

                # MAE

                html.Div(
                    [
                        html.H4("MAE"),
                        html.H2(
                            f"{proposed_mae:,.2f}"
                        ),
                        html.P(
                            "Lower is better"
                        )
                    ],

                    style={
                        "flex": "1",
                        "padding": "20px",
                        "backgroundColor": "white",
                        "textAlign": "center",
                        "borderRadius": "10px",
                        "boxShadow": (
                            "0 2px 5px rgba(0,0,0,0.1)"
                        )
                    }
                ),


                # RMSE

                html.Div(
                    [
                        html.H4("RMSE"),
                        html.H2(
                            f"{proposed_rmse:,.2f}"
                        ),
                        html.P(
                            "Lower is better"
                        )
                    ],

                    style={
                        "flex": "1",
                        "padding": "20px",
                        "backgroundColor": "white",
                        "textAlign": "center",
                        "borderRadius": "10px",
                        "boxShadow": (
                            "0 2px 5px rgba(0,0,0,0.1)"
                        )
                    }
                ),


                # MAPE

                html.Div(
                    [
                        html.H4("MAPE"),
                        html.H2(
                            f"{proposed_mape:.2f}%"
                        ),
                        html.P(
                            "Lower is better"
                        )
                    ],

                    style={
                        "flex": "1",
                        "padding": "20px",
                        "backgroundColor": "white",
                        "textAlign": "center",
                        "borderRadius": "10px",
                        "boxShadow": (
                            "0 2px 5px rgba(0,0,0,0.1)"
                        )
                    }
                ),


                # R2

                html.Div(
                    [
                        html.H4("R² Score"),
                        html.H2(
                            f"{proposed_r2:.4f}"
                        ),
                        html.P(
                            "Higher is better"
                        )
                    ],

                    style={
                        "flex": "1",
                        "padding": "20px",
                        "backgroundColor": "white",
                        "textAlign": "center",
                        "borderRadius": "10px",
                        "boxShadow": (
                            "0 2px 5px rgba(0,0,0,0.1)"
                        )
                    }
                )
            ]
        ),


        # =================================================
        # FORECAST KPI CARDS
        # =================================================

        html.Div(

            style={
                "display": "flex",
                "gap": "20px",
                "marginBottom": "30px"
            },

            children=[

                html.Div(
                    [
                        html.H4(
                            "30-Day Forecast Sales"
                        ),
                        html.H2(
                            f"{total_forecast:,.0f}"
                        )
                    ],

                    style={
                        "flex": "1",
                        "padding": "20px",
                        "backgroundColor": "white",
                        "textAlign": "center",
                        "borderRadius": "10px"
                    }
                ),

                html.Div(
                    [
                        html.H4(
                            "Average Daily Forecast"
                        ),
                        html.H2(
                            f"{average_forecast:,.0f}"
                        )
                    ],

                    style={
                        "flex": "1",
                        "padding": "20px",
                        "backgroundColor": "white",
                        "textAlign": "center",
                        "borderRadius": "10px"
                    }
                ),

                html.Div(
                    [
                        html.H4(
                            "Minimum Forecast"
                        ),
                        html.H2(
                            f"{minimum_forecast:,.0f}"
                        )
                    ],

                    style={
                        "flex": "1",
                        "padding": "20px",
                        "backgroundColor": "white",
                        "textAlign": "center",
                        "borderRadius": "10px"
                    }
                ),

                html.Div(
                    [
                        html.H4(
                            "Maximum Forecast"
                        ),
                        html.H2(
                            f"{maximum_forecast:,.0f}"
                        )
                    ],

                    style={
                        "flex": "1",
                        "padding": "20px",
                        "backgroundColor": "white",
                        "textAlign": "center",
                        "borderRadius": "10px"
                    }
                )
            ]
        ),


        # =================================================
        # MODEL PERFORMANCE
        # =================================================

        html.Div(
            [
                dcc.Graph(
                    figure=model_fig
                )
            ],

            style={
                "backgroundColor": "white",
                "padding": "15px",
                "marginBottom": "25px",
                "borderRadius": "10px"
            }
        ),


        # =================================================
        # ACTUAL VS PREDICTED
        # =================================================

        html.Div(
            [
                dcc.Graph(
                    figure=actual_vs_pred_fig
                )
            ],

            style={
                "backgroundColor": "white",
                "padding": "15px",
                "marginBottom": "25px",
                "borderRadius": "10px"
            }
        ),


        # =================================================
        # ERROR COMPARISON
        # =================================================

        html.Div(
            [
                dcc.Graph(
                    figure=error_fig
                )
            ],

            style={
                "backgroundColor": "white",
                "padding": "15px",
                "marginBottom": "25px",
                "borderRadius": "10px"
            }
        ),


        # =================================================
        # FORECAST
        # =================================================

        html.Div(
            [
                dcc.Graph(
                    figure=forecast_fig
                )
            ],

            style={
                "backgroundColor": "white",
                "padding": "15px",
                "marginBottom": "25px",
                "borderRadius": "10px"
            }
        ),


        # =================================================
        # SHAP EXPLAINABILITY
        # =================================================

        html.Div(
            [

                html.H2(
                    "Explainability Using SHAP",
                    style={
                        "textAlign": "center"
                    }
                ),

                html.P(
                    "SHAP identifies the features that "
                    "contribute most strongly to the "
                    "XGBoost base-model predictions. "
                    "The residual Random Forest correction "
                    "layer is not directly explained by "
                    "this SHAP analysis.",
                    style={
                        "textAlign": "center",
                        "fontSize": "15px"
                    }
                ),

                dcc.Graph(
                    figure=shap_fig
                )

            ],

            style={
                "backgroundColor": "white",
                "padding": "15px",
                "marginBottom": "25px",
                "borderRadius": "10px"
            }
        ),


        # =================================================
        # FOOTER
        # =================================================

        html.Hr(),

        html.P(
            "An Explainable Machine Learning Approach "
            "for Sales Forecasting and Demand Prediction",
            style={
                "textAlign": "center",
                "fontSize": "14px"
            }
        )

    ]
)


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    print("\n==============================================")
    print("STARTING FINAL DASHBOARD")
    print("==============================================")

    print(
        "\nProposed Model:",
        proposed_model
    )

    print(
        "R²:",
        f"{proposed_r2:.4f}"
    )

    print(
        "MAE:",
        f"{proposed_mae:.2f}"
    )

    print(
        "RMSE:",
        f"{proposed_rmse:.2f}"
    )

    print(
        "MAPE:",
        f"{proposed_mape:.2f}%"
    )

    print(
        "\n30-Day Forecast:",
        f"{total_forecast:,.2f}"
    )

    app.run(
        debug=True
    )