"""
Module 1: Demand Forecasting
Uses time-series analysis to predict future product demand.
BigQuery ML equivalent: CREATE MODEL ARIMA_PLUS
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

def run(transactions_df, products_df):
    print("\n📦 MODULE 1: Demand Forecasting")
    print("-" * 45)

    transactions_df["transaction_date"] = pd.to_datetime(transactions_df["transaction_date"])
    transactions_df["year_month"] = transactions_df["transaction_date"].dt.to_period("M")

    monthly_demand = (
        transactions_df.groupby(["product_id", "year_month"])["quantity"]
        .sum()
        .reset_index()
    )
    monthly_demand["month_num"] = monthly_demand["year_month"].apply(
        lambda x: x.year * 12 + x.month
    )

    results = []
    for product_id in monthly_demand["product_id"].unique():
        prod_data = monthly_demand[monthly_demand["product_id"] == product_id].copy()
        if len(prod_data) < 3:
            continue

        prod_data = prod_data.sort_values("month_num")
        X = prod_data[["month_num"]].values
        y = prod_data["quantity"].values

        # Add trend and seasonality features
        prod_data["trend"] = range(len(prod_data))
        prod_data["month_of_year"] = prod_data["year_month"].apply(lambda x: x.month)

        X_features = prod_data[["trend", "month_of_year"]].values
        model = LinearRegression()
        model.fit(X_features[:-1], y[:-1])

        next_trend = len(prod_data)
        next_month = (prod_data["month_of_year"].iloc[-1] % 12) + 1
        predicted = max(0, int(model.predict([[next_trend, next_month]])[0]))

        mae = mean_absolute_error(y[1:], model.predict(X_features[:-1]))

        results.append({
            "product_id": product_id,
            "avg_monthly_demand": round(y.mean(), 1),
            "predicted_next_month": predicted,
            "mae": round(mae, 2),
        })

    results_df = pd.DataFrame(results).merge(
        products_df[["product_id", "product_name", "category"]], on="product_id"
    )

    avg_accuracy = max(0, 100 - (results_df["mae"] / results_df["avg_monthly_demand"] * 100).mean())
    print(f"   Products forecasted:  {len(results_df)}")
    print(f"   Forecast accuracy:    {avg_accuracy:.1f}%")
    print(f"   Avg MAE:              {results_df['mae'].mean():.2f} units")
    print(f"\n   Top 5 High-Demand Products (Next Month):")
    top5 = results_df.nlargest(5, "predicted_next_month")[["product_name", "predicted_next_month"]]
    for _, row in top5.iterrows():
        print(f"   • {row['product_name']:<30} → {row['predicted_next_month']} units")

    return results_df
