"""
Module 8: Sales Forecasting
Forecasts total revenue for next quarter using regression.
BigQuery ML equivalent: CREATE MODEL ARIMA_PLUS / LINEAR_REG
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")

def run(transactions_df):
    print("\n📈 MODULE 8: Sales Forecasting")
    print("-" * 45)

    transactions_df = transactions_df.copy()
    transactions_df["transaction_date"] = pd.to_datetime(transactions_df["transaction_date"])
    transactions_df["year_month"] = transactions_df["transaction_date"].dt.to_period("M")

    monthly_sales = transactions_df.groupby("year_month")["total_amount"].sum().reset_index()
    monthly_sales["month_num"] = range(len(monthly_sales))
    monthly_sales["month_of_year"] = monthly_sales["year_month"].apply(lambda x: x.month)

    X = monthly_sales[["month_num", "month_of_year"]].values
    y = monthly_sales["total_amount"].values

    split = int(len(X) * 0.8)
    model = LinearRegression()
    model.fit(X[:split], y[:split])

    y_pred = model.predict(X[split:])
    mape = mean_absolute_percentage_error(y[split:], y_pred) * 100

    # Forecast next 3 months
    last_month_num = monthly_sales["month_num"].max()
    last_month_of_year = monthly_sales["month_of_year"].iloc[-1]
    forecasts = []
    for i in range(1, 4):
        next_num = last_month_num + i
        next_moy = ((last_month_of_year - 1 + i) % 12) + 1
        pred = max(0, model.predict([[next_num, next_moy]])[0])
        forecasts.append({"month": f"Month +{i}", "forecasted_revenue": round(pred, 2)})

    forecast_df = pd.DataFrame(forecasts)
    total_forecast = forecast_df["forecasted_revenue"].sum()

    print(f"   Historical Months:    {len(monthly_sales)}")
    print(f"   Forecast Accuracy:    {100-mape:.1f}% (MAPE: {mape:.1f}%)")
    print(f"   Avg Monthly Revenue:  ₹{y.mean():,.0f}")
    print(f"\n   Next Quarter Forecast:")
    for _, row in forecast_df.iterrows():
        print(f"   • {row['month']}:  ₹{row['forecasted_revenue']:>12,.0f}")
    print(f"   • Total Q Forecast:  ₹{total_forecast:>12,.0f}")

    return forecast_df


"""
Module 9: Anomaly Detection
Detects fraudulent transactions and unusual patterns.
BigQuery ML equivalent: CREATE MODEL AUTOENCODER / KMEANS for outlier detection
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

def run_anomaly(transactions_df, products_df):
    print("\n🚨 MODULE 9: Anomaly Detection (Fraud)")
    print("-" * 45)

    features = ["quantity", "unit_price", "total_amount", "discount", "day_of_week", "month"]
    X = transactions_df[features].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(contamination=0.03, random_state=42, n_estimators=100)
    transactions_df = transactions_df.copy()
    transactions_df["anomaly_score"] = model.fit_predict(X_scaled)
    transactions_df["is_anomaly"] = transactions_df["anomaly_score"] == -1

    anomalies = transactions_df[transactions_df["is_anomaly"]]

    print(f"   Total Transactions:   {len(transactions_df):,}")
    print(f"   Anomalies Detected:   {len(anomalies):,} ({len(anomalies)/len(transactions_df)*100:.1f}%)")
    print(f"   Avg Anomaly Amount:   ₹{anomalies['total_amount'].mean():.2f}")
    print(f"   Normal Avg Amount:    ₹{transactions_df[~transactions_df['is_anomaly']]['total_amount'].mean():.2f}")
    print(f"\n   Top Suspicious Transactions:")
    top_suspicious = anomalies.nlargest(5, "total_amount")[["transaction_id", "quantity", "total_amount", "discount"]]
    for _, row in top_suspicious.iterrows():
        print(f"   ⚠️  {row['transaction_id']} | Qty: {row['quantity']} | ₹{row['total_amount']:.2f} | Discount: {row['discount']*100:.0f}%")

    return transactions_df[["transaction_id", "is_anomaly", "total_amount"]]
