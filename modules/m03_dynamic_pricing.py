"""
Module 3: Dynamic Pricing Engine
Adjusts prices based on demand, competition, and inventory levels.
BigQuery ML equivalent: CREATE MODEL LINEAR_REG
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

def run(transactions_df, products_df):
    print("\n💰 MODULE 3: Dynamic Pricing Engine")
    print("-" * 45)

    # Aggregate transaction features per product
    agg = transactions_df.groupby("product_id").agg(
        total_quantity=("quantity", "sum"),
        avg_discount=("discount", "mean"),
        total_revenue=("total_amount", "sum"),
        transaction_count=("transaction_id", "count"),
        return_rate=("is_returned", "mean"),
    ).reset_index()

    df = products_df.merge(agg, on="product_id", how="left").fillna(0)
    df["demand_score"] = df["total_quantity"] / (df["total_quantity"].max() + 1)
    df["inventory_pressure"] = (df["stock_quantity"] - df["reorder_point"]) / (df["stock_quantity"] + 1)
    df["popularity_score"] = df["review_count"] / df["review_count"].max()

    features = ["demand_score", "inventory_pressure", "popularity_score",
                "avg_discount", "return_rate", "rating"]
    X = df[features].fillna(0).values
    y = df["base_price"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)

    predicted_prices = model.predict(X_scaled)
    r2 = r2_score(y, predicted_prices)

    # Dynamic price adjustment logic
    df["predicted_optimal_price"] = np.round(predicted_prices, 2)
    df["price_adjustment_pct"] = ((df["predicted_optimal_price"] - df["base_price"]) / df["base_price"] * 100).round(1)
    df["recommended_price"] = np.where(
        df["demand_score"] > 0.7,
        df["base_price"] * 1.08,   # High demand → increase price
        np.where(
            df["stock_quantity"] > df["reorder_point"] * 3,
            df["base_price"] * 0.92,  # Overstock → discount
            df["base_price"]
        )
    ).round(2)

    margin_improvement = ((df["recommended_price"] - df["base_price"]) / df["base_price"] * 100).mean()

    print(f"   Model R² Score:         {r2:.3f}")
    print(f"   Products Optimized:     {len(df)}")
    print(f"   Avg Margin Improvement: {abs(margin_improvement):.1f}%")
    print(f"\n   Pricing Recommendations:")
    for _, row in df.iterrows():
        direction = "↑" if row["recommended_price"] > row["base_price"] else "↓"
        print(f"   • {row['product_name']:<28} ₹{row['base_price']:.2f} → ₹{row['recommended_price']:.2f} {direction}")

    return df[["product_id", "product_name", "base_price", "recommended_price", "price_adjustment_pct"]]
