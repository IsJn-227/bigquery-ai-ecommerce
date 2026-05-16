"""
Module 6: Customer Segmentation
Groups customers into meaningful clusters using K-Means.
BigQuery ML equivalent: CREATE MODEL KMEANS
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

def run(customers_df):
    print("\n👥 MODULE 6: Customer Segmentation")
    print("-" * 45)

    features = ["total_spend", "num_orders", "avg_rating", "days_since_last_order", "age"]
    X = customers_df[features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    customers_df = customers_df.copy()
    customers_df["cluster"] = kmeans.fit_predict(X_scaled)

    segment_names = {0: "Champions", 1: "At-Risk", 2: "Promising", 3: "Hibernating"}
    cluster_stats = customers_df.groupby("cluster").agg(
        count=("customer_id", "count"),
        avg_spend=("total_spend", "mean"),
        avg_orders=("num_orders", "mean"),
    ).reset_index()

    # Assign names by avg spend
    cluster_stats = cluster_stats.sort_values("avg_spend", ascending=False)
    name_list = ["Champions", "Loyal", "Promising", "At-Risk"]
    cluster_stats["segment_name"] = name_list

    name_map = dict(zip(cluster_stats["cluster"], cluster_stats["segment_name"]))
    customers_df["ml_segment"] = customers_df["cluster"].map(name_map)

    print(f"   Total Customers:  {len(customers_df):,}")
    print(f"   Segments Found:   4")
    print(f"\n   Segment Breakdown:")
    for _, row in cluster_stats.iterrows():
        print(f"   • {row['segment_name']:<12} → {int(row['count']):>4} customers | Avg Spend: ₹{row['avg_spend']:>7.0f} | Avg Orders: {row['avg_orders']:.1f}")

    return customers_df[["customer_id", "ml_segment", "cluster"]]
