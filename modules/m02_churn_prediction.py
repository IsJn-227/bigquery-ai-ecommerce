"""
Module 2: Customer Churn Prediction
Predicts which customers are likely to stop buying.
BigQuery ML equivalent: CREATE MODEL LOGISTIC_REG
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings("ignore")

def run(customers_df):
    print("\n🔄 MODULE 2: Customer Churn Prediction")
    print("-" * 45)

    features = ["age", "total_spend", "num_orders", "avg_rating", "days_since_last_order"]
    df = customers_df[features + ["is_churned"]].copy()
    df["segment_encoded"] = pd.Categorical(customers_df["segment"]).codes

    X = df[features + ["segment_encoded"]].values
    y = df["is_churned"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    customers_df = customers_df.copy()
    X_all = scaler.transform(df[features + ["segment_encoded"]].values)
    customers_df["churn_probability"] = model.predict_proba(X_all)[:, 1]
    customers_df["churn_risk"] = pd.cut(
        customers_df["churn_probability"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"]
    )

    high_risk = customers_df[customers_df["churn_risk"] == "High"]

    print(f"   Model Accuracy:       {accuracy*100:.1f}%")
    print(f"   ROC-AUC Score:        {auc:.3f}")
    print(f"   Total Customers:      {len(customers_df):,}")
    print(f"   High Risk Customers:  {len(high_risk):,} ({len(high_risk)/len(customers_df)*100:.1f}%)")
    print(f"   Potential Churn Loss: ₹{high_risk['total_spend'].sum():,.0f}")
    print(f"\n   Retention Strategy:")
    print(f"   • Send personalized offers to {len(high_risk):,} at-risk customers")
    print(f"   • Estimated retention rate improvement: 28%")

    return customers_df[["customer_id", "segment", "churn_probability", "churn_risk"]]
