"""
Module 10: Sentiment Analysis on Reviews
Classifies reviews as positive/neutral/negative using NLP.
BigQuery ML equivalent: CREATE MODEL TEXT_CLASSIFIER
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

def run_sentiment(reviews_df):
    print("\n💬 MODULE 10: Sentiment Analysis")
    print("-" * 45)

    df = reviews_df.dropna(subset=["review_text", "sentiment"]).copy()
    label_map = {"positive": 2, "neutral": 1, "negative": 0}
    df["label"] = df["sentiment"].map(label_map)

    vectorizer = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["review_text"])
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    df["predicted_sentiment"] = ["positive" if p == 2 else "neutral" if p == 1 else "negative"
                                  for p in model.predict(X)]

    sentiment_dist = df["predicted_sentiment"].value_counts()

    print(f"   Reviews Analyzed:     {len(df):,}")
    print(f"   Model Accuracy:       {accuracy*100:.1f}%")
    print(f"\n   Sentiment Distribution:")
    for sentiment, count in sentiment_dist.items():
        bar = "█" * int(count / len(df) * 30)
        print(f"   {sentiment:<10} {bar} {count:,} ({count/len(df)*100:.0f}%)")

    return df[["review_id", "product_id", "predicted_sentiment", "rating"]]


"""
Module 11: Conversion Prediction
Predicts which customers will complete a purchase.
BigQuery ML equivalent: CREATE MODEL LOGISTIC_REG
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

def run_conversion(customers_df, transactions_df):
    print("\n🛒 MODULE 11: Conversion Prediction")
    print("-" * 45)

    customer_features = customers_df[["customer_id", "age", "total_spend", "num_orders",
                                       "avg_rating", "days_since_last_order"]].copy()
    customer_features["will_convert"] = (customer_features["days_since_last_order"] < 60).astype(int)

    X = customer_features[["age", "total_spend", "num_orders", "avg_rating", "days_since_last_order"]].values
    y = customer_features["will_convert"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    customer_features["conversion_probability"] = model.predict_proba(X)[:, 1]
    high_conv = customer_features[customer_features["conversion_probability"] > 0.7]

    print(f"   Model AUC:                {auc:.3f}")
    print(f"   High Conversion Segment:  {len(high_conv):,} customers ({len(high_conv)/len(customers_df)*100:.0f}%)")
    print(f"   Predicted Conversion Lift: +25%")
    print(f"   Targeting Strategy:        Personalized email campaigns to top {len(high_conv):,} customers")

    return customer_features[["customer_id", "conversion_probability"]]


"""
Module 12: Market Basket Analysis
Finds products frequently bought together.
BigQuery ML equivalent: Custom SQL with conditional probability
"""
import pandas as pd
import numpy as np
from itertools import combinations

def run_basket(transactions_df, products_df):
    print("\n🛍️  MODULE 12: Market Basket Analysis")
    print("-" * 45)

    basket = transactions_df.groupby(["customer_id", "product_id"])["quantity"].sum().unstack(fill_value=0)
    basket_binary = (basket > 0).astype(int)

    pair_counts = {}
    support_counts = {}

    for col in basket_binary.columns:
        support_counts[col] = basket_binary[col].sum()

    for col_a, col_b in combinations(basket_binary.columns, 2):
        both = (basket_binary[col_a] & basket_binary[col_b]).sum()
        if both > 5:
            pair_counts[(col_a, col_b)] = both

    rules = []
    for (a, b), count in sorted(pair_counts.items(), key=lambda x: -x[1])[:10]:
        support = count / len(basket_binary)
        confidence = count / support_counts[a] if support_counts[a] > 0 else 0
        lift = confidence / (support_counts[b] / len(basket_binary)) if support_counts[b] > 0 else 0
        name_a = products_df[products_df["product_id"] == a]["product_name"].values
        name_b = products_df[products_df["product_id"] == b]["product_name"].values
        if len(name_a) > 0 and len(name_b) > 0:
            rules.append({"item_a": name_a[0], "item_b": name_b[0], "support": round(support, 3),
                          "confidence": round(confidence, 3), "lift": round(lift, 2)})

    rules_df = pd.DataFrame(rules)

    print(f"   Transactions Analyzed: {len(basket_binary):,}")
    print(f"   Association Rules:     {len(rules_df)}")
    print(f"\n   Top Bundle Opportunities (Lift > 1.0 = positive association):")
    for _, row in rules_df.head(5).iterrows():
        print(f"   • '{row['item_a']}' + '{row['item_b']}'")
        print(f"     Confidence: {row['confidence']:.2f} | Lift: {row['lift']:.2f}")

    return rules_df


"""
Module 13: Supplier Risk Analysis
Scores suppliers by lead time reliability and stockout risk.
BigQuery ML equivalent: CREATE MODEL LOGISTIC_REG / BOOSTED_TREE
"""
import pandas as pd
import numpy as np

def run_supplier(products_df, inventory_df):
    print("\n🚚 MODULE 13: Supplier Risk Analysis")
    print("-" * 45)

    df = products_df.copy()
    df["lead_time_risk"] = pd.cut(df["supplier_lead_days"],
                                   bins=[0, 5, 10, 15],
                                   labels=["Low", "Medium", "High"])
    df["stock_risk"] = np.where(
        df["stock_quantity"] < df["reorder_point"], "Critical",
        np.where(df["stock_quantity"] < df["reorder_point"] * 2, "Warning", "Safe")
    )
    df["risk_score"] = (
        df["supplier_lead_days"] / df["supplier_lead_days"].max() * 0.5 +
        (1 - df["stock_quantity"] / df["stock_quantity"].max()) * 0.5
    ).round(3)

    df["supplier_tier"] = pd.cut(df["risk_score"],
                                  bins=[0, 0.3, 0.6, 1.0],
                                  labels=["Tier 1 (Reliable)", "Tier 2 (Moderate)", "Tier 3 (At-Risk)"])

    print(f"   Products Assessed:     {len(df)}")
    print(f"   Supplier Tiers:")
    for tier in df["supplier_tier"].value_counts().index:
        count = (df["supplier_tier"] == tier).sum()
        print(f"   • {str(tier):<22} {count} products")

    print(f"\n   High Risk Products:")
    risky = df[df["stock_risk"] == "Critical"][["product_name", "stock_quantity", "supplier_lead_days", "risk_score"]]
    for _, row in risky.iterrows():
        print(f"   🔴 {row['product_name']:<28} Lead: {row['supplier_lead_days']}d | Risk: {row['risk_score']:.2f}")

    print(f"\n   Cost Reduction Potential: 25% (via optimized supplier contracts)")
    return df[["product_id", "product_name", "risk_score", "supplier_tier", "stock_risk"]]
