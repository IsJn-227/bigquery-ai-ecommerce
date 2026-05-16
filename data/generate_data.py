import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__))

CATEGORIES = ["Electronics", "Clothing", "Home & Kitchen", "Sports", "Books", "Beauty", "Toys"]
PRODUCTS = [
    ("Wireless Headphones", "Electronics", 79.99),
    ("Running Shoes", "Sports", 59.99),
    ("Yoga Mat", "Sports", 29.99),
    ("Coffee Maker", "Home & Kitchen", 49.99),
    ("Python Programming Book", "Books", 39.99),
    ("Bluetooth Speaker", "Electronics", 34.99),
    ("Face Moisturizer", "Beauty", 24.99),
    ("Resistance Bands", "Sports", 14.99),
    ("Laptop Stand", "Electronics", 44.99),
    ("Water Bottle", "Home & Kitchen", 19.99),
    ("Sunglasses", "Clothing", 22.99),
    ("Backpack", "Clothing", 54.99),
    ("LED Desk Lamp", "Home & Kitchen", 32.99),
    ("Protein Powder", "Sports", 44.99),
    ("Kindle", "Electronics", 89.99),
    ("LEGO Set", "Toys", 49.99),
    ("Board Game", "Toys", 34.99),
    ("Hair Dryer", "Beauty", 39.99),
    ("Smart Watch", "Electronics", 129.99),
    ("Cookware Set", "Home & Kitchen", 89.99),
]

def generate_customers(n=2000):
    segments = np.random.choice(["Premium", "Regular", "Budget"], n, p=[0.2, 0.5, 0.3])
    data = {
        "customer_id": [f"C{str(i).zfill(5)}" for i in range(1, n+1)],
        "age": np.random.randint(18, 70, n),
        "segment": segments,
        "city": np.random.choice(["Delhi", "Mumbai", "Bangalore", "Chennai", "Hyderabad", "Pune"], n),
        "join_date": [
            (datetime(2022, 1, 1) + timedelta(days=random.randint(0, 900))).strftime("%Y-%m-%d")
            for _ in range(n)
        ],
        "total_spend": np.round(np.random.exponential(200, n), 2),
        "num_orders": np.random.randint(1, 50, n),
        "avg_rating": np.round(np.random.uniform(2.5, 5.0, n), 1),
        "days_since_last_order": np.random.randint(1, 365, n),
        "is_churned": np.random.choice([0, 1], n, p=[0.72, 0.28]),
    }
    return pd.DataFrame(data)

def generate_products():
    rows = []
    for i, (name, cat, base_price) in enumerate(PRODUCTS):
        rows.append({
            "product_id": f"P{str(i+1).zfill(3)}",
            "product_name": name,
            "category": cat,
            "base_price": base_price,
            "stock_quantity": random.randint(10, 500),
            "reorder_point": random.randint(20, 80),
            "supplier_lead_days": random.randint(3, 14),
            "rating": round(random.uniform(3.5, 5.0), 1),
            "review_count": random.randint(50, 5000),
            "description": f"High quality {name.lower()} for everyday use. Great value and durability.",
        })
    return pd.DataFrame(rows)

def generate_transactions(customers_df, products_df, n=50000):
    rows = []
    start_date = datetime(2023, 1, 1)
    customer_ids = customers_df["customer_id"].tolist()
    product_ids = products_df["product_id"].tolist()
    base_prices = dict(zip(products_df["product_id"], products_df["base_price"]))

    for i in range(n):
        pid = random.choice(product_ids)
        base = base_prices[pid]
        discount = round(random.uniform(0, 0.3), 2)
        qty = random.randint(1, 5)
        sale_price = round(base * (1 - discount), 2)
        date = start_date + timedelta(days=random.randint(0, 730))
        rows.append({
            "transaction_id": f"T{str(i+1).zfill(7)}",
            "customer_id": random.choice(customer_ids),
            "product_id": pid,
            "quantity": qty,
            "unit_price": sale_price,
            "discount": discount,
            "total_amount": round(sale_price * qty, 2),
            "transaction_date": date.strftime("%Y-%m-%d"),
            "month": date.month,
            "day_of_week": date.weekday(),
            "is_returned": random.choice([0, 0, 0, 0, 1]),
        })
    return pd.DataFrame(rows)

def generate_reviews(customers_df, products_df, n=10000):
    sentiments = ["positive", "neutral", "negative"]
    texts = {
        "positive": [
            "Absolutely love this product! Works perfectly.",
            "Great quality, fast shipping. Highly recommend!",
            "Exceeded my expectations. Will buy again.",
            "Best purchase I made this year. Amazing!",
        ],
        "neutral": [
            "Product is okay. Does what it says.",
            "Average quality for the price.",
            "Decent product, nothing special.",
        ],
        "negative": [
            "Disappointed with the quality. Not as described.",
            "Stopped working after a week. Poor quality.",
            "Would not recommend. Waste of money.",
        ]
    }
    rows = []
    for i in range(n):
        sentiment = random.choices(sentiments, weights=[0.65, 0.20, 0.15])[0]
        rows.append({
            "review_id": f"R{str(i+1).zfill(6)}",
            "customer_id": random.choice(customers_df["customer_id"].tolist()),
            "product_id": random.choice(products_df["product_id"].tolist()),
            "rating": {"positive": random.randint(4, 5), "neutral": 3, "negative": random.randint(1, 2)}[sentiment],
            "sentiment": sentiment,
            "review_text": random.choice(texts[sentiment]),
            "review_date": (datetime(2023, 1, 1) + timedelta(days=random.randint(0, 730))).strftime("%Y-%m-%d"),
        })
    return pd.DataFrame(rows)

def generate_all():
    print("Generating synthetic e-commerce data...")
    customers = generate_customers(2000)
    products = generate_products()
    transactions = generate_transactions(customers, products, 50000)
    reviews = generate_reviews(customers, products, 10000)

    customers.to_csv(os.path.join(OUTPUT_DIR, "customers.csv"), index=False)
    products.to_csv(os.path.join(OUTPUT_DIR, "products.csv"), index=False)
    transactions.to_csv(os.path.join(OUTPUT_DIR, "transactions.csv"), index=False)
    reviews.to_csv(os.path.join(OUTPUT_DIR, "reviews.csv"), index=False)

    print(f"✅ Customers:    {len(customers):,} records")
    print(f"✅ Products:     {len(products):,} records")
    print(f"✅ Transactions: {len(transactions):,} records")
    print(f"✅ Reviews:      {len(reviews):,} records")
    print("Data saved to /data/")

if __name__ == "__main__":
    generate_all()
