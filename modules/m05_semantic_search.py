"""
Module 5: Semantic Search
Finds relevant products based on natural language queries using TF-IDF vectors.
BigQuery ML equivalent: CREATE MODEL TEXT_EMBEDDING + vector search
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def run(products_df):
    print("\n🔍 MODULE 5: Semantic Search Engine")
    print("-" * 45)

    corpus = (
        products_df["product_name"] + " " +
        products_df["category"] + " " +
        products_df["description"]
    ).tolist()

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    test_queries = [
        "wireless audio device for music",
        "exercise equipment for home workout",
        "smart technology wearable",
        "kitchen appliance for coffee",
        "outdoor adventure footwear",
    ]

    print(f"   Products Indexed:     {len(products_df)}")
    print(f"   Vocabulary Size:      {len(vectorizer.vocabulary_):,} terms")
    print(f"\n   Search Results:")

    all_scores = []
    for query in test_queries:
        q_vec = vectorizer.transform([query])
        scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
        top_idx = scores.argsort()[::-1][:3]
        top_score = scores[top_idx[0]]
        all_scores.append(top_score)

        print(f"\n   Query: '{query}'")
        for idx in top_idx:
            print(f"   → {products_df.iloc[idx]['product_name']:<30} (relevance: {scores[idx]:.2f})")

    avg_relevance = np.mean(all_scores)
    print(f"\n   Average Search Relevance: {avg_relevance*100:.0f}%")

    def search(query, top_k=5):
        q_vec = vectorizer.transform([query])
        scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
        top_idx = scores.argsort()[::-1][:top_k]
        results = products_df.iloc[top_idx].copy()
        results["relevance_score"] = scores[top_idx]
        return results[["product_name", "category", "base_price", "relevance_score"]]

    return search, vectorizer
