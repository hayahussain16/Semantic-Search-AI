import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2")

df = pd.DataFrame({
    "id": range(1, 29),
    "text": [
        "How to reset your account password",
        "Steps to change your login credentials",
        "Our refund policy for cancelled orders",
        "Delivery usually takes 3 to 5 business days",
        "How to track the status of your shipment",
        "Python is a popular programming language",
        "Learn to code with beginner-friendly tutorials",
        "The weather today is cloudy with light rain",
        "How to update your account email",
        "How to change your username",
        "How to enable two-factor authentication",
        "How to delete my account",
        "What payment methods do you accept?",
        "How can I cancel my subscription?",
        "Can I get a refund for my subscription?",
        "How do I contact customer support?",
        "Why has my payment failed?",
        "How to update my billing information",
        "What are the subscription plans?",
        "How can I download my invoice?",
        "How do I reset my security questions?",
        "How can I change my profile picture?",
        "How do I create a new account?",
        "What should I do if my account is locked?",
        "How can I change my password?",
        "Where can I see my order history?",
        "How do I return a product?",
        "How long does a refund take?"
    ],
})

# 1. Embed all documents once (normalize so cosine = dot product)
doc_vectors = model.encode(
    df["text"].tolist(),
    normalize_embeddings=True
)

np.save("doc_vectors.npy", doc_vectors)


def search(query: str, top_k: int = 3) -> pd.DataFrame:
    q = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    scores = doc_vectors @ q

    top = np.argsort(scores)[::-1][:top_k]

    result = df.iloc[top].copy()
    result["score"] = scores[top].round(3)
    result = result[result["score"] >= 0.2]

    return result


for q in [
    "I forgot my password",
    "Where is my package?",
    "start learning programming",
    "quantum physics and black holes"
]:
    print(f"\nQuery: {q}")
    print(search(q).to_string(index=False))


def keyword_search(query: str, top_k: int = 3):
    words = set(query.lower().split())

    df2 = df.copy()

    df2["hits"] = df2["text"].str.lower().apply(
        lambda t: len(words & set(t.split()))
    )

    return df2.sort_values("hits", ascending=False).head(top_k)


print(keyword_search("I forgot my password").to_string(index=False))

print("\n--- COMPARISON 1 ---")

query = "I cannot access my account"

print("\nSemantic Search:")
print(search(query).to_string(index=False))

print("\nKeyword Search:")
print(keyword_search(query).to_string(index=False))