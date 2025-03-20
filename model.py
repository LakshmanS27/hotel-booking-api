import pandas as pd
import faiss
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

# Load the cleaned dataset
file_path = "cleaned_hotel_bookings.csv"  # Update path if needed
df = pd.read_csv(file_path)

# Convert data into text format for embeddings
df['text'] = df.apply(lambda row: f"Hotel booking in {row['country']} on {row['reservation_status_date']} "
                                  f"with {row['stays_in_week_nights']} week nights and {row['stays_in_weekend_nights']} weekend nights. "
                                  f"Price: {row['adr']}. Canceled: {bool(row['is_canceled'])}.", axis=1)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
embeddings = model.encode(df['text'].tolist(), convert_to_numpy=True)

# Initialize FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# ----- Function to Answer User Queries -----
def answer_query(query, top_k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        results.append(df.iloc[idx]['text'])

    return results

# ----- Example Queries -----
user_question = "Which locations had the highest booking cancellations?"
response = answer_query(user_question)

print("🔍 Answer:")
for res in response:
    print(res)

