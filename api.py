from fastapi import FastAPI
import pandas as pd
import faiss
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load the cleaned dataset
file_path = "cleaned_hotel_bookings.csv"
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


# ---------------------- API MODELS ----------------------
class QueryRequest(BaseModel):
    question: str


# ---------------------- ANALYTICS FUNCTIONS ----------------------
def calculate_analytics():
    """Compute booking analytics."""
    analytics = {
        "total_bookings": len(df),
        "total_revenue": df["adr"].sum(),
        "cancellation_rate": (df["is_canceled"].sum() / len(df)) * 100,
        "avg_booking_price": df["adr"].mean(),
        "top_cancellation_countries": df[df["is_canceled"] == 1]["country"].value_counts().head(5).to_dict(),
    }
    return analytics


# ---------------------- FAISS RAG FUNCTION ----------------------
def answer_query(query, top_k=3):
    """Retrieve most relevant bookings."""
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        results.append(df.iloc[idx]['text'])

    return results


# ---------------------- API ENDPOINTS ----------------------
@app.get("/health")
def health_check():
    """Check if API is running."""
    return {"status": "API is running smoothly!"}


@app.post("/analytics")
def get_analytics():
    """Return booking analytics."""
    analytics = calculate_analytics()
    return analytics


@app.post("/ask")
def ask_question(request: QueryRequest):
    """Return answers for user questions."""
    response = answer_query(request.question)
    return {"question": request.question, "answers": response}


# ---------------------- RUN SERVER ----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
