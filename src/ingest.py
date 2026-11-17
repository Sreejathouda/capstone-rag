
import os
import torch
from transformers import AutoTokenizer, AutoModel
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")

if not MONGODB_URI:
    raise ValueError("MONGODB_URI is missing from .env")

# Model configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384
BATCH_SIZE = 16

# Load embedding model once
print("Loading embedding model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

print("Embedding model loaded successfully!")

# Connect to MongoDB
mongo_client = MongoClient(MONGODB_URI)

db = mongo_client["sample_airbnb"]
source_collection = db["listingsAndReviews"]
target_collection = db["rag_chunks"]


# Split text into smaller chunks
def split_text(text, chunk_size=1000, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


# Extract text from Airbnb listings
def extract_listing_text(listing):
    fields = [
        "description",
        "summary",
        "space",
        "neighborhood_overview"
    ]

    parts = []

    for field in fields:
        value = listing.get(field, "")

        if isinstance(value, str) and value.strip():
            parts.append(f"{field}: {value.strip()}")

    # Extract guest reviews
    for review in listing.get("reviews", []):
        comment = review.get("comments", "")

        if isinstance(comment, str) and comment.strip():
            parts.append(f"Guest review: {comment.strip()}")

    return "\n\n".join(parts)


# Generate normalized embeddings
def generate_embeddings(texts):
    all_embeddings = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs)

            attention_mask = inputs["attention_mask"].unsqueeze(-1)

            embeddings = (
                outputs.last_hidden_state * attention_mask
            ).sum(dim=1) / attention_mask.sum(dim=1)

            embeddings = torch.nn.functional.normalize(
                embeddings,
                p=2,
                dim=1
            )

        all_embeddings.extend(embeddings.cpu().numpy().tolist())

    return all_embeddings


# Ingest Airbnb data and save embeddings
def ingest_data():

    listings = source_collection.find(
        {},
        {
            "name": 1,
            "description": 1,
            "summary": 1,
            "space": 1,
            "neighborhood_overview": 1,
            "reviews.comments": 1
        }
    ).limit(200)

    total_chunks = 0

    for listing in listings:

        listing_text = extract_listing_text(listing)

        if not listing_text.strip():
            continue

        chunks = split_text(listing_text)

        # Generate embeddings for this listing's chunks
        embeddings = generate_embeddings(chunks)

        operations = []

        for chunk_number, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):

            chunk_id = f"{listing['_id']}_{chunk_number}"

            document = {
                "text": chunk,
                "listing_id": str(listing["_id"]),
                "name": listing.get("name", "Unknown"),
                "chunk_number": chunk_number,
                "embedding": embedding,
                "embedding_model": MODEL_NAME
            }

            operations.append(
                UpdateOne(
                    {"_id": chunk_id},
                    {"$set": document},
                    upsert=True
                )
            )

        if operations:
            target_collection.bulk_write(
                operations,
                ordered=False
            )

            total_chunks += len(operations)

        print(f"Processed: {listing.get('name', 'Unknown')}")

    print("\nEmbedding ingestion completed!")
    print(f"Total chunks processed: {total_chunks}")
    print(f"Embedding dimensions: {EMBEDDING_DIMENSIONS}")


if __name__ == "__main__":
    try:
        ingest_data()
    finally:
        mongo_client.close()
