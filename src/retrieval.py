
import os
import torch
from pymongo import MongoClient
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel

load_dotenv()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


def generate_query_embedding(query):
    inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    mask = inputs["attention_mask"].unsqueeze(-1)
    embeddings = (outputs.last_hidden_state * mask).sum(dim=1)
    embeddings = embeddings / mask.sum(dim=1)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings[0].tolist()


def search(query, k=5):
    client = MongoClient(os.getenv("MONGODB_URI"))
    collection = client["sample_airbnb"]["rag_chunks"]

    query_vector = generate_query_embedding(query)

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 50,
                "limit": k
            }
        },
        {
            "$project": {
                "_id": 0,
                "name": 1,
                "text": 1,
                "listing_id": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    results = list(collection.aggregate(pipeline))
    client.close()

    return results


if __name__ == "__main__":
    results = search("Apartments near the beach", 5)

    for result in results:
        print("\nListing:", result["name"])
        print("Score:", result["score"])
        print("Text:", result["text"][:300])
