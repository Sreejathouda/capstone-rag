
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/content")

from src.rag_agent import app, REFUSAL, validate_answer

DATASET_FILE = Path("/content/evals/datasets/qa.jsonl")
OUTPUT_FILE = Path("/content/evals/evaluation_results_latest.json")

with open(DATASET_FILE, "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f]

results = []

for i, item in enumerate(dataset, start=1):
    print(f"\nQuestion {i}/{len(dataset)}: {item['question']}")

    attempts = 0
    while True:
        try:
            result = app.invoke({
                "question": item["question"],
                "search_query": item["question"],
                "documents": [],
                "relevant": False,
                "retry_count": 0,
                "answer": ""
            })
            break
        except Exception as e:
            attempts += 1
            error = str(e)
            temporary = any(x in error for x in [
                "429", "RESOURCE_EXHAUSTED", "503",
                "UNAVAILABLE", "500", "INTERNAL"
            ])
            if temporary and attempts < 5:
                print("Temporary model error. Waiting 30 seconds...")
                time.sleep(30)
            else:
                raise

    documents = result.get("documents", [])
    answer = result.get("answer", "").strip()
    relevant_ids = list(dict.fromkeys(
        str(x) for x in item["relevant_listing_ids"]
    ))

    retrieved_ids = []
    for doc in documents:
        listing_id = doc.metadata.get("listing_id")
        if listing_id is not None and str(listing_id) not in retrieved_ids:
            retrieved_ids.append(str(listing_id))

    recall = None
    if relevant_ids:
        recall = (
            len(set(retrieved_ids[:5]) & set(relevant_ids))
            / len(relevant_ids)
        )

    is_refusal = answer == REFUSAL
    refusal_correct = (
        is_refusal if not relevant_ids else not is_refusal
    )

    citation = None
    if relevant_ids and not is_refusal:
        citation = bool(validate_answer(answer, documents))

    results.append({
        "question": item["question"],
        "recall": recall,
        "citation": citation,
        "refusal": refusal_correct,
        "answer": answer,
        "retrieved_ids": retrieved_ids
    })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Recall:", recall,
          "| Citation:", citation,
          "| Refusal correct:", refusal_correct)

    time.sleep(5)

recalls = [r["recall"] for r in results if r["recall"] is not None]
citations = [r["citation"] for r in results if r["citation"] is not None]
refusals = [r["refusal"] for r in results]

avg_recall = sum(recalls) / len(recalls) if recalls else 0
citation_rate = sum(citations) / len(citations) if citations else 0
refusal_rate = sum(refusals) / len(refusals) if refusals else 0

print("\n========== FRESH EVALUATION RESULTS ==========")
print("Total questions:", len(results))
print("Recall@5:", round(avg_recall, 3))
print("Citation groundedness:", round(citation_rate, 3))
print("Refusal correctness:", round(refusal_rate, 3))
print("Citation denominator:", len(citations))
print("Refusal denominator:", len(refusals))

passed = (
    len(results) == len(dataset)
    and avg_recall >= 0.70
    and citation_rate >= 0.90
    and refusal_rate == 1.0
)

print("\nREGRESSION GATE:", "PASSED" if passed else "FAILED")
print("Results saved to:", OUTPUT_FILE)

if not passed:
    sys.exit(1)
