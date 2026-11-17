
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from retrieval import search


# 1. Define the agent state
class RAGState(TypedDict):
    question: str
    documents: List[dict]
    relevant: bool
    answer: str


# 2. Retrieve relevant documents
def retrieve(state: RAGState):
    print("\nRetrieving relevant documents...")

    documents = search(state["question"], k=5)

    return {"documents": documents}


# 3. Grade retrieved documents
def grade_relevance(state: RAGState):
    print("\nChecking document relevance...")

    documents = state["documents"]

    relevant = any(
        doc.get("score", 0) >= 0.70
        for doc in documents
    )

    return {"relevant": relevant}


# 4. Generate an answer
def generate_answer(state: RAGState):
    print("\nGenerating answer...")

    if not state["relevant"]:
        return {
            "answer": "I don't know based on the available Airbnb information."
        }

    answer = "\n\n".join(
        f"Listing: {doc['name']}\n{doc['text'][:500]}"
        for doc in state["documents"][:3]
    )

    return {"answer": answer}


# 5. Build the LangGraph workflow
workflow = StateGraph(RAGState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_relevance", grade_relevance)
workflow.add_node("generate_answer", generate_answer)

workflow.set_entry_point("retrieve")

workflow.add_edge("retrieve", "grade_relevance")

workflow.add_conditional_edges(
    "grade_relevance",
    lambda state: (
        "generate_answer"
        if state["relevant"]
        else END
    ),
    {
        "generate_answer": "generate_answer",
        END: END
    }
)

workflow.add_edge("generate_answer", END)

rag_app = workflow.compile()


# 6. Run the agent
if __name__ == "__main__":
    question = input("Ask a question about Airbnb listings: ")

    result = rag_app.invoke({
        "question": question,
        "documents": [],
        "relevant": False,
        "answer": ""
    })

    print("\nFinal Answer:")
    print(result["answer"])
