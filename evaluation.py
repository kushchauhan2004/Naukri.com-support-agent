# Part 1 : Dataset Design and Rag core
# Task 5 : Evaluate and compare both chunking strategies.
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

# load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# connect to chroma database
client = chromadb.PersistentClient(path="./chroma_db")

fixed_collection = client.get_collection(
    name="fixed_size_chunks"
)

sentence_collection = client.get_collection(
    name="sentence_chunks"
)

# test queries and relevant document ids
queries = [
    {
        "query": "What is the salary range for a Data Analyst?",
        "relevant_docs": ["KB003"]
    },
    {
        "query": "What does Interview Scheduled mean?",
        "relevant_docs": ["KB009"]
    },
    {
        "query": "What are the responsibilities of an HR Executive?",
        "relevant_docs": ["KB005"]
    },
    {
        "query": "What does the Offered status indicate?",
        "relevant_docs": ["KB010"]
    },
    {
        "query": "What is the priority review policy",
        "relevant_docs": ["KB012"]
    }
]

# retrieve top 3 unique documents
def retrieve_documents(query, collection):

    query_embedding = model.encode([query])[0]

    data = collection.get(
        include=["documents", "embeddings", "metadatas"]
    )

    results = []

    for index, embedding in enumerate(data["embeddings"]):

        score = np.dot(query_embedding, embedding) / (
            np.linalg.norm(query_embedding) *
            np.linalg.norm(embedding)
        )

        results.append({
            "document_id": data["metadatas"][index]["document_id"],
            "similarity": float(score)
        })

    results.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    # remove duplicate parent documents
    unique_results = []
    seen = set()

    for result in results:

        doc_id = result["document_id"]

        if doc_id not in seen:
            unique_results.append(result)
            seen.add(doc_id)

        if len(unique_results) == 3:
            break

    return unique_results


# calculate precision and recall
def evaluate_collection(collection, collection_name):

    total_precision = 0
    total_recall = 0

    print("\n", collection_name)
    print("-" * 50)

    for item in queries:

        query = item["query"]
        relevant_docs = set(item["relevant_docs"])

        retrieved = retrieve_documents(query, collection)

        retrieved_docs = {
            result["document_id"] for result in retrieved
        }

        correct = retrieved_docs.intersection(relevant_docs)

        precision = len(correct) / 3
        recall = len(correct) / len(relevant_docs)

        total_precision += precision
        total_recall += recall

        print("\nquery:", query)
        print("relevant documents:", relevant_docs)
        print("retrieved documents:", retrieved_docs)
        print("correct documents:", correct)

        print(
            "precision@3:",
            len(correct), "/ 3 =",
            round(precision, 2)
        )

        print(
            "recall@3:",
            len(correct), "/",
            len(relevant_docs), "=",
            round(recall, 2)
        )

    average_precision = total_precision / len(queries)
    average_recall = total_recall / len(queries)

    print("\naverage precision@3:", round(average_precision, 2))
    print("average recall@3:", round(average_recall, 2))

    return average_precision, average_recall


# evaluate both strategies
fixed_precision, fixed_recall = evaluate_collection(
    fixed_collection,
    "fixed size chunking"
)

sentence_precision, sentence_recall = evaluate_collection(
    sentence_collection,
    "sentence based chunking"
)


# compare both strategies
print("\nfinal comparison")
print("-" * 50)

print(
    "fixed size precision:",
    round(fixed_precision, 2)
)

print(
    "fixed size recall:",
    round(fixed_recall, 2)
)

print(
    "sentence precision:",
    round(sentence_precision, 2)
)

print(
    "sentence recall:",
    round(sentence_recall, 2)
)

if sentence_precision + sentence_recall > fixed_precision + fixed_recall:

    print("\nrecommendation: use sentence based chunking")

elif sentence_precision + sentence_recall < fixed_precision + fixed_recall:

    print("\nrecommendation: use fixed size chunking")

else:

    print("\nboth strategies have equal average scores")
