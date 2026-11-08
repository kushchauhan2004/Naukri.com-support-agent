# Part 1 : Dataset design and Rag core
# Task 3 : implement and index two chunking strategies

import json
import re
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

# load documents
with open("knowledge_base.json", "r", encoding="utf-8") as file:
    data = json.load(file)

documents = data["knowledge_base"]

# load embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    local_files_only=True
)

# create chroma database
client = chromadb.PersistentClient(path="./chroma_db")

fixed_collection = client.get_or_create_collection(
    name="fixed_size_chunks"
)

sentence_collection = client.get_or_create_collection(
    name="sentence_chunks"
)

# fixed size chunking with overlap
fixed_chunks = []
fixed_ids = []
fixed_metadata = []

for doc in documents:
    words = doc["content"].split()

    chunk_size = 40
    overlap = 10
    start = 0
    chunk_index = 0

    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])

        fixed_chunks.append(chunk)
        fixed_ids.append(
            f'{doc["document_id"]}_fixed_{chunk_index}'
        )
        fixed_metadata.append({
            "document_id": doc["document_id"],
            "title": doc["title"]
        })

        start += chunk_size - overlap
        chunk_index += 1

# sentence based chunking
sentence_chunks = []
sentence_ids = []
sentence_metadata = []

for doc in documents:
    sentences = re.split(r'(?<=[.!?])\s+', doc["content"])

    for index, sentence in enumerate(sentences):
        sentence = sentence.strip()

        if sentence:
            sentence_chunks.append(sentence)
            sentence_ids.append(
                f'{doc["document_id"]}_sentence_{index}'
            )
            sentence_metadata.append({
                "document_id": doc["document_id"],
                "title": doc["title"]
            })

# generate embeddings
fixed_embeddings = model.encode(fixed_chunks).tolist()
sentence_embeddings = model.encode(sentence_chunks).tolist()

# store fixed size chunks
fixed_collection.upsert(
    ids=fixed_ids,
    documents=fixed_chunks,
    embeddings=fixed_embeddings,
    metadatas=fixed_metadata
)

# store sentence chunks
sentence_collection.upsert(
    ids=sentence_ids,
    documents=sentence_chunks,
    embeddings=sentence_embeddings,
    metadatas=sentence_metadata
)

# display results
print("total documents:", len(documents))
print("fixed size chunks:", len(fixed_chunks))
print("sentence chunks:", len(sentence_chunks))

print("fixed collection:", fixed_collection.count())
print("sentence collection:", sentence_collection.count())

print("both chunking strategies indexed successfully!")

# Task 4 : Implement grounded generation

# select collection
active_collection = sentence_collection

# calculate cosine similarity and retrieve top chunks
def retrieve_chunks(query, collection, top_k=3):
    query_embedding = model.encode([query])[0]

    stored_data = collection.get(
        include=["documents", "embeddings", "metadatas"]
    )

    if not stored_data["documents"]:
        return []

    similarities = []

    for index, embedding in enumerate(stored_data["embeddings"]):
        denominator = (
            np.linalg.norm(query_embedding)
            * np.linalg.norm(embedding)
        )

        if denominator == 0:
            score = 0.0
        else:
            score = float(
                np.dot(query_embedding, embedding) / denominator
            )

        similarities.append({
            "text": stored_data["documents"][index],
            "score": score,
            "metadata": stored_data["metadatas"][index]
        })

    similarities.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return similarities[:top_k]

# calibration queries
in_scope_queries = [
    "Explain the job application process",
    "What does a Software Engineer do?",
    "What does a Data Analyst do?",
    "What does a Product Manager do?",
    "What is the role of an HR Executive?",
    "What does a Sales Associate do?",
    "What does Applied status mean?",
    "What happens during Screening?",
    "What does Interview Scheduled mean?",
    "What is the priority review policy?"
]

out_scope_queries = [
    "Explain how photosynthesis works",
    "What is the capital of Japan?",
    "How do airplanes fly?",
    "Explain nuclear fusion",
    "What are coral reefs?"
]

# measure actual similarity values
in_scope_scores = []
out_scope_scores = []

print("\ncalibration results")

for query in in_scope_queries:

    results = retrieve_chunks(query, active_collection, 1)
    score = results[0]["score"]

    in_scope_scores.append(score)

    print("in scope:", query)
    print("similarity:", round(score, 4))


for query in out_scope_queries:

    results = retrieve_chunks(query, active_collection, 1)
    score = results[0]["score"]

    out_scope_scores.append(score)

    print("out of scope:", query)
    print("similarity:", round(score, 4))


# calculate empirical threshold
min_in_scope = min(in_scope_scores)
max_out_scope = max(out_scope_scores)

if max_out_scope >= min_in_scope:
    raise ValueError(
        "similarity clusters overlap. choose better calibration "
        "queries or improve the knowledge base before setting a threshold."
    )

threshold = (min_in_scope + max_out_scope) / 2

print("\nminimum in scope:", round(min_in_scope, 4))
print("maximum out of scope:", round(max_out_scope, 4))
print("selected threshold:", round(threshold, 4))


# save calibration results in readme
readme_text = ""

try:
    with open("README.md", "r", encoding="utf-8") as file:
        readme_text = file.read()
except FileNotFoundError:
    pass

section_start = "<!-- calibration start -->"
section_end = "<!-- calibration end -->"

if section_start in readme_text:
    readme_text = readme_text.split(section_start)[0].rstrip()

calibration_report = "\n\n".join([
    section_start,
    "## grounded generation calibration",
    "Embedding model: all-MiniLM-L6-v2",
    "Collection: sentence_chunks",
    "In-scope similarity scores: " + str(
        [round(score, 4) for score in in_scope_scores]
    ),
    "Out-of-scope similarity scores: " + str(
        [round(score, 4) for score in out_scope_scores]
    ),
    "Minimum in-scope similarity: " + str(round(min_in_scope, 4)),
    "Maximum out-of-scope similarity: " + str(round(max_out_scope, 4)),
    "Selected empirical threshold: " + str(round(threshold, 4)),
    "Threshold formula: (minimum in-scope + maximum out-of-scope) / 2",
    "Fallback: I don't know based on the available documents.",
    section_end
])

with open("README.md", "w", encoding="utf-8") as file:
    file.write(readme_text + calibration_report)


# mock llm grounded answer
def grounded_answer(query, collection, top_k=3):
    results = retrieve_chunks(query, collection, top_k)

    if not results:
        return "I don't know based on the available documents."

    if results[0]["score"] < threshold:
        return "I don't know based on the available documents."

    relevant_results = [
        item for item in results
        if item["score"] >= threshold
    ]

    context = "\n".join(
        item["text"] for item in relevant_results
    )

    sources = list(set(
        item["metadata"].get("title", "Unknown")
        for item in relevant_results
    ))

    return (
        f"Answer based on the available documents:\n{context}\n"
        f"Sources: {', '.join(sources)}"
    )


# demonstrate five in-scope queries
test_queries = [
    "What is the salary range for a Data Analyst?",
    "What does Interview Scheduled mean?",
    "What are the responsibilities of an HR Executive?",
    "What does the Offered status indicate?",
    "How many days can an application be old?"
]

print("\n\nin scope query demonstrations")

for query in test_queries:

    print("\nquery:", query)

    print(
        grounded_answer(
            query,
            active_collection,
            top_k=3
        )
    )


# demonstrate one out-of-scope query
print("\n\nout of scope query demonstration")

query = "How do coral reefs reproduce?"

print("query:", query)

print(
    grounded_answer(
        query,
        active_collection,
        top_k=3
    )
)
