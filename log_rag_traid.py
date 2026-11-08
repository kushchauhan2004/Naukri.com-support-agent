# Part 3 : Evaluation, Observability & FastAPI Deployment
# Task 12 : Add structured logging.
import json
import re
import time
import uuid
from datetime import datetime, timezone

from chunking import retrieve_chunks, active_collection, grounded_answer


def mask_pii(text):

    text = re.sub(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "[EMAIL MASKED]",
        text
    )

    text = re.sub(
        r"(?<!\d)(?:\+91[- ]?)?[6-9]\d{9}(?!\d)",
        "[PHONE MASKED]",
        text
    )

    return text


# mask every string inside a json request

def mask_json_values(value):

    if isinstance(value, dict):

        return {
            key: mask_json_values(item)
            for key, item in value.items()
        }

    if isinstance(value, list):

        return [mask_json_values(item) for item in value]

    if isinstance(value, str):

        return mask_pii(value)

    return value


# json lines logging middleware

class JsonlLoggingMiddleware:

    def __init__(self, app, log_file="api_requests.jsonl"):

        self.app = app
        self.log_file = log_file

    async def __call__(self, scope, receive, send):

        if scope["type"] != "http":

            await self.app(scope, receive, send)
            return

        trace_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        # read request body

        body_parts = []

        while True:

            message = await receive()

            if message["type"] == "http.request":

                body_parts.append(message.get("body", b""))

                if not message.get("more_body", False):
                    break

            elif message["type"] == "http.disconnect":
                break

        original_body = b"".join(body_parts)

        # prepare masked request text for logging

        try:

            request_data = json.loads(original_body)

            safe_data = mask_json_values(request_data)

            safe_request_text = json.dumps(safe_data)

        except Exception:

            safe_request_text = mask_pii(
                original_body.decode("utf-8", errors="replace")
            )

        # replay original request body to fastapi

        body_sent = False

        async def receive_again():

            nonlocal body_sent

            if not body_sent:

                body_sent = True

                return {
                    "type": "http.request",
                    "body": original_body,
                    "more_body": False
                }

            return await receive()

        status_code = 500

        async def send_with_status(message):

            nonlocal status_code

            if message["type"] == "http.response.start":

                status_code = message["status"]

            await send(message)

        try:

            await self.app(
                scope,
                receive_again,
                send_with_status
            )

        finally:

            duration_ms = round(
                (time.perf_counter() - start_time) * 1000,
                2
            )

            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trace_id": trace_id,
                "method": scope["method"],
                "path": scope["path"],
                "status_code": status_code,
                "duration_ms": duration_ms,
                "request_text": safe_request_text
            }

            with open(self.log_file, "a", encoding="utf-8") as file:

                file.write(json.dumps(log_entry) + "\n")


# register middleware in fastapi

def register_jsonl_logging(api):

    api.add_middleware(JsonlLoggingMiddleware)


# Task 13 : Evaluate with the RAG triad at scale.

MOCK_LLM = True


# words ignored during scoring

stop_words = {
    "the", "is", "a", "an", "and", "or",
    "to", "of", "in", "on", "for", "what",
    "how", "does", "do", "are", "can"
}

def get_words(text):
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

    return {
        word for word in words
        if word not in stop_words
    }


# mock llm judge prompt

def mock_llm_judge(query, context, answer):

    query_words = get_words(query)
    context_words = get_words(context)
    answer_words = get_words(answer)

    if not context_words:
        context_relevance = 1
    else:
        context_overlap = (
            len(query_words & context_words)
            / max(len(query_words), 1)
        )

        context_relevance = round(
            context_overlap * 5, 2
        )

    if not answer_words:
        answer_relevance = 1
    else:
        answer_overlap = (
            len(query_words & answer_words)
            / max(len(query_words), 1)
        )

        answer_relevance = round(
            answer_overlap * 5, 2
        )

    if not answer_words:
        groundedness = 1
    else:
        grounded_overlap = (
            len(answer_words & context_words)
            / max(len(answer_words), 1)
        )

        groundedness = round(
            grounded_overlap * 5, 2
        )

    return {
        "context_relevance": context_relevance,
        "groundedness": groundedness,
        "answer_relevance": answer_relevance
    }

# test set containing 15 queries

test_queries = [

    "What is the salary range for a Software Engineer?",

    "What is the salary range for a Data Analyst?",

    "What are the responsibilities of an HR Executive?",

    "What does the Screening application status mean?",

    "What does Interview Scheduled status mean?",

    "What does Offered status mean?",

    "How is application age measured in days?",

    "What is the priority review policy for applications?",

    "How can I identify an application flagged for priority review?",

    "What happens when an application is in Screening?",

    "What is the difference between Interview Scheduled and Offered?",

    "How many days since an application was created?",

    "What salary information is available for Data Analysts?",

    "Explain how coral reefs are formed.",

    "Explain stellar nuclear fusion."
]


# run evaluation

def run_evaluation():

    results = []

    for number, query in enumerate(test_queries, start=1):

        print("\nEvaluating query", number)
        print("Question:", query)

        # retrieve context

        retrieved = retrieve_chunks(
            query,
            active_collection,
            top_k=3
        )

        context = "\n".join(
            item["text"] for item in retrieved
        )

        # generate answer using existing rag function

        answer = grounded_answer(
            query,
            active_collection,
            top_k=3
        )

        # evaluate with mock llm judge

        scores = mock_llm_judge(
            query,
            context,
            answer
        )

        result = {
            "query": query,
            "context_relevance": scores["context_relevance"],
            "groundedness": scores["groundedness"],
            "answer_relevance": scores["answer_relevance"]
        }

        results.append(result)

        print("Context relevance:", scores["context_relevance"])
        print("Groundedness:", scores["groundedness"])
        print("Answer relevance:", scores["answer_relevance"])

    # calculate average scores

    total = len(results)

    averages = {

        "context_relevance": round(
            sum(r["context_relevance"] for r in results) / total, 2
        ),

        "groundedness": round(
            sum(r["groundedness"] for r in results) / total, 2
        ),

        "answer_relevance": round(
            sum(r["answer_relevance"] for r in results) / total, 2
        )
    }

    # save evaluation report

    report = {
        "mock_llm": MOCK_LLM,
        "total_queries": total,
        "results": results,
        "average_scores": averages
    }

    with open("rag_triad_results.json", "w") as file:

        json.dump(report, file, indent=4)

    print("\nRAG TRIAD EVALUATION REPORT")

    print(json.dumps(averages, indent=4))

    print("\nReport saved to rag_triad_results.json")


# demonstrate pii masking

if __name__ == "__main__":

    print("PII MASKING TEST")

    test_pii = (
        "My email is student@example.com "
        "and my phone is 9876543210"
    )

    print("Original:", test_pii)
    print("Masked:", mask_pii(test_pii))

    print("\nStarting RAG triad evaluation")

    run_evaluation()
