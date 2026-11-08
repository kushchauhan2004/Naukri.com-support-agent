# Job Application Assistant – RAG & LangGraph Project

## Project Overview

This project implements an intelligent Job Application Assistant using Retrieval-Augmented Generation (RAG), LangGraph, persistent memory, guardrails, FastAPI, MCP, SQLite checkpointing, and retry/timeout mechanisms.

The system uses a synthetic dataset of 50 job applications and a knowledge base containing 12 documents. Users can ask questions about job roles, salary ranges, application statuses, and priority review policies. They can also check individual application records using their record IDs.

The project is divided into four parts and contains 16 tasks, carrying a total of 100 marks.

---

# 1. Project Structure

Suggested file organization:

```text
job_application_assistant/
│
├── dataset.py
├── knowledge_base.json
│
├── chunking.py
├── evaluation.py
│
├── job_status_tool.py
├── langagent.py
├── memory.py
├── structured_output.py
├── guardrails.py
│
├── api.py
├── logging_middleware.py
├── rag_triad.py
│
├── mcp_server.py
├── mcp_client.py
│
├── checkpointing.py
├── resilience.py
│
├── conversation_memory.json
├── api_documents.json
├── api_requests.jsonl
├── rag_triad_results.json
├── checkpoints.sqlite
├── chroma_db/
│
└── README.md
```

*Note: The filenames are suggested. Adjust imports to match the actual filenames in your project.*

---

# 2. Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Random | Synthetic dataset generation |
| JSON | Knowledge base and persistent storage |
| NumPy | Vector similarity calculations |
| Sentence Transformers | Text embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| ChromaDB | Vector database |
| LangGraph | Agent workflow orchestration |
| TypedDict | Agent state definition |
| JSON Schema | Structured output validation |
| FastAPI | REST API deployment |
| Pydantic | API request and response validation |
| FastMCP | Model Context Protocol server and client |
| SQLite | Persistent graph checkpointing |
| asyncio | Asynchronous execution |
| re | Text matching, PII masking, and intent detection |
| uuid | Unique trace IDs |
| datetime | UTC timestamps |
| time | Execution duration measurement |
| math | Percentile calculation |
| os | File existence and file operations |

---

# PART 1: Dataset Design & RAG Core

## Task 1: Design and Validate the Dataset

**File:** `dataset.py`

Created a synthetic dataset containing 50 job application records.

### Variables

- `random`
- `seed`
- `random_generator`
- `categories`
- `statuses`
- `salary_ranges`
- `category_list`
- `status_list`
- `job_applications`
- `index`
- `category`
- `minimum_salary`
- `maximum_salary`
- `record`
- `JOB_APPLICATIONS`
- `category_counts`
- `status_counts`
- `flagged_count`
- `flagged_percentage`

### Dataset fields

| Field | Description |
|---|---|
| record_id | Unique application identifier |
| category | Job role |
| status | Current application status |
| expected_salary_inr | Expected annual salary |
| days_since_created | Application age in days |
| flagged_priority_review | Priority review indicator |

### Job categories

1. Software Engineer
2. Data Analyst
3. Product Manager
4. HR Executive
5. Sales Associate

### Application statuses

1. Applied
2. Screening
3. Interview Scheduled
4. Offered
5. Rejected

### Validation

The implementation includes assertions to verify:

- Exactly 50 records.
- At least three records per category.
- All five application statuses are represented.
- Priority review percentage is between 10% and 30%.
- Application age is between 0 and 30 days.

The random seed is set to `2` to ensure reproducible dataset generation.

---

## Task 2: Create the Knowledge Base

**File:** `knowledge_base.json`

Created a knowledge base containing 12 documents.

### Main variables

- `knowledge_base`
- `document_id`
- `title`
- `content`
- `data`
- `documents`

### Knowledge base documents

| Document ID | Document title |
|---|---|
| KB001 | Job Application Process |
| KB002 | Software Engineer Role |
| KB003 | Data Analyst Role |
| KB004 | Product Manager Role |
| KB005 | HR Executive Role |
| KB006 | Sales Associate Role |
| KB007 | Application Status: Applied |
| KB008 | Application Status: Screening |
| KB009 | Application Status: Interview Scheduled |
| KB010 | Application Status: Offered |
| KB011 | Application Status: Rejected |
| KB012 | Priority Review Policy |

Each document contains information about job responsibilities, salary ranges, application statuses, or recruitment policies.

---

## Task 3: Implement and Index Two Chunking Strategies

**File:** `chunking.py`

Implemented two different text chunking strategies using Sentence Transformers and ChromaDB.

### Libraries

- `json`
- `re`
- `chromadb`
- `SentenceTransformer`
- `numpy`

### Variables

- `file`
- `data`
- `documents`
- `model`
- `client`
- `fixed_collection`
- `sentence_collection`
- `fixed_chunks`
- `fixed_ids`
- `fixed_metadata`
- `fixed_embeddings`
- `sentence_chunks`
- `sentence_ids`
- `sentence_metadata`
- `sentence_embeddings`
- `doc`
- `words`
- `chunk_size`
- `overlap`
- `start`
- `chunk_index`
- `chunk`
- `sentences`
- `index`
- `sentence`

### Strategy 1: Fixed-size chunking

- Chunk size: 40 words.
- Overlap: 10 words.
- Chunks are generated using a sliding window.

### Strategy 2: Sentence-based chunking

- Splits documents into individual sentences.
- Uses regular expressions to identify sentence boundaries.
- Maintains document metadata.

### Embedding model

`all-MiniLM-L6-v2`

### ChromaDB collections

- `fixed_size_chunks`
- `sentence_chunks`

Both collections are stored persistently in:

`./chroma_db`

The `upsert()` method stores document chunks, embeddings, IDs, and metadata.

---

## Task 4: Implement Grounded Generation

**File:** `chunking.py`

Implemented retrieval and grounded response generation using cosine similarity.

### Variables

- `active_collection`
- `query`
- `collection`
- `top_k`
- `query_embedding`
- `stored_data`
- `similarities`
- `index`
- `embedding`
- `score`
- `item`
- `result`
- `answer`
- `in_scope_queries`
- `out_scope_queries`
- `in_scope_scores`
- `out_scope_scores`
- `min_in_scope`
- `max_out_scope`
- `threshold`
- `readme_text`
- `section_start`
- `section_end`
- `calibration_report`
- `test_queries`

### Functions

**1. `retrieve_chunks(query, collection, top_k=3)`**

- Converts the query into an embedding.
- Retrieves stored embeddings from ChromaDB.
- Calculates cosine similarity.
- Sorts results by similarity.
- Returns the top matching chunks.

**2. `grounded_answer(query, collection, top_k=3)`**

- Retrieves relevant chunks.
- Checks the similarity threshold.
- Returns supporting context.
- Uses the fallback response when retrieval confidence is insufficient.

### Calibration

Three in-scope queries and two out-of-scope queries are used.

The threshold is calculated using:

`threshold = (min_in_scope + max_out_scope) / 2`

The calibration results are saved in `README.md` between calibration markers.

### Fallback response

"I don't know based on the available documents."

---

## Task 5: Evaluate and Compare Chunking Strategies

**File:** `evaluation.py`

Evaluated fixed-size and sentence-based chunking using Precision@3 and Recall@3.

### Variables

- `model`
- `client`
- `fixed_collection`
- `sentence_collection`
- `queries`
- `query`
- `relevant_docs`
- `query_embedding`
- `data`
- `results`
- `index`
- `embedding`
- `score`
- `unique_results`
- `seen`
- `result`
- `doc_id`
- `retrieved`
- `retrieved_docs`
- `correct`
- `precision`
- `recall`
- `total_precision`
- `total_recall`
- `average_precision`
- `average_recall`
- `fixed_precision`
- `fixed_recall`
- `sentence_precision`
- `sentence_recall`

### Functions

**1. `retrieve_documents(query, collection)`**

Retrieves the top three unique parent documents.

**2. `evaluate_collection(collection, collection_name)`**

Calculates Precision@3 and Recall@3 for each test query.

### Evaluation formulas

Precision@3:

`Precision@3 = Relevant retrieved documents / 3`

Recall@3:

`Recall@3 = Relevant retrieved documents / Total relevant documents`

Both chunking strategies are compared using average precision and recall.

The strategy with the better combined score is recommended.

---

# PART 2: LangGraph Agent with Tools, Memory & Guardrails 

## Task 6: Build the Application Status Tool

**File:** `job_status_tool.py`

Implemented an application lookup tool with an escalation scoring mechanism.

### Variables

- `math`
- `JOB_APPLICATIONS`
- `scores`
- `record`
- `priority`
- `recency`
- `score`
- `sorted_scores`
- `position`
- `threshold`
- `record_id`
- `result`

### Function

`calculate_escalation_score(record)`

Calculates the escalation score using:

`score = (0.6 × priority) + (0.4 × days_since_created / 30)`

The 80th percentile is used as the escalation threshold.

### Function

`check_job_application_status(record_id)`

Returns:

- Record ID
- Application status
- Expected salary
- Escalation score
- Escalation recommendation

Returns an error when the application record does not exist.

---

## Task 7: Build the LangGraph Agent

**File:** `langagent.py`

Created a LangGraph workflow containing four nodes and conditional routing.

### Libraries

- `re`
- `TypedDict`
- `StateGraph`
- `END`

### Variables

- `AgentState`
- `query`
- `route`
- `response`
- `has_record_id`
- `match`
- `record_id`
- `result`
- `answer`
- `workflow`
- `app`
- `rag_query`
- `rag_result`
- `status_query`
- `status_result`

### Agent state

`AgentState` contains:

- `query`
- `route`
- `response`

### Four graph nodes

1. `detect_intent`
2. `rag_node`
3. `status_node`
4. `final_node`

### Conditional routing

The `detect_intent` node identifies the user's query.

- RAG questions are routed to `rag_node`.
- Application record lookup questions are routed to `status_node`.

Both nodes connect to `final_node`, which connects to `END`.

### Imported functions

- `grounded_answer`
- `active_collection`
- `check_job_application_status`

The graph is compiled using:

`app = workflow.compile()`

---

## Task 8: Add Persisted Memory

**File:** `memory.py`

Implemented conversation memory using JSON file storage.

### Variables

- `json`
- `os`
- `memory_file`
- `conversation_id`
- `memory`
- `history`
- `user_message`
- `message`
- `item`
- `previous_message`
- `response`
- `query1`
- `query2`
- `query3`
- `response1`
- `response2`
- `response3`
- `fresh_conversation_id`

### Functions

**1. `load_memory()`**

Loads previously saved conversations.

**2. `save_memory(memory)`**

Saves conversation history into JSON.

**3. `clear_conversation(conversation_id)`**

Removes one conversation from stored memory.

**4. `memory_chat(conversation_id, user_message)`**

Maintains conversation history and retrieves previously stored preferred-role information.

### Memory file

`conversation_memory.json`

The implementation demonstrates:

- First conversation.
- Second turn in the same conversation.
- Fresh conversation without previous information.
- Persistent JSON conversation history.

---

## Task 9: Add Structured Output Schema

**File:** `structured_output.py`

Implemented JSON Schema validation for agent responses.

### Variables

- `json`
- `validate`
- `ValidationError`
- `response_schema`
- `agent_response`
- `error`
- `result`
- `conversation_id`
- `user_message`
- `response`
- `memory_used`
- `status`
- `agent_result`
- `invalid_result`

### Response fields

| Field | Data type |
|---|---|
| conversation_id | String |
| user_message | String |
| response | String |
| memory_used | Boolean |
| status | String |

### Allowed status values

- `success`
- `no_memory`
- `error`

Additional properties are disabled.

### Functions

**1. `validate_response(agent_response)`**

Validates an agent response against the JSON Schema.

**2. `structured_response(conversation_id, user_message, response, memory_used, status)`**

Creates and validates a structured response.

The implementation also tests an invalid response where `memory_used` is provided as a string instead of a Boolean.

---

## Task 10: Add Guardrails

**File:** `guardrails.py`

Implemented input and output guardrails to improve application safety.

### Libraries

- `re`
- `retrieve_chunks`
- `active_collection`
- `threshold`

### Variables

- `text`
- `suspicious_patterns`
- `pattern`
- `query`
- `safe_query`
- `results`
- `context`
- `stop_words`
- `query_words`
- `matching_words`
- `word`
- `overlap`
- `answer`
- `result`
- `pii_query`
- `injection_query`
- `unsupported_query`
- `normal_query`
- `response`

### Functions

**1. `mask_pii(text)`**

Masks email addresses and mobile phone numbers.

**2. `detect_prompt_injection(text)`**

Detects suspicious instructions, including:

- Ignore previous instructions.
- Reveal hidden instructions.
- Act as an unrestricted system.
- Bypass guardrails.

**3. `grounded_answer_with_guardrail(query)`**

Checks retrieved context, similarity scores, and lexical overlap before returning supporting information.

**4. `process_query(query)`**

Combines input masking, prompt injection detection, retrieval, and groundedness checking.

### Guardrail tests

1. PII masking.
2. Prompt injection detection.
3. Unsupported question rejection.
4. Normal supported question.

---

# PART 3: Evaluation, Observability & FastAPI Deployment

## Task 11: Deploy Using FastAPI

**File:** `api.py`

Created a FastAPI application exposing REST endpoints.

### Libraries

- `json`
- `os`
- `FastAPI`
- `HTTPException`
- `BaseModel`
- `agent_app`

### Variables and classes

- `api`
- `AskRequest`
- `AskResponse`
- `DocumentRequest`
- `DocumentResponse`
- `document_file`
- `documents`
- `title`
- `content`
- `request`
- `result`
- `error`

### Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | API health check |
| `/ask` | POST | Submit questions to the agent |
| `/add-document` | POST | Save additional documents |

### Function

`save_document(title, content)`

Saves additional documents in `api_documents.json`.

### FastAPI response models

- `AskResponse`
- `DocumentResponse`

The `/ask` endpoint invokes the compiled LangGraph agent and returns the query, response, and selected route.

---

## Task 12: Add Structured Logging

**File:** `logging_middleware.py`

Implemented JSON Lines request logging middleware.

### Libraries

- `json`
- `re`
- `time`
- `uuid`
- `datetime`
- `timezone`

### Variables

- `text`
- `value`
- `key`
- `item`
- `app`
- `log_file`
- `scope`
- `receive`
- `send`
- `trace_id`
- `start_time`
- `message`
- `body_parts`
- `original_body`
- `request_data`
- `safe_data`
- `safe_request_text`
- `body_sent`
- `status_code`
- `duration_ms`
- `log_entry`
- `file`

### Functions

**1. `mask_pii(text)`**

Masks sensitive email and phone information.

**2. `mask_json_values(value)`**

Recursively masks string values inside JSON objects and lists.

**3. `register_jsonl_logging(api)`**

Registers the custom logging middleware with FastAPI.

### Class

`JsonlLoggingMiddleware`

The middleware records:

- UTC timestamp.
- Unique trace ID.
- HTTP method.
- Request path.
- HTTP status code.
- Request processing duration.
- Masked request content.

### Log file

`api_requests.jsonl`

Each request is stored as one JSON object per line.

---

## Task 13: Evaluate Using the RAG Triad at Scale

**File:** `rag_triad.py`

Implemented a mock evaluation system for three RAG quality metrics.

### Variables

- `MOCK_LLM`
- `stop_words`
- `text`
- `word`
- `query`
- `context`
- `answer`
- `prompt`
- `query_words`
- `context_words`
- `answer_words`
- `context_overlap`
- `answer_overlap`
- `refusal`
- `answer_context_overlap`
- `groundedness`
- `answer_relevance`
- `context_relevance`
- `test_queries`
- `results`
- `number`
- `retrieved`
- `item`
- `scores`
- `result`
- `total`
- `averages`
- `report`

### Functions

**1. `get_words(text)`**

Extracts meaningful words while excluding stop words.

**2. `mock_llm_judge(query, context, answer)`**

Produces three evaluation scores using reproducible lexical-overlap calculations.

**3. `run_evaluation()`**

Runs evaluation against 15 test queries, calculates average scores, and saves the results.

### RAG triad metrics

| Metric | Description |
|---|---|
| Context Relevance | Measures whether retrieved context relates to the question |
| Groundedness | Measures whether the answer is supported by context |
| Answer Relevance | Measures whether the answer addresses the question |

### Evaluation report

`rag_triad_results.json`

The report includes:

- Mock LLM setting.
- Total number of queries.
- Individual evaluation results.
- Average metric scores.

**Note:** This implementation uses a mock lexical evaluator, not an actual LLM judge.

---

# PART 4: Resilience & Interoperability

## Task 14: Expose the Lookup Tool Using MCP

**Files:** `mcp_server.py` and `mcp_client.py`

Implemented an MCP server and client using FastMCP.

### Server libraries

- `FastMCP`
- `check_job_application_status`

### Server variables

- `mcp`
- `record_id`
- `result`

### Server function

`get_application_status(record_id)`

Exposes the application lookup function as an MCP tool.

### MCP server configuration

- Host: `127.0.0.1`
- Port: `8001`
- Transport: HTTP
- Endpoint: `http://127.0.0.1:8001/mcp`

### Client libraries

- `asyncio`
- `json`
- `Client`

### Client variables

- `MCP_URL`
- `client`
- `tools`
- `tool`
- `record_ids`
- `record_id`
- `result`

### Client function

`main()`

The client:

1. Connects to the MCP server.
2. Lists available tools.
3. Calls `get_application_status`.
4. Tests application records 5 and 10.
5. Prints the returned results.

---

## Task 15: Add SQLite-Based Checkpointing

**File:** `checkpointing.py`

Implemented persistent graph checkpointing and human-in-the-loop interruption using SQLite.

### Libraries

- `TypedDict`
- `StateGraph`
- `END`
- `interrupt`
- `Command`
- `SqliteSaver`

### Variables

- `execution_count`
- `AgentState`
- `state`
- `answer`
- `workflow`
- `checkpointer`
- `app`
- `config`
- `result`
- `saved_state`

### Agent state fields

- `message`
- `first_result`
- `second_result`
- `final_result`

### Nodes

1. `first_node`
2. `second_node`
3. `pause_node`
4. `final_node`

### Functions

- `first_node(state)`
- `second_node(state)`
- `pause_node(state)`
- `final_node(state)`

### Checkpoint database

`checkpoints.sqlite`

### Workflow

The execution sequence is:

`first_node → second_node → pause_node → final_node`

The `interrupt()` function pauses execution.

The `Command(resume="continue")` operation resumes the same thread using the stored checkpoint.

The `execution_count` dictionary tracks node executions before and after resuming.

---

## Task 16: Add Timeouts and Retries

**File:** `resilience.py`

Implemented retry policies and timeout handling using LangGraph and asyncio.

### Libraries

- `asyncio`
- `TypedDict`
- `StateGraph`
- `END`
- `RetryPolicy`
- `TimeoutPolicy`

### Variables

- `AgentState`
- `message`
- `result`
- `attempt_count`
- `state`
- `retry_workflow`
- `retry_app`
- `node_timeout_workflow`
- `node_timeout_app`
- `global_timeout_workflow`
- `global_timeout_app`
- `error`

### Functions

**1. `unreliable_node(state)`**

Simulates temporary failures and succeeds after retries.

**2. `slow_node(state)`**

Demonstrates per-node timeout handling.

**3. `long_running_node(state)`**

Demonstrates a long-running operation.

**4. `main()`**

Runs all three resilience demonstrations.

### Demonstration 1: Retry policy

- Maximum attempts: 4.
- Initial interval: 0.1 seconds.
- Backoff factor: 2.
- Maximum interval: 0.5 seconds.
- Jitter enabled.
- Retry exception: `ValueError`.

The node deliberately fails twice before succeeding.

### Demonstration 2: Per-node timeout

- Node: `slow_node`
- Timeout: 0.3 seconds.
- Simulated operation duration: 2 seconds.

### Demonstration 3: Global graph timeout

Uses `asyncio.wait_for()` to cancel a graph execution after 0.5 seconds.

The graph node has its own timeout configuration of 5 seconds.

---

# 3. Execution Instructions

## Step 1: Install dependencies

Run:

```bash
pip install numpy chromadb sentence-transformers
pip install langgraph langgraph-checkpoint-sqlite
pip install fastapi uvicorn pydantic
pip install fastmcp jsonschema
```

## Step 2: Prepare the knowledge base

Create `knowledge_base.json` and insert the 12 knowledge base documents.

## Step 3: Generate the dataset

```bash
python dataset.py
```

## Step 4: Build the vector database

```bash
python chunking.py
```

Ensure the embedding model is available locally if `local_files_only=True` is enabled.

## Step 5: Run chunking evaluation

```bash
python evaluation.py
```

## Step 6: Test application status

```bash
python job_status_tool.py
```

## Step 7: Run the LangGraph agent

```bash
python langagent.py
```

## Step 8: Test memory and structured output

```bash
python memory.py
python structured_output.py
```

## Step 9: Test guardrails

```bash
python guardrails.py
```

## Step 10: Start FastAPI

```bash
uvicorn api:api --reload
```

Open the API documentation:

`http://127.0.0.1:8000/docs`

## Step 11: Run MCP server

Open a separate terminal:

```bash
python mcp_server.py
```

## Step 12: Run MCP client

Open another terminal:

```bash
python mcp_client.py
```

## Step 13: Test checkpointing

```bash
python checkpointing.py
```

## Step 14: Test resilience

```bash
python resilience.py
```

---

# 4. Important Implementation Improvements

The following improvements are recommended before final submission.

| Task | Improvement required |
|---|---|
| Task 3 | Consider deleting stale ChromaDB records before re-indexing changed documents. |
| Task 4 | Improve calibration with a larger validation set and filter retrieved chunks individually. |
| Task 5 | Correct the ground-truth mapping for the application-age query. KB012 describes priority review, not application age. |
| Task 6 | Document that percentile ties may result in more than 20% of applications meeting the threshold. |
| Task 7 | Verify the record ID regex is `r"(record|application)\s*id\s*#?\s*(\d+)"`. |
| Task 8 | Improve preferred-role extraction and handle multiple user preferences. |
| Task 9 | Use the `no_memory` status when no previous information is available. |
| Task 10 | Verify the PII regex patterns in the actual Python files. |
| Task 11 | Connect `/add-document` to the indexing process if newly added documents must become searchable. |
| Task 12 | Call `register_jsonl_logging(api)` to activate the middleware. |
| Task 13 | Import all required dependencies independently and clearly label the evaluator as a mock evaluator. |
| Task 14 | Test the MCP server and client in separate terminals. |
| Task 15 | Verify checkpoint persistence across application restarts. |
| Task 16 | Verify the installed LangGraph version supports the specified retry and timeout policies. |

### Important corrected PII regex examples

Email:

```python
r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
```

Indian mobile number:

```python
r"(?<!\d)(?:\+91[- ]?)?[6-9]\d{9}(?!\d)"
```

### Important logging registration

Add the following after creating the FastAPI application:

```python
from logging_middleware import register_jsonl_logging

register_jsonl_logging(api)
```

### Important application-age knowledge gap

The question:

`How many days can an application be old?`

Currently has no dedicated knowledge base document.

Consider adding a document explaining that the synthetic dataset represents application ages from 0 to 30 days.

---

# 5. Marks Evaluation

The following is my estimated assessment based on the submitted implementation.

| Part | Tasks | Maximum marks | Estimated marks |
|---|---|---:|---:|
| Part 1: Dataset Design & RAG Core | 1–5 | 30 | 24 |
| Part 2: LangGraph, Memory & Guardrails | 6–10 | 30 | 24 |
| Part 3: Evaluation & FastAPI | 11–13 | 20 | 14 |
| Part 4: Resilience & Interoperability | 14–16 | 20 | 18 |
| **Total** | **1–16** | **100** | **80** |

## Detailed task-wise marks

| Task | Description | Maximum | Awarded |
|---|---|---:|---:|
| 1 | Dataset design and validation | 6 | 6 |
| 2 | Knowledge base creation | 6 | 6 |
| 3 | Two chunking strategies | 6 | 5 |
| 4 | Grounded generation and calibration | 6 | 3 |
| 5 | Chunking evaluation | 6 | 4 |
| 6 | Escalation scoring tool | 6 | 5.5 |
| 7 | LangGraph agent | 6 | 5 |
| 8 | Persisted memory | 6 | 5 |
| 9 | Structured output | 6 | 5.5 |
| 10 | Guardrails | 6 | 3 |
| 11 | FastAPI deployment | 6.67 | 5 |
| 12 | Structured logging | 6.67 | 4 |
| 13 | RAG triad evaluation | 6.66 | 5 |
| 14 | MCP server and client | 6.67 | 6 |
| 15 | SQLite checkpointing | 6.67 | 6 |
| 16 | Timeouts and retries | 6.66 | 6 |
| **TOTAL** | | **100** | **80** |

<small>Marks are approximate and may change depending on execution results, the instructor's rubric, and the completeness of the submitted files.</small>

---

# 6. Final Project Assessment

**Estimated score: 80/100**

### Strong areas

- Reproducible synthetic dataset.
- Two different chunking approaches.
- Persistent vector storage with ChromaDB.
- Conditional routing using LangGraph.
- Application status lookup with escalation scoring.
- JSON-based conversation memory.
- Structured output validation.
- MCP interoperability.
- SQLite checkpointing.
- Retry and timeout demonstrations.

### Areas needing improvement

- Grounded generation calibration and evaluation.
- PII regex verification.
- Integration of newly added API documents into RAG.
- Activation of structured logging middleware.
- Stronger RAG triad evaluation.
- Independent imports and integration testing across files.

## Conclusion

The Job Application Assistant demonstrates the implementation of a complete AI application architecture, from synthetic data generation and RAG retrieval to agent orchestration, API deployment, interoperability, and resilience.

All 16 tasks have corresponding implementation code in the submitted work. The estimated score is **80 out of 100**, with the main opportunities for improvement being integration, groundedness validation, and runtime testing.

**Project status:** All 16 task implementations are represented in the submitted code. Runtime verification and the recommended corrections are still required before final submission.<!-- calibration start -->

## grounded generation calibration

Embedding model: all-MiniLM-L6-v2

Collection: sentence_chunks

In-scope similarity scores: [0.8071, 0.7484, 0.7865, 0.7817, 0.7381, 0.7277, 0.8072, 0.6221, 0.6662, 0.5597]

Out-of-scope similarity scores: [0.1715, 0.0787, 0.1177, 0.1321, 0.1173]

Minimum in-scope similarity: 0.5597

Maximum out-of-scope similarity: 0.1715

Selected empirical threshold: 0.3656

Threshold formula: (minimum in-scope + maximum out-of-scope) / 2

Fallback: I don't know based on the available documents.

<!-- calibration end -->