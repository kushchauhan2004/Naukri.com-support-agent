# Part 2 : LangGraph Agent with Tools, Memory & Guardrails
# Task 7 : Build the LangGraph agent.
import re
from typing import TypedDict
from langgraph.graph import StateGraph, END
from chunking import grounded_answer, active_collection
from job_status_tool import check_job_application_status
# define agent state

class AgentState(TypedDict):
    query: str
    route: str
    response: str


# detect query intent

def detect_intent(state: AgentState):

    query = state["query"].lower()

    has_record_id = re.search(
        r"(record|application)\s*id\s*#?\s*\d+",
        query
    )

    if has_record_id:
        return {"route": "status"}

    return {"route": "rag"}


# rag node

def rag_node(state: AgentState):

    answer = grounded_answer(
        state["query"],
        active_collection,
        top_k=3
    )

    return {"response": answer}


# application status node

def status_node(state: AgentState):

    match = re.search(
        r"(record|application)\s*id\s*#?\s*(\d+)",
        state["query"].lower()
    )

    if not match:
        return {
            "response": "Please provide a valid application record id."
        }

    record_id = match.group(2)

    result = check_job_application_status(record_id)

    return {"response": str(result)}


# final response node

def final_node(state: AgentState):

    return {"response": state["response"]}


# create workflow

workflow = StateGraph(AgentState)

workflow.add_node("detect_intent", detect_intent)
workflow.add_node("rag_node", rag_node)
workflow.add_node("status_node", status_node)
workflow.add_node("final_node", final_node)


# set starting node

workflow.set_entry_point("detect_intent")


# add conditional routing

workflow.add_conditional_edges(
    "detect_intent",
    lambda state: state["route"],
    {
        "rag": "rag_node",
        "status": "status_node"
    }
)


# connect remaining nodes

workflow.add_edge("rag_node", "final_node")
workflow.add_edge("status_node", "final_node")
workflow.add_edge("final_node", END)


# compile graph

app = workflow.compile()

print("langgraph agent created successfully")


# test rag route

rag_query = "What is the salary range for a Data Analyst?"

rag_result = app.invoke({
    "query": rag_query,
    "route": "",
    "response": ""
})

print("\nrag route")
print("query:", rag_query)
print("selected route:", rag_result["route"])
print("response:", rag_result["response"])


# test application status route

status_query = "Check application status for record id 5"

status_result = app.invoke({
    "query": status_query,
    "route": "",
    "response": ""
})

print("\nstatus route")
print("query:", status_query)
print("selected route:", status_result["route"])
print("response:", status_result["response"])


# display graph

print("\nworkflow diagram")
print(app.get_graph().draw_mermaid())