# Part 4 : Resilience & Interoperability
# Task 15 : Add SQLite-based checkpointing.
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver


# track node execution

execution_count = {
    "first_node": 0,
    "second_node": 0,
    "pause_node": 0,
    "final_node": 0
}


# define graph state

class AgentState(TypedDict):
    message: str
    first_result: str
    second_result: str
    final_result: str


# node 1

def first_node(state: AgentState):

    execution_count["first_node"] += 1

    print("first node executed")

    return {
        "first_result": "first node completed"
    }


# node 2

def second_node(state: AgentState):

    execution_count["second_node"] += 1

    print("second node executed")

    return {
        "second_result": "second node completed"
    }


# node 3

def pause_node(state: AgentState):

    execution_count["pause_node"] += 1

    print("pause node reached")

    answer = interrupt(
        "Execution paused. Send continue to resume."
    )

    print("resuming pause node:", answer)

    return {}


# node 4

def final_node(state: AgentState):

    execution_count["final_node"] += 1

    print("final node executed")

    return {
        "final_result": "workflow completed successfully"
    }


# create workflow

workflow = StateGraph(AgentState)

workflow.add_node("first_node", first_node)
workflow.add_node("second_node", second_node)
workflow.add_node("pause_node", pause_node)
workflow.add_node("final_node", final_node)

workflow.set_entry_point("first_node")

workflow.add_edge("first_node", "second_node")
workflow.add_edge("second_node", "pause_node")
workflow.add_edge("pause_node", "final_node")
workflow.add_edge("final_node", END)


# use sqlite checkpointing

with SqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:

    app = workflow.compile(checkpointer=checkpointer)

    # same thread id for both runs

    config = {
        "configurable": {
            "thread_id": "job_application_thread_1"
        }
    }

    # first run

    print("\nfirst run")

    result = app.invoke(
        {
            "message": "check application",
            "first_result": "",
            "second_result": "",
            "final_result": ""
        },
        config
    )

    print("\nexecution paused")

    # show saved checkpoint state

    saved_state = app.get_state(config)

    print("\nsaved checkpoint values:")

    print(saved_state.values)

    print("\nnext node:", saved_state.next)

    print("\nexecution counts before resume:")

    print(execution_count)

    # resume the same thread

    print("\nresuming same thread")

    result = app.invoke(
        Command(resume="continue"),
        config
    )

    print("\nfinal result:")

    print(result)

    print("\nexecution counts after resume:")

    print(execution_count)