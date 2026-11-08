# Part 4 : Resilience & Interoperability
# Task 16 : Add timeouts and retries.
import asyncio
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.types import RetryPolicy, TimeoutPolicy


# define state

class AgentState(TypedDict):
    message: str
    result: str


# demonstration 1: retry after transient failures

attempt_count = 0


async def unreliable_node(state: AgentState):

    global attempt_count

    attempt_count += 1

    print("attempt:", attempt_count)

    # deliberately fail first two attempts

    if attempt_count <= 2:

        print("temporary error occurred")

        raise ValueError("temporary service unavailable")

    print("operation successful")

    return {
        "result": "operation completed after retries"
    }


retry_workflow = StateGraph(AgentState)

retry_workflow.add_node(
    "unreliable_node",
    unreliable_node,
    retry_policy=RetryPolicy(
    max_attempts=4,
    initial_interval=0.1,
    backoff_factor=2,
    max_interval=0.5,
    jitter=True,
    retry_on=(ValueError,)
))

retry_workflow.set_entry_point("unreliable_node")
retry_workflow.add_edge("unreliable_node", END)

retry_app = retry_workflow.compile()


# demonstration 2: per-node timeout

async def slow_node(state: AgentState):

    print("slow node started")

    await asyncio.sleep(2)

    return {
        "result": "slow operation completed"
    }


node_timeout_workflow = StateGraph(AgentState)

node_timeout_workflow.add_node(
    "slow_node",
    slow_node,
    timeout=TimeoutPolicy(run_timeout=0.3)
)

node_timeout_workflow.set_entry_point("slow_node")
node_timeout_workflow.add_edge("slow_node", END)

node_timeout_app = node_timeout_workflow.compile()


# demonstration 3: global graph timeout

async def long_running_node(state: AgentState):

    print("long running node started")

    await asyncio.sleep(3)

    return {
        "result": "long operation completed"
    }


global_timeout_workflow = StateGraph(AgentState)

global_timeout_workflow.add_node(
    "long_running_node",
    long_running_node,
    timeout=TimeoutPolicy(run_timeout=5)
)

global_timeout_workflow.set_entry_point("long_running_node")
global_timeout_workflow.add_edge("long_running_node", END)

global_timeout_app = global_timeout_workflow.compile()


# run all demonstrations

async def main():

    global attempt_count

    print("\nTEST 1: RETRY POLICY")

    attempt_count = 0

    try:

        result = await retry_app.ainvoke({
            "message": "test retry",
            "result": ""
        })

        print("final result:", result["result"])
        print("total attempts:", attempt_count)

    except Exception as error:

        print("retry test failed:", error)


    print("\nTEST 2: PER-NODE TIMEOUT")

    try:

        await node_timeout_app.ainvoke({
            "message": "test node timeout",
            "result": ""
        })
    except Exception as error:
        print("per-node timeout fired successfully")
        print("error type:", type(error).__name__)
        print("error:", str(error))


    print("\nTEST 3: GLOBAL GRAPH TIMEOUT")

    try:

        await asyncio.wait_for(
            global_timeout_app.ainvoke({
                "message": "test global timeout",
                "result": ""
            }),
            timeout=0.5
        )
    except (asyncio.TimeoutError, TimeoutError) as error:
        print("global graph timeout fired successfully")
        print("graph execution was cancelled")
        print("error:", type(error).__name__)


if __name__ == "__main__":

    asyncio.run(main())
