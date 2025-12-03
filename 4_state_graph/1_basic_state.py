from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
import os

os.environ["LANGCHAIN_TRACING_V2"] = "false"

class SimpleState(TypedDict):
    counter: int

def increment(state: SimpleState) -> SimpleState:
    return {"counter": state["counter"] + 1}

def should_continue(state: SimpleState) -> Literal["continue", "end"]:
    if state["counter"] > 5:
        return "end"
    return "continue"

graph = StateGraph(SimpleState)

graph.add_node("increment", increment)

graph.add_edge(START, "increment")
graph.add_conditional_edges("increment", should_continue, {"continue": "increment", "end": END})

workflow = graph.compile()

response = workflow.invoke({"counter": 0})

print(response)