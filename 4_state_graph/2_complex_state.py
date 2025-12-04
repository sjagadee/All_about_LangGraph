from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
import os

os.environ["LANGCHAIN_TRACING_V2"] = "false"

class ComplexState(TypedDict):
    counter: int
    sum: int
    history: list[int]

def increment(state: ComplexState) -> ComplexState:
    cum_sum = state.get("sum", 0) + state["counter"]
    return {
        "counter": state["counter"] + 1, 
        "sum": cum_sum, 
        "history": state.get("history", []) + [cum_sum]
    }

def should_continue(state: ComplexState) -> Literal["continue", "end"]:
    if state["counter"] <= 5:
        return "continue"
    return "end"

graph = StateGraph(ComplexState)

graph.add_node("increment", increment)

graph.add_edge(START, "increment")
graph.add_conditional_edges("increment", should_continue, {"continue": "increment", "end": END})

workflow = graph.compile()

response = workflow.invoke({"counter": 0})

print(response)