from typing import TypedDict, Literal, Annotated
from langgraph.graph import StateGraph, START, END
import os, operator

os.environ["LANGCHAIN_TRACING_V2"] = "false"

class ComplexState(TypedDict):
    counter: int
    sum: Annotated[int, operator.add]
    history: Annotated[list[int], operator.concat]

def increment(state: ComplexState) -> ComplexState:
    new_counter = state["counter"] + 1
    return {
        "counter": new_counter, 
        "sum": new_counter, 
        "history": [new_counter]
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