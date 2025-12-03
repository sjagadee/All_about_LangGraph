from typing import List
from langgraph.graph import StateGraph, MessagesState, END, START
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage

from chains import revise_responser_chain, first_responser_chain
from execute_tools import execute_tools

REVISE = "revise"
DRAFT = "draft"
EXECUTE_TOOL = "execute_tool"
MAX_ITERATIONS = 2

graph = StateGraph(MessagesState)

def revise_node(state):
    response = revise_responser_chain.invoke(state)
    return {"messages": [response]}

def respond_node(state):
    response = first_responser_chain.invoke(state)
    return {"messages": [response]}

def execute_tool_node(state):
    tool_messages = execute_tools(state['messages'])
    return {"messages": tool_messages}

def should_continue(state) -> str:
    count_tool_visits = sum(isinstance(message, ToolMessage) for message in state['messages'])
    if count_tool_visits > MAX_ITERATIONS:
        return "end"
    return "execute_tool"

graph.add_node(REVISE, revise_node)
graph.add_node(DRAFT, respond_node)
graph.add_node(EXECUTE_TOOL, execute_tool_node)

graph.add_edge(START, DRAFT)
graph.add_edge(DRAFT, EXECUTE_TOOL)
graph.add_edge(EXECUTE_TOOL, REVISE)
graph.add_conditional_edges(REVISE, should_continue, {"execute_tool": EXECUTE_TOOL, "end": END})

workflow = graph.compile()

print(workflow.get_graph().draw_mermaid())
workflow.get_graph().print_ascii()

response = workflow.invoke({"messages": [HumanMessage(content="How can small businesses leverage AI to grow their business?")]})

for message in response['messages']:
    message.pretty_print()