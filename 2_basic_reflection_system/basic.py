from langgraph.graph import StateGraph, MessagesState, END, START
from langchain_core.messages import BaseMessage, HumanMessage

from typing import List, Sequence
from dotenv import load_dotenv
import os

load_dotenv()

from chains import generation_chain, reflection_chain
os.environ["LANGCHAIN_PROJECT"] = "Reflection Agentic Pattern"

graph = StateGraph(MessagesState)

GENERATE = "generate"
REFLECT = "reflect"

def generate_node(state):
    # print("###### generate_node state ######", state)
    response = generation_chain.invoke(state)
    return {"messages": [response]}

def reflect_node(state):
    response = reflection_chain.invoke(state)
    # Ensure content is not empty to prevent API errors
    content = response.content if response.content else "Please provide feedback on the generated post."
    return {"messages": [HumanMessage(content=content)]}

def should_continue(state):
    if len(state['messages']) > 4:
        return END
    return REFLECT

graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)

graph.add_edge(START, GENERATE)
graph.add_conditional_edges(GENERATE, should_continue, {REFLECT: REFLECT, END: END})
graph.add_edge(REFLECT, GENERATE)

workflow = graph.compile()

# print(workflow.get_graph().draw_mermaid())
# workflow.get_graph().print_ascii()

response = workflow.invoke({"messages": [HumanMessage(content="AI Agents taking over content creation")]})

for message in response['messages']:
    message.pretty_print()