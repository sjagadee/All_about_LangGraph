from langchain_groq import ChatGroq
from langchain.messages import SystemMessage, HumanMessage, AnyMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, List, Annotated
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "false"

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)


search_tool = TavilySearch()

tools = [search_tool]

llm_with_tools = llm.bind_tools(tools)

class BasicChatState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

def chatbot_node(state: BasicChatState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

tool_node = ToolNode(tools)

graph = StateGraph(BasicChatState)

graph.add_node("generate", chatbot_node)

graph.add_node("tools", tool_node)


graph.add_edge(START, "generate")
graph.add_conditional_edges("generate", tools_condition)
graph.add_edge("tools", "generate")
graph.add_edge("generate", END)

workflow = graph.compile()

while True:

    user_input = input("Human: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    response = workflow.invoke({"messages": [HumanMessage(content=user_input)]})
    for message in response['messages']:
        message.pretty_print()
