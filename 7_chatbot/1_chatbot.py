from langchain_groq import ChatGroq
from langchain.messages import SystemMessage, HumanMessage, AnyMessage
from langgraph.graph import StateGraph, START, END, add_messages
from typing import TypedDict, List, Annotated
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "false"

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

class BasicChatState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

def chatbot_node(state: BasicChatState):
    return {"messages": [llm.invoke(state["messages"])]}

graph = StateGraph(BasicChatState)

graph.add_node("generate", chatbot_node)

graph.add_edge(START, "generate")
graph.add_edge("generate", END)

workflow = graph.compile()

while True:

    user_input = input("Human: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    response = workflow.invoke({"messages": [HumanMessage(content=user_input)]})
    for message in response['messages']:
        message.pretty_print()
