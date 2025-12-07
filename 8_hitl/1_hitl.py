from langchain_groq import ChatGroq
from langchain.messages import SystemMessage, HumanMessage, AnyMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, List, Annotated
from dotenv import load_dotenv
import sqlite3
import os

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "false"

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

search_tool = TavilySearch(
    max_results=3,
    search_depth="basic",
    include_images=False
)

tools = [search_tool]

# llm_with_tools = llm.bind_tools(tools)

class BasicChatState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

GENERATE_POST = "generate_post"
GET_REVIEW_DECISION = "get_review_decision"
POST = "post"
COLLECT_FEEDBACK = "collect_feedback"

def generate_post_node(state: BasicChatState):
    return {"messages": [llm.invoke(state["messages"])]}

tool_node = ToolNode(tools)

def get_review_decision_node(state: BasicChatState):
    post_content = state["messages"][-1].content

    print("\n\n++++ POST CONTENT ++++\n\n")
    print(post_content)

    decision = input("Post approved? (y/n): ")

    if decision.lower() in ['y', 'yes']:
        return "post"
    else:
        return "collect_feedback"
    
def post_node(state: BasicChatState):
    final_post = state["messages"][-1].content

    print("\n\n++++ FINAL POST CONTENT ++++\n\n")
    print(final_post)

    print("\n --- Post was approved and is Live on LinkedIn --- \n")

def collect_feedback_node(state: BasicChatState):
    feedback = input("Please provide feedback: ")
    return {"messages": [HumanMessage(content=feedback)]}

graph = StateGraph(BasicChatState)

graph.add_node(GENERATE_POST, generate_post_node)
graph.add_node(GET_REVIEW_DECISION, get_review_decision_node)
graph.add_node(POST, post_node)
graph.add_node(COLLECT_FEEDBACK, collect_feedback_node)

graph.add_edge(START, GENERATE_POST)
graph.add_conditional_edges(GENERATE_POST, get_review_decision_node, {"post": POST, "collect_feedback": COLLECT_FEEDBACK})
graph.add_edge(POST, END)
graph.add_edge(COLLECT_FEEDBACK, GENERATE_POST)

# while compiling add the memory
workflow = graph.compile()

initial_state = {
    "messages": [HumanMessage("Write a LinkedIn Post on AI Agents taking over content creation.")]
}

response = workflow.invoke(initial_state)



