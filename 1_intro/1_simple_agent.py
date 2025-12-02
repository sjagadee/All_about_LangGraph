from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
# TO MAKE THIS A REACT AGENT:
# 1. Import: from langgraph.prebuilt import create_react_agent
# 2. Replace create_agent with create_react_agent
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

# Initialize the LLM (Language Model)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

# print(llm.invoke("tell how is the weather today in dallas?").content)

# Create a search tool - this allows the agent to search the web
search_tool = TavilySearch()

# List of tools available to the agent
tools = [search_tool]

# CURRENT: Using basic agent
# TO CONVERT TO REACT: Change this to create_react_agent(llm, tools)
# ReAct agents explicitly show their reasoning (Thought) before taking actions (Action)
agent = create_agent(llm, tools)

# The system prompt will be set dynamically based on context
# Invoke the agent with a user message
result = agent.invoke({"messages": [{"role": "user", "content": "tell how is the weather today in dallas?"}]})

# REACT BENEFIT: With ReAct, you'll see:
# - Thought: "I need to search for current weather in Dallas"
# - Action: Call search_tool with query
# - Observation: Search results
# - Thought: "Based on results, I can answer"
# - Final Answer: Weather information
print(result)

