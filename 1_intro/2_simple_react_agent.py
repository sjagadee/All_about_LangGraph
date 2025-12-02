from langchain_google_genai import ChatGoogleGenerativeAI
# from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import datetime

load_dotenv()

# Initialize the LLM (Language Model)
# Use a Gemini model that supports function/tool calling
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",  # This model has better tool calling support
    temperature=0.7
)

# Create a search tool - this allows the agent to search the web
# max_results: number of search results to return
# include_answer: whether to include a generated answer in the response
search_tool = TavilySearch(
    max_results=3,
    include_answer=True,
    include_raw_content=False
)

@tool
def get_system_time():
    """This function returns the current system time in the format "YYYY-MM-DD HH:MM:SS" """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# List of tools available to the agent
tools = [search_tool, get_system_time]

# Create a ReAct agent using LangGraph's prebuilt function
# This agent follows the classic ReAct pattern with messages showing:
# - AI reasoning (thinking about what to do)
# - Tool calls (actions taken)
# - Tool messages (observations from tools)
# - Final AI response

# Use a system prompt that encourages the ReAct pattern
# The agent will naturally show its reasoning through the message flow
system_prompt = """You are a helpful AI assistant that uses the ReAct pattern.

Before using any tool, briefly explain why you need it.
When you need any other information, check the tools first.
After getting results, explain what you learned before providing your final answer."""

agent = create_agent(
    llm,
    tools,
    system_prompt=system_prompt
)

# Invoke the agent with a user question
# LangGraph's ReAct agent will show its reasoning process through messages:
# - The agent thinks about what to do
# - Calls the appropriate tool (Action)
# - Receives results (Observation)
# - Repeats until it can answer
# - Provides the final answer

print("\n" + "="*60)
print("REACT AGENT IN ACTION")
print("="*60 + "\n")

# Stream the agent's execution to see the ReAct pattern in real-time
# response = agent.stream(
#     {"messages": [{"role": "user", "content": "tell how is the weather today in dallas?"}]},
#     stream_mode="values"
# )

response = agent.stream(
    {"messages": [{"role": "user", "content": "When did spacex launch happen? and how many days has been since the launch?"}]},
    stream_mode="values"
)
step = 1
for chunk in response:
    # Print each step of the agent's reasoning with ReAct labels
    if "messages" in chunk:
        last_message = chunk["messages"][-1]

        # Label each type of message according to ReAct pattern
        if last_message.__class__.__name__ == "HumanMessage":
            print(f"\n📝 QUESTION:")
            print(f"   {last_message.content}")

        elif last_message.__class__.__name__ == "AIMessage":
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                # This is the ACTION step
                print(f"\n💡 THOUGHT {step}:")
                if last_message.content:
                    print(f"   {last_message.content}")
                print(f"\n🔧 ACTION {step}:")
                print(f"   Tool: {last_message.tool_calls[0]['name']}")
                print(f"   Input: {last_message.tool_calls[0]['args']}")
            else:
                # This is the final answer
                print(f"\n✅ FINAL ANSWER:")
                print(f"   {last_message.content}")

        elif last_message.__class__.__name__ == "ToolMessage":
            # This is the OBSERVATION step
            print(f"\n👁️  OBSERVATION {step}:")
            # Parse and show a clean summary instead of raw JSON
            import json
            try:
                data = json.loads(last_message.content)
                if 'answer' in data:
                    print(f"   {data['answer']}")
                else:
                    print(f"   Search completed successfully")
            except:
                print(f"   {last_message.content[:200]}...")
            step += 1

print("\n" + "="*60)

