from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langsmith import Client
from dotenv import load_dotenv
import datetime

load_dotenv()

# Initialize the LLM (Language Model)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.7
)

# Create a search tool - this allows the agent to search the web
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

# Pull the prompt from LangChain Hub
# This is the standard ReAct chat prompt from Harrison Chase
client = Client()
prompt = client.pull_prompt("hwchase17/react-chat")

# Extract the template string from the prompt object
# The prompt is a PromptTemplate, we need to get the actual string template
system_prompt = prompt.template

# Create the ReAct agent using the hub prompt
react_agent = create_agent(
    llm,
    tools,
    system_prompt=system_prompt
)

# Invoke the agent with a user question
print("\n" + "="*60)
print("REACT AGENT WITH HUB PROMPT")
print("="*60 + "\n")

result = react_agent.invoke({
    "messages": [{"role": "user", "content": "when did spacex launch happen? and how many days has been since the launch?"}]
})

print("\nAgent Response:")
# Print all messages in the conversation
for message in result["messages"]:
    msg_type = message.__class__.__name__
    if msg_type == "HumanMessage":
        print(f"\n👤 USER: {message.content}")
    elif msg_type == "AIMessage":
        if hasattr(message, 'tool_calls') and message.tool_calls:
            print(f"\n🤖 AI (calling tool): {message.tool_calls[0]['name']}")
        else:
            print(f"\n🤖 AI: {message.content}")
    elif msg_type == "ToolMessage":
        print(f"\n🔧 TOOL: {message.content[:200]}...")

print("\n" + "="*60)
# Stream the agent's execution
# response = react_agent.stream(
#     {"messages": [{"role": "user", "content": "When did spacex launch happen? and how many days has been since the launch?"}]},
#     stream_mode="values"
# )

# step = 1
# for chunk in response:
#     # Print each step of the agent's reasoning with ReAct labels
#     if "messages" in chunk:
#         last_message = chunk["messages"][-1]

#         # Label each type of message according to ReAct pattern
#         if last_message.__class__.__name__ == "HumanMessage":
#             print(f"\n📝 QUESTION:")
#             print(f"   {last_message.content}")

#         elif last_message.__class__.__name__ == "AIMessage":
#             if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
#                 # This is the ACTION step
#                 print(f"\n💡 THOUGHT {step}:")
#                 if last_message.content:
#                     print(f"   {last_message.content}")
#                 print(f"\n🔧 ACTION {step}:")
#                 print(f"   Tool: {last_message.tool_calls[0]['name']}")
#                 print(f"   Input: {last_message.tool_calls[0]['args']}")
#             else:
#                 # This is the final answer
#                 print(f"\n✅ FINAL ANSWER:")
#                 print(f"   {last_message.content}")

#         elif last_message.__class__.__name__ == "ToolMessage":
#             # This is the OBSERVATION step
#             print(f"\n👁️  OBSERVATION {step}:")
#             # Parse and show a clean summary instead of raw JSON
#             import json
#             try:
#                 data = json.loads(last_message.content)
#                 if 'answer' in data:
#                     print(f"   {data['answer']}")
#                 else:
#                     print(f"   Search completed successfully")
#             except:
#                 print(f"   {last_message.content[:200]}...")
#             step += 1

# print("\n" + "="*60)
