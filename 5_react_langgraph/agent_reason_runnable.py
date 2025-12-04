from langchain.chat_models import init_chat_model
from langchain.agents import tool, create_agent
from langchain_tavily import TavilySearch
from langsmith import Client

from dotenv import load_dotenv
import datetime
import os

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "ReAct Agent LangGraph"

llm = init_chat_model(model="gpt-4.1-minimum", temperature=0.7)


@tool
def get_system_time():
    """This function returns the current system time in the format "YYYY-MM-DD HH:MM:SS" """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

search_tool = TavilySearch(
    max_results=3,
    include_answer=True,
    include_raw_content=False
)

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