from typing import TypedDict, Literal, Annotated, Union
from langgraph.graph import StateGraph, START, END

from langchain_core.agents import AgentAction, AgentFinish
import operator

class AgentState(TypedDict):
    input: str
    agent_output: Union[AgentAction, AgentFinish]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]