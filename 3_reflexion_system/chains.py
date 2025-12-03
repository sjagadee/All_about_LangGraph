from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.messages import HumanMessage
from schema import AnswerQuestion
from dotenv import load_dotenv
import datetime
import os

load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "Reflexion Pattern"

generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """You are an expert AI researcher.\n
Current Time: {time}\n\n
1. {first_instruction}\n
2. Reflect and cretique your answer. Be sure to maximize improvement.\n
3. After reflection, come up with **list of 2-4 search queries seperately** for researching improvements. 
Do ot include these questions as part of the reflection cretique.\n"""
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system",
         "Answer the user's question using the above required format. ")
    ]
).partial(time=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

first_responser_prompt_template = generation_prompt.partial(first_instruction="Generate a ~250 worded detailed answer to the question.")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.7)
pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion])

first_responser_chain = first_responser_prompt_template | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion") | pydantic_parser

response = first_responser_chain.invoke({
    "messages": [
        HumanMessage(content="Write a blog post about how small businesses can leverage AI to grow their business.")
    ]    
})

print(response)