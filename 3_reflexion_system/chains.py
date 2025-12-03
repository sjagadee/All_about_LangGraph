from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.messages import HumanMessage
from schema import AnswerQuestion, ReviseAnswer
from dotenv import load_dotenv
import datetime
import os

load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "Reflexion Pattern"
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.7)
pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",
         """You are an expert AI researcher.\n
Current Time: {time}\n\n
1. {first_instruction}\n
2. Reflect and critique your answer. Be sure to maximize improvement.\n
3. After reflection, come up with **list of 2-4 search queries separately** for researching improvements. 
Do not include these questions as part of the reflection critique.\n"""
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system",
         "Answer the user's question using the above required format. ")
    ]
).partial(time=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# initial responder section

first_responser_prompt_template = actor_prompt_template.partial(first_instruction="Generate a ~250 worded detailed answer to the question.")

# revisor section

revise_instruction = """
Revise the previous answer using the new information available.\n
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

revisor_prompt_template = actor_prompt_template.partial(first_instruction=revise_instruction)

first_responser_chain = first_responser_prompt_template | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")

revise_responser_chain = revisor_prompt_template | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer") 

# response = first_responser_chain.invoke({
#     "messages": [
#         HumanMessage(content="Write a blog post about how small businesses can leverage AI to grow their business.")
#     ]    
# })

# print(response)