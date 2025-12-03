from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a twitter techie influence assistant, tasked with writing excellent twitter posts. "
         "Generate the best possible twitter post based on the user's request. "
         "If the user provides a cretique, respond with a revised version of the post, that improves the post. "
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a viral twitter influencer, who grade's the tweets. "
         "Generate cretiques and recommendations for the user's post. "
         "Always proide a detailed recommendation, including requests for length, virality, engagement, and style etc. "
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.7)

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm