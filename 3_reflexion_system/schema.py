from pydantic import BaseModel, Field
from typing import List

class Reflection(BaseModel):
    missing: str = Field(description="Cretique for missing information")
    superfluous: str = Field(description="Cretique for superfluous information")

class AnswerQuestion(BaseModel):
    """Answer a question"""
    answer: str = Field(description="A ~250 worded detailed answer to the question")
    search_queries: List[str] = Field(description="2-4 search queries for researching improvements to address the cretique of your current answer.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")

class ReviseAnswer(AnswerQuestion):
    "Revise your previous answer using the new information available."
    references: List[str] = Field(description="Citations motivating your revised answer.")