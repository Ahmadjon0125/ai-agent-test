import os
from pydantic import BaseModel,Field
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent 
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

# 1. Model va qidiruv tizimini sozlash
# llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)
llm = ChatOpenAI(
    model="google/gemma-4-31B-it", # Your model name
    openai_api_base="http://192.168.100.54:8000/v1",   # Your server's API endpoint
    openai_api_key="not-needed",
    temperature=0,
    timeout=30,
    max_retries=1
)

class WriterOutput(BaseModel):
    code: str = Field(..., description="Generated Python code")
    explanation: str = Field(..., description="Explanation of the generated code")


# 2. Yangi standart bo'yicha tizim prompti va agentni yaratish
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Python coding assistant. Generate clean, efficient, correct code."),
    ("human", "Please write Python code for the following task:\n{task}")
])

writer_chain = writer_prompt | llm.with_structured_output(WriterOutput)

def writer_node(state):
    result: WriterOutput = writer_chain.invoke({"task": state["task"]})
    return {"code": result.code, "explanation": result.explanation}