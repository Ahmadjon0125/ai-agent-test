from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.4)

# llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.4)

post_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional LinkedIn content creator specializing in technology."),
    ("human",
     "Write a clear, engaging LinkedIn post about this topic: \n {topic}")
])
post_chain = post_prompt | llm


def post_node(state):
    response = post_chain.invoke({"topic": state['topic']})
    return {"post_text": response.content}