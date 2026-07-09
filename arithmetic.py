import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from dotenv import load_dotenv
from langchain_protocol import Any
load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)

# @tool
# def multiply(a: int, b: int) -> int:
#     """Multiply two integers."""
#     return a * b

@tool
def multiply(a: Any, b: Any) -> int:
    """Multiply two integers."""
    # Kelgan qiymatlarni majburiy butun songa o'giramiz
    return int(a) * int(b)

tools = [multiply]

prompt = ChatPromptTemplate.from_messages([
    ("system", "Siz ajoyib yordamchi agentisiz. Sizga berilgan vazifani bajarish uchun kerakli vositalardan foydalanishingiz mumkin."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent_runnable = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent = AgentExecutor(agent=agent_runnable, tools=tools, verbose=True)

if __name__ == "__main__":
    out = agent.invoke({"input": "5 va 3 ni ko'paytirganda javobi nima bo'ladi?"})
    print(out['output'])