import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_protocol import Any
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)

search_tool = DuckDuckGoSearchRun()

@tool
def multiply(a: Any, b: Any) -> int:
    """Multiply two integers."""
    # Kelgan qiymatlarni majburiy butun songa o'giramiz
    return int(a) * int(b)

tools = [search_tool, multiply]

prompt = ChatPromptTemplate.from_messages([
    ("system", "Siz ajoyib yordamchi agentisiz. ReAct sifatida fikr yuriting va faqat kerak bo'lganda vositalardan foydalaning. "
               "Agar sizda ma'lumot bo'lmasa yoki biron vosita yordam bera olmasa, savolni qaytarmang, "
               "balki foydalanuvchiga javob berishga harakat qiling. "),

    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent_runnable = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent = AgentExecutor(agent=agent_runnable, tools=tools)

if __name__ == "__main__":
    out = agent.invoke({"input": "Salom , Mening ismim Ahmadjon"})
    print(out['output'])
    
    out = agent.invoke({"input": "Bugun qaysi kun?, bugungi sana kerak, yil oy kun formatida,"})
    print(out['output'])

    out = agent.invoke({"input": "5 va 3 ni ko'paytirganda javobi nima bo'ladi?"})
    print(out['output'])

    out = agent.invoke({"input": " Men kimman?"})
    print(out['output'])