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
tools = [search_tool]

prompt = ChatPromptTemplate.from_messages([
    ("system", "Siz ajoyib yordamchi agentisiz. Sizga berilgan vazifani bajarish uchun kerakli vositalardan foydalanishingiz mumkin."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent_runnable = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent = AgentExecutor(agent=agent_runnable, tools=tools, verbose=True)

if __name__ == "__main__":
    out = agent.invoke({"input": "Bugun qaysi kun?, bugungi sana kerak, yil oy kun formatida,"})
    print(out['output'])