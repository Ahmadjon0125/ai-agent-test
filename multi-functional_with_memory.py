import os
from typing import Dict, List, Any
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent  # Yangi standart agent yaratuvchisi
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
load_dotenv()

# 1. Model va qidiruv tizimini sozlash
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)
search_tool = DuckDuckGoSearchRun()

@tool
def multiply(a, b):
    """Multiply two integers."""
    return int(a) * int(b)

tools = [search_tool, multiply]

# 2. Yangi standart bo'yicha tizim prompti va agentni yaratish
system_prompt = (
    "Siz ajoyib yordamchi agentisiz. ReAct sifatida fikr yuriting va faqat kerak bo'lganda vositalardan foydalaning. "
               "Agar sizda ma'lumot bo'lmasa yoki biron vosita yordam bera olmasa, savolni qaytarmang, "
               "balki foydalanuvchiga javob berishga harakat qiling. "
)
agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)

# 3. Har bir sessiya uchun suhbat tarixini saqlovchi xotira ombori
_memory_store: Dict[str, List[dict]] = {}

def get_session_history(session_id: str) -> List[dict]:
    if session_id not in _memory_store:
        _memory_store[session_id] = []
    return _memory_store[session_id]

def ask(session_id: str, text: str) -> str:
    # Sessiyaga tegishli eski suhbatlarni yuklaymiz
    history = get_session_history(session_id)
    
    # Yangi xabarni umumiy tarix ketma-ketligiga qo'shamiz
    current_messages = history + [{"role": "user", "content": text}]
    
    # Agentni yangi xabarlar oqimi bilan chaqiramiz
    result = agent.invoke({"messages": current_messages})
    
    # Model qaytargan yakuniy javob matni
    final_output = result["messages"][-1].content
    
    # Tarixni kelgusi so'rovlar uchun yangilab qo'yamiz
    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": final_output})
    
    return final_output


if __name__ == "__main__":
    print("Ko'p funksiyali xotirali agent ishga tushdi...\n")
    session_id = "user_1"
    
    out = ask(session_id, "Salom, Mening ismim Ahmadjon")
    print(f"Bot: {out}\n" + "-"*50)
    
    out = ask(session_id, "O'zbekistonda nechta viloyat bor?")
    print(f"Bot: {out}\n" + "-"*50)
    
    out = ask(session_id, "5 va 3 ni ko'paytirganda javobi nima bo'ladi?")
    print(f"Bot: {out}\n" + "-"*50)
    
    out = ask(session_id, "Mening ismim nima?")
    print(f"Bot: {out}\n" + "-"*50)
