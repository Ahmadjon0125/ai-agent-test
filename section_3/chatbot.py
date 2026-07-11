import os
from typing import Dict, List, Any
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent  # Yangi standart agent yaratuvchisi
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

# 1. Model va qidiruv tizimini sozlash
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)


emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
persist_directory = os.path.join(BASE_DIR, "chroma_db")
if not os.path.exists(persist_directory):
    raise FileNotFoundError(f"Chroma bazasi topilmadi. Iltimos, 'db_build.py' skriptini ishga tushiring va bazani yarating: {persist_directory}")

vs = Chroma(
    embedding_function=emb, 
    collection_name="customer_db", 
    persist_directory=persist_directory
)

retriever = vs.as_retriever(search_kwargs={"k": 1})

@tool
def db_search(text: str) -> str:
    """Search the database for relevant information."""
    res = retriever.invoke(text)
    if not res:
        return "No relevant information found in the database."
    return "\n\n".join([doc.page_content for doc in res])

tools = [db_search]

# 2. Yangi standart bo'yicha tizim prompti va agentni yaratish
system_prompt = (
    "Siz mijozlarga xizmat ko'rsatish chatbotisiz, faqat mavjud manbalardan foydalanib javob bering. Agar topilmasa to'g'risini ayting. "
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
    session_id = "user_1"
    
    out = ask(session_id, "Salom, Mening ismim Ahmadjon")
    print(f"Bot: {out}\n" + "-"*50)
    
    out = ask(session_id, "Sizga mahsulotni qanday qaytratish mumkinligi haqida ma'lumot kerak.")
    print(f"Bot: {out}\n" + "-"*50)
    

