import os
import json
import urllib.parse
import urllib.request
from typing import Dict, List, Any
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent  # Yangi standart agent yaratuvchisi
from dotenv import load_dotenv
load_dotenv()

# 1. Modelni sozlash
llm = ChatOpenAI(
    model="google/gemma-4-31B-it", # Your model name
    openai_api_base="http://192.168.100.54:8000/v1",   # Your server's API endpoint
    openai_api_key="not-needed",
    temperature=0,
    timeout=30,
    max_retries=1
)

# --- 1-ASBOB: Valyuta konvertatsiya qilish funksiyasi ---
def _fx_convert_impl(amount: float, base: str, target: str) -> dict:
    base_up = base.upper()
    target_up = target.upper()
    url = f"https://er-api.com{base_up}"

    with urllib.request.urlopen(url, timeout=10) as response:
        data = json.loads(response.read())

    rate = data["rates"].get(target_up)
    result = amount * rate 
    return {
            "amount": amount,
            "base": base_up,
            "target": target_up,
            "result": result,
            "rate": rate,
            "date": data["time_last_update_utc"]
        }


@tool
def fx_convert(amount: float, base: str, target: str) -> str:
    """Use this tool ALWAYS when the user wants to convert currencies, calculate exchange rates, or mentions money (e.g., USD, EUR, UZS)."""
    try:
        res = _fx_convert_impl(float(amount), base, target)
        return (
            f"{res['amount']} {res['base']} = {res['result']} {res['target']} "
            f"(rate: {res['rate']}, date: {res['date']})"
        )
    except Exception as e:
        return f"Conversion failed: {e}"


# --- 2-ASBOB: Xavfsiz maxsus Wikipedia asbobi (User-Agent bilan) ---
@tool
def custom_wikipedia_search(query: str) -> str:
    """Use this tool to search Wikipedia for travel information, city guides, museums, history, landmarks, or general knowledge."""
    try:
        # Wikipedia API qidiruv manzili
        encoded_query = urllib.parse.quote(query)
        search_url = f"https://wikipedia.org{encoded_query}&format=json"
        
        # Bloklanib qolmaslik uchun so'rovga sarlavha qo'shamiz
        req = urllib.request.Request(
            search_url, 
            headers={'User-Agent': 'MyLangChainAgent/1.0 (ahmadjon@example.com)'}
        )
        
        with urllib.request.urlopen(req, timeout=10) as response:
            search_data = json.loads(response.read())
            
        search_results = search_data.get("query", {}).get("search", [])
        if not search_results:
            return f"Wikipedia could not find any results for '{query}'."
            
        # Topilgan eng birinchi yaxshi natijaning snippet (qisqa matn) qismini olamiz
        output = []
        for i, res in enumerate(search_results[:2]):
            clean_snippet = res['snippet'].replace('<span class="searchmatch">', '').replace('</span>', '')
            output.append(f"Result {i+1} [{res['title']}]: {clean_snippet}")
            
        return "\n".join(output)
    except Exception as e:
        return f"Wikipedia search failed: {e}. Please answer using your general knowledge."


# --- HAMMA ASBOBLARNI BITTA RO'YXATGA JAMLAYMIZ ---
tools = [fx_convert, custom_wikipedia_search]

# --- YAGONA VA KUCHLI REACT AGENT PROMPTI ---
system_prompt = (
    "You are an expert multi-functional AI Assistant equipped with specialized tools. "
    "1. Use `fx_convert` for any currency exchange, conversion, or money math queries. "
    "2. Use `custom_wikipedia_search` for travel information, city guides, museums, history, or landmarks. "
    "3. You MUST remember previous conversations, user context, and user names. "
    "Be concise, direct, and factual."
)

# Yagona agentni yaratamiz
agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)


# --- XOTIRA TIZIMI ---
_memory_store: Dict[str, List[dict]] = {}

def get_session_history(session_id: str) -> List[dict]:
    if session_id not in _memory_store:
        _memory_store[session_id] = []
    return _memory_store[session_id]

def ask(session_id: str, text: str) -> str:
    history = get_session_history(session_id)
    
    current_messages = history + [{"role": "user", "content": text}]
    
    # Agentni chaqiramiz
    result = agent.invoke({"messages": current_messages})
    final_output = result["messages"][-1].content
    
    # Xotirani yangilaymiz
    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": final_output})

    
    return final_output


if __name__ == "__main__":
    session_id = "user_1"

    print("Foydalanuvchi: Parijdagi qaysi muzeyni tavsiya qilasiz?")
    print(f"Bot: {ask(session_id, 'Parijdagi qaysi muzeyni tavsiya qilasiz?')}\n" + "-"*50)
    
    print("Foydalanuvchi: Sayohat uchun 500 USD bor, EUR da qancha bo'ladi?")
    print(f"Bot: {ask(session_id, "Sayohat uchun 500 USD bor, EUR da qancha bo'ladi?")}\n" + "-"*50)
