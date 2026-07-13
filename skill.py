import os
import json
from langchain.tools import tool
from langchain_groq import ChatGroq  # Qayta Groq importiga o'tamiz
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

# =====================================================================
# 🛠️ 1. ASBOBLAR (Tools)
# =====================================================================

@tool
def fetch_sales_data(year: int) -> str:
    """Ma'lumotlar bazasidan berilgan yil uchun savdo raqamlarini oladi."""
    mock_db = {
        2025: {"smartphones": 5000, "laptops": 2500, "headphones": 8000},
        2026: {"smartphones": 6200, "laptops": 1900, "headphones": 9500}
    }
    return json.dumps(mock_db.get(year, {}))

@tool
def calculate_growth(old_val: int, new_val: int) -> float:
    """Ikkita qiymat o'rtasidagi o'sish foizini hisoblaydi."""
    if old_val == 0: return 0.0
    return ((new_val - old_val) / old_val) * 100

# =====================================================================
# 🧠 2. KO'NIKMA (Skill)
# =====================================================================

class ReportGeneratorSkill:
    def __init__(self, llm_model):
        self.llm = llm_model
        self.tools = {
            "fetch_sales_data": fetch_sales_data,
            "calculate_growth": calculate_growth
        }
        
    def __call__(self, state: dict) -> dict:
        year = state.get("target_year", 2026)
        print(f"\n[Skill Boshlandi]: {year}-yil uchun tahliliy hisobot tayyorlanmoqda...")
        
        # .invoke() orqali to'g'ri chaqiramiz
        raw_data = self.tools["fetch_sales_data"].invoke({"year": year})
        past_data = self.tools["fetch_sales_data"].invoke({"year": year - 1})
        
        sales = json.loads(raw_data)
        prev_sales = json.loads(past_data)
        
        # Ikkinchi tool ham invoke yordamida
        smartphone_growth = self.tools["calculate_growth"].invoke({
            "old_val": prev_sales.get("smartphones", 0), 
            "new_val": sales.get("smartphones", 0)
        })
        
        # Groq uchun prompt
        prompt = ChatPromptTemplate.from_template(
            "Berilgan savdo natijalari va o'sish foizini tahlil qilib, "
            "rahbariyat uchun o'zbek tilida qisqa va lo'nda tahliliy hisobot yozib ber.\n\n"
            "Joriy yilgi savdo: {sales}\n"
            "Smartfonlar o'sishi: {growth:.2f}%"
        )
        
        chain = prompt | self.llm
        ai_response = chain.invoke({"sales": raw_data, "growth": smartphone_growth})
        
        return {
            "report_text": ai_response.content,
            "status": "COMPLETED"
        }

# =====================================================================
# 🚀 3. ISHGA TUSHIRISH (Groq Buluti orqali)
# =====================================================================

# Groq platformasidagi eng baquvvat mantiqiy model
groq_llm = ChatGroq(
    model="llama-3.1-8b-instant",  # Yoki loyihangizda ishlatayotgan boshqa faol Groq modeli
    temperature=0
)

# Skill obyektini yaratamiz
generate_report_node = ReportGeneratorSkill(llm_model=groq_llm)

# Davlat simulyatsiyasi
current_state = {"target_year": 2026}

# Ishga tushiramiz
updated_state = generate_report_node(current_state)

print("\n=== YAKUNIY HISOBOT (GROQ) ===")
print(updated_state["report_text"])