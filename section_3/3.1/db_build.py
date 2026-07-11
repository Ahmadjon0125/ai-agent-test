# from langchain_openai import OpenAIEmbeddings
# from langchain_chroma import Chroma
# from langchain_community.document_loaders import TextLoader
# from dotenv import load_dotenv
# import os
# load_dotenv()

# sources = [
#     TextLoader("data/refund_policy.txt", encoding="utf-8").load(),
#     TextLoader("data/support_info.txt", encoding="utf-8").load(),
#     TextLoader("data/working_hours.txt", encoding="utf-8").load()
# ]

# docs = [d for sub in sources for d in sub]

# emb = OpenAIEmbeddings(
#     model="google/gemma-4-31B-it", # Your model name
#     openai_api_base="http://192.168.100.54:8000/v1",   # Your server's API endpoint
#     openai_api_key="not-needed"
# )

# persist_directory = "./chroma_db"
# os.makedirs(persist_directory, exist_ok=True)
# vs = Chroma.from_documents(docs, emb, collection_name="customer_db", persist_directory=persist_directory)

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
# YANGI IMPORT:
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Skript qayerdan ishga tushsa ham data papkasini to'g'ri topishi uchun dinamik yo'l
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sources = [
    TextLoader(os.path.join(BASE_DIR, "data", "refund_policy.txt"), encoding="utf-8").load(),
    TextLoader(os.path.join(BASE_DIR, "data", "support_info.txt"), encoding="utf-8").load(),
    TextLoader(os.path.join(BASE_DIR, "data", "working_hours.txt"), encoding="utf-8").load()
]

docs = [d for sub in sources for d in sub]

# LOCAL EMBEDDING MODEL: Serverga so'rov yubormaydi, to'liq local ishlaydi
print("Embedding model yuklanmoqda (MiniLM)...")
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Bazani saqlash joyi
persist_directory = os.path.join(BASE_DIR, "chroma_db")
os.makedirs(persist_directory, exist_ok=True)

print("Vektor bazasi (Chroma) qurilmoqda...")
vs = Chroma.from_documents(
    documents=docs, 
    embedding=emb, 
    collection_name="customer_db", 
    persist_directory=persist_directory
)

print("🎉 Ajoyib! Chroma bazasi muvaffaqiyatli qurildi va saqlandi!")