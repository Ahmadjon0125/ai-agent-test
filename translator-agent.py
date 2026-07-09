from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=1)

template = """Siz tarjimon agentisiz. Sizga berilgan matnni o'zbek tiliga tarjima qilishingiz kerak.
Matn: {text}
Tarjima: """

prompt = PromptTemplate(input_variables=["text"], template=template)
chain = prompt | llm

output = chain.invoke({"text": "namedonam"})
print(output)