from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain.messages import HumanMessage


llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

message = HumanMessage(
    content=[
        {"type": "text", "text": "rasmni tushuntir. o'zbek tilida va 100 so'zdan oshmasin."},
        {
            "type": "image_url",
            "image_url": {"url": "https://media2.dev.to/dynamic/image/width=1280,height=720,fit=cover,gravity=auto,format=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fgxlqa8x598fcl888zcnm.jpg"},
        },
    ]
)

response = llm.invoke([message])
print(response.content)


