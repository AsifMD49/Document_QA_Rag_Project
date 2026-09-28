from dotenv import load_dotenv
import os

load_dotenv()

print("API key exists:", bool(os.getenv("GROQ_API_KEY")))

print(os.getenv("GROQ_API_KEY"))
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)

print("ChatGroq initialized successfully")

response = llm.invoke("Say hello in one sentence.")

print(response.content)
