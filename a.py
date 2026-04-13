from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
load_dotenv()
api_key=os.getenv("GOOGLE_API_KEY")

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview",google_api_key=api_key)

# Send a simple prompt
response = llm.invoke([
    HumanMessage(content="Explain Langchain and whats benefits of langchain?")
])

with open("aireposnse.txt", "w") as file:
    
    file.write(str(response))

print("File written successfully")