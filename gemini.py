from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os 
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
gemini_model = ChatGoogleGenerativeAI(
    model = "gemini-3.1-flash-lite-preview",
    temperature = 0.4 ,
    api_key = api_key
)
query = "write a documents  on langchain"
for chunk in gemini_model.stream(query):
    print(chunk.text,end="",flush=True)


