from langchain_google_genai import ChatGoogleGenerativeAI
from dataclasses import dataclass
from langchain.agents import create_agent
from dotenv import load_dotenv
import os 
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

@dataclass
class ContactInfo:
    """Contact information for a person"""
    name:str
    age:int
    city:str

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.1-flash-lite-preview",
    google_api_key = api_key
)


agent = create_agent(
    model = llm,
    response_format = ContactInfo

)
response = agent.invoke({
    "messages": [
        {"role": "user", "content": "Ali is 24 years old and lives in Faisalabad"}
    ]
})
print(response)
