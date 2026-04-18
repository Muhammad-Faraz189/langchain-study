from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage,HumanMessage
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY") 

gemini_model = ChatGoogleGenerativeAI(
    google_api_key = api_key,
    model = "gemini-3.1-flash-lite-preview"



)

my_messages = [
    AIMessage(content="you are gen-z assistant,who always answer in fun way"),
    HumanMessage(content="bro! tell me a fun fact")
]

result = gemini_model.invoke(my_messages)
print(result.content)