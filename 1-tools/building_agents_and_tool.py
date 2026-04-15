from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from dotenv import load_dotenv
import os
load_dotenv()

gemini_api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview",api_key = gemini_api_key)

@tool
def get_weather(location:str)->str:
    """give the accurate weather condition of location."""
    print("give your location, where you want to about the weather condition.")
    return f"The weather of {location} is rainy today."
#step one
message = ("what will be the answer of this question 23/5")
ai_msg = llm.invoke(message)
message.append(ai_msg)
#step two excecute tools and collect result
for tool_call in ai_msg.tool_calls:
    tool_result =get_weather.invoke(tool_call)
    message.append(tool_result)
#step 3
response = llm.invoke(message)
print(response)

    