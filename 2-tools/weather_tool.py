from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from dotenv import load_dotenv
import os 
load_dotenv()

gemini_api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview",api_key = gemini_api_key)

@tool
def get_weather(location:str)->str:
    """get the weather of particular given loaction"""
    return f"The weather of {location} is sunny"
bind_tool = llm.bind_tools([get_weather])
result = bind_tool.invoke("what is  the weather of karachi",config = {"tool_choice": "auto"})


if result.tool_calls:
    tool_call = result.tool_calls[0]
    tool_output = get_weather.invoke(tool_call["args"])
    print(tool_output)
else:
    print(result.content)





# print("Content:", result.content)
# print("Tool Calls:", result.tool_calls)
