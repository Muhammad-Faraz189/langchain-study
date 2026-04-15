from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv
import os 
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.1-flash-lite-preview",
    google_api_key = api_key

)

tavily_search_tool = TavilySearch(
    max_results=5,
    topic="general"
    
)

@tool("calculator", description="Performs arithmetic calculations. Use this for any math problems.")
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))

agents = create_agent(
    model =llm,
    tools = [tavily_search_tool,calc]
)

query = "what is the best langchain learning platform on internet"

for chunk in tavily_search_tool.stream(query):
    print(chunk,end="")[-1].pretty_print()