from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.messages import HumanMessage
from langchain.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key
)


@tool
def get_hotel_info(city: str) -> str:
    """Get hotel recommendations for a given city"""
    hotels = {
        "faisalabad": ["Hotel One Faisalabad", "Serena Hotel Faisalabad"],
        "lahore": ["Pearl Continental Lahore", "Avari Lahore"],
        "islamabad": ["Serena Islamabad", "Marriott Islamabad"]
    }

    city = city.lower()

    if city in hotels:
        return f"Hotels in {city}: {', '.join(hotels[city])}"
    return "No hotel data available."


agents = create_agent(
    model=llm,
    tools=[get_hotel_info],
    checkpointer=MemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model=llm,
            trigger=("tokens", 20),
            keep=("tokens", 10)
        )
    ]
)

config = {"configurable": {"thread_id": "test_1"}}

questions = [
    "what is the answer of 2*3?",
    "what is he answer of 5/3?",
    "what is the capital of United Kingdom?",
    "what is the average salary in Pakistan?",
    "suggest hotels in Faisalabad"
]

for q in questions:
    response = agents.invoke(
        {"messages": [HumanMessage(content=q)]},
        config
    )
    print(response)
    print(len(response["messages"]))