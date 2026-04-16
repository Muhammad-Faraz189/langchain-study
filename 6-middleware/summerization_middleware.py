from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.messages import HumanMessage,AIMessage
from dotenv import load_dotenv
import os 
load_dotenv()


api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.1-flash-lite-preview",
    google_api_key = api_key
)

agents = create_agent(
    model = llm,
    checkpointer = MemorySaver(),
    middleware =[
        SummarizationMiddleware(
            model = llm,
            trigger = ("tokens",20),
            keep = ("tokens",10)
        )
        
    ]

)

config = {"configurable": {"thread_id": "test_1"}}

questions = [
    "what is the answer of 2*3?",
    "what is he answer of 5/3?",
    "what is  the capital of United Kingdom?",
    "what is the average sallary in Pakistan?"
]

for q in questions:
    response= agents.invoke({"messages":[HumanMessage(content =q )]},config)
    print(f"Messages:{response}")
    print(f"Messages: {len(response['messages'])}")
