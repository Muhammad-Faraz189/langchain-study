from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage,HumanMessage
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model = "gemini-3-flash-preview",
    google_api_key = api_key

)

agent = create_agent(
    model = llm,
    checkpointer=InMemorySaver()


)

#Run with thread_id
config = {"configurable":{"thread_id":"test_1"}}


response = agent.invoke({"messages":[HumanMessage(content="Hi my name is  Faraz")]},config=config)

# response = agent.invoke({"messages":[HumanMessage(content="what is my naeme")]},congif=config)
print(response)