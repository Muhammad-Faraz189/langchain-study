from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict, Annotated
from dotenv import load_dotenv
import os  

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key
)

class Movie(TypedDict):
    """About movie detail"""
    title: Annotated[str, ..., "The title of the movie"]
    year: Annotated[int, ..., "Which year when the movie released"]
    director: Annotated[str, ..., "The director of the movie"]
    rating: Annotated[float, ..., "The movie rating out of 10"]

model_output = llm.with_structured_output(Movie)

response = model_output.invoke("tell me about inception")
print(response)