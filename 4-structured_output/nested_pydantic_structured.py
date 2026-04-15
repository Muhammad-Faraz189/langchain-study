from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel,Field
from dotenv import load_dotenv
import os 
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")



class actor(BaseModel):
    name:str
    role:str


class movie(BaseModel):
    """Give me output in that format"""
    title:str=Field(description="Name of the movie.")
    cast:list[actor]       # nested 
    year:int=Field(description="thw year when movie was released.")
    rating:float=Field(description="the rating of the movie.")
    director:str=Field(description="the director of this movie.")


        

model = ChatGoogleGenerativeAI(
    model = "gemini-3.1-flash-lite-preview",
    api_key=api_key

)

model_output = model.with_structured_output(movie)
response =model_output.invoke("tell me about inception")
print(response)
