from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY") 

gemini_model = ChatGoogleGenerativeAI(
    google_api_key = api_key,
    model = "gemini-3.1-flash-lite-preview",
    temperature = 0.4
)

input_prompt = input("Enter a topic about fun fact: ")

dynamic_prompt = PromptTemplate.from_template("write a fun fact about {topic}")
final_prompt = dynamic_prompt.format(topic=input_prompt)
result = gemini_model.invoke(final_prompt)
print(result.content)
