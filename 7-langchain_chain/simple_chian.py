from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY") 

prompt = PromptTemplate(
    template= 'generate 5 lines about this {topic}',
    input_variables= ['topic']
)

gemini_model = ChatGoogleGenerativeAI(
    google_api_key = api_key,
    model = "gemini-3.1-flash-lite-preview",
    temperature = 0.4
)

parser = StrOutputParser()

chains = prompt | gemini_model | parser

result =chains.invoke({'topic': 'cricket'})

print(result)

chains.get_graph().print_ascii()

