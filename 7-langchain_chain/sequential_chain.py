from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")


prompt1 = PromptTemplate(
    template = 'Generate a detailed report on this {topic}',
    input_variable = ['topic']
)

prompt2 = PromptTemplate(
    template = "Generate a 5 pointer summary on a following text\n {text}",
    input_variable = ["text"]
)


gemini_model = ChatGoogleGenerativeAI(
    google_api_key = api_key,
    model = "gemini-3.1-flash-lite-preview",
    temperature = 0.8


)

parser = StrOutputParser()

chains = prompt1 | gemini_model | parser | prompt2 | gemini_model | parser

result =  chains.invoke({'topic':'development in pakistan'})

print(result)

chains.get_graph().print_ascii()