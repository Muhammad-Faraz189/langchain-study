from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

gemini_model = ChatGoogleGenerativeAI(
    google_api_key = api_key,
    model = "gemini-3.1-flash-lite-preview",
    temperature = 0.8


)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['postive','negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template = 'classify the sentiment of the following feedback  text into positive and negative\n {feedback}\n {format_instruction}',
    input_variables = ['feedback'],
    partial_variables= {'format_instruction': parser2.get_format_instructions()} 
)

classifier_chain = prompt1 | gemini_model | parser2

prompt2 = PromptTemplate(
    template='write an appropriate response to this  positive feedback\n {feedback}',
    input_variables= ['feedback']
)


prompt3 = PromptTemplate(
    template='write an appropriate response to this  negative feedback\n {feedback}',
    input_variables= ['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment=="positive", prompt1 | gemini_model | parser),
    (lambda x:x.sentiment=="negative", prompt2 | gemini_model | parser),
    RunnableLambda(lambda x:'could not find sentiment')
)

merge_chain = classifier_chain | branch_chain

result = merge_chain.invoke({'feedback': 'This  is a terrible phone'})

print(result)