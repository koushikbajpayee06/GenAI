from langchain_openai import OpenAI
from dotenv import load_dotenv
load_dotenv()   

llm = OpenAI(model='gpt-3.5-turbo-instruct')

result = llm.invoke("who has higher cAhNCES TO  win todays ipl match")
print(result)   